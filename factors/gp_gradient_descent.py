"""Gradient-descent refinement for GP factor trees.

This module keeps the classic GP path untouched unless explicitly enabled.
It optimizes continuous parameters attached to a fixed GP tree:
- edge weights (shared by operator type for GPGD, independent per edge for OPGD),
- numeric constant leaves,
- time-series window parameters through a differentiable soft window blend.

The optimized tree is converted back to ordinary ``FactorNode`` formulas by
materializing edge weights as ``Mul(ConstNode(w), child)`` and rounded windows
as integer window arguments. No soft operator name is persisted.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
import traceback
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, cast

import numpy as np
import pandas as pd

from utils.logging import log
from utils.params import FEE
from .gp_gradient_descent_config import validate_gradient_descent_fitness_indicators
from .factor_ops import (
    BINARY_CHILD_OPS,
    BINARY_TS_OPS,
    TERNARY_CHILD_OPS,
    UNARY_CHILD_OPS,
    UNARY_TS_OPS,
    ConstNode,
    DataNode,
    FactorDataType,
    FactorNode,
    OpAbs,
    OpAdd,
    OpAmihud,
    OpBias,
    OpBodyRatio,
    OpDelta,
    OpDiv,
    OpEma,
    OpGt,
    OpIfElse,
    OpInv,
    OpLogReturn,
    OpLowerShadowRatio,
    OpLt,
    OpMaRibbon,
    OpMax,
    OpMin,
    OpMul,
    OpNanMean,
    OpNeg,
    OpOiTrendConviction,
    OpPriceAcceleration,
    OpRangePosition,
    OpReturn,
    OpRollNorm,
    OpSig,
    OpSign,
    OpSqrtAbs,
    OpStochasticK,
    OpSub,
    OpTsArgmax,
    OpTsArgmin,
    OpTsBeta,
    OpTsCorr,
    OpTsCov,
    OpTsDecayExp,
    OpTsDelay,
    OpTsDelta,
    OpTsEntropy,
    OpTsKurt,
    OpTsMax,
    OpTsMean,
    OpTsMin,
    OpTsPctDelta,
    OpTsRank,
    OpTsRankCorr,
    OpTsResidual,
    OpTsSkew,
    OpTsStd,
    OpTsSum,
    OpTsTimeWeightedMean,
    OpTrueAmplitude,
    OpTurnoverShock,
    OpTypicalPrice,
    OpUpperShadowRatio,
    OpVolatility,
    OpVolumeStd,
    OpVolumeZScore,
    OpVpDivergence,
    infer_node_type,
)

try:  # Torch is an optional runtime dependency for the legacy GP path.
    import torch
except Exception:  # pragma: no cover - exercised only when torch missing.
    torch = None  # type: ignore

_TorchModuleBase = torch.nn.Module if torch is not None else object
_UNARY_TS_OPS_TUPLE: Tuple[Type[Any], ...] = tuple(UNARY_TS_OPS)
_BINARY_TS_OPS_TUPLE: Tuple[Type[Any], ...] = tuple(BINARY_TS_OPS)
_TERNARY_CHILD_OPS_TUPLE: Tuple[Type[Any], ...] = tuple(TERNARY_CHILD_OPS)
_GD_RUN_COUNTER = itertools.count(1)


@dataclass
class GradientDescentConfig:
    """Configuration for one GP tree gradient-descent refinement.

    Attributes
    ----------
    enable_gradient_descent:
        Master switch. False keeps the legacy GP behavior exactly unchanged.
    gradient_descent_method:
        ``alternated`` means GP evolves one generation then GD refines individuals
        before elite-archive comparison. ``consecutive`` means GD runs only on the
        final elite archive after all GP generations.
    generation_per_gradient_descent:
        Run GD every N generations in alternated mode. ``1`` means every generation.
    gradient_descent_steps:
        Maximum optimizer steps per GD call.
    parametric_method:
        ``opgd`` gives each operator edge/window its own parameter. ``gpgd`` shares
        parameters among the same operator class and child edge name.
    gradient_descent_optimizer:
        Torch optimizer name. Supported: adam, adamw, sgd, rmsprop, adagrad.
    learning_rate:
        Optimizer learning rate.
    early_stopping_steps:
        Stop GD when surrogate fitness does not improve for this many steps.
    gradient_clip_norm:
        Optional max gradient norm for numerical stability.
    soft_temperature:
        Temperature for soft max/min/rank and soft window blending.
    window_soft_temperature:
        Independent temperature for differentiable window blending. It is kept
        lower than operator softmax temperature by default so window parameters
        can receive usable gradients instead of being locked at the initial
        discrete choice.
    window_neighbor_radius:
        Add integer windows around the original window to the candidate set.
        This makes windows such as 19/21 reachable even when GP initialization
        only samples coarse choices like 10/20/30.
    min_window / max_window / window_choices:
        Bounds/candidates used when optimizing discrete time-series windows.
    device:
        Torch device. ``None`` selects cuda when available, otherwise cpu.
        MPS is not selected automatically because rolling-window backward relies
        on operators such as ``aten::unfold_backward`` that are not implemented
        on MPS in current PyTorch releases.
    log_progress:
        Whether to emit sampled progress logs for GD refinement. Logs are
        sampled by run id to avoid flooding when alternated mode refines
        hundreds of individuals each generation.
    progress_log_first_n_runs / progress_log_run_interval:
        Always log the first N GD calls, then log every M-th call.
    progress_log_step_interval:
        Within a logged GD call, print step metrics every N optimizer steps.
    """

    enable_gradient_descent: bool = False
    gradient_descent_method: str = 'alternated'
    generation_per_gradient_descent: int = 1
    gradient_descent_steps: int = 100
    parametric_method: str = 'opgd'
    gradient_descent_optimizer: str = 'adam'
    learning_rate: float = 0.05
    early_stopping_steps: int = 20
    gradient_clip_norm: float = 1.0
    soft_temperature: float = 10.0
    window_soft_temperature: float = 2.0
    window_neighbor_radius: int = 2
    min_window: int = 2
    max_window: int = 60
    window_choices: Optional[Sequence[int]] = None
    device: Optional[str] = None
    log_progress: bool = True
    progress_log_first_n_runs: int = 3
    progress_log_run_interval: int = 50
    progress_log_step_interval: int = 5

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'GradientDescentConfig':
        field_names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in kwargs.items() if k in field_names})
        cfg.gradient_descent_method = str(cfg.gradient_descent_method or 'alternated').lower()
        cfg.parametric_method = str(cfg.parametric_method or 'opgd').lower()
        cfg.gradient_descent_optimizer = str(cfg.gradient_descent_optimizer or 'adam').lower()
        if cfg.gradient_descent_method not in {'alternated', 'consecutive'}:
            raise ValueError('gradient_descent_method must be alternated or consecutive.')
        if cfg.parametric_method not in {'gpgd', 'opgd'}:
            raise ValueError('parametric_method must be gpgd or opgd.')
        cfg.generation_per_gradient_descent = max(1, int(cfg.generation_per_gradient_descent))
        cfg.gradient_descent_steps = max(0, int(cfg.gradient_descent_steps))
        cfg.early_stopping_steps = max(0, int(cfg.early_stopping_steps))
        cfg.learning_rate = float(cfg.learning_rate)
        cfg.gradient_clip_norm = float(cfg.gradient_clip_norm)
        cfg.soft_temperature = max(0.1, float(cfg.soft_temperature))
        cfg.window_soft_temperature = max(0.1, float(cfg.window_soft_temperature))
        cfg.window_neighbor_radius = max(0, int(cfg.window_neighbor_radius))
        cfg.min_window = max(1, int(cfg.min_window))
        cfg.max_window = max(cfg.min_window, int(cfg.max_window))
        cfg.log_progress = bool(cfg.log_progress)
        cfg.progress_log_first_n_runs = max(0, int(cfg.progress_log_first_n_runs))
        cfg.progress_log_run_interval = max(1, int(cfg.progress_log_run_interval))
        cfg.progress_log_step_interval = max(1, int(cfg.progress_log_step_interval))
        return cfg


def _torch_device(device: Optional[str]):
    if torch is None:
        raise ImportError('enable_gradient_descent=True requires PyTorch, but torch is not importable.')
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _is_mps_backward_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        'mps' in msg
        and (
            'unfold_backward' in msg
            or 'not currently implemented for the mps device' in msg
            or 'not implemented for the mps device' in msg
        )
    )


def _all_gradients_finite(params: Iterable[Any]) -> bool:
    for param in params:
        grad = getattr(param, 'grad', None)
        if grad is not None and not bool(torch.isfinite(grad).all().detach().cpu().item()):
            return False
    return True
def _gradient_norm_and_status(params: Iterable[Any]) -> Tuple[float, int, bool]:
    """Return global L2 grad norm, number of tensors with grad, and finite flag."""
    total_sq = 0.0
    grad_tensor_count = 0
    finite = True
    for param in params:
        grad = getattr(param, 'grad', None)
        if grad is None:
            continue
        grad_tensor_count += 1
        if not bool(torch.isfinite(grad).all().detach().cpu().item()):
            finite = False
            continue
        total_sq += float((grad.detach() * grad.detach()).sum().cpu().item())
    return math.sqrt(max(total_sq, 0.0)), grad_tensor_count, finite


def _max_parameter_delta(model: Any, initial_params: Dict[str, Any]) -> float:
    max_delta = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in initial_params:
                continue
            delta = torch.max(torch.abs(param.detach() - initial_params[name])).detach().cpu().item()
            max_delta = max(max_delta, float(delta))
    return max_delta


def _safe_tensor(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _soft_abs(x, k: float):
    return x * torch.tanh(k * x)


def _safe_div(a, b, eps: float = 1e-6):
    a = _safe_tensor(a)
    b = _safe_tensor(b)
    eps_tensor = torch.full_like(b, float(eps))
    signed_eps = torch.where(b < 0, -eps_tensor, eps_tensor)
    safe_b = torch.where(torch.abs(b) < float(eps), signed_eps, b)
    return a / safe_b


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _pearson_corr(x, y, eps: float = 1e-6, min_std: float = 1e-4):
    """Differentiable Pearson correlation with minimum-std guard.

    The minimum-std clamp prevents gradient explosion in the backward pass
    when the factor has near-zero variance (e.g. bounded operators like
    StochasticK produce nearly constant slices).  Without this guard the
    gradient of 1/(std_x * std_y) blows up when either std ≈ 0.
    """
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.detach().sum().item()) < 2:
        return x.sum() * 0.0
    xv = x[mask]
    yv = y[mask]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    cov = (xv * yv).mean()
    x_std = torch.sqrt((xv * xv).mean() + 1e-12)
    y_std = torch.sqrt((yv * yv).mean() + 1e-12)
    _min = torch.tensor(float(min_std), dtype=x.dtype, device=x.device)
    x_std = torch.maximum(x_std, _min)
    y_std = torch.maximum(y_std, _min)
    return cov / (x_std * y_std + float(eps))


def _rolling_apply_unary(x, slices: Sequence[slice], window: int, fn, fill: float = float('nan')):
    """
    对输入的时序数据，按每个品种独立执行滚动窗口计算，最后把结果拼接成完整的输出序列。
    这是所有单输入时序算子（ts_mean、ts_std、ts_max 等）的通用底层函数。
    x: 输入的完整时序数据，是一个一维的 PyTorch 张量，长度等于 DataFrame 的行数。
    slices: 刚才生成的品种分组索引	[slice(0,3), slice(3,6)]
    window: 滚动窗口大小	2
    fn: 要在每个窗口上执行的函数	计算均值的函数
    fill: 前 window-1 行的填充值	nan

    关键特点：
    绝对不会跨品种计算：每个品种的滚动计算完全独立
    完全可微：所有操作都是 PyTorch 原生张量操作，梯度可以正确传递
    通用灵活：可以传入任何函数来实现不同的滚动指标（均值、标准差、最大值等）

    为什么不用 pandas 的groupby().rolling()？
    原因有三个：
    PyTorch 兼容性：整个代码是用 PyTorch 写的，需要保持所有操作都在 PyTorch 张量上进行，这样才能自动微分
    计算效率：在 GPU 上，PyTorch 的张量操作比 pandas 快得多，特别是当品种数量很多时
    灵活性：_rolling_apply_unary可以传入任何自定义函数，包括平滑版的 max/min 等，而 pandas 的 rolling 只能用内置函数
    """
    parts = []
    for sl in slices:  # 遍历每个品种的 slice
        xs = x[sl]
        n = int(xs.shape[0])
        if n < window or window <= 0:
            parts.append(torch.full_like(xs, fill))
            continue
        unfolded = xs.unfold(0, window, 1)
        vals = fn(unfolded)  # 每个时序窗口（window）上执行的操作
        # 填充前 window-1 个值，因为滚动窗口的结果从第 window 行开始才有意义
        pad = torch.full((window - 1,), fill, dtype=x.dtype, device=x.device)
        # 将每个窗口上执行的操作，得到的结果拼接，得到最终结果。
        parts.append(torch.cat([pad, vals], dim=0))
    # 拼接所有品种的结果并返回
    # e.g.
    # 第一个品种结果parts[0]：[nan, 1.5, 2.5]
    # 第二个品种结果parts[1]：[nan, 4.5, 5.5]
    # 最终输出：[nan, 1.5, 2.5, nan, 4.5, 5.5]
    return torch.cat(parts, dim=0)


def _rolling_apply_binary(x, y, slices: Sequence[slice], window: int, fn, fill: float = float('nan')):
    parts = []
    for sl in slices:
        xs = x[sl]
        ys = y[sl]
        n = int(xs.shape[0])
        if n < window or window <= 0:
            parts.append(torch.full_like(xs, fill))
            continue
        xu = xs.unfold(0, window, 1)
        yu = ys.unfold(0, window, 1)
        vals = fn(xu, yu)
        pad = torch.full((window - 1,), fill, dtype=x.dtype, device=x.device)
        parts.append(torch.cat([pad, vals], dim=0))
    return torch.cat(parts, dim=0)


def _shift_by_group(x, slices: Sequence[slice], periods: int):
    parts = []
    for sl in slices:
        xs = x[sl]
        n = int(xs.shape[0])
        out = torch.full_like(xs, float('nan'))
        p = int(periods)
        if p == 0:
            out = xs
        elif p > 0 and n > p:
            out[p:] = xs[:-p]
        elif p < 0 and n > abs(p):
            out[:p] = xs[-p:]
        parts.append(out)
    return torch.cat(parts, dim=0)


def _diff_by_group(x, slices: Sequence[slice], periods: int):
    return x - _shift_by_group(x, slices, periods)


def _pct_change_by_group(x, slices: Sequence[slice], periods: int):
    shifted = _shift_by_group(x, slices, periods)
    return _safe_div(x - shifted, shifted)


def _rolling_mean(x, slices, window: int):
    return _rolling_apply_unary(x, slices, window, lambda u: u.mean(dim=-1))


def _rolling_sum(x, slices, window: int):
    return _rolling_apply_unary(x, slices, window, lambda u: u.sum(dim=-1))


def _sample_std_last_dim(u, eps_inside_sqrt: float = 1e-12):
    """Sample std with ddof=1, matching pandas rolling().std() for full finite windows."""
    n = int(u.shape[-1])
    if n <= 1:
        return torch.full(u.shape[:-1], float('nan'), dtype=u.dtype, device=u.device)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    centered = u - u.mean(dim=-1, keepdim=True)
    # Add epsilon inside sqrt.  Without this, windows with exactly zero variance
    # have finite forward value but infinite/NaN backward through sqrt(0).
    return torch.sqrt((centered * centered).sum(dim=-1) / float(n - 1) + float(eps_inside_sqrt))


def _sample_std_1d(x, eps_inside_sqrt: float = 1e-12, min_std: float = 1e-4):
    """Sample std with ddof=1, matching pandas std() on non-degenerate data.

    Returns at least ``min_std`` to prevent gradient explosion in downstream
    divisions (e.g. Sharpe = ret / vol).
    """
    n = int(x.shape[0])
    if n <= 1:
        return x.sum() * 0.0 + torch.tensor(float(min_std), dtype=x.dtype, device=x.device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centered = x - x.mean()
    raw = torch.sqrt((centered * centered).sum() / float(n - 1) + float(eps_inside_sqrt))
    return torch.maximum(raw, torch.tensor(float(min_std), dtype=x.dtype, device=x.device))


def _rolling_std(x, slices, window: int):
    return _rolling_apply_unary(x, slices, window, _sample_std_last_dim)


def _ewm_mean(x, slices: Sequence[slice], span: int):
    """Differentiable group-wise EMA matching pandas ewm(span, adjust=False).mean() for finite inputs."""
    alpha = 2.0 / (float(max(1, int(span))) + 1.0)
    parts = []
    for sl in slices:
        xs = torch.nan_to_num(x[sl], nan=0.0, posinf=0.0, neginf=0.0)
        if int(xs.shape[0]) == 0:
            parts.append(xs)
            continue
        prev = xs[0]
        out_vals = [prev]
        for i in range(1, int(xs.shape[0])):
            prev = alpha * xs[i] + (1.0 - alpha) * prev
            out_vals.append(prev)
        parts.append(torch.stack(out_vals, dim=0))
    return torch.cat(parts, dim=0)


def _rolling_max(x, slices, window: int, k: float):
    return _rolling_apply_unary(x, slices, window, lambda u: (torch.softmax(k * u, dim=-1) * u).sum(dim=-1))


def _rolling_min(x, slices, window: int, k: float):
    return -_rolling_max(-x, slices, window, k)


def _rolling_corr(x, y, slices, window: int):
    def fn(xu, yu):
        xc = xu - xu.mean(dim=-1, keepdim=True)
        yc = yu - yu.mean(dim=-1, keepdim=True)
        cov = (xc * yc).mean(dim=-1)
        return cov / (_sample_std_last_dim(xu) * _sample_std_last_dim(yu) + 1e-6)
    return _rolling_apply_binary(x, y, slices, window, fn)


def _rolling_cov(x, y, slices, window: int):
    def fn(xu, yu):
        n = int(xu.shape[-1])
        if n <= 1:
            return torch.full(xu.shape[:-1], float('nan'), dtype=xu.dtype, device=xu.device)
        return ((xu - xu.mean(dim=-1, keepdim=True)) * (yu - yu.mean(dim=-1, keepdim=True))).sum(dim=-1) / float(n - 1)
    return _rolling_apply_binary(x, y, slices, window, fn)


def _rolling_rank_last(x, slices, window: int, k: float):
    def fn(u):
        last = u[..., -1:]
        return torch.sigmoid(k * (last - u)).mean(dim=-1)
    return _rolling_apply_unary(x, slices, window, fn)


def _rolling_argext_distance(x, slices, window: int, mode: str, k: float):
    positions = torch.arange(window, dtype=x.dtype, device=x.device)
    distance = float(window - 1) - positions
    if mode == 'max':
        return _rolling_apply_unary(x, slices, window, lambda u: (torch.softmax(k * u, dim=-1) * distance).sum(dim=-1))
    return _rolling_apply_unary(x, slices, window, lambda u: (torch.softmax(-k * u, dim=-1) * distance).sum(dim=-1))


def _rolling_time_weighted_mean(x, slices, window: int):
    weights = torch.arange(1, window + 1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return _rolling_apply_unary(x, slices, window, lambda u: (u * weights).sum(dim=-1))


def _rolling_norm(x, slices, window: int, min_periods: int, eps: float, clip: float):
    """
    z = (raw[t] - mean_hist) / (std_hist + eps)
    为什么常数乘法被「吸收」了？
    假设我把整个因子的输出乘以常数 C：
    raw' = C * raw
    那么历史均值也跟着缩放：
    mean_hist' = rolling_mean(C * raw_hist) = C * mean_hist
    历史标准差也同样缩放：
    std_hist' = rolling_std(C * raw_hist) = |C| * std_hist
    代入 rolling norm 公式：

    z' = (C*raw[t] - C*mean_hist) / (|C|*std_hist + eps)

    当 |C|*std_hist >> eps 时：

    ≈ C*(raw[t] - mean_hist) / (|C|*std_hist)
    = sign(C) * (raw[t] - mean_hist) / std_hist
    = sign(C) * z

    结论：无论 C 是 0.8、1.0、还是 1.5，经过 rolling norm 后的输出要么等于原始 z，要么等于 -z（符号翻转）。幅度完全不变。
    """
    # For differentiable refinement, use full-window historical z-score. min_periods is
    # approximated by max(1, min(min_periods, window)) to preserve leakage-free shift.
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    hist = torch.nan_to_num(_shift_by_group(x, slices, 1), nan=0.0, posinf=0.0, neginf=0.0)
    w = max(1, int(window))
    mean = torch.nan_to_num(_rolling_mean(hist, slices, w), nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.nan_to_num(_rolling_std(hist, slices, w), nan=0.0, posinf=0.0, neginf=0.0)
    z = _safe_div(x - mean, std + float(eps), eps=max(float(eps), 1e-6))
    return torch.clamp(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -float(clip), float(clip))


class _ParametricTorchEvaluator(_TorchModuleBase):  # type: ignore[misc]
    def __init__(self,
                 root: FactorNode,
                 df: pd.DataFrame,
                 cfg: GradientDescentConfig,
                 apply_rolling_norm: bool,
                 rolling_norm_window: int,
                 rolling_norm_min_periods: int,
                 rolling_norm_eps: float,
                 rolling_norm_clip: float,
                 target_col: str = 'future_ret'):
        super().__init__()
        self.root = copy.deepcopy(root)
        self.cfg = cfg
        self.apply_rolling_norm = bool(apply_rolling_norm)
        self.rolling_norm_window = int(rolling_norm_window)
        self.rolling_norm_min_periods = int(rolling_norm_min_periods)
        self.rolling_norm_eps = float(rolling_norm_eps)
        self.rolling_norm_clip = float(rolling_norm_clip)
        self.device = _torch_device(cfg.device)
        self.dtype = torch.float32
        self.parametric_method = cfg.parametric_method
        self.soft_temperature = float(cfg.soft_temperature)

        sorted_df = df.sort_values(['instrument_id', 'time']).reset_index(drop=True) if 'instrument_id' in df.columns and 'time' in df.columns else df.reset_index(drop=True)
        self.df_index = sorted_df.index
        self.fields: Dict[str, Any] = {}
        for col in sorted_df.columns:
            if col in {'time', 'instrument_id'}:
                continue
            values = pd.to_numeric(sorted_df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).values
            """
            作用：加载并存储固定的原始数据
            
            什么是 "缓冲区 (buffer)"？
            这是 PyTorch 里的一个特殊概念，你可以把它理解为模型里的 "只读数据区"：
            ✅ 数据是固定不变的，不会被梯度下降优化
            ✅ 会自动和模型一起移动到 GPU/CPU，不需要你手动处理
            ✅ 会和模型一起保存 / 加载，非常方便
            为什么不用torch.nn.Parameter？
            因为原始数据（比如close价格）是客观事实，不是需要优化的参数。如果用 Parameter，PyTorch 会尝试去优化这些价格，这显然是错误的。
            最终得到了什么？
            模型内部有了一组以_field_开头的变量，比如_field_close、_field_volume、_field_future_ret
            每个变量都是一个一维数组，长度等于你的 DataFrame 的行数（时间步数）
            self.fields字典就像一个索引：self.fields['close']直接就能拿到收盘价的一维数组
            """
            self.register_buffer(f'_field_{col}', torch.tensor(values, dtype=self.dtype, device=self.device))
            self.fields[col] = getattr(self, f'_field_{col}')
        if 'oi' not in self.fields and 'position' in self.fields:
            self.fields['oi'] = self.fields['position']
        if 'position' not in self.fields and 'oi' in self.fields:
            self.fields['position'] = self.fields['oi']

        target = pd.to_numeric(sorted_df[target_col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).values
        self.register_buffer('future_ret', torch.tensor(target, dtype=self.dtype, device=self.device))

        if 'instrument_id' in sorted_df.columns:
            self.instrument_ids = [str(x) for x in sorted_df['instrument_id'].tolist()]
        else:
            self.instrument_ids = ['UNKNOWN'] * len(sorted_df)
        self.slices = self._build_slices(self.instrument_ids)
        if 'time' in sorted_df.columns:
            years = pd.to_datetime(sorted_df['time'], errors='coerce').dt.year.fillna(-1).astype(int).tolist()
        else:
            years = [0] * len(sorted_df)
        self.year_slices = self._build_year_slices(self.slices, years)
        fee_values = [float(FEE.get(ins, 0.0)) for ins in self.instrument_ids]
        self.register_buffer('fee', torch.tensor(fee_values, dtype=self.dtype, device=self.device))

        self.edge_params = torch.nn.ParameterDict()
        self.const_params = torch.nn.ParameterDict()
        self.window_params = torch.nn.ParameterDict()
        self._material_param_names: Dict[Tuple[str, str], str] = {}
        self._register_tree_params(self.root, path='root')

    @staticmethod
    def _build_slices(ids: Sequence[str]) -> List[slice]:
        """
        把按 "品种 + 时间" 排序的长表格，切分成一个个独立的品种数据块，记录每个品种在整个数据中的起始和结束位置。

        # 假设你的DataFrame有6行，包含2个品种，各3天数据
        self.instrument_ids = [
            'RB2305', 'RB2305', 'RB2305',  # 螺纹钢2305合约，3天数据
            'I2305', 'I2305', 'I2305'      # 铁矿石2305合约，3天数据
        ]
        _build_slices函数做了什么？
        它会遍历instrument_ids，找出所有连续的相同品种的行，生成对应的slice对象（Python 的切片对象，表示起始和结束索引）。
        对于上面的例子，_build_slices会返回：
        python
        运行
        self.slices = [
            slice(0, 3),  # 第一个品种RB2305在第0、1、2行
            slice(3, 6)   # 第二个品种I2305在第3、4、5行
        ]
        """
        if not ids:
            return []
        slices: List[slice] = []
        start = 0
        prev = ids[0]
        for i, val in enumerate(ids[1:], start=1):
            if val != prev:
                slices.append(slice(start, i))
                start = i
                prev = val
        slices.append(slice(start, len(ids)))
        return slices

    @staticmethod
    def _build_year_slices(instrument_slices: Sequence[slice], years: Sequence[int]) -> List[List[slice]]:
        """Split each instrument slice into contiguous calendar-year slices."""
        out: List[List[slice]] = []
        for ins_sl in instrument_slices:
            start = int(ins_sl.start or 0)
            end = int(ins_sl.stop or start)
            if end <= start:
                out.append([])
                continue
            local: List[slice] = []
            year_start = start
            prev_year = int(years[start]) if start < len(years) else 0
            for idx in range(start + 1, end):
                year = int(years[idx]) if idx < len(years) else prev_year
                if year != prev_year:
                    local.append(slice(year_start, idx))
                    year_start = idx
                    prev_year = year
            local.append(slice(year_start, end))
            out.append(local)
        return out

    def _ts_icir(self, factor, future_ret, year_slices: Sequence[slice]):
        """Differentiable TS ICIR: mean(yearly TS IC) / (std(yearly TS IC) + eps)."""
        ic_values = []
        for year_sl in year_slices:
            if int((year_sl.stop or 0) - (year_sl.start or 0)) < 2:
                continue
            ic_values.append(_pearson_corr(factor[year_sl], future_ret[year_sl]))
        if len(ic_values) < 2:
            return factor.sum() * 0.0
        ic_tensor = torch.stack(ic_values)
        centered = ic_tensor - ic_tensor.mean()
        # pandas Series.std() uses ddof=1. Add a tiny value inside sqrt to keep
        # gradients finite when yearly IC values are identical, then add the
        # public 1e-6 denominator epsilon used by factor_indicators.py.
        ic_std = torch.sqrt((centered * centered).sum() / float(len(ic_values) - 1) + 1e-12)
        return ic_tensor.mean() / (ic_std + 1e-6)

    def _edge_key(self, node: FactorNode, path: str, child_name: str) -> str:
        if self.parametric_method == 'gpgd':
            return f'{node.__class__.__name__}_{child_name}'
        return f'{path}_{child_name}'

    def _window_key(self, node: FactorNode, path: str, window_name: str = 'window') -> str:
        if self.parametric_method == 'gpgd':
            return f'{node.__class__.__name__}_{window_name}'
        return f'{path}_{window_name}'

    @staticmethod
    def _safe_param_name(key: str) -> str:
        """
        把所有特殊字符替换成下划线。
        PyTorch 对参数名有严格限制：不能包含点.、斜杠/、方括号[]等特殊字符。
        而参数原始 key（比如root.left.right）包含大量点号，直接作为 PyTorch 参数名会报错。
        """
        return key.replace('.', '_').replace('/', '_').replace('[', '_').replace(']', '_').replace(':', '_')

    def _get_or_create_param(self, container, raw_key: str, init_value: float) -> str:
        name = self._safe_param_name(raw_key)
        if name not in container:
            safe_init = _finite_float(init_value, 1.0)
            container[name] = torch.nn.Parameter(torch.tensor(safe_init, dtype=self.dtype, device=self.device))
        """
        self._material_param_names：参数名映射表
        container._get_name()：参数所在的容器名称，可能是edge_params、const_params或window_params
        raw_key：原始的参数 key（比如root.left.right）
        name：转换后的安全参数名（比如root_left_right）
        格式例子：
        self._material_param_names = {
            # 边权重参数映射
            ('edge_params', 'root_left'): 'root_left',
            ('edge_params', 'root_right'): 'root_right',
            ('edge_params', 'root_left_child'): 'root_left_child',
            ('edge_params', 'root_right_child'): 'root_right_child',
            
            # 常数参数映射
            ('const_params', 'root.left.window'): 'root_left_window',
            ('const_params', 'root.right.window'): 'root_right_window',
            
            # 窗口参数映射（如果用软窗口的话）
            ('window_params', 'root.left.window'): 'root_left_window',
            ('window_params', 'root.right.window'): 'root_right_window',
        }
        在最后一步materialize()物化因子树的时候，会根据原始的 path，
        通过self._material_param_names这个字典找到对应的 PyTorch 参数，取出优化后的值写回原始的 FactorNode。
        """
        self._material_param_names[(container._get_name(), raw_key)] = name
        return name

    def _register_edge(self, node: FactorNode, path: str, child_name: str) -> None:
        """
        注册边权重参数。
        为因子树中每个算子的每个子节点(child)连接，注册一个可学习的权重参数。这是整个 OPGD（每个算子实例独立权重）方法的核心。
        注意：对于常数节点(其实都不叫child，而是称为window)，不会注册权重参数。
        比如ts_mean(close, 20)这个算子，只会给close(child)注册一个权重，而不会给20(window)注册一个权重
        什么是 "边"？
        在因子树中，算子节点和它的子节点之间的连接就叫 "边"。
        二元算子（如OpAdd、OpSub）有两条边：left和right
        一元算子（如OpTsMean、OpNeg）有一条边：child
        三元算子（如OpIfElse）有三条边：cond、left、right
        什么是“节点”？
        节点可以是算子，也可以是常数，也可以是原始行情数据（open, high, low, close, position, volume）

        例子：对于原始因子树：20日均线 - 60日均线
        factor_tree = OpSub(
            left=OpTsMean(
                child=DataNode('close'),
                window=ConstNode(20)
            ),
            right=OpTsMean(
                child=DataNode('close'),
                window=ConstNode(60)
            )
        )这个tree，
        原始表达式是factor = (ts_mean(close, 20)) - (ts_mean(close, 60))
        最终的用于梯度下降的表达式：
        factor = (w1 * ts_mean(w3 * close, x1)) - (w2 * ts_mean(w4 * close, x2))
        其中w1,w2,w3,w4,x1,x2都是可学习的参数

        node：当前的算子节点
        path：当前节点在因子树中的路径
        child_name：子节点的名称（left/right/child等）
        """
        self._get_or_create_param(self.edge_params, self._edge_key(node, path, child_name), 1.0)

    def _register_window(self, node: FactorNode, path: str, window_name: str, init_value: float) -> None:
        # Add small random noise to break exact-integer symmetry so the
        # soft-window blend gives non-zero gradient to neighbouring candidates
        # from step 1.  Without noise the softmax concentrates on the original
        # integer and ∂wp/∂score ≈ 0.
        noise = (random.random() - 0.5) * 0.6
        init = float(init_value) + noise
        init = float(min(self.cfg.max_window, max(self.cfg.min_window, init)))
        self._get_or_create_param(self.window_params, self._window_key(node, path, window_name), init)

    def _register_tree_params(self, node: FactorNode, path: str) -> None:
        if isinstance(node, ConstNode):
            # 当遇到一个常数节点（比如ConstNode(20)）
            # 就注册一个可学习参数，初始值就是这个常数的值
            # path是这个常数节点在因子树中的唯一地址，用来区分不同的常数节点。
            # 比如因子树OpAdd(ConstNode(20), ConstNode(30))
            # 第一个 ConstNode 的 path 是root.left
            # 第二个 ConstNode 的 path 是root.right
            self._get_or_create_param(self.const_params, path, float(node.value))
            return
        if isinstance(node, DataNode):
            return
        if isinstance(node, OpNanMean):
            for i, child in enumerate(node.children):
                child_name = f'child{i}'
                self._register_edge(node, path, child_name)
                self._register_tree_params(child, f'{path}.{child_name}')
            return
        if isinstance(node, _TERNARY_CHILD_OPS_TUPLE):
            node_any = cast(Any, node)
            for child_name in ['cond', 'left', 'right']:
                self._register_edge(node, path, child_name)
                self._register_tree_params(getattr(node_any, child_name), f'{path}.{child_name}')
            return
        if isinstance(node, BINARY_CHILD_OPS):
            node_any = cast(Any, node)
            for child_name in ['left', 'right']:
                self._register_edge(node, path, child_name)
                self._register_tree_params(getattr(node_any, child_name), f'{path}.{child_name}')
            if isinstance(node, _BINARY_TS_OPS_TUPLE):
                # 可学习的window参数的初始值采用node本身的window属性
                self._register_window(node, path, 'window', float(getattr(node_any, 'window', 1)))
            return
        if isinstance(node, UNARY_CHILD_OPS):
            node_any = cast(Any, node)
            self._register_edge(node, path, 'child')
            self._register_tree_params(node_any.child, f'{path}.child')
            if isinstance(node, _UNARY_TS_OPS_TUPLE):
                self._register_window(node, path, 'window', float(getattr(node_any, 'window', 1)))
            return
        if isinstance(node, OpMaRibbon):
            self._register_edge(node, path, 'child')
            self._register_tree_params(node.child, f'{path}.child')
            self._register_window(node, path, 'window1', float(node.window1))
            self._register_window(node, path, 'window2', float(node.window2))
            self._register_window(node, path, 'window3', float(node.window3))

    def _edge_param(self, node: FactorNode, path: str, child_name: str):
        name = self._safe_param_name(self._edge_key(node, path, child_name))
        return self.edge_params[name]

    def _window_param(self, node: FactorNode, path: str, window_name: str = 'window'):
        name = self._safe_param_name(self._window_key(node, path, window_name))
        return self.window_params[name]

    def _const_param(self, path: str):
        return self.const_params[self._safe_param_name(path)]

    def _child(self, node: FactorNode, path: str, child_name: str):
        child = getattr(cast(Any, node), child_name)
        return self._edge_param(node, path, child_name) * self._eval(child, f'{path}.{child_name}')

    def _window_candidates(self, init_window: int) -> List[int]:
        """
        首先会生成所有可能的窗口候选值，默认包含：
            原始窗口大小：20
            原始窗口附近的整数窗口：18, 19, 20, 21, 22（半径由 window_neighbor_radius 控制）
            配置的最小 / 最大窗口：2~60
            配置的窗口候选列表（如果有的话）
        最终会去重并排序，得到比如：[3, 5, 18, 19, 20, 21, 22, 30]

        这样做的原因：GP 生成树时通常只从粗粒度 window_choices（如 3/5/10/20/30）
        里采样；如果 GD 阶段也只在这些粗粒度窗口之间做 soft blend，窗口参数在 20
        附近很难学到 19/21 这类局部改进，看起来就像窗口完全没有优化。
        """
        base = list(self.cfg.window_choices or [])
        init = int(round(float(init_window)))
        base.append(init)
        radius = int(getattr(self.cfg, 'window_neighbor_radius', 0))
        if radius > 0:
            base.extend(range(init - radius, init + radius + 1))
        base.extend([self.cfg.min_window, self.cfg.max_window])
        out = sorted({int(x) for x in base if self.cfg.min_window <= int(x) <= self.cfg.max_window})
        return out or [max(1, int(init_window))]

    def _soft_window_weights(self, window_param, candidates: Sequence[int]):
        """Return smooth weights for discrete window candidates.

        Window optimization uses a separate, moderate temperature and squared
        distance. Compared with ``-soft_temperature * abs(wp - w)``, this keeps
        the initial hard-window approximation close to the GP tree while still
        providing non-negligible gradients to neighbouring windows.
        """
        logits = []
        temp = float(getattr(self.cfg, 'window_soft_temperature', self.soft_temperature))
        for w in candidates:
            dist = window_param - float(w)
            logits.append(-temp * dist * dist)
        return torch.softmax(torch.stack(logits), dim=0)

    def _safe_window_int(self, node: FactorNode, path: str, window_name: str = 'window', fallback: Optional[int] = None) -> int:
        if fallback is None:
            fallback = int(getattr(cast(Any, node), window_name, self.cfg.min_window))
        fallback = int(min(self.cfg.max_window, max(self.cfg.min_window, int(fallback))))
        try:
            param = self._window_param(node, path, window_name)
            raw = float(param.detach().cpu().item())
        except Exception:
            raw = float(fallback)
        if not math.isfinite(raw):
            raw = float(fallback)
        return int(min(self.cfg.max_window, max(self.cfg.min_window, round(raw))))

    def _sanitize_parameters_(self) -> None:
        """Keep learnable parameters finite after optimizer steps."""
        with torch.no_grad():
            for param in self.parameters():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e6, neginf=-1e6)
                param.data.clamp_(-1e6, 1e6)
            for param in self.window_params.values():
                fallback = float((self.cfg.min_window + self.cfg.max_window) / 2.0)
                param.data = torch.nan_to_num(param.data, nan=fallback, posinf=float(self.cfg.max_window), neginf=float(self.cfg.min_window))
                param.data.clamp_(float(self.cfg.min_window), float(self.cfg.max_window))

    def _soft_window_unary(self, node: FactorNode, path: str, x, op_fn, init_window: int):
        """
        _soft_window_unary通过"多窗口加权平均" 的平滑技巧，让窗口大小变成了连续可微的参数，
        同时保持了和原始硬窗口几乎相同的计算结果，且保持计算结果具有实际意义

        node: 当前的时序算子节点。例如OpTsMean(close, 20)
        path: 该节点在因子树中的唯一路径。例如root.left
        x: 算子的输入时序数据。例如收盘价数组
        op_fn: 原始的硬窗口算子函数。例如lambda z, w: _rolling_mean(z, self.slices, w)
        init_window: 原始因子树中的窗口大小。例如20

        return: 平滑后的算子输出时序。例如和原始ts_mean(close, 20)几乎相同的数组
        """

        # wp是可学习的窗口参数，是一个连续值。例如21.3
        wp = self._window_param(node, path, 'window')
        candidates = self._window_candidates(init_window)
        outs = []
        # 计算每个候选窗口的硬输出
        # 这一步会用原始的硬窗口算子计算每个候选窗口的输出
        # 所有计算都用标准时序算子，没有任何近似
        # 比如对于ts_mean(close, 20)，如果候选窗口是[19, 20, 21]，就会计算出三个数组：
        # ts_mean(close, 19)，ts_mean(close, 20)，ts_mean(close, 21)
        for w in candidates:  # 每个候选窗口大小
            # outs是一个长度 = 候选窗口的数量的列表。
            # 列表中的每个元素，都是单个候选窗口对应的完整时序计算结果
            # 每个元素都是一个一维 PyTorch 张量，长度 = 你的 DataFrame 总行数（时间步数 T）
            outs.append(op_fn(x, int(w)))
        # softmax 归一化权重
        # softmax 会把所有原始权重转换成 0~1 之间的数，且总和为 1
        # 对于上面的例子：
            # 窗口 20 的权重 ≈ 0.000045（几乎为 0）
            # 窗口 21 的权重 ≈ 0.982（占 98.2%）
            # 窗口 22 的权重 ≈ 0.018（占 1.8%）
        weights = self._soft_window_weights(wp, candidates)
        # 加权平均得到最终输出
        # 最终输出 = 0.000045×ts_mean (20) + 0.982×ts_mean (21) + 0.018×ts_mean (22)
        # 其中ts_mean (21)、ts_mean (20)、ts_mean (22)都是一个数组
        # 上述结果和直接计算ts_mean(close,21)几乎完全一样，但却是完全可微的
        """
        以两个候选窗口为例展开说明：
    
        假设：
        你的数据有 6 行（T=6），2 个品种
        候选窗口集合 = [2, 3]
        输入 x = [1, 2, 3, 4, 5, 6]（收盘价数组）
        op_fn = _rolling_mean（计算滚动均值）
        那么outs列表的内容是：
        outs = [
            # 第一个元素：窗口=2的滚动均值结果（长度6的一维张量）
            tensor([nan, 1.5, 2.5, nan, 4.5, 5.5]),

            # 第二个元素：窗口=3的滚动均值结果（长度6的一维张量）
            tensor([nan, nan, 2.0, nan, nan, 5.0])
        ]
            
        torch.nan_to_num是数值稳定性处理，把所有会导致计算错误的特殊值替换成 0：
        nan → 0.0（滚动计算前 window-1 行的缺失值）
        +inf → 0.0（除以零等异常）
        -inf → 0.0
        处理后，上面的outs变成：
        python
        运行
        outs_processed = [
            tensor([0.0, 1.5, 2.5, 0.0, 4.5, 5.5]),
            tensor([0.0, 0.0, 2.0, 0.0, 0.0, 5.0])
        ]
        
        对于上面的例子：
        候选窗口数量 K=2
        时间步数 T=6
        所以stacked.shape = (2, 6)
        最终stacked的具体内容
        python
        运行
        stacked = tensor([
            [0.0, 1.5, 2.5, 0.0, 4.5, 5.5],  # 第0行：窗口=2的结果
            [0.0, 0.0, 2.0, 0.0, 0.0, 5.0]   # 第1行：窗口=3的结果
        ]
        
        weighted = tensor([
            [0.0*0.982, 1.5*0.982, 2.5*0.982, 0.0*0.982, 4.5*0.982, 5.5*0.982],
            [0.0*0.018, 0.0*0.018, 2.0*0.018, 0.0*0.018, 0.0*0.018, 5.0*0.018]
        ])
        
        final_output = tensor([
            0.0,
            1.5*0.982 + 0.0*0.018 = 1.473,
            2.5*0.982 + 2.0*0.018 = 2.491,
            0.0,
            4.5*0.982 + 0.0*0.018 = 4.419,
            5.5*0.982 + 5.0*0.018 = 5.491
        ])
        """
        stacked = torch.stack([torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0) for o in outs], dim=0)
        # weights.view(-1, 1)：权重形状从(2,)变成(2,1)，方便广播相乘
        # weights_reshaped = weights.view(-1, 1)  # shape=(2,1)
        return (weights.view(-1, 1) * stacked).sum(dim=0)

    def _soft_window_binary(self, node: FactorNode, path: str, x, y, op_fn, init_window: int):
        wp = self._window_param(node, path, 'window')
        candidates = self._window_candidates(init_window)
        outs = []
        for w in candidates:
            outs.append(op_fn(x, y, int(w)))
        weights = self._soft_window_weights(wp, candidates)
        stacked = torch.stack([torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0) for o in outs], dim=0)
        return (weights.view(-1, 1) * stacked).sum(dim=0)

    def _eval(self, node: FactorNode, path: str):
        if isinstance(node, DataNode):
            if node.field not in self.fields:
                raise KeyError(f'Field `{node.field}` is not available for differentiable GP evaluation.')
            return self.fields[node.field]
        if isinstance(node, ConstNode):
            # self._const_param(path): 根据路径获取常数节点的可学习参数
            # 核心作用: 找到因子树中某个特定常数节点对应的、需要被梯度下降优化的参数值。
            return torch.ones_like(self.future_ret) * self._const_param(path)

        if isinstance(node, OpAdd):
            return self._child(node, path, 'left') + self._child(node, path, 'right')
        if isinstance(node, OpSub):
            return self._child(node, path, 'left') - self._child(node, path, 'right')
        if isinstance(node, OpMul):
            return self._child(node, path, 'left') * self._child(node, path, 'right')
        if isinstance(node, OpDiv):
            return _safe_div(self._child(node, path, 'left'), self._child(node, path, 'right'))
        if isinstance(node, OpMax):
            l = self._child(node, path, 'left')
            r = self._child(node, path, 'right')
            w = torch.softmax(self.soft_temperature * torch.stack([l, r], dim=0), dim=0)
            return w[0] * l + w[1] * r
        if isinstance(node, OpMin):
            l = self._child(node, path, 'left')
            r = self._child(node, path, 'right')
            w = torch.softmax(-self.soft_temperature * torch.stack([l, r], dim=0), dim=0)
            return w[0] * l + w[1] * r
        if isinstance(node, OpLt):
            return torch.sigmoid(self.soft_temperature * (self._child(node, path, 'left') - self._child(node, path, 'right')))
        if isinstance(node, OpGt):
            return torch.sigmoid(self.soft_temperature * (self._child(node, path, 'right') - self._child(node, path, 'left')))
        if isinstance(node, OpOiTrendConviction):
            close = self._child(node, path, 'left')
            oi = self._child(node, path, 'right')
            return torch.tanh(self.soft_temperature * _diff_by_group(close, self.slices, 1)) * _diff_by_group(oi, self.slices, 1)

        if isinstance(node, OpNeg):
            return -self._child(node, path, 'child')
        if isinstance(node, OpSqrtAbs):
            return torch.sqrt(_soft_abs(self._child(node, path, 'child'), self.soft_temperature) + 1e-6)
        if isinstance(node, OpAbs):
            return _soft_abs(self._child(node, path, 'child'), self.soft_temperature)
        if isinstance(node, OpInv):
            return _safe_div(torch.ones_like(self.future_ret), self._child(node, path, 'child'))
        if isinstance(node, OpSig):
            return torch.sigmoid(torch.clamp(self._child(node, path, 'child'), -50.0, 50.0))
        if isinstance(node, OpSign):
            return torch.tanh(self.soft_temperature * self._child(node, path, 'child'))
        if isinstance(node, OpDelta):
            return _diff_by_group(self._child(node, path, 'child'), self.slices, 1)
        if isinstance(node, OpBodyRatio):
            return _safe_div(self.fields['close'] - self.fields['open'], self.fields['high'] - self.fields['low'] + 1e-8)
        if isinstance(node, OpUpperShadowRatio):
            return _safe_div(self.fields['high'] - torch.maximum(self.fields['open'], self.fields['close']), self.fields['high'] - self.fields['low'] + 1e-8)
        if isinstance(node, OpLowerShadowRatio):
            return _safe_div(torch.minimum(self.fields['open'], self.fields['close']) - self.fields['low'], self.fields['high'] - self.fields['low'] + 1e-8)
        if isinstance(node, OpStochasticK):
            return _safe_div(self.fields['close'] - self.fields['low'], self.fields['high'] - self.fields['low'] + 1e-8)
        if isinstance(node, OpTypicalPrice):
            return (self.fields['high'] + self.fields['low'] + self.fields['close']) / 3.0
        if isinstance(node, OpNanMean):
            vals = [self._edge_param(node, path, f'child{i}') * self._eval(c, f'{path}.child{i}') for i, c in enumerate(node.children)]
            return torch.stack(vals, dim=0).mean(dim=0)

        if isinstance(node, OpRollNorm):
            child = self._child(node, path, 'child')
            return _rolling_norm(child, self.slices, node.window, node.min_periods, node.eps, node.clip)

        if isinstance(node, _UNARY_TS_OPS_TUPLE):
            # x: 子节点的完整时序向量数据（长度为 T 的一维张量）这里是通过递归_eval函数的方式计算得到具体数值的
            x = self._child(node, path, 'child')
            init_w = int(getattr(node, 'window', 1))
            if isinstance(node, OpTsMean):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_mean(z, self.slices, w), init_w)
            if isinstance(node, (OpEma, OpTsDecayExp)):
                return self._soft_window_unary(node, path, x, lambda z, w: _ewm_mean(z, self.slices, w), init_w)
            if isinstance(node, OpTsStd):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_std(z, self.slices, w), init_w)
            if isinstance(node, OpTsDelta):
                return self._soft_window_unary(node, path, x, lambda z, w: _diff_by_group(z, self.slices, w), init_w)
            if isinstance(node, OpTsPctDelta):
                return self._soft_window_unary(node, path, x, lambda z, w: _pct_change_by_group(z, self.slices, w), init_w)
            if isinstance(node, (OpReturn,)):
                return self._soft_window_unary(node, path, x, lambda z, w: _pct_change_by_group(z, self.slices, w), init_w)
            if isinstance(node, OpLogReturn):
                return self._soft_window_unary(node, path, torch.log(torch.clamp(_soft_abs(x, self.soft_temperature), min=1e-6)), lambda z, w: _diff_by_group(z, self.slices, w), init_w)
            if isinstance(node, OpVolatility):
                vol_input = _pct_change_by_group(x, self.slices, 1) if getattr(node.child, 'data_type', None) == FactorDataType.PRICE else x
                return self._soft_window_unary(node, path, vol_input, lambda z, w: _rolling_std(z, self.slices, w), init_w)
            if isinstance(node, OpVolumeStd):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_std(z, self.slices, w), init_w)
            if isinstance(node, OpVolumeZScore):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(z - _rolling_mean(z, self.slices, w), _rolling_std(z, self.slices, w)), init_w)
            if isinstance(node, OpTurnoverShock):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(z, _rolling_mean(z, self.slices, w)) - 1.0, init_w)
            if isinstance(node, OpBias):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(z, _rolling_mean(z, self.slices, w)) - 1.0, init_w)
            if isinstance(node, OpRangePosition):
                close = x
                high = self.fields['high']
                low = self.fields['low']
                return self._soft_window_unary(node, path, close, lambda z, w: _safe_div(z - _rolling_min(low, self.slices, w, self.soft_temperature), _rolling_max(high, self.slices, w, self.soft_temperature) - _rolling_min(low, self.slices, w, self.soft_temperature) + 1e-8), init_w)
            if isinstance(node, OpPriceAcceleration):
                return self._soft_window_unary(node, path, x, lambda z, w: _diff_by_group(z, self.slices, w) - _shift_by_group(_diff_by_group(z, self.slices, w), self.slices, w), init_w)
            if isinstance(node, OpTrueAmplitude):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(self.fields['high'] - self.fields['low'], _shift_by_group(z, self.slices, w)), init_w)
            if isinstance(node, OpTsDelay):
                return self._soft_window_unary(node, path, x, lambda z, w: _shift_by_group(z, self.slices, w), init_w)
            if isinstance(node, OpTsSum):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_sum(z, self.slices, w), init_w)
            if isinstance(node, OpTsMax):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_max(z, self.slices, w, self.soft_temperature), init_w)
            if isinstance(node, OpTsMin):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_min(z, self.slices, w, self.soft_temperature), init_w)
            if isinstance(node, OpTsArgmax):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_argext_distance(z, self.slices, w, 'max', self.soft_temperature), init_w)
            if isinstance(node, OpTsArgmin):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_argext_distance(z, self.slices, w, 'min', self.soft_temperature), init_w)
            if isinstance(node, OpTsTimeWeightedMean):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_time_weighted_mean(z, self.slices, w), init_w)
            if isinstance(node, OpTsRank):
                return self._soft_window_unary(node, path, x, lambda z, w: _rolling_rank_last(z, self.slices, w, self.soft_temperature), init_w)
            if isinstance(node, OpTsEntropy):
                return self._soft_window_unary(node, path, x, lambda z, w: torch.log(_rolling_std(z, self.slices, w) + 1e-6), init_w)
            if isinstance(node, OpTsSkew):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(_rolling_apply_unary(z, self.slices, w, lambda u: ((u - u.mean(dim=-1, keepdim=True)) ** 3).mean(dim=-1)), _rolling_std(z, self.slices, w) ** 3 + 1e-6), init_w)
            if isinstance(node, OpTsKurt):
                return self._soft_window_unary(node, path, x, lambda z, w: _safe_div(_rolling_apply_unary(z, self.slices, w, lambda u: ((u - u.mean(dim=-1, keepdim=True)) ** 4).mean(dim=-1)), _rolling_std(z, self.slices, w) ** 4 + 1e-6), init_w)

        if isinstance(node, _BINARY_TS_OPS_TUPLE):
            x = self._child(node, path, 'left')
            y = self._child(node, path, 'right')
            init_w = int(getattr(node, 'window', 1))
            if isinstance(node, (OpTsCorr, OpTsRankCorr, OpVpDivergence)):
                return self._soft_window_binary(node, path, x, y, lambda a, b, w: _rolling_corr(a, b, self.slices, w), init_w)
            if isinstance(node, (OpTsCov, OpTsResidual)):
                if isinstance(node, OpTsResidual):
                    return self._soft_window_binary(node, path, x, y, lambda a, b, w: a - (_rolling_cov(a, b, self.slices, w) / (_rolling_std(b, self.slices, w) ** 2 + 1e-6)) * b, init_w)
                return self._soft_window_binary(node, path, x, y, lambda a, b, w: _rolling_cov(a, b, self.slices, w), init_w)
            if isinstance(node, OpTsBeta):
                return self._soft_window_binary(node, path, x, y, lambda a, b, w: _safe_div(_rolling_cov(a, b, self.slices, w), _rolling_std(b, self.slices, w) ** 2), init_w)
            if isinstance(node, OpAmihud):
                return self._soft_window_binary(node, path, x, y, lambda close, volume, w: _rolling_mean(_safe_div(_soft_abs(_diff_by_group(close, self.slices, 1), self.soft_temperature), close) / (volume + 1e-6), self.slices, w), init_w)

        if isinstance(node, OpIfElse):
            c = torch.sigmoid(self.soft_temperature * self._child(node, path, 'cond'))
            l = self._child(node, path, 'left')
            r = self._child(node, path, 'right')
            return c * l + (1.0 - c) * r

        if isinstance(node, OpMaRibbon):
            x = self._child(node, path, 'child')
            w1 = self._safe_window_int(node, path, 'window1', int(node.window1))
            w2 = self._safe_window_int(node, path, 'window2', int(node.window2))
            w3 = self._safe_window_int(node, path, 'window3', int(node.window3))
            ma1 = _rolling_mean(x, self.slices, w1)
            ma2 = _rolling_mean(x, self.slices, w2)
            ma3 = _rolling_mean(x, self.slices, w3)
            mx = torch.maximum(torch.maximum(ma1, ma2), ma3)
            mn = torch.minimum(torch.minimum(ma1, ma2), ma3)
            center = _soft_abs((ma1 + ma2 + ma3) / 3.0, self.soft_temperature)
            return _safe_div(mx - mn, center)

        raise NotImplementedError(f'Differentiable evaluator does not support {type(node).__name__}.')

    def forward(self):
        out = self._eval(self.root, 'root')
        if self.apply_rolling_norm:
            out = _rolling_norm(out, self.slices, self.rolling_norm_window, self.rolling_norm_min_periods, self.rolling_norm_eps, self.rolling_norm_clip)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def score(self, factor, fitness_indicator_dict: Dict[str, float]):
        """
        计算口径和GP因子挖掘过程中的_metric_score()的
        metric_value = float(np.mean(vals))
        score += float(weight) * metric_value 逻辑保持一致：

        for each instrument:
            if instrument length < 50:
                skip

            先对 factor / future_ret 做和 GP 一致的清洗:
                factor: nan/inf -> 0, clip(-20, 20)
                future_ret: nan/inf -> 0, clip(-1, 1)

            按年份切片:
                TS IC: 每年一个值
                TS ICIR: 每个品种一个值 = mean(yearly IC) / (std(yearly IC) + 1e-6)
                Gross Return: 每年一个值 = yearly mean(gross) * 252
                Net Return: 每年一个值 = yearly mean(net) * 252
                Gross Volatility: 每年一个值 = yearly sample std(gross) * sqrt(252)
                Net Volatility: 每年一个值 = yearly sample std(net) * sqrt(252)
                Gross Sharpe: 每年一个值 = annual_ret / annual_vol
                Net Sharpe: 每年一个值 = annual_ret / annual_vol
                Turnover: 每年一个值 = yearly mean(turnover)，不再乘 252

        最后:
            每个 indicator 自己跨所有品种/年份取 mean
            再按 fitness_indicator_dict 加权求和
        """
        requested = {
            str(indicator): float(weight)
            for indicator, weight in dict(fitness_indicator_dict or {}).items()
            if abs(float(weight)) > 1e-12
        }
        if not requested:
            return factor.sum() * 0.0

        indicator_values: Dict[str, List[Any]] = {indicator: [] for indicator in requested}
        annual = 252.0
        sqrt_annual = math.sqrt(annual)
        factor_safe = torch.clamp(torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
        ret_safe = torch.clamp(torch.nan_to_num(self.future_ret, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

        for instrument_idx, sl in enumerate(self.slices):
            ins_start = int(sl.start or 0)
            ins_stop = int(sl.stop or ins_start)
            if ins_stop - ins_start < 50:
                continue

            f = factor_safe[sl]
            r = ret_safe[sl]
            fee = self.fee[ins_start] if ins_start < int(self.fee.shape[0]) else self.fee.mean()
            gross = f * r
            diff = f - _shift_by_group(f, [slice(0, len(f))], 1)
            turnover = torch.nan_to_num(torch.abs(diff), nan=0.0)
            if int(turnover.shape[0]) > 0:
                turnover = turnover.clone()
                turnover[0] = torch.abs(f[0])
            net = gross - turnover * fee

            absolute_year_slices = self.year_slices[instrument_idx] if instrument_idx < len(self.year_slices) else []
            year_slices = [
                slice(int(ys.start or ins_start) - ins_start, int(ys.stop or ins_start) - ins_start)
                for ys in absolute_year_slices
                if int(ys.stop or ins_start) > int(ys.start or ins_start)
            ]

            if 'TS ICIR' in requested:
                indicator_values['TS ICIR'].append(self._ts_icir(f, r, year_slices))

            for year_sl in year_slices:
                if int((year_sl.stop or 0) - (year_sl.start or 0)) <= 0:
                    continue
                yf = f[year_sl]
                yr = r[year_sl]
                ygross = gross[year_sl]
                ynet = net[year_sl]
                yturnover = turnover[year_sl]

                if 'TS IC' in requested:
                    indicator_values['TS IC'].append(_pearson_corr(yf, yr))
                if 'Gross Return' in requested:
                    indicator_values['Gross Return'].append(ygross.mean() * annual)
                if 'Net Return' in requested:
                    indicator_values['Net Return'].append(ynet.mean() * annual)
                if 'Gross Volatility' in requested or 'Gross Sharpe' in requested:
                    gross_vol = _sample_std_1d(ygross) * sqrt_annual
                    if 'Gross Volatility' in requested:
                        indicator_values['Gross Volatility'].append(gross_vol)
                    if 'Gross Sharpe' in requested:
                        gross_ret = ygross.mean() * annual
                        indicator_values['Gross Sharpe'].append(gross_ret / gross_vol)
                if 'Net Volatility' in requested or 'Net Sharpe' in requested:
                    net_vol = _sample_std_1d(ynet) * sqrt_annual
                    if 'Net Volatility' in requested:
                        indicator_values['Net Volatility'].append(net_vol)
                    if 'Net Sharpe' in requested:
                        net_ret = ynet.mean() * annual
                        indicator_values['Net Sharpe'].append(net_ret / net_vol)
                if 'Turnover' in requested:
                    # Align with get_annualized_turnover(): annual value is the
                    # yearly mean turnover, without multiplying by 252.
                    indicator_values['Turnover'].append(yturnover.mean())

        score = factor.sum() * 0.0
        has_valid_metric = False
        for indicator, weight in requested.items():
            vals = indicator_values.get(indicator, [])
            if not vals:
                continue
            metric_value = torch.nan_to_num(torch.stack(vals).mean(), nan=0.0, posinf=0.0, neginf=0.0)
            score = score + weight * metric_value
            has_valid_metric = True
        if not has_valid_metric:
            return factor.sum() * 0.0
        return score

    def materialize(self) -> FactorNode:
        """
        将最终经过梯度下降优化后的参数值应用到原始因子树上，生成一个新的、具体的因子树。
        """
        root = self._materialize_node(self.root, 'root')
        try:
            infer_node_type(root)
        except Exception:
            pass
        return root

    def _materialized_child(self, node: FactorNode, path: str, child_name: str) -> FactorNode:
        child = getattr(cast(Any, node), child_name)
        out = self._materialize_node(child, f'{path}.{child_name}')
        weight = _finite_float(self._edge_param(node, path, child_name).detach().cpu().item(), 1.0)
        if abs(weight - 1.0) <= 1e-4:
            return out
        # Collapse consecutive Mul(Const, Mul(Const, x)) into Mul(Const, x)
        # 检测连续 Mul(ConstNode, Mul(ConstNode, x)) 模式，合并为 Mul(ConstNode(w1*w2), x)
        if isinstance(out, OpMul) and isinstance(out.left, ConstNode):
            merged = _finite_float(weight * float(out.left.value), weight)
            if abs(merged - 1.0) <= 1e-4:
                return out.right
            return OpMul(ConstNode(merged), out.right)
        return OpMul(ConstNode(weight), out)

    def _rounded_window(self, node: FactorNode, path: str, window_name: str = 'window') -> int:
        return self._safe_window_int(node, path, window_name)

    def _materialize_node(self, node: FactorNode, path: str) -> FactorNode:
        if isinstance(node, DataNode):
            return copy.deepcopy(node)
        if isinstance(node, ConstNode):
            return ConstNode(_finite_float(self._const_param(path).detach().cpu().item(), float(node.value)))
        if isinstance(node, OpNanMean):
            return OpNanMean(*[self._materialized_child(node, path, f'child{i}') for i in range(len(node.children))])
        if isinstance(node, OpRollNorm):
            return OpRollNorm(self._materialized_child(node, path, 'child'), node.window, node.min_periods, node.eps, node.clip)
        if isinstance(node, OpMaRibbon):
            return OpMaRibbon(
                self._materialized_child(node, path, 'child'),
                self._rounded_window(node, path, 'window1'),
                self._rounded_window(node, path, 'window2'),
                self._rounded_window(node, path, 'window3'),
            )
        if isinstance(node, _TERNARY_CHILD_OPS_TUPLE):
            op_cls = cast(Any, node.__class__)
            return op_cls(
                self._materialized_child(node, path, 'cond'),
                self._materialized_child(node, path, 'left'),
                self._materialized_child(node, path, 'right'),
            )
        if isinstance(node, _BINARY_TS_OPS_TUPLE):
            op_cls = cast(Any, node.__class__)
            return op_cls(
                self._materialized_child(node, path, 'left'),
                self._materialized_child(node, path, 'right'),
                self._rounded_window(node, path),
            )
        if isinstance(node, BINARY_CHILD_OPS):
            op_cls = cast(Any, node.__class__)
            return op_cls(self._materialized_child(node, path, 'left'), self._materialized_child(node, path, 'right'))
        if isinstance(node, _UNARY_TS_OPS_TUPLE):
            op_cls = cast(Any, node.__class__)
            return op_cls(self._materialized_child(node, path, 'child'), self._rounded_window(node, path))
        if isinstance(node, UNARY_CHILD_OPS):
            op_cls = cast(Any, node.__class__)
            return op_cls(self._materialized_child(node, path, 'child'))
        return copy.deepcopy(node)


def _make_optimizer(name: str, params: Iterable[Any], learning_rate: float):
    opt = str(name or 'adam').lower()
    if opt == 'adam':
        return torch.optim.Adam(params, lr=learning_rate)
    if opt == 'adamw':
        return torch.optim.AdamW(params, lr=learning_rate)
    if opt == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    if opt == 'rmsprop':
        return torch.optim.RMSprop(params, lr=learning_rate)
    if opt == 'adagrad':
        return torch.optim.Adagrad(params, lr=learning_rate)
    raise ValueError(f'Unsupported gradient_descent_optimizer={name}. Use adam/adamw/sgd/rmsprop/adagrad.')


def optimize_tree_with_gradient_descent(
    tree: FactorNode,
    df: pd.DataFrame,
    fitness_indicator_dict: Dict[str, float],
    config: GradientDescentConfig,
    apply_rolling_norm: bool = True,
    rolling_norm_window: int = 30,
    rolling_norm_min_periods: int = 20,
    rolling_norm_eps: float = 1e-8,
    rolling_norm_clip: float = 5.0,
    target_col: str = 'future_ret',
) -> FactorNode:
    """Return a new tree after differentiable parameter refinement.

    Any optimization failure falls back to a deepcopy of the original tree so the
    GP pipeline remains robust.
    """
    if not config.enable_gradient_descent or config.gradient_descent_steps <= 0:
        return copy.deepcopy(tree)
    validate_gradient_descent_fitness_indicators(fitness_indicator_dict)
    if torch is None:
        raise ImportError('enable_gradient_descent=True requires PyTorch.')

    run_id = next(_GD_RUN_COUNTER)
    original_formula = tree.to_formula()
    requested_indicators = {
        str(k): float(v)
        for k, v in dict(fitness_indicator_dict or {}).items()
        if abs(float(v)) > 1e-12
    }

    def _should_log_run(run_config: GradientDescentConfig) -> bool:
        return bool(
            run_config.log_progress
            and (
                run_id <= run_config.progress_log_first_n_runs
                or run_id % run_config.progress_log_run_interval == 0
            )
        )

    def _format_float(value: float) -> str:
        if not math.isfinite(float(value)):
            return str(value)
        return f'{float(value):.6g}'

    def _run_once(run_config: GradientDescentConfig) -> FactorNode:
        should_log = _should_log_run(run_config)
        model = _ParametricTorchEvaluator(
            root=tree,
            df=df,
            cfg=run_config,
            apply_rolling_norm=apply_rolling_norm,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            target_col=target_col,
        )
        optimizer = _make_optimizer(run_config.gradient_descent_optimizer, model.parameters(), run_config.learning_rate)
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        initial_params = {name: p.detach().clone() for name, p in model.named_parameters()}
        best_score = -float('inf')
        initial_score = None
        final_score = None
        last_score_pos = None
        last_score_neg = None
        last_loss = None
        last_grad_norm = 0.0
        last_grad_tensor_count = 0
        last_grad_finite = True
        last_stop_reason = 'completed_all_steps'
        actual_steps = 0
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0

        if should_log:
            log.info(
                f'[GPGD][run={run_id}] start: device={model.device}, optimizer={run_config.gradient_descent_optimizer}, '
                f'lr={run_config.learning_rate}, steps={run_config.gradient_descent_steps}, '
                f'params={trainable_param_count}, method={run_config.parametric_method}, '
                f'indicators={requested_indicators}, apply_rolling_norm={apply_rolling_norm}, '
                f'formula={original_formula}'
            )

        for step in range(run_config.gradient_descent_steps):
            # 把所有可学习参数的梯度值重置为 None，避免上一步的梯度累加到当前步
            # 为什么要做：PyTorch 默认会自动累加梯度，如果不清零，每一步的梯度都会是之前所有步的和，导致参数更新错误
            # set_to_none=True的好处：比设置为 0 更节省内存，会直接释放梯度张量的内存空间
            optimizer.zero_grad(set_to_none=True)
            # factor: 整个时间序列的因子值。torch.Tensor，形状 =(T,)（一维张量，长度等于时间步数）
            factor = model.forward()
            score_pos = model.score(factor, fitness_indicator_dict)
            score_neg = model.score(-factor, fitness_indicator_dict)
            score = torch.maximum(score_pos, score_neg)
            loss = -score
            if not torch.isfinite(loss):
                last_stop_reason = f'non_finite_loss_at_step_{step + 1}'
                if should_log:
                    log.warning(
                        f'[GPGD][run={run_id}] step={step + 1}: non-finite loss encountered; '
                        f'score_pos={score_pos.detach().cpu().item()}, score_neg={score_neg.detach().cpu().item()}'
                    )
                break

            score_value = float(score.detach().cpu().item())
            last_score_pos = float(score_pos.detach().cpu().item())
            last_score_neg = float(score_neg.detach().cpu().item())
            last_loss = float(loss.detach().cpu().item())
            final_score = score_value
            if initial_score is None:
                initial_score = score_value
            if score_value > best_score + 1e-10:
                best_score = score_value
                # The score above was computed from the current parameters, so
                # save state before optimizer.step(). Saving after step can
                # accidentally persist NaN/inf parameters produced by an
                # unstable update while attributing them to the pre-step score.
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            loss.backward()
            # Sanitize non-finite gradients before clipping. Non-finite
            # gradients occur when factor has near-zero variance (common for
            # bounded operators like StochasticK, UpperShadowRatio), causing
            # Pearson correlation backward to divide by near-zero std.
            had_non_finite = False
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        had_non_finite = True
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            if had_non_finite and should_log:
                log.warning(
                    f'[GPGD][run={run_id}] step={step + 1}: non-finite gradient sanitized '
                    f'(NaN/inf → 0) before clipping.'
                )
            if run_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=run_config.gradient_clip_norm)
            last_grad_norm, last_grad_tensor_count, last_grad_finite = _gradient_norm_and_status(model.parameters())

            actual_steps = step + 1
            if should_log and (step == 0 or (step + 1) % run_config.progress_log_step_interval == 0 or step + 1 == run_config.gradient_descent_steps):
                log.info(
                    f'[GPGD][run={run_id}] step={step + 1}/{run_config.gradient_descent_steps}: '
                    f'loss={_format_float(last_loss)}, score={_format_float(score_value)}, '
                    f'score_pos={_format_float(last_score_pos)}, score_neg={_format_float(last_score_neg)}, '
                    f'best_score={_format_float(best_score)}, grad_norm={_format_float(last_grad_norm)}, '
                    f'grad_tensors={last_grad_tensor_count}, grad_finite={last_grad_finite}, '
                    f'no_improve={no_improve}'
                )

            optimizer.step()
            model._sanitize_parameters_()

            if run_config.early_stopping_steps > 0 and no_improve >= run_config.early_stopping_steps:
                last_stop_reason = f'early_stopping_no_improve_{no_improve}_steps'
                break

        model.load_state_dict(best_state)
        param_delta = _max_parameter_delta(model, initial_params)
        refined_tree = model.materialize()
        refined_formula = refined_tree.to_formula()
        formula_changed = refined_formula != original_formula

        if should_log:
            if formula_changed:
                unchanged_reason = ''  # formula actually changed, no unchanged reason
            elif actual_steps <= 0:
                unchanged_reason = last_stop_reason
            elif not math.isfinite(best_score) or best_score <= (initial_score if initial_score is not None else best_score) + 1e-10:
                unchanged_reason = 'no_surrogate_score_improvement_saved_initial_state'
            elif param_delta <= 1e-8:
                unchanged_reason = 'optimized_parameters_almost_equal_initial'
            elif last_grad_tensor_count == 0:
                unchanged_reason = 'no_parameter_received_gradient'
            elif last_grad_norm <= 1e-12:
                unchanged_reason = 'gradient_norm_near_zero_scale_invariant_or_flat_fitness'
            else:
                unchanged_reason = 'parameters_changed_but_materialized_formula_same_rounding_or_tiny_weights'

            if formula_changed:
                log.info(
                    f'[GPGD][run={run_id}] summary: steps={actual_steps}/{run_config.gradient_descent_steps}, '
                    f'stop_reason={last_stop_reason}, initial_score={_format_float(initial_score if initial_score is not None else float("nan"))}, '
                    f'final_score={_format_float(final_score if final_score is not None else float("nan"))}, '
                    f'best_score={_format_float(best_score)}, last_loss={_format_float(last_loss if last_loss is not None else float("nan"))}, '
                    f'last_grad_norm={_format_float(last_grad_norm)}, grad_tensors={last_grad_tensor_count}, '
                    f'grad_finite={last_grad_finite}, max_param_delta={_format_float(param_delta)}, '
                    f'formula_changed=True'
                )
            else:
                log.info(
                    f'[GPGD][run={run_id}] summary: steps={actual_steps}/{run_config.gradient_descent_steps}, '
                    f'stop_reason={last_stop_reason}, initial_score={_format_float(initial_score if initial_score is not None else float("nan"))}, '
                    f'final_score={_format_float(final_score if final_score is not None else float("nan"))}, '
                    f'best_score={_format_float(best_score)}, last_loss={_format_float(last_loss if last_loss is not None else float("nan"))}, '
                    f'last_grad_norm={_format_float(last_grad_norm)}, grad_tensors={last_grad_tensor_count}, '
                    f'grad_finite={last_grad_finite}, max_param_delta={_format_float(param_delta)}, '
                    f'formula_changed=False, unchanged_reason={unchanged_reason}'
                )
            if not formula_changed:
                log.info(
                    f'[GPGD][run={run_id}] unchanged_formula_detail: '
                    f'fitness like TS IC/TS ICIR/Sharpe can be scale-invariant, rolling_norm can absorb edge/root scales, '
                    f'and windows are rounded back to integers during materialization. original_formula={original_formula}'
                )
            else:
                log.info(f'[GPGD][run={run_id}] refined_formula={refined_formula}')

        return refined_tree

    try:
        return _run_once(config)
    except Exception as exc:
        if str(config.device or '').lower().startswith('mps') and _is_mps_backward_unsupported_error(exc):
            log.warning(f'[GPGD] MPS gradient descent refinement failed due to unsupported backward op: {exc}. Retry on CPU.')
            try:
                cpu_config = copy.copy(config)
                cpu_config.device = 'cpu'
                return _run_once(cpu_config)
            except Exception as cpu_exc:
                log.warning(f'[GPGD] CPU retry after MPS failure also failed: {cpu_exc}. Fallback to original tree.')
                return copy.deepcopy(tree)
        log.warning(f'[GPGD] gradient descent refinement failed: {exc}. Fallback to original tree. Traceback:\n{traceback.format_exc()}')
        return copy.deepcopy(tree)





