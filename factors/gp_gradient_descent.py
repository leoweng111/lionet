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
import math
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
    min_window / max_window / window_choices:
        Bounds/candidates used when optimizing discrete time-series windows.
    device:
        Torch device. ``None`` selects cuda/mps when available, otherwise cpu.
    """

    enable_gradient_descent: bool = False
    gradient_descent_method: str = 'alternated'
    generation_per_gradient_descent: int = 1
    gradient_descent_steps: int = 20
    parametric_method: str = 'opgd'
    gradient_descent_optimizer: str = 'adam'
    learning_rate: float = 0.01
    early_stopping_steps: int = 5
    gradient_clip_norm: float = 1.0
    soft_temperature: float = 10.0
    min_window: int = 2
    max_window: int = 60
    window_choices: Optional[Sequence[int]] = None
    device: Optional[str] = None

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
        cfg.min_window = max(1, int(cfg.min_window))
        cfg.max_window = max(cfg.min_window, int(cfg.max_window))
        return cfg


def _torch_device(device: Optional[str]):
    if torch is None:
        raise ImportError('enable_gradient_descent=True requires PyTorch, but torch is not importable.')
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _safe_tensor(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _soft_abs(x, k: float):
    return x * torch.tanh(k * x)


def _safe_div(a, b, eps: float = 1e-6):
    return a / torch.where(torch.abs(b) < eps, torch.full_like(b, eps), b)


def _pearson_corr(x, y, eps: float = 1e-6):
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.detach().sum().item()) < 3:
        return x.sum() * 0.0
    xv = x[mask]
    yv = y[mask]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    return (xv * yv).mean() / (xv.std(unbiased=False) * yv.std(unbiased=False) + eps)


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


def _sample_std_last_dim(u):
    """Sample std with ddof=1, matching pandas rolling().std() for full finite windows."""
    n = int(u.shape[-1])
    if n <= 1:
        return torch.full(u.shape[:-1], float('nan'), dtype=u.dtype, device=u.device)
    centered = u - u.mean(dim=-1, keepdim=True)
    return torch.sqrt((centered * centered).sum(dim=-1) / float(n - 1))


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
    # For differentiable refinement, use full-window historical z-score. min_periods is
    # approximated by max(1, min(min_periods, window)) to preserve leakage-free shift.
    hist = _shift_by_group(x, slices, 1)
    w = max(1, int(window))
    mean = _rolling_mean(hist, slices, w)
    std = _rolling_std(hist, slices, w)
    z = (x - mean) / (std + float(eps))
    return torch.clamp(z, -float(clip), float(clip))


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
        fee_values = [float(FEE.get(ins, 0.0)) for ins in self.instrument_ids]
        self.register_buffer('fee', torch.tensor(fee_values, dtype=self.dtype, device=self.device))

        self.edge_params = torch.nn.ParameterDict()
        self.const_params = torch.nn.ParameterDict()
        self.window_params = torch.nn.ParameterDict()
        # root_scale是这整个因子(tree)的可学习缩放参数
        # 初始时，整个因子树的输出不做任何缩放(1.0)
        # 优化后：root_scale会变成比如 0.8 或 1.2，用来调整整个因子的量级，让 IC/ICIR 最大化
        self.root_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=self.dtype, device=self.device))
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
            container[name] = torch.nn.Parameter(torch.tensor(float(init_value), dtype=self.dtype, device=self.device))
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
        self._get_or_create_param(self.window_params, self._window_key(node, path, window_name), init_value)

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
            配置的最小 / 最大窗口：2~60
            配置的窗口候选列表（如果有的话）
        最终会去重并排序，得到比如：[20, 21, 22]
        """
        base = list(self.cfg.window_choices or [])
        base.append(int(round(float(init_window))))
        base.extend([self.cfg.min_window, self.cfg.max_window])
        out = sorted({int(x) for x in base if self.cfg.min_window <= int(x) <= self.cfg.max_window})
        return out or [max(1, int(init_window))]

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
        weights_raw = []
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
            # 计算每个窗口的原始权重
            # 权重公式：原始权重 = -k × |当前窗口参数 - 候选窗口大小|
            # 离当前参数wp越近的窗口，原始权重越大（负数越小）
            # e.g. 对于wp=21.3：
                # 窗口 20 的原始权重 = -10 × |21.3-20| = -13
                # 窗口 21 的原始权重 = -10 × |21.3-21| = -3
                # 窗口 22 的原始权重 = -10 × |21.3-22| = -7
            weights_raw.append(-self.soft_temperature * torch.abs(wp - float(w)))
        # softmax 归一化权重
        # softmax 会把所有原始权重转换成 0~1 之间的数，且总和为 1
        # 对于上面的例子：
            # 窗口 20 的权重 ≈ 0.000045（几乎为 0）
            # 窗口 21 的权重 ≈ 0.982（占 98.2%）
            # 窗口 22 的权重 ≈ 0.018（占 1.8%）
        weights = torch.softmax(torch.stack(weights_raw), dim=0)
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
        weights_raw = []
        for w in candidates:
            outs.append(op_fn(x, y, int(w)))
            weights_raw.append(-self.soft_temperature * torch.abs(wp - float(w)))
        weights = torch.softmax(torch.stack(weights_raw), dim=0)
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
            w1 = int(torch.clamp(torch.round(self._window_param(node, path, 'window1')).detach(), self.cfg.min_window, self.cfg.max_window).item())
            w2 = int(torch.clamp(torch.round(self._window_param(node, path, 'window2')).detach(), self.cfg.min_window, self.cfg.max_window).item())
            w3 = int(torch.clamp(torch.round(self._window_param(node, path, 'window3')).detach(), self.cfg.min_window, self.cfg.max_window).item())
            ma1 = _rolling_mean(x, self.slices, w1)
            ma2 = _rolling_mean(x, self.slices, w2)
            ma3 = _rolling_mean(x, self.slices, w3)
            mx = torch.maximum(torch.maximum(ma1, ma2), ma3)
            mn = torch.minimum(torch.minimum(ma1, ma2), ma3)
            center = _soft_abs((ma1 + ma2 + ma3) / 3.0, self.soft_temperature)
            return _safe_div(mx - mn, center)

        raise NotImplementedError(f'Differentiable evaluator does not support {type(node).__name__}.')

    def forward(self):
        out = self.root_scale * self._eval(self.root, 'root')
        if self.apply_rolling_norm:
            out = _rolling_norm(out, self.slices, self.rolling_norm_window, self.rolling_norm_min_periods, self.rolling_norm_eps, self.rolling_norm_clip)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def score(self, factor, fitness_indicator_dict: Dict[str, float]):
        scores = []
        annual = 252.0
        for sl in self.slices:
            f = factor[sl]
            r = self.future_ret[sl]
            fee = self.fee[sl]
            gross = f * r
            diff = f - _shift_by_group(f, [slice(0, len(f))], 1)
            turnover = torch.nan_to_num(_soft_abs(diff, self.soft_temperature), nan=0.0)
            net = gross - turnover * fee
            local: Dict[str, Any] = {
                'TS IC': _pearson_corr(f, r),
                'Gross Return': gross.mean() * annual,
                'Net Return': net.mean() * annual,
                'Gross Volatility': gross.std(unbiased=False) * math.sqrt(annual),
                'Net Volatility': net.std(unbiased=False) * math.sqrt(annual),
                'Gross Sharpe': gross.mean() / (gross.std(unbiased=False) + 1e-6) * math.sqrt(annual),
                'Net Sharpe': net.mean() / (net.std(unbiased=False) + 1e-6) * math.sqrt(annual),
                'Turnover': turnover.mean() * annual,
            }
            score = f.sum() * 0.0
            for indicator, weight in fitness_indicator_dict.items():
                if indicator in local and abs(float(weight)) > 1e-12:
                    score = score + float(weight) * local[indicator]
            scores.append(score)
        if not scores:
            return factor.sum() * 0.0
        return torch.stack(scores).mean()

    def materialize(self) -> FactorNode:
        """
        将最终经过梯度下降优化后的参数值应用到原始因子树上，生成一个新的、具体的因子树。

        """
        root = self._materialize_node(self.root, 'root')
        scale = float(self.root_scale.detach().cpu().item())
        if abs(scale - 1.0) > 1e-8:
            root = OpMul(ConstNode(scale), root)
        try:
            infer_node_type(root)
        except Exception:
            pass
        return root

    def _materialized_child(self, node: FactorNode, path: str, child_name: str) -> FactorNode:
        child = getattr(cast(Any, node), child_name)
        out = self._materialize_node(child, f'{path}.{child_name}')
        weight = float(self._edge_param(node, path, child_name).detach().cpu().item())
        if abs(weight - 1.0) <= 1e-8:
            return out
        return OpMul(ConstNode(weight), out)

    def _rounded_window(self, node: FactorNode, path: str, window_name: str = 'window') -> int:
        val = float(self._window_param(node, path, window_name).detach().cpu().item())
        return int(min(self.cfg.max_window, max(self.cfg.min_window, round(val))))

    def _materialize_node(self, node: FactorNode, path: str) -> FactorNode:
        if isinstance(node, DataNode):
            return copy.deepcopy(node)
        if isinstance(node, ConstNode):
            return ConstNode(float(self._const_param(path).detach().cpu().item()))
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

    try:
        model = _ParametricTorchEvaluator(
            root=tree,
            df=df,
            cfg=config,
            apply_rolling_norm=apply_rolling_norm,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            target_col=target_col,
        )
        optimizer = _make_optimizer(config.gradient_descent_optimizer, model.parameters(), config.learning_rate)
        best_score = -float('inf')
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0

        for step in range(config.gradient_descent_steps):
            optimizer.zero_grad(set_to_none=True)
            factor = model.forward()
            score_pos = model.score(factor, fitness_indicator_dict)
            score_neg = model.score(-factor, fitness_indicator_dict)
            score = torch.maximum(score_pos, score_neg)
            loss = -score
            if not torch.isfinite(loss):
                break
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()

            score_value = float(score.detach().cpu().item())
            if score_value > best_score + 1e-10:
                best_score = score_value
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if config.early_stopping_steps > 0 and no_improve >= config.early_stopping_steps:
                break

        model.load_state_dict(best_state)
        return model.materialize()
    except Exception as exc:
        log.warning(f'[GPGD] gradient descent refinement failed: {exc}. Fallback to original tree.')
        return copy.deepcopy(tree)





