"""Genetic programming engine for factor mining.

This module is self-contained so it can be reused by framework classes
without changing existing gp.py scripts.
"""

import copy
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from utils.logging import log
from utils.params import FUTURES_CONTRACT_MULTIPLIER
from .factor_indicators import (
    get_annualized_ret,
    get_annualized_sharpe,
    get_annualized_ts_ic_and_t_corr,
    get_annualized_volatility,
)
from .factor_utils import rolling_normalize_features
from .factor_ops import (
    BINARY_CHILD_OPS,
    BINARY_OPS,
    BINARY_TS_OPS,
    UNARY_CHILD_OPS,
    UNARY_OPS,
    UNARY_TS_OPS,
    ConstNode,
    DataNode,
    FactorNode,
    OpNeg,
    infer_node_type,
)

# Shock mode hyper-parameters (higher exploration)
SHOCK_CROSSOVER_PROB: float = 0.2
SHOCK_MUTATION_PROB: float = 0.75
SHOCK_TOURNAMENT_SIZE: int = 3
SHOCK_ROOT_CUT_PROB: float = 0.25
SHOCK_HOIST_PROB: float = 0.35


def _generate_valid_random_tree(
    data_fields: Sequence[str],
    max_depth: int,
    current_depth: int,
    window_choices: Sequence[int],
    const_prob: float,
    leaf_prob: float,
    rng: random.Random,
    log_context: Optional[str] = None,
) -> FactorNode:
    """Generate a random GP tree and ensure it passes semantic type validation."""

    def _build() -> FactorNode:
        if current_depth >= max_depth:
            if rng.random() < const_prob:
                return ConstNode(rng.uniform(-2.0, 2.0))
            return DataNode(rng.choice(list(data_fields)))

        if rng.random() < leaf_prob:
            if rng.random() < const_prob:
                return ConstNode(rng.uniform(-2.0, 2.0))
            return DataNode(rng.choice(list(data_fields)))

        op_pick = rng.random()
        if op_pick < 0.5:
            op_cls = rng.choice(BINARY_OPS)
            left = _generate_valid_random_tree(
                data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
            )
            right = _generate_valid_random_tree(
                data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
            )
            return op_cls(left, right)

        if op_pick < 0.75:
            op_cls = rng.choice(UNARY_OPS)
            child = _generate_valid_random_tree(
                data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
            )
            return op_cls(child)

        if op_pick < 0.9:
            op_cls = rng.choice(UNARY_TS_OPS)
            child = _generate_valid_random_tree(
                data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
            )
            return op_cls(child, int(rng.choice(list(window_choices))))

        op_cls = rng.choice(BINARY_TS_OPS)
        left = _generate_valid_random_tree(
            data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
        )
        right = _generate_valid_random_tree(
            data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng
        )
        return op_cls(left, right, int(rng.choice(list(window_choices))))

    max_attempts = 25
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        tree = _build()
        try:
            infer_node_type(tree)
            if log_context and attempts > 8:
                log.warning(f'{log_context} valid tree built after {attempts} attempts.')
            return tree
        except TypeError:
            continue

    # Fallback without validation to avoid deadlock in rare cases.
    if log_context:
        log.warning(f'{log_context} failed to build valid tree after {attempts} attempts, using fallback.')
    return _build()


def get_all_nodes_with_parents(node: FactorNode, parent=None, direction: Optional[str] = None):
    nodes = [(node, parent, direction)]
    node_any = cast(Any, node)
    if isinstance(node, BINARY_CHILD_OPS):
        nodes.extend(get_all_nodes_with_parents(node_any.left, node, 'left'))
        nodes.extend(get_all_nodes_with_parents(node_any.right, node, 'right'))
    elif isinstance(node, UNARY_CHILD_OPS):
        nodes.extend(get_all_nodes_with_parents(node_any.child, node, 'child'))
    return nodes


def get_tree_depth(node: FactorNode) -> int:
    """Return max depth of AST, where leaf depth is 1."""
    node_any = cast(Any, node)
    if isinstance(node, (DataNode, ConstNode)):
        return 1
    if isinstance(node, BINARY_CHILD_OPS):
        return 1 + max(get_tree_depth(node_any.left), get_tree_depth(node_any.right))
    if isinstance(node, UNARY_CHILD_OPS):
        return 1 + get_tree_depth(node_any.child)
    return 1


def _calc_depth_penalty(depth: int,
                        depth_penalty_coef: float = 0.0,
                        depth_penalty_start_depth: int = 0,
                        depth_penalty_linear_coef: float = 0.0,
                        depth_penalty_quadratic_coef: float = 0.0) -> float:
    """
    Compute depth penalty with dynamic ramp-up:
    - base part: depth_penalty_coef * depth
    - dynamic part: 0 when depth <= start_depth
      else linear_coef * extra_depth + quadratic_coef * extra_depth^2
    """
    d = max(int(depth), 0)
    start = max(int(depth_penalty_start_depth), 0)
    extra = max(d - start, 0)

    base_penalty = float(depth_penalty_coef) * float(d)
    dynamic_penalty = float(depth_penalty_linear_coef) * float(extra) + \
        float(depth_penalty_quadratic_coef) * float(extra) * float(extra)
    return base_penalty + dynamic_penalty


def mutate_tree(
    root_node: FactorNode,
    data_fields: Sequence[str],
    max_depth: int,
    window_choices: Sequence[int],
    const_prob: float,
    leaf_prob: float,
    rng: random.Random,
    gen_idx: Optional[int] = None,
) -> FactorNode:
    new_root = copy.deepcopy(root_node)
    nodes_info = get_all_nodes_with_parents(new_root)
    target_node, parent, direction = rng.choice(nodes_info)

    if parent is None:
        return _generate_valid_random_tree(
            data_fields,
            max_depth,
            0,
            window_choices,
            const_prob,
            leaf_prob,
            rng,
            log_context=f'[GP][mutate_tree][gen={gen_idx}] root-mutation' if gen_idx is not None else '[GP][mutate_tree] root-mutation',
        )

    new_branch = _generate_valid_random_tree(
        data_fields,
        min(max_depth, 3),
        0,
        window_choices,
        const_prob,
        leaf_prob,
        rng,
        log_context=f'[GP][mutate_tree][gen={gen_idx}] subtree-mutation' if gen_idx is not None else '[GP][mutate_tree] subtree-mutation',
    )
    old_child = None
    try:
        infer_node_type(new_branch)
        target_type = infer_node_type(target_node)
        if new_branch.data_type != target_type:
            # If type mismatch, keep original subtree to avoid invalid mix.
            return new_root
        old_child = getattr(parent, direction)
        setattr(parent, direction, new_branch)
        infer_node_type(new_root)
    except TypeError:
        # Restore old subtree when mutation breaks semantic constraints.
        if old_child is not None:
            try:
                setattr(parent, direction, old_child)
            except Exception:
                pass
        return new_root
    _ = target_node
    return new_root


def macro_subtree_mutation(
    root_node: FactorNode,
    data_fields: Sequence[str],
    max_depth: int,
    window_choices: Sequence[int],
    const_prob: float,
    leaf_prob: float,
    rng: random.Random,
    gen_idx: Optional[int] = None,
) -> FactorNode:
    """Macro subtree mutation: cut one child under the root and regrow a new subtree."""
    new_root = copy.deepcopy(root_node)

    if isinstance(new_root, (DataNode, ConstNode)):
        return _generate_valid_random_tree(
            data_fields,
            max_depth,
            0,
            window_choices,
            const_prob,
            leaf_prob,
            rng,
            log_context=f'[GP][macro_mutation][gen={gen_idx}] root-leaf',
        )

    target_child_attr: Optional[str] = None
    if isinstance(new_root, BINARY_CHILD_OPS):
        target_child_attr = 'left' if rng.random() < 0.5 else 'right'
    elif isinstance(new_root, UNARY_CHILD_OPS):
        target_child_attr = 'child'

    if target_child_attr is None:
        return new_root

    new_branch = _generate_valid_random_tree(
        data_fields,
        max_depth,
        0,
        window_choices,
        const_prob,
        leaf_prob,
        rng,
        log_context=f'[GP][macro_mutation][gen={gen_idx}] root-cut',
    )

    old_child = None
    try:
        old_child = getattr(new_root, target_child_attr)
        setattr(new_root, target_child_attr, new_branch)
        infer_node_type(new_root)
    except TypeError:
        if old_child is not None:
            try:
                setattr(new_root, target_child_attr, old_child)
            except Exception:
                pass
        return root_node

    return new_root


def hoist_mutation(
    root_node: FactorNode,
    rng: random.Random,
    gen_idx: Optional[int] = None,
) -> FactorNode:
    """Hoist mutation: promote a random subtree to be the new root."""
    new_root = copy.deepcopy(root_node)
    nodes_info = get_all_nodes_with_parents(new_root)
    if not nodes_info:
        return new_root

    # Prefer non-root nodes to make the mutation effective.
    for _ in range(10):
        node, parent, _ = rng.choice(nodes_info)
        if parent is not None:
            try:
                infer_node_type(node)
                if gen_idx is not None:
                    log.warning(f'[GP][hoist_mutation][gen={gen_idx}] hoisted subtree as new root.')
                return copy.deepcopy(node)
            except TypeError:
                continue

    return new_root


def crossover_trees(tree_a: FactorNode, tree_b: FactorNode, rng: random.Random):
    child_a = copy.deepcopy(tree_a)
    child_b = copy.deepcopy(tree_b)

    try:
        infer_node_type(child_a)
        infer_node_type(child_b)
    except TypeError:
        return child_a, child_b

    nodes_a = get_all_nodes_with_parents(child_a)
    nodes_b = get_all_nodes_with_parents(child_b)

    for _ in range(20):
        node_a, parent_a, dir_a = rng.choice(nodes_a)
        node_b, parent_b, dir_b = rng.choice(nodes_b)
        if parent_a is None or parent_b is None:
            continue
        if getattr(node_a, 'data_type', None) != getattr(node_b, 'data_type', None):
            continue
        setattr(parent_a, dir_a, node_b)
        setattr(parent_b, dir_b, node_a)
        return child_a, child_b

    return child_a, child_b


@dataclass
class GPCandidate:
    node: FactorNode
    formula: str
    fitness: float
    original_fitness: float
    penalized_fitness: float


def _collect_yearly_metric_values(metric_df: pd.DataFrame,
                                   value_col: str) -> List[float]:
    values: List[float] = []
    if value_col not in metric_df.columns:
        return values
    for idx, val in metric_df[value_col].items():
        if str(idx) == 'all':
            continue
        if pd.isna(val):
            # fill the nan with 0, this can be helpful when calculating sharpe with zero volatility
            values.append(0.0)
            continue
        values.append(float(val))
    return values


def _calc_ic_fitness_from_factor_indicators(eval_df: pd.DataFrame,
                                            factor_col: str = 'factor') -> float:
    yearly_values: List[float] = []
    for _, df_ins in eval_df.groupby('instrument_id', sort=False):
        if len(df_ins) < 50:
            continue
        try:
            ic_df, _ = get_annualized_ts_ic_and_t_corr(
                df_ins[['time', 'instrument_id', 'future_ret', factor_col]].copy(),
                fc_col=factor_col,
                fc_freq='1d',
                portfolio_adjust_method='1D',
            )
        except Exception:
            continue

        vals = _collect_yearly_metric_values(ic_df, 'TS RankIC')
        yearly_values.extend(vals)

    if not yearly_values:
        return 0.0
    return float(np.mean(yearly_values))


def _calc_sharpe_fitness_from_factor_indicators(eval_df: pd.DataFrame,
                                                factor_col: str = 'factor') -> float:
    yearly_values: List[float] = []
    for _, df_ins in eval_df.groupby('instrument_id', sort=False):
        if len(df_ins) < 50:
            continue

        ret_df = df_ins[['time']].copy()
        signal = pd.to_numeric(df_ins[factor_col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
        future_ret = pd.to_numeric(df_ins['future_ret'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Keep GP fitness numerically stable on extreme formulas.
        signal = signal.clip(-20.0, 20.0)
        strategy_ret = (signal * future_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        strategy_ret = strategy_ret.clip(-1.0, 1.0)
        ret_df[factor_col] = strategy_ret

        try:
            # Use simple annualization to avoid cumprod overflow in compound path.
            annual_ret = get_annualized_ret(ret_df, factor_col, interest_method='simple')
            annual_vol = get_annualized_volatility(ret_df, factor_col)
            annual_sharpe = get_annualized_sharpe(annual_ret[[factor_col]], annual_vol[[factor_col]])
        except Exception:
            continue

        vals = _collect_yearly_metric_values(annual_sharpe, factor_col)
        yearly_values.extend(vals)

    if not yearly_values:
        return 0.0
    return float(np.mean(yearly_values))


def _metric_score(eval_df: pd.DataFrame, fitness_metric: str) -> float:
    if fitness_metric == 'ic':
        return _calc_ic_fitness_from_factor_indicators(eval_df, factor_col='factor')
    if fitness_metric == 'sharpe':
        return _calc_sharpe_fitness_from_factor_indicators(eval_df, factor_col='factor')
    raise ValueError(f'Unsupported fitness_metric: {fitness_metric}')


def _infer_root_instrument(instrument_id: Any) -> str:
    ins = str(instrument_id).upper().strip()
    if not ins:
        return ''
    if ins.endswith('0') and len(ins) > 1:
        return ins[:-1]
    m = re.match(r'([A-Z]+)', ins)
    return m.group(1) if m else ins


def _calc_small_factor_penalty(eval_df: pd.DataFrame,
                               factor_col: str,
                               open_col: str,
                               instrument_col: str,
                               assumed_initial_capital: float,
                               small_factor_penalty_coef: float) -> float:
    if small_factor_penalty_coef <= 0:
        return 0.0
    if assumed_initial_capital <= 0:
        return 0.0
    if factor_col not in eval_df.columns or open_col not in eval_df.columns or instrument_col not in eval_df.columns:
        return 0.0

    df = eval_df[[factor_col, open_col, instrument_col]].copy()
    df[factor_col] = pd.to_numeric(df[factor_col], errors='coerce')
    df[open_col] = pd.to_numeric(df[open_col], errors='coerce')

    root_series = df[instrument_col].map(_infer_root_instrument)
    multiplier_series = root_series.map(FUTURES_CONTRACT_MULTIPLIER)
    multiplier_series = pd.to_numeric(multiplier_series, errors='coerce')
    min_factor_to_open_1lot = (df[open_col] * multiplier_series) / float(assumed_initial_capital)

    valid = np.isfinite(df[factor_col]) & np.isfinite(min_factor_to_open_1lot) & (min_factor_to_open_1lot > 0)
    if not bool(valid.any()):
        return 0.0

    tradable_ratio = float((df.loc[valid, factor_col].abs() >= min_factor_to_open_1lot.loc[valid]).mean())
    return float(small_factor_penalty_coef) * (1.0 - tradable_ratio)


def calc_fitness_and_sign(tree: FactorNode,
                          df: pd.DataFrame,
                          fitness_metric: str = 'ic',
                          target_col: str = 'future_ret',
                          depth_penalty_coef: float = 0.0,
                          depth_penalty_start_depth: int = 0,
                          depth_penalty_linear_coef: float = 0.0,
                          depth_penalty_quadratic_coef: float = 0.0,
                          apply_rolling_norm: bool = True,
                          rolling_norm_window: int = 30,
                          rolling_norm_min_periods: int = 20,
                          rolling_norm_eps: float = 1e-8,
                          rolling_norm_clip: float = 10.0,
                          small_factor_penalty_coef: float = 0.0,
                          assumed_initial_capital: float = 1_000_000.0) -> Tuple[float, float, int]:
    try:
        factor_series = pd.to_numeric(tree.calc(df), errors='coerce')
        if apply_rolling_norm:
            norm_df = pd.DataFrame({'factor': factor_series})
            if 'time' in df.columns:
                norm_df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if 'instrument_id' in df.columns:
                norm_df['instrument_id'] = df['instrument_id']

            norm_df = rolling_normalize_features(
                df=norm_df,
                factor_cols='factor',
                rolling_norm_window=rolling_norm_window,
                rolling_norm_min_periods=rolling_norm_min_periods,
                rolling_norm_eps=rolling_norm_eps,
                rolling_norm_clip=rolling_norm_clip,
                instrument_col='instrument_id',
            )
            factor_series = norm_df['factor']

        eval_df = pd.DataFrame({
            'time': pd.to_datetime(df['time'], errors='coerce') if 'time' in df.columns else pd.NaT,
            'instrument_id': df['instrument_id'] if 'instrument_id' in df.columns else 'UNKNOWN',
            'open': pd.to_numeric(df['open'], errors='coerce') if 'open' in df.columns else np.nan,
            'future_ret': pd.to_numeric(df[target_col], errors='coerce'),
            'factor': pd.to_numeric(factor_series, errors='coerce'),
        }).dropna(subset=['future_ret', 'factor'])
        if eval_df.empty:
            return 0.0, 0.0, 1

        fit_pos = _metric_score(eval_df, fitness_metric)
        eval_df_neg = eval_df.copy()
        eval_df_neg['factor'] = -eval_df_neg['factor']
        fit_neg = _metric_score(eval_df_neg, fitness_metric)

        # Guard against NaN/inf from metric calculations.
        if not np.isfinite(fit_pos):
            fit_pos = 0.0
        if not np.isfinite(fit_neg):
            fit_neg = 0.0

        best_fit = fit_neg if fit_neg > fit_pos else fit_pos
        sign = -1 if fit_neg > fit_pos else 1
        if not np.isfinite(best_fit):
            best_fit = 0.0
        original_best_fit = float(best_fit)

        depth_penalty = _calc_depth_penalty(
            depth=get_tree_depth(tree),
            depth_penalty_coef=depth_penalty_coef,
            depth_penalty_start_depth=depth_penalty_start_depth,
            depth_penalty_linear_coef=depth_penalty_linear_coef,
            depth_penalty_quadratic_coef=depth_penalty_quadratic_coef,
        )
        if depth_penalty > 0:
            best_fit = best_fit - depth_penalty

        small_factor_penalty = _calc_small_factor_penalty(
            eval_df=eval_df,
            factor_col='factor',
            open_col='open',
            instrument_col='instrument_id',
            assumed_initial_capital=assumed_initial_capital,
            small_factor_penalty_coef=small_factor_penalty_coef,
        )
        if small_factor_penalty > 0:
            best_fit = best_fit - small_factor_penalty

        return float(best_fit), original_best_fit, sign
    except Exception:
        return 0.0, 0.0, 1


class EliteArchive:
    """精英库（Elite Archive）：维护一组低相关、高适应度的精英因子。

    在 GP 进化过程中，每当一个新因子评分完毕，都会尝试加入精英库。
    精英库通过相关性检测保证库内因子的多样性（低相关），同时通过
    适应度比较保证每个"流派"只保留最强个体。

    核心逻辑
    --------
    1. **查重**：计算新因子与库内所有精英的 Pearson 相关系数绝对值，
       找出所有 ``|corr| > elite_relative_threshold`` 的"同流派"精英。
    2. **同流派 PK（情况 A）**：若存在同流派精英——
       - 新因子 fitness **高于所有** 同流派精英的最强者 → 删除全部同流派精英，
         新因子入库（"合并抹杀"，去冗余 + 收敛）。
       - 否则 → 新因子被淘汰（劣质克隆体）。
    3. **全新流派（情况 B）**：若不存在同流派精英——
       - 库未满 → 无条件入库（冷启动，收集多样性）。
       - 库已满 → 与库内 fitness 最低的"垫底精英"比较：
         * 新因子 > 垫底精英 → 踢掉垫底，新因子入库（门槛自动提升）。
         * 否则 → 淘汰。

    注意：合并抹杀可能使库容量短暂低于 ``max_size``，这属于正常的
    "基因收敛与去冗余"，空出的名额会被后续全新流派填充。

    Parameters
    ----------
    max_size : int
        精英库最大容量（对应原 ``elite_size`` 参数）。
    elite_relative_threshold : float, default 0.75
        判定"同流派"的相关性阈值。若新因子与某精英的
        ``|Pearson corr| > elite_relative_threshold``，则视为同流派。
    min_corr_samples : int, default 10
        计算相关系数时要求的最少有效（非 NaN / 非 Inf）样本数。
        样本不足时视为不相关（corr = 0）。
    """

    def __init__(self,
                 max_size: int,
                 elite_relative_threshold: float = 0.75,
                 min_corr_samples: int = 10):
        self.max_size: int = max(1, int(max_size))
        self.elite_relative_threshold: float = float(elite_relative_threshold)
        self.min_corr_samples: int = max(2, int(min_corr_samples))
        # 内部存储：每个精英为 (FactorNode, fitness, factor_values_ndarray)
        self._elites: List[Tuple[FactorNode, float, np.ndarray]] = []

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def try_add(self,
                node: FactorNode,
                fitness: float,
                factor_values: np.ndarray) -> Tuple[bool, int]:
        """尝试将一个新因子加入精英库。

        Parameters
        ----------
        node : FactorNode
            新因子的 AST 节点（会被 deepcopy 后存入）。
        fitness : float
            新因子的适应度分数（penalized_fitness）。
        factor_values : np.ndarray
            新因子在全量数据上的因子值（一维数组），用于计算相关性。

        Returns
        -------
        Tuple[bool, int]
            (added, removed_count)
            - added: True 表示新因子成功入库
            - removed_count: 被踢出的精英数量（用于统计变动规模）
        """
        # ---- 空库 → 直接入库 ----
        if len(self._elites) == 0:
            self._elites.append((copy.deepcopy(node), fitness, factor_values.copy()))
            return True, 0

        # ---- 第一步：与库内所有精英计算相关系数绝对值 ----
        correlations = np.array([
            self._calc_abs_corr(factor_values, elite_vals)
            for _, _, elite_vals in self._elites
        ])

        # ---- 找出所有"同流派"精英（相关系数 > 阈值）----
        similar_mask = correlations > self.elite_relative_threshold
        similar_indices = np.where(similar_mask)[0]

        if len(similar_indices) > 0:
            # ============================================================
            # 情况 A：存在同流派精英 —— 直接 PK，无需看容量
            # ============================================================
            # 找出同流派中 fitness 最高的"最强者"
            strongest_fitness = max(self._elites[i][1] for i in similar_indices)

            if fitness > strongest_fitness:
                # 新因子比所有同流派精英都强 → "合并抹杀"
                # 从后往前删除，避免索引偏移
                for idx in sorted(similar_indices, reverse=True):
                    self._elites.pop(idx)
                self._elites.append((copy.deepcopy(node), fitness, factor_values.copy()))
                log.debug(
                    f'[EliteArchive] 新因子(fitness={fitness:.6f})击败 '
                    f'{len(similar_indices)} 个同流派精英，合并入库。'
                    f'当前库容量: {len(self._elites)}/{self.max_size}'
                )
                return True, int(len(similar_indices))
            else:
                # 新因子不如最强者 → 淘汰
                return False, 0
        else:
            # ============================================================
            # 情况 B：全新流派（与所有精英的相关性均 <= 阈值）
            # ============================================================
            if len(self._elites) < self.max_size:
                # 库未满 → 无条件入库（冷启动，收集多样性）
                self._elites.append((copy.deepcopy(node), fitness, factor_values.copy()))
                log.debug(
                    f'[EliteArchive] 新流派因子(fitness={fitness:.6f})入库（冷启动）。'
                    f'当前库容量: {len(self._elites)}/{self.max_size}'
                )
                return True, 0
            else:
                # 库已满 → 和垫底精英 PK
                worst_idx = min(range(len(self._elites)), key=lambda i: self._elites[i][1])
                worst_fitness = self._elites[worst_idx][1]

                if fitness > worst_fitness:
                    # 踢掉垫底精英，新因子入库（准入门槛自动提升）
                    self._elites.pop(worst_idx)
                    self._elites.append((copy.deepcopy(node), fitness, factor_values.copy()))
                    log.debug(
                        f'[EliteArchive] 新流派因子(fitness={fitness:.6f})踢掉垫底精英'
                        f'(fitness={worst_fitness:.6f})入库。'
                        f'当前库容量: {len(self._elites)}/{self.max_size}'
                    )
                    return True, 1
                else:
                    # 连垫底都不如 → 淘汰
                    return False, 0

    def get_elites_sorted(self) -> List[Tuple[FactorNode, float]]:
        """返回当前精英库中所有精英，按 fitness 从高到低排序。

        Returns
        -------
        List[Tuple[FactorNode, float]]
            ``(node, fitness)`` 列表，节点已做 deepcopy。
        """
        sorted_elites = sorted(self._elites, key=lambda x: x[1], reverse=True)
        return [(copy.deepcopy(e[0]), e[1]) for e in sorted_elites]

    @property
    def size(self) -> int:
        """当前精英库中的精英数量。"""
        return len(self._elites)

    def summary_str(self) -> str:
        """返回精英库的摘要信息字符串，用于日志输出。"""
        if not self._elites:
            return f'EliteArchive(size=0/{self.max_size})'
        fitnesses = [f for _, f, _ in self._elites]
        return (
            f'EliteArchive(size={len(self._elites)}/{self.max_size}, '
            f'best={max(fitnesses):.6f}, worst={min(fitnesses):.6f}, '
            f'mean={np.mean(fitnesses):.6f})'
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _calc_abs_corr(self, values_a: np.ndarray, values_b: np.ndarray) -> float:
        """计算两个因子值序列的 Pearson 相关系数的绝对值。

        仅在共同非 NaN / 非 Inf 位置上计算。样本不足时返回 0.0。

        Parameters
        ----------
        values_a, values_b : np.ndarray
            等长的一维因子值数组。

        Returns
        -------
        float
            ``|Pearson r|``，范围 [0, 1]。无法计算时返回 0.0。
        """
        valid = np.isfinite(values_a) & np.isfinite(values_b)
        n_valid = int(valid.sum())
        if n_valid < self.min_corr_samples:
            return 0.0
        a = values_a[valid]
        b = values_b[valid]
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-12 or std_b < 1e-12:
            # 常数序列视为不相关
            return 0.0
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            return 0.0
        return abs(float(corr))


def _tournament_select(scored_pop: List[Tuple[FactorNode, float]],
                       tournament_size: int,
                       rng: random.Random) -> FactorNode:
    """
    从 scored_pop 随机抽 k=min(tournament_size, len(scored_pop)) 个候选
    按 fitness 从高到低排序
    返回这批候选中最好的那个

    tournament_size 越大，越偏向强者；越小，随机性越强。
    """
    participants = rng.sample(scored_pop, k=min(tournament_size, len(scored_pop)))
    participants.sort(key=lambda x: x[1], reverse=True)
    return participants[0][0]


def run_gp_evolution(
    df: pd.DataFrame,
    data_fields: Sequence[str],
    fitness_metric: str,
    max_factor_count: int,
    generations: int,
    population_size: int,
    max_depth: int,
    elite_size: int,
    tournament_size: int,
    crossover_prob: float,
    mutation_prob: float,
    window_choices: Sequence[int],
    const_prob: float,
    leaf_prob: float,
    random_seed: Optional[int] = None,
    early_stopping_generation_count: int = 8,
    log_interval: int = 5,
    depth_penalty_coef: float = 0.0,
    depth_penalty_start_depth: int = 0,
    depth_penalty_linear_coef: float = 0.0,
    depth_penalty_quadratic_coef: float = 0.0,
    apply_rolling_norm: bool = True,
    rolling_norm_window: int = 30,
    rolling_norm_min_periods: int = 20,
    rolling_norm_eps: float = 1e-8,
    rolling_norm_clip: float = 10.0,
    small_factor_penalty_coef: float = 0.0,
    assumed_initial_capital: float = 1_000_000.0,
    elite_relative_threshold: float = 0.75,
    elite_stagnation_generation_count: int = 5,
    max_shock_generation: int = 3,
) -> List[GPCandidate]:
    """运行遗传规划（GP）进行因子挖掘。
+
+    参数说明（中文）：
+    - df: 用于 GP 评估的原始数据（含 time/instrument_id/OHLCV 等列）。
+    - data_fields: 叶子节点可用的字段名列表（如 open/high/low/close/volume/position）。
+    - fitness_metric: 适应度指标（ic 或 sharpe）。
+    - max_factor_count: 最终返回的因子数量上限。
+    - generations: 演化代数。
+    - population_size: 每代种群规模。
+    - max_depth: 树的最大深度。
+    - elite_size: 精英库最大容量（Elite Archive）。
+    - tournament_size: 锦标赛选择规模，越大越偏强者。
+    - crossover_prob: 交叉概率。
+    - mutation_prob: 变异概率。
+    - window_choices: 时序算子可用的窗口长度集合。
+    - const_prob: 叶子节点选常数的概率。
+    - leaf_prob: 生成叶子节点的概率（控制树的复杂度）。
+    - random_seed: 随机种子。
+    - early_stopping_generation_count: 连续多少代无提升则早停（<=0 表示关闭）。
+    - log_interval: 多少代输出一次汇总日志。
+    - depth_penalty_coef: 深度惩罚系数（线性部分）。
+    - depth_penalty_start_depth: 深度惩罚的起始深度。
+    - depth_penalty_linear_coef: 深度惩罚线性系数（超过起始深度后）。
+    - depth_penalty_quadratic_coef: 深度惩罚二次系数（超过起始深度后）。
+    - apply_rolling_norm: 是否对因子值做滚动归一化。
+    - rolling_norm_window: 滚动归一化窗口。
+    - rolling_norm_min_periods: 滚动归一化的最小样本数。
+    - rolling_norm_eps: 滚动归一化的数值稳定项。
+    - rolling_norm_clip: 滚动归一化后的截断范围。
+    - small_factor_penalty_coef: 小因子惩罚系数（越大越惩罚不可交易因子）。
+    - assumed_initial_capital: 小因子惩罚中的资金假设。
+    - elite_relative_threshold: 精英库“同流派”判定阈值（相关性）。
+    - elite_stagnation_generation_count: 连续多少代精英库无更新触发 Shock。
+    - max_shock_generation: Shock 模式最多持续代数（无更新则退出）。
+    """
    rng = random.Random(random_seed)

    if log_interval <= 0:
        log_interval = 1

    early_stopping_generation_count = int(early_stopping_generation_count)

    log.info(
        'GP start: '
        f'fitness_metric={fitness_metric}, generations={generations}, population_size={population_size}, '
        f'max_depth={max_depth}, elite_size={elite_size}, elite_relative_threshold={elite_relative_threshold}, '
        f'max_factor_count={max_factor_count}, '
        f'elite_stagnation_generation_count={elite_stagnation_generation_count}, '
        f'max_shock_generation={max_shock_generation}, '
        f'early_stopping_generation_count={early_stopping_generation_count}, '
        f'depth_penalty_coef={depth_penalty_coef}, '
        f'depth_penalty_start_depth={depth_penalty_start_depth}, '
        f'depth_penalty_linear_coef={depth_penalty_linear_coef}, '
        f'depth_penalty_quadratic_coef={depth_penalty_quadratic_coef}, '
        f'apply_rolling_norm={apply_rolling_norm}, '
        f'small_factor_penalty_coef={small_factor_penalty_coef}, '
        f'assumed_initial_capital={assumed_initial_capital}'
    )

    # ------------------------------------------------------------------
    # 初始化精英库（Elite Archive）
    # 精英库在整个进化过程中持续维护，跨代累积优秀且多样的因子。
    # ------------------------------------------------------------------
    elite_archive = EliteArchive(
        max_size=elite_size,
        elite_relative_threshold=elite_relative_threshold,
    )

    population: List[FactorNode] = []
    for i in range(population_size):
        population.append(
            _generate_valid_random_tree(
                data_fields,
                max_depth,
                0,
                window_choices,
                const_prob,
                leaf_prob,
                rng,
                log_context=f'[GP][init] seed tree {i + 1}/{population_size}',
            )
        )

    global_best: Dict[str, GPCandidate] = {}
    best_round_fitness = -np.inf
    no_improve_generations = 0

    elite_stagnation_generation_count = max(1, int(elite_stagnation_generation_count))
    max_shock_generation = max(1, int(max_shock_generation))
    mode = 'NORMAL'
    stagnation_count = 0
    shock_count = 0

    for gen_idx in range(generations):
        log.info(f'GP generation {gen_idx + 1}/{generations} scoring started.')
        scored_pop: List[Tuple[FactorNode, float]] = []
        penalized_fitness_values: List[float] = []
        original_fitness_values: List[float] = []
        round_best = -np.inf
        round_best_original = -np.inf
        elite_archive_updated = False
        elite_add_count = 0
        elite_remove_count = 0
        tree_log_interval = max(1, int(population_size / 10))
        for tree_idx, tree in enumerate(population, start=1):  # 每棵树代表了一个因子
            penalized_fitness, original_fitness, sign = calc_fitness_and_sign(
                tree,
                df,
                fitness_metric=fitness_metric,
                depth_penalty_coef=depth_penalty_coef,
                depth_penalty_start_depth=depth_penalty_start_depth,
                depth_penalty_linear_coef=depth_penalty_linear_coef,
                depth_penalty_quadratic_coef=depth_penalty_quadratic_coef,
                apply_rolling_norm=apply_rolling_norm,
                rolling_norm_window=rolling_norm_window,
                rolling_norm_min_periods=rolling_norm_min_periods,
                rolling_norm_eps=rolling_norm_eps,
                rolling_norm_clip=rolling_norm_clip,
                small_factor_penalty_coef=small_factor_penalty_coef,
                assumed_initial_capital=assumed_initial_capital,
            )
            scored_pop.append((tree, penalized_fitness))
            penalized_fitness_values.append(float(penalized_fitness))
            original_fitness_values.append(float(original_fitness))
            if penalized_fitness > round_best:
                round_best = penalized_fitness
            if original_fitness > round_best_original:
                round_best_original = original_fitness
            """
            global_best是一个历史档案字典：
            键：formula（因子公式字符串）
            值：该公式历史上出现过的最佳 GPCandidate(node, formula, fitness)
            
            ConstNode.to_formula() 目前是：
            f"{self.value:.6g}"（只保留 6 位有效数字）
            这会导致不同常数值被打印成同一个字符串，例如：
            0.12345671 和 0.12345674 都可能显示成 0.123457
            于是会出现：
            结构类似、常数略不同的两个树，formula 字符串相同
            实际计算值不同 -> fitness 不同
            """
            oriented_node = copy.deepcopy(tree) if sign > 0 else OpNeg(copy.deepcopy(tree))
            formula = oriented_node.to_formula()
            old = global_best.get(formula)
            if old is None or penalized_fitness > old.fitness:  # 对每个公式，只保留历史上 penalized fitness 最高的一次
                global_best[formula] = GPCandidate(
                    node=oriented_node,
                    formula=formula,
                    fitness=penalized_fitness,
                    original_fitness=original_fitness,
                    penalized_fitness=penalized_fitness,
                )

            # ----------------------------------------------------------
            # 尝试将当前因子加入精英库（Elite Archive）
            # 使用原始（未取向）的因子值进行相关性比较，因为 |corr| 对
            # 符号翻转不变，无需使用 oriented_node。
            # ----------------------------------------------------------
            try:
                factor_values_for_archive = pd.to_numeric(
                    tree.calc(df), errors='coerce'
                ).values.astype(float)
                added, removed_count = elite_archive.try_add(tree, penalized_fitness, factor_values_for_archive)
                if added or removed_count > 0:
                    elite_archive_updated = True
                elite_add_count += int(added)
                elite_remove_count += int(removed_count)
            except Exception as e:
                log.warning(f'GP generation {gen_idx + 1}/{generations} tree {tree_idx}/{population_size}: '
                            f'factor value calculation failed with error: {e}. Skipping EliteArchive insertion.')
                pass  # 因子值计算异常时跳过，不影响主流程

            """
            当前树进度：tree i/population_size
            当前树 fitness：fitness=...
            本代已见最好值：round_best=...
            累积唯一公式数：unique_formulas=...
            """
            if tree_idx == 1 or tree_idx % tree_log_interval == 0 or tree_idx == population_size:
                log.info(
                    f'GP generation {gen_idx + 1}/{generations} tree {tree_idx}/{population_size}: '
                    f'penalized_fitness={penalized_fitness:.6f}, original_fitness={original_fitness:.6f}, '
                    f'round_best_penalized={round_best:.6f}, round_best_original={round_best_original:.6f}, '
                    f'unique_formulas={len(global_best)}'
                )

        scored_pop.sort(key=lambda x: x[1], reverse=True)

        # if elite_add_count > 0 or elite_remove_count > 0:
        merged_count = max(0, elite_remove_count - elite_add_count)
        log.info(
            '[EliteArchive][Update] '
            f'gen={gen_idx + 1}, '
            f'added={elite_add_count}, '
            f'removed={elite_remove_count}, '
            f'merged_removed={merged_count}, '
            f'current_size={elite_archive.size}/{elite_archive.max_size}'
        )

        # -----------------------
        # Shock state machine
        # -----------------------
        if mode == 'NORMAL':
            if elite_archive_updated:
                stagnation_count = 0
            else:
                stagnation_count += 1
            if stagnation_count >= elite_stagnation_generation_count:
                mode = 'SHOCK'
                shock_count = 0
                log.info(f'GP shock mode entered at generation {gen_idx + 1}: elite archive stagnated.')
        else:
            if elite_archive_updated:
                mode = 'NORMAL'
                stagnation_count = 0
                shock_count = 0
                log.info(f'GP shock mode exited at generation {gen_idx + 1}: elite archive updated.')
            else:
                shock_count += 1
                if shock_count >= max_shock_generation:
                    mode = 'NORMAL'
                    stagnation_count = 0
                    shock_count = 0
                    log.info(
                        f'GP shock mode exited at generation {gen_idx + 1}: '
                        f'no update for {max_shock_generation} generations.'
                    )

        if scored_pop:
            best_fitness_current = float(scored_pop[0][1])
            avg_fitness_current = float(np.mean(penalized_fitness_values))
            best_original_fitness_current = float(max(original_fitness_values))
            avg_original_fitness_current = float(np.mean(original_fitness_values))
            global_best_fitness = float(max(x.fitness for x in global_best.values())) if global_best else 0.0
            global_best_original_fitness = float(max(x.original_fitness for x in global_best.values())) if global_best else 0.0
        else:
            best_fitness_current = 0.0
            avg_fitness_current = 0.0
            best_original_fitness_current = 0.0
            avg_original_fitness_current = 0.0
            global_best_fitness = 0.0
            global_best_original_fitness = 0.0

        should_log = ((gen_idx + 1) % log_interval == 0) or (gen_idx == 0) or (gen_idx == generations - 1)
        if should_log:
            log.info(
                '=' * 100,
                f'GP generation {gen_idx + 1}/{generations}: '
                f'current_best_penalized={best_fitness_current:.6f}, current_avg_penalized={avg_fitness_current:.6f}, '
                f'current_best_original={best_original_fitness_current:.6f}, '
                f'current_avg_original={avg_original_fitness_current:.6f}, '
                f'global_best_penalized={global_best_fitness:.6f}, '
                f'global_best_original={global_best_original_fitness:.6f}, '
                f'unique_formulas={len(global_best)}, '
                f'{elite_archive.summary_str()}',
                '=' * 100
            )

        if best_fitness_current > best_round_fitness:
            best_round_fitness = best_fitness_current
            no_improve_generations = 0
        else:
            no_improve_generations += 1

        if 0 < early_stopping_generation_count <= no_improve_generations:
            log.info(
                f'GP early stopping triggered at generation {gen_idx + 1}/{generations}: '
                f'no improvement in round_best for {no_improve_generations} consecutive generations. '
                f'best_round_fitness={best_round_fitness:.6f}'
            )
            break

        # ------------------------------------------------------------------
        # 构建下一代种群（next_gen）
        # 精英库（Elite Archive）中的个体优先注入下一代，保证优秀且多样
        # 的基因不丢失。剩余名额通过锦标赛选择 + 交叉/变异/复制填充。
        # ------------------------------------------------------------------
        effective_crossover_prob = crossover_prob
        effective_mutation_prob = mutation_prob
        effective_tournament_size = tournament_size
        if mode == 'SHOCK':
            effective_crossover_prob = SHOCK_CROSSOVER_PROB
            effective_mutation_prob = SHOCK_MUTATION_PROB
            effective_tournament_size = SHOCK_TOURNAMENT_SIZE

        archive_elites = elite_archive.get_elites_sorted()
        next_gen: List[FactorNode] = [copy.deepcopy(node) for node, _ in archive_elites]
        next_gen_log_interval = max(1, int(population_size / 5))
        if next_gen:
            log.info(
                f'[MODE={mode}] GP generation {gen_idx + 1}/{generations} next_gen seeded from EliteArchive: '
                f'{len(next_gen)}/{population_size} ({elite_archive.summary_str()})'
            )

        while len(next_gen) < population_size:  # 把 next_gen 补满到 population_size。
            r = rng.random()
            if r < effective_crossover_prob and len(scored_pop) >= 2:  # 交叉分支
                # tournament_size 越大，越偏向强者；越小，随机性越强
                p1 = _tournament_select(scored_pop, effective_tournament_size, rng)
                p2 = _tournament_select(scored_pop, effective_tournament_size, rng)
                c1, c2 = crossover_trees(p1, p2, rng)
                next_gen.append(c1)
                if len(next_gen) < population_size:
                    next_gen.append(c2)
            elif r < effective_crossover_prob + effective_mutation_prob:  # 变异分支
                p = _tournament_select(scored_pop, effective_tournament_size, rng)
                if mode == 'SHOCK':
                    shock_pick = rng.random()
                    if shock_pick < SHOCK_ROOT_CUT_PROB:
                        m = macro_subtree_mutation(
                            p,
                            data_fields,
                            max_depth,
                            window_choices,
                            const_prob,
                            leaf_prob,
                            rng,
                            gen_idx=gen_idx + 1,
                        )
                    elif shock_pick < SHOCK_ROOT_CUT_PROB + SHOCK_HOIST_PROB:
                        m = hoist_mutation(p, rng, gen_idx=gen_idx + 1)
                    else:
                        m = mutate_tree(
                            p,
                            data_fields,
                            max_depth,
                            window_choices,
                            const_prob,
                            leaf_prob,
                            rng,
                            gen_idx=gen_idx + 1,
                        )
                else:
                    m = mutate_tree(
                        p,
                        data_fields,
                        max_depth,
                        window_choices,
                        const_prob,
                        leaf_prob,
                        rng,
                        gen_idx=gen_idx + 1,
                    )
                next_gen.append(m)
            else:  # 复制分支
                p = _tournament_select(scored_pop, effective_tournament_size, rng)
                next_gen.append(copy.deepcopy(p))

            if len(next_gen) % next_gen_log_interval == 0 or len(next_gen) == population_size:
                log.info(
                    f'GP generation {gen_idx + 1}/{generations} next_gen building progress: '
                    f'{len(next_gen)}/{population_size}'
                )

        population = next_gen[:population_size]

    candidates = sorted(global_best.values(), key=lambda x: x.fitness, reverse=True)
    log.info(
        f'GP finished: total_unique_formulas={len(global_best)}, '
        f'return_top_n={min(max_factor_count, len(candidates))}'
    )
    return candidates[:max_factor_count]

