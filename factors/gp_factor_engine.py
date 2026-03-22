"""Genetic programming engine for factor mining.

This module is self-contained so it can be reused by framework classes
without changing existing gp.py scripts.
"""

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from utils.logging import log
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
)


def generate_random_tree(
    data_fields: Sequence[str],
    max_depth: int,
    current_depth: int,
    window_choices: Sequence[int],
    const_prob: float,
    leaf_prob: float,
    rng: random.Random,
) -> FactorNode:
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
        left = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        right = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        return op_cls(left, right)

    if op_pick < 0.75:
        op_cls = rng.choice(UNARY_OPS)
        child = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        return op_cls(child)

    if op_pick < 0.9:
        op_cls = rng.choice(UNARY_TS_OPS)
        child = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        return op_cls(child, int(rng.choice(list(window_choices))))

    op_cls = rng.choice(BINARY_TS_OPS)
    left = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
    right = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
    return op_cls(left, right, int(rng.choice(list(window_choices))))


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
) -> FactorNode:
    new_root = copy.deepcopy(root_node)
    nodes_info = get_all_nodes_with_parents(new_root)
    target_node, parent, direction = rng.choice(nodes_info)

    if parent is None:
        return generate_random_tree(data_fields, max_depth, 0, window_choices, const_prob, leaf_prob, rng)

    new_branch = generate_random_tree(data_fields, min(max_depth, 3), 0, window_choices, const_prob, leaf_prob, rng)
    setattr(parent, direction, new_branch)
    _ = target_node
    return new_root


def crossover_trees(tree_a: FactorNode, tree_b: FactorNode, rng: random.Random):
    child_a = copy.deepcopy(tree_a)
    child_b = copy.deepcopy(tree_b)

    nodes_a = get_all_nodes_with_parents(child_a)
    nodes_b = get_all_nodes_with_parents(child_b)

    node_a, parent_a, dir_a = rng.choice(nodes_a)
    node_b, parent_b, dir_b = rng.choice(nodes_b)

    if parent_a is None or parent_b is None:
        return child_a, child_b

    setattr(parent_a, dir_a, node_b)
    setattr(parent_b, dir_b, node_a)
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
                          rolling_norm_clip: float = 10.0) -> Tuple[float, float, int]:
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
            'future_ret': pd.to_numeric(df[target_col], errors='coerce'),
            'factor': pd.to_numeric(factor_series, errors='coerce'),
        }).dropna(subset=['future_ret', 'factor'])
        if eval_df.empty:
            return 0.0, 0.0, 1

        fit_pos = _metric_score(eval_df, fitness_metric)
        eval_df_neg = eval_df.copy()
        eval_df_neg['factor'] = -eval_df_neg['factor']
        fit_neg = _metric_score(eval_df_neg, fitness_metric)

        best_fit = fit_neg if fit_neg > fit_pos else fit_pos
        sign = -1 if fit_neg > fit_pos else 1
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

        return float(best_fit), original_best_fit, sign
    except Exception:
        return 0.0, 0.0, 1


def _tournament_select(scored_pop: List[Tuple[FactorNode, float]],
                       tournament_size: int,
                       rng: random.Random) -> FactorNode:
    """
    从 scored_pop 随机抽 k=min(tournament_size, len(scored_pop)) 个候选
    按 fitness 从高到低排序
    返回这批里最好的那个

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
) -> List[GPCandidate]:
    rng = random.Random(random_seed)

    if log_interval <= 0:
        log_interval = 1

    early_stopping_generation_count = int(early_stopping_generation_count)

    log.info(
        'GP start: '
        f'fitness_metric={fitness_metric}, generations={generations}, population_size={population_size}, '
        f'max_depth={max_depth}, elite_size={elite_size}, max_factor_count={max_factor_count}, '
        f'early_stopping_generation_count={early_stopping_generation_count}, '
        f'depth_penalty_coef={depth_penalty_coef}, '
        f'depth_penalty_start_depth={depth_penalty_start_depth}, '
        f'depth_penalty_linear_coef={depth_penalty_linear_coef}, '
        f'depth_penalty_quadratic_coef={depth_penalty_quadratic_coef}, '
        f'apply_rolling_norm={apply_rolling_norm}'
    )

    population = [
        generate_random_tree(data_fields, max_depth, 0, window_choices, const_prob, leaf_prob, rng)
        for _ in range(population_size)
    ]

    global_best: Dict[str, GPCandidate] = {}
    best_round_fitness = -np.inf
    no_improve_generations = 0

    for gen_idx in range(generations):
        log.info(f'GP generation {gen_idx + 1}/{generations} scoring started.')
        scored_pop: List[Tuple[FactorNode, float]] = []
        penalized_fitness_values: List[float] = []
        original_fitness_values: List[float] = []
        round_best = -np.inf
        round_best_original = -np.inf
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
                f'GP generation {gen_idx + 1}/{generations}: '
                f'current_best_penalized={best_fitness_current:.6f}, current_avg_penalized={avg_fitness_current:.6f}, '
                f'current_best_original={best_original_fitness_current:.6f}, '
                f'current_avg_original={avg_original_fitness_current:.6f}, '
                f'global_best_penalized={global_best_fitness:.6f}, '
                f'global_best_original={global_best_original_fitness:.6f}, '
                f'unique_formulas={len(global_best)}'
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
        # scored_pop：当前代全部个体和 fitness（已排序）
        # next_gen先放入精英保留（elite_size 个，直接复制）
        next_gen: List[FactorNode] = [copy.deepcopy(x[0]) for x in scored_pop[:min(elite_size, len(scored_pop))]]
        next_gen_log_interval = max(1, int(population_size / 5))
        if next_gen:
            log.info(
                f'GP generation {gen_idx + 1}/{generations} next_gen initialized with elites: '
                f'{len(next_gen)}/{population_size}'
            )

        while len(next_gen) < population_size:  # 把 next_gen 补满到 population_size。
            r = rng.random()
            if r < crossover_prob and len(scored_pop) >= 2:  # 交叉分支
                # tournament_size 越大，越偏向强者；越小，随机性越强
                p1 = _tournament_select(scored_pop, tournament_size, rng)
                p2 = _tournament_select(scored_pop, tournament_size, rng)
                c1, c2 = crossover_trees(p1, p2, rng)
                next_gen.append(c1)
                if len(next_gen) < population_size:
                    next_gen.append(c2)
            elif r < crossover_prob + mutation_prob:  # 变异分支
                p = _tournament_select(scored_pop, tournament_size, rng)
                m = mutate_tree(p, data_fields, max_depth, window_choices, const_prob, leaf_prob, rng)
                next_gen.append(m)
            else:  # 复制分支
                p = _tournament_select(scored_pop, tournament_size, rng)
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

