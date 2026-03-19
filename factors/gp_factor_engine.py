"""Genetic programming engine for factor mining.

This module is self-contained so it can be reused by framework classes
without changing existing gp.py scripts.
"""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.logging import log
from .factor_indicators import (
    get_annualized_ret,
    get_annualized_sharpe,
    get_annualized_ts_ic_and_t_corr,
    get_annualized_volatility,
)


class FactorNode:
    """Base class for AST nodes."""

    def calc(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def to_formula(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_formula()


class DataNode(FactorNode):
    def __init__(self, field: str):
        self.field = field

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[self.field], errors='coerce')

    def to_formula(self) -> str:
        return self.field


class ConstNode(FactorNode):
    def __init__(self, value: float):
        self.value = float(value)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)

    def to_formula(self) -> str:
        return f"{self.value:.6g}"


class OpAdd(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) + self.right.calc(df)

    def to_formula(self) -> str:
        return f"Add({self.left}, {self.right})"


class OpSub(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) - self.right.calc(df)

    def to_formula(self) -> str:
        return f"Sub({self.left}, {self.right})"


class OpMul(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) * self.right.calc(df)

    def to_formula(self) -> str:
        return f"Mul({self.left}, {self.right})"


class OpDiv(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        denominator = self.right.calc(df).replace(0, np.nan)
        return self.left.calc(df) / denominator

    def to_formula(self) -> str:
        return f"Div({self.left}, {self.right})"


class OpNeg(FactorNode):
    # 取负算子不会出现在原子算子列表中，因为这里会人工确认方向，保证了更快的收敛速度。
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return -self.child.calc(df)

    def to_formula(self) -> str:
        return f"Neg({self.child})"


class OpTsMean(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.calc(df)
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).mean())
        return s.rolling(self.window).mean()

    def to_formula(self) -> str:
        return f"TsMean({self.child}, {self.window})"


class OpTsStd(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.calc(df)
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).std())
        return s.rolling(self.window).std()

    def to_formula(self) -> str:
        return f"TsStd({self.child}, {self.window})"


BINARY_OPS = [OpAdd, OpSub, OpMul, OpDiv]
UNARY_TS_OPS = [OpTsMean, OpTsStd]


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

    if rng.random() < 0.65:
        op_cls = rng.choice(BINARY_OPS)
        left = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        right = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
        return op_cls(left, right)

    op_cls = rng.choice(UNARY_TS_OPS)
    child = generate_random_tree(data_fields, max_depth, current_depth + 1, window_choices, const_prob, leaf_prob, rng)
    return op_cls(child, int(rng.choice(list(window_choices))))


def get_all_nodes_with_parents(node: FactorNode, parent=None, direction: Optional[str] = None):
    nodes = [(node, parent, direction)]
    if isinstance(node, (OpAdd, OpSub, OpMul, OpDiv)):
        nodes.extend(get_all_nodes_with_parents(node.left, node, 'left'))
        nodes.extend(get_all_nodes_with_parents(node.right, node, 'right'))
    elif isinstance(node, (OpTsMean, OpTsStd)):
        nodes.extend(get_all_nodes_with_parents(node.child, node, 'child'))
    elif isinstance(node, OpNeg):
        nodes.extend(get_all_nodes_with_parents(node.child, node, 'child'))
    return nodes


def get_tree_depth(node: FactorNode) -> int:
    """Return max depth of AST, where leaf depth is 1."""
    if isinstance(node, (DataNode, ConstNode)):
        return 1
    if isinstance(node, (OpAdd, OpSub, OpMul, OpDiv)):
        return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))
    if isinstance(node, (OpTsMean, OpTsStd, OpNeg)):
        return 1 + get_tree_depth(node.child)
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
        ret_df[factor_col] = pd.to_numeric(df_ins[factor_col], errors='coerce').fillna(0.0) * \
            pd.to_numeric(df_ins['future_ret'], errors='coerce').fillna(0.0)

        try:
            annual_ret = get_annualized_ret(ret_df, factor_col, interest_method='compound')
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
                          depth_penalty_quadratic_coef: float = 0.0) -> Tuple[float, int]:
    try:
        factor_series = tree.calc(df)
        eval_df = pd.DataFrame({
            'time': pd.to_datetime(df['time'], errors='coerce') if 'time' in df.columns else pd.NaT,
            'instrument_id': df['instrument_id'] if 'instrument_id' in df.columns else 'UNKNOWN',
            'future_ret': pd.to_numeric(df[target_col], errors='coerce'),
            'factor': pd.to_numeric(factor_series, errors='coerce'),
        }).dropna(subset=['future_ret', 'factor'])
        if eval_df.empty:
            return 0.0, 1

        fit_pos = _metric_score(eval_df, fitness_metric)
        eval_df_neg = eval_df.copy()
        eval_df_neg['factor'] = -eval_df_neg['factor']
        fit_neg = _metric_score(eval_df_neg, fitness_metric)

        best_fit = fit_neg if fit_neg > fit_pos else fit_pos
        sign = -1 if fit_neg > fit_pos else 1

        depth_penalty = _calc_depth_penalty(
            depth=get_tree_depth(tree),
            depth_penalty_coef=depth_penalty_coef,
            depth_penalty_start_depth=depth_penalty_start_depth,
            depth_penalty_linear_coef=depth_penalty_linear_coef,
            depth_penalty_quadratic_coef=depth_penalty_quadratic_coef,
        )
        if depth_penalty > 0:
            best_fit = best_fit - depth_penalty

        return float(best_fit), sign
    except Exception:
        return 0.0, 1


def _tournament_select(scored_pop: List[Tuple[FactorNode, float]],
                       tournament_size: int,
                       rng: random.Random) -> FactorNode:
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
    log_interval: int = 5,
    depth_penalty_coef: float = 0.0,
    depth_penalty_start_depth: int = 0,
    depth_penalty_linear_coef: float = 0.0,
    depth_penalty_quadratic_coef: float = 0.0,
) -> List[GPCandidate]:
    rng = random.Random(random_seed)

    if log_interval <= 0:
        log_interval = 1

    log.info(
        'GP start: '
        f'fitness_metric={fitness_metric}, generations={generations}, population_size={population_size}, '
        f'max_depth={max_depth}, elite_size={elite_size}, max_factor_count={max_factor_count}, '
        f'depth_penalty_coef={depth_penalty_coef}, '
        f'depth_penalty_start_depth={depth_penalty_start_depth}, '
        f'depth_penalty_linear_coef={depth_penalty_linear_coef}, '
        f'depth_penalty_quadratic_coef={depth_penalty_quadratic_coef}'
    )

    population = [
        generate_random_tree(data_fields, max_depth, 0, window_choices, const_prob, leaf_prob, rng)
        for _ in range(population_size)
    ]

    global_best: Dict[str, GPCandidate] = {}

    for gen_idx in range(generations):
        scored_pop: List[Tuple[FactorNode, float]] = []
        for tree in population:
            fitness, sign = calc_fitness_and_sign(
                tree,
                df,
                fitness_metric=fitness_metric,
                depth_penalty_coef=depth_penalty_coef,
                depth_penalty_start_depth=depth_penalty_start_depth,
                depth_penalty_linear_coef=depth_penalty_linear_coef,
                depth_penalty_quadratic_coef=depth_penalty_quadratic_coef,
            )
            scored_pop.append((tree, fitness))

            oriented_node = copy.deepcopy(tree) if sign > 0 else OpNeg(copy.deepcopy(tree))
            formula = oriented_node.to_formula()
            old = global_best.get(formula)
            if old is None or fitness > old.fitness:
                global_best[formula] = GPCandidate(node=oriented_node, formula=formula, fitness=fitness)

        scored_pop.sort(key=lambda x: x[1], reverse=True)

        if scored_pop:
            best_fitness_current = float(scored_pop[0][1])
            avg_fitness_current = float(np.mean([x[1] for x in scored_pop]))
            global_best_fitness = float(max(x.fitness for x in global_best.values())) if global_best else 0.0
        else:
            best_fitness_current = 0.0
            avg_fitness_current = 0.0
            global_best_fitness = 0.0

        should_log = ((gen_idx + 1) % log_interval == 0) or (gen_idx == 0) or (gen_idx == generations - 1)
        if should_log:
            log.info(
                f'GP generation {gen_idx + 1}/{generations}: '
                f'current_best={best_fitness_current:.6f}, current_avg={avg_fitness_current:.6f}, '
                f'global_best={global_best_fitness:.6f}, unique_formulas={len(global_best)}'
            )

        next_gen: List[FactorNode] = [copy.deepcopy(x[0]) for x in scored_pop[:min(elite_size, len(scored_pop))]]

        while len(next_gen) < population_size:
            r = rng.random()
            if r < crossover_prob and len(scored_pop) >= 2:
                p1 = _tournament_select(scored_pop, tournament_size, rng)
                p2 = _tournament_select(scored_pop, tournament_size, rng)
                c1, c2 = crossover_trees(p1, p2, rng)
                next_gen.append(c1)
                if len(next_gen) < population_size:
                    next_gen.append(c2)
            elif r < crossover_prob + mutation_prob:
                p = _tournament_select(scored_pop, tournament_size, rng)
                m = mutate_tree(p, data_fields, max_depth, window_choices, const_prob, leaf_prob, rng)
                next_gen.append(m)
            else:
                p = _tournament_select(scored_pop, tournament_size, rng)
                next_gen.append(copy.deepcopy(p))

        population = next_gen[:population_size]

    candidates = sorted(global_best.values(), key=lambda x: x.fitness, reverse=True)
    log.info(
        f'GP finished: total_unique_formulas={len(global_best)}, '
        f'return_top_n={min(max_factor_count, len(candidates))}'
    )
    return candidates[:max_factor_count]

