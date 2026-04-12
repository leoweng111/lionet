"""Daily entry for genetic programming factor generation.

Example:
    python daily/factor_generate/gp_factor_generate.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator
from utils.logging import log


def _parse_instrument_id(value: str):
    ids = [item.strip() for item in value.split(',') if item.strip()]
    return ids if len(ids) > 1 else ids[0]


def run_gp_factor_generate(
    instrument_type: str = 'futures_continuous_contract',
    instrument_id_list='C0',
    fc_freq: str = '1d',
    data: Optional[pd.DataFrame] = None,
    start_time: Optional[str] = '20200101',
    end_time: Optional[str] = '20241231',
    portfolio_adjust_method: str = '1D',
    interest_method: str = 'simple',
    risk_free_rate: bool = False,
    calculate_baseline: bool = True,
    n_jobs: int = 5,
    base_col_list: Optional[Sequence[str]] = None,
    min_window_size: int = 30,
    max_factor_count: int = 50,
    apply_rolling_norm: bool = True,
    rolling_norm_window: int = 30,
    rolling_norm_min_periods: int = 20,
    rolling_norm_eps: float = 1e-8,
    rolling_norm_clip: float = 5.0,
    check_leakage_count: int = 20,
    check_relative: bool = True,
    relative_threshold: float = 0.7,
    relative_check_version_list: Optional[Sequence[str]] = None,
    gp_generations: int = 60,
    gp_population_size: int = 500,
    gp_max_depth: int = 6,
    gp_elite_size: int = 50,
    gp_elite_relative_threshold: float = 0.65,
    gp_tournament_size: int = 3,
    gp_crossover_prob: float = 0.3,
    gp_mutation_prob: float = 0.7,
    gp_leaf_prob: float = 0.2,
    gp_const_prob: float = 0.02,
    gp_window_choices: Optional[Sequence[int]] = None,
    fitness_metric: str = 'ic',
    random_seed: Optional[int] = None,
    gp_early_stopping_generation_count: int = 20,
    gp_depth_penalty_coef: float = 0.0,
    gp_depth_penalty_start_depth: int = 6,
    gp_depth_penalty_linear_coef: float = 0.03,
    gp_depth_penalty_quadratic_coef: float = 0.0,
    gp_log_interval: int = 5,
    gp_small_factor_penalty_coef: float = 0.0,
    gp_assumed_initial_capital: float = 100_000.0,
    gp_elite_stagnation_generation_count: int = 4,
    gp_max_shock_generation: int = 3,
    attempt_time: int = 3,
    version: Optional[str] = '20260411_gp_test_1',
):
    version = version or datetime.now().strftime('%Y%m%d')
    total_attempts = max(1, int(attempt_time))
    last_result = None
    for attempt_idx in range(total_attempts):
        fg = GeneticFactorGenerator(
            instrument_type=instrument_type,
            instrument_id_list=instrument_id_list,
            fc_freq=fc_freq,
            data=data,
            start_time=start_time,
            end_time=end_time,
            portfolio_adjust_method=portfolio_adjust_method,
            interest_method=interest_method,
            risk_free_rate=risk_free_rate,
            calculate_baseline=calculate_baseline,
            n_jobs=n_jobs,
            base_col_list=base_col_list,
            min_window_size=min_window_size,
            max_factor_count=max_factor_count,
            apply_rolling_norm=apply_rolling_norm,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            check_leakage_count=check_leakage_count,
            check_relative=check_relative,
            relative_threshold=relative_threshold,
            relative_check_version_list=relative_check_version_list,
            gp_generations=gp_generations,
            gp_population_size=gp_population_size,
            gp_max_depth=gp_max_depth,
            gp_elite_size=gp_elite_size,
            gp_elite_relative_threshold=gp_elite_relative_threshold,
            gp_tournament_size=gp_tournament_size,
            gp_crossover_prob=gp_crossover_prob,
            gp_mutation_prob=gp_mutation_prob,
            gp_leaf_prob=gp_leaf_prob,
            gp_const_prob=gp_const_prob,
            gp_window_choices=gp_window_choices,
            fitness_metric=fitness_metric,
            random_seed=random_seed,
            gp_early_stopping_generation_count=gp_early_stopping_generation_count,
            gp_depth_penalty_coef=gp_depth_penalty_coef,
            gp_depth_penalty_start_depth=gp_depth_penalty_start_depth,
            gp_depth_penalty_linear_coef=gp_depth_penalty_linear_coef,
            gp_depth_penalty_quadratic_coef=gp_depth_penalty_quadratic_coef,
            gp_log_interval=gp_log_interval,
            gp_small_factor_penalty_coef=gp_small_factor_penalty_coef,
            gp_assumed_initial_capital=gp_assumed_initial_capital,
            gp_elite_stagnation_generation_count=gp_elite_stagnation_generation_count,
            gp_max_shock_generation=gp_max_shock_generation,
            version=version,
        )
        result = fg.auto_mine_select_and_save_fc(
            filter_indicator_dict={
                'Net Return': (0.05, 0.03, 1),
                'Net Sharpe': (0.5, 0.3, 1),
            },
            n_jobs=n_jobs,
        )
        last_result = result

        selected_count = len(result.get('selected_fc_name_list', []))
        if selected_count > 0 and result.get('config_ref'):
            if attempt_idx > 0:
                log.info(f'[gp] Success after attempts: attempt={attempt_idx + 1}/{total_attempts}, selected_count={selected_count}')
            return result

        if attempt_idx < total_attempts - 1:
            log.warning(f'[gp] No factor persisted on attempt {attempt_idx + 1}/{total_attempts}, retrying...')

    log.warning(f'[gp] No factor persisted after all attempts: total_attempts={total_attempts}')
    return last_result or {
        'config_ref': None,
        'config_path': None,
        'selected_fc_name_list': [],
        'message': 'No factor persisted after retries.',
    }


def _parse_window_choices(value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    vals = [x.strip() for x in value.split(',') if x.strip()]
    if not vals:
        return None
    return [int(x) for x in vals]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate factors with genetic programming and run one-step mine/select/save.')
    parser.add_argument('--instrument_id', type=str, default='C0',
                        help='One instrument id (C0) or comma-separated ids (C0,FG0).')
    parser.add_argument('--start_time', type=str, default='20200101', help='Backtest start time in YYYYMMDD.')
    parser.add_argument('--end_time', type=str, default='20241231', help='Backtest end time in YYYYMMDD.')
    parser.add_argument('--version', type=str, default='20260411_gp_test_1',
                        help='Config version suffix. Default: 20260411_gp_test_1.')

    parser.add_argument('--instrument_type', type=str, default='futures_continuous_contract',
                        choices=['futures_continuous_contract'])
    parser.add_argument('--fc_freq', type=str, default='1d', choices=['1m', '5m', '1d'])
    parser.add_argument('--portfolio_adjust_method', type=str, default='1D', choices=['min', '1D', '1M', '1Q'])
    parser.add_argument('--interest_method', type=str, default='simple', choices=['simple', 'compound'])
    parser.add_argument('--risk_free_rate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--calculate_baseline', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--n_jobs', type=int, default=5)
    parser.add_argument('--base_col_list', type=str, default='open,high,low,close,volume,position',
                        help='Comma-separated base columns used for formula evaluation.')

    parser.add_argument('--min_window_size', type=int, default=30)
    parser.add_argument('--max_factor_count', type=int, default=50)

    parser.add_argument('--apply_rolling_norm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rolling_norm_window', type=int, default=30)
    parser.add_argument('--rolling_norm_min_periods', type=int, default=20)
    parser.add_argument('--rolling_norm_eps', type=float, default=1e-8)
    parser.add_argument('--rolling_norm_clip', type=float, default=5.0)
    parser.add_argument('--check_leakage_count', type=int, default=20)

    parser.add_argument('--check_relative', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--relative_threshold', type=float, default=0.7)
    parser.add_argument('--relative_check_version_list', type=str, default=None,
                        help='Comma-separated versions for relative check. Default None means all versions.')

    parser.add_argument('--gp_generations', type=int, default=60)
    parser.add_argument('--gp_population_size', type=int, default=500)
    parser.add_argument('--gp_max_depth', type=int, default=6)
    parser.add_argument('--gp_elite_size', type=int, default=50)
    parser.add_argument('--gp_elite_relative_threshold', type=float, default=0.65,
                        help='Elite archive correlation threshold for "same school" detection. '
                             'Factors with |corr| > threshold are considered same school.')
    parser.add_argument('--gp_tournament_size', type=int, default=3)
    parser.add_argument('--gp_crossover_prob', type=float, default=0.3)
    parser.add_argument('--gp_mutation_prob', type=float, default=0.7)
    parser.add_argument('--gp_leaf_prob', type=float, default=0.2)
    parser.add_argument('--gp_const_prob', type=float, default=0.02)
    parser.add_argument('--gp_window_choices', type=str, default='3,5,10,20,30',
                        help='Comma-separated integers, e.g. 5,10,20')
    parser.add_argument('--fitness_metric', type=str, default='ic', choices=['ic', 'sharpe'])
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--gp_early_stopping_generation_count', type=int, default=20)
    parser.add_argument('--gp_depth_penalty_coef', type=float, default=0.0)
    parser.add_argument('--gp_depth_penalty_start_depth', type=int, default=6)
    parser.add_argument('--gp_depth_penalty_linear_coef', type=float, default=0.03)
    parser.add_argument('--gp_depth_penalty_quadratic_coef', type=float, default=0.0)
    parser.add_argument('--gp_log_interval', type=int, default=5)
    parser.add_argument('--gp_small_factor_penalty_coef', type=float, default=0.0)
    parser.add_argument('--gp_assumed_initial_capital', type=float, default=100000.0)
    parser.add_argument('--gp_elite_stagnation_generation_count', type=int, default=4,
                        help='Enter shock mode when elite archive stagnates for N generations.')
    parser.add_argument('--gp_max_shock_generation', type=int, default=3,
                        help='Exit shock mode after N generations without elite updates.')
    parser.add_argument('--attempt_time', type=int, default=3,
                        help='Total attempts when mining GP factors. Must be >= 1.')

    return parser


def _parse_base_col_list(value: Optional[str]) -> Optional[Sequence[str]]:
    if value is None:
        return None
    cols = [x.strip() for x in value.split(',') if x.strip()]
    return cols if cols else None


def main(argv: Optional[Sequence[str]] = None):
    if argv is None and 'ipykernel' in sys.modules:
        argv = []

    args = build_parser().parse_args(argv)
    result = run_gp_factor_generate(
        instrument_type=args.instrument_type,
        instrument_id_list=_parse_instrument_id(args.instrument_id),
        fc_freq=args.fc_freq,
        start_time=args.start_time,
        end_time=args.end_time,
        portfolio_adjust_method=args.portfolio_adjust_method,
        interest_method=args.interest_method,
        risk_free_rate=args.risk_free_rate,
        calculate_baseline=args.calculate_baseline,
        n_jobs=args.n_jobs,
        base_col_list=_parse_base_col_list(args.base_col_list),
        min_window_size=args.min_window_size,
        max_factor_count=args.max_factor_count,
        apply_rolling_norm=args.apply_rolling_norm,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
        check_leakage_count=args.check_leakage_count,
        check_relative=args.check_relative,
        relative_threshold=args.relative_threshold,
        relative_check_version_list=[x.strip() for x in args.relative_check_version_list.split(',') if x.strip()]
        if args.relative_check_version_list else None,
        gp_generations=args.gp_generations,
        gp_population_size=args.gp_population_size,
        gp_max_depth=args.gp_max_depth,
        gp_elite_size=args.gp_elite_size,
        gp_elite_relative_threshold=args.gp_elite_relative_threshold,
        gp_tournament_size=args.gp_tournament_size,
        gp_crossover_prob=args.gp_crossover_prob,
        gp_mutation_prob=args.gp_mutation_prob,
        gp_leaf_prob=args.gp_leaf_prob,
        gp_const_prob=args.gp_const_prob,
        gp_window_choices=_parse_window_choices(args.gp_window_choices),
        fitness_metric=args.fitness_metric,
        random_seed=args.random_seed,
        gp_early_stopping_generation_count=args.gp_early_stopping_generation_count,
        gp_depth_penalty_coef=args.gp_depth_penalty_coef,
        gp_depth_penalty_start_depth=args.gp_depth_penalty_start_depth,
        gp_depth_penalty_linear_coef=args.gp_depth_penalty_linear_coef,
        gp_depth_penalty_quadratic_coef=args.gp_depth_penalty_quadratic_coef,
        gp_log_interval=args.gp_log_interval,
        gp_small_factor_penalty_coef=args.gp_small_factor_penalty_coef,
        gp_assumed_initial_capital=args.gp_assumed_initial_capital,
        gp_elite_stagnation_generation_count=args.gp_elite_stagnation_generation_count,
        gp_max_shock_generation=args.gp_max_shock_generation,
        attempt_time=args.attempt_time,
        version=args.version,
    )
    print(f"[gp] config_ref={result.get('config_ref')}")
    print(f"[gp] config_path={result.get('config_path')}")
    print(f"[gp] selected_factor_count={len(result.get('selected_fc_name_list', []))}")
    return result


if __name__ == '__main__':
    main()

