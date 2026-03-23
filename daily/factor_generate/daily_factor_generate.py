"""One-click daily factor generation for llm_prompt + genetic programming.

Example:
    python daily/factor_generate/daily_factor_generate.py
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import schedule

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from llm_prompt_factor_generate import run_llm_prompt_factor_generate
from gp_factor_generate import run_gp_factor_generate


def _parse_instrument_id(value: str):
    ids = [item.strip() for item in value.split(',') if item.strip()]
    return ids if len(ids) > 1 else ids[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run llm_prompt and genetic_programming factor generation in one command.')
    parser.add_argument('--instrument_id', type=str, default='C0',
                        help='One instrument id (C0) or comma-separated ids (C0,FG0).')
    parser.add_argument('--start_time', type=str, default='20200101', help='Backtest start time in YYYYMMDD.')
    parser.add_argument('--end_time', type=str, default='20241231', help='Backtest end time in YYYYMMDD.')
    parser.add_argument('--version', type=str, default=None,
                        help='Version suffix. Default: today YYYYMMDD.')
    parser.add_argument('--n_jobs', type=int, default=5)

    parser.add_argument('--instrument_type', type=str, default='futures_continuous_contract',
                        choices=['futures_continuous_contract'])
    parser.add_argument('--portfolio_adjust_method', type=str, default='1D', choices=['min', '1D', '1M', '1Q'])
    parser.add_argument('--interest_method', type=str, default='compound', choices=['simple', 'compound'])
    parser.add_argument('--fc_freq', type=str, default='1d', choices=['1m', '5m', '1d'])
    parser.add_argument('--risk_free_rate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--calculate_baseline', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--base_col_list', type=str, default='open,high,low,close,volume,position',
                        help='Comma-separated base columns used for formula evaluation.')

    parser.add_argument('--apply_rolling_norm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rolling_norm_window', type=int, default=30)
    parser.add_argument('--rolling_norm_min_periods', type=int, default=20)
    parser.add_argument('--rolling_norm_eps', type=float, default=1e-8)
    parser.add_argument('--rolling_norm_clip', type=float, default=10.0)
    parser.add_argument('--check_relative', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--relative_threshold', type=float, default=0.7)
    parser.add_argument('--relative_check_version_list', type=str, default=None,
                        help='Comma-separated versions for relative check. Default None means all versions.')
    parser.add_argument('--min_window_size', type=int, default=30)

    parser.add_argument('--llm_max_factor_count', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='deepseek')
    parser.add_argument('--llm_temperature', type=float, default=0.7)
    parser.add_argument('--llm_factor_count', type=int, default=5)
    parser.add_argument('--llm_early_stopping_round', type=int, default=20)
    parser.add_argument('--llm_user_requirement', type=str, default='生成期货的日频量价因子')

    parser.add_argument('--gp_max_factor_count', type=int, default=50)
    parser.add_argument('--gp_generations', type=int, default=50)
    parser.add_argument('--gp_population_size', type=int, default=200)
    parser.add_argument('--gp_max_depth', type=int, default=4)
    parser.add_argument('--gp_elite_size', type=int, default=20)
    parser.add_argument('--gp_tournament_size', type=int, default=6)
    parser.add_argument('--gp_crossover_prob', type=float, default=0.7)
    parser.add_argument('--gp_mutation_prob', type=float, default=0.25)
    parser.add_argument('--gp_leaf_prob', type=float, default=0.2)
    parser.add_argument('--gp_const_prob', type=float, default=0.02)
    parser.add_argument('--gp_window_choices', type=str, default='5,10,20')
    parser.add_argument('--fitness_metric', type=str, default='ic', choices=['ic', 'sharpe'])
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--gp_early_stopping_generation_count', type=int, default=8)
    parser.add_argument('--gp_depth_penalty_coef', type=float, default=0.0)
    parser.add_argument('--gp_depth_penalty_start_depth', type=int, default=3)
    parser.add_argument('--gp_depth_penalty_linear_coef', type=float, default=0.0)
    parser.add_argument('--gp_depth_penalty_quadratic_coef', type=float, default=0.0)
    parser.add_argument('--gp_log_interval', type=int, default=5)

    parser.add_argument('--check_leakage_count', type=int, default=20)
    parser.add_argument('--only', type=str, default='all', choices=['all', 'llm_prompt', 'gp'])
    parser.add_argument('--schedule', action=argparse.BooleanOptionalAction, default=False,
                        help='If enabled, run daily at 22:00 using schedule library.')
    return parser


def _parse_window_choices(value: Optional[str]):
    if not value:
        return None
    vals = [x.strip() for x in value.split(',') if x.strip()]
    if not vals:
        return None
    return [int(x) for x in vals]


def _parse_base_col_list(value: Optional[str]):
    if not value:
        return None
    cols = [x.strip() for x in value.split(',') if x.strip()]
    return cols if cols else None


def run_daily_once(args) -> dict:
    instrument_id = _parse_instrument_id(args.instrument_id)
    version = args.version or datetime.now().strftime('%Y%m%d')
    relative_versions = [x.strip() for x in args.relative_check_version_list.split(',') if x.strip()] \
        if args.relative_check_version_list else None

    llm_result = None
    gp_result = None

    if args.only in ['all', 'llm_prompt']:
        llm_result = run_llm_prompt_factor_generate(
            instrument_type=args.instrument_type,
            instrument_id_list=instrument_id,
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
            max_factor_count=args.llm_max_factor_count,
            apply_rolling_norm=args.apply_rolling_norm,
            rolling_norm_window=args.rolling_norm_window,
            rolling_norm_min_periods=args.rolling_norm_min_periods,
            rolling_norm_eps=args.rolling_norm_eps,
            rolling_norm_clip=args.rolling_norm_clip,
            check_leakage_count=args.check_leakage_count,
            check_relative=args.check_relative,
            relative_threshold=args.relative_threshold,
            relative_check_version_list=relative_versions,
            model_name=args.model_name,
            llm_temperature=args.llm_temperature,
            llm_factor_count=args.llm_factor_count,
            llm_early_stopping_round=args.llm_early_stopping_round,
            llm_user_requirement=args.llm_user_requirement,
            version=version,
        )
        print(f"[daily][llm_prompt] config_path={llm_result.get('config_path')}")
        print(f"[daily][llm_prompt] selected_factor_count={len(llm_result.get('selected_fc_name_list', []))}")

    if args.only in ['all', 'gp']:
        gp_result = run_gp_factor_generate(
            instrument_type=args.instrument_type,
            instrument_id_list=instrument_id,
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
            max_factor_count=args.gp_max_factor_count,
            apply_rolling_norm=args.apply_rolling_norm,
            rolling_norm_window=args.rolling_norm_window,
            rolling_norm_min_periods=args.rolling_norm_min_periods,
            rolling_norm_eps=args.rolling_norm_eps,
            rolling_norm_clip=args.rolling_norm_clip,
            check_leakage_count=args.check_leakage_count,
            check_relative=args.check_relative,
            relative_threshold=args.relative_threshold,
            relative_check_version_list=relative_versions,
            gp_generations=args.gp_generations,
            gp_population_size=args.gp_population_size,
            gp_max_depth=args.gp_max_depth,
            gp_elite_size=args.gp_elite_size,
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
            version=version,
        )
        print(f"[daily][gp] config_path={gp_result.get('config_path')}")
        print(f"[daily][gp] selected_factor_count={len(gp_result.get('selected_fc_name_list', []))}")

    return {
        'version': version,
        'llm_prompt': llm_result,
        'genetic_programming': gp_result,
    }


def main(argv: Optional[Sequence[str]] = None):
    # In notebooks, default argparse on sys.argv may include kernel args.
    if argv is None and 'ipykernel' in sys.modules:
        argv = []
    args = build_parser().parse_args(argv)
    if not args.schedule:
        return run_daily_once(args)

    def _job():
        args.version = datetime.now().strftime('%Y%m%d')
        print(f"[schedule] Trigger daily factor generation at 22:00, version={args.version}")
        run_daily_once(args)

    schedule.every().day.at('22:00').do(_job)
    print('[schedule] Started. Daily factor generation will run at 22:00.')
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == '__main__':
    main()


