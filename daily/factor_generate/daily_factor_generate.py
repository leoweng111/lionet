"""One-click daily factor generation for llm_prompt.

Example:
    python daily/factor_generate/daily_factor_generate.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from llm_prompt_factor_generate import run_llm_prompt_factor_generate


def _parse_instrument_id(value: str):
    ids = [item.strip() for item in value.split(',') if item.strip()]
    return ids if len(ids) > 1 else ids[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run llm_prompt factor generation in one command.')
    parser.add_argument('--instrument_id', type=str, default='C0',
                        help='One instrument id (C0) or comma-separated ids (C0,FG0).')
    parser.add_argument('--start_time', type=str, default='20200101', help='Backtest start time in YYYYMMDD.')
    parser.add_argument('--end_time', type=str, default='20241231', help='Backtest end time in YYYYMMDD.')
    parser.add_argument('--version', type=str, default=None,
                        help='Version suffix. Default: today YYYYMMDD.')
    parser.add_argument('--n_jobs', type=int, default=5)

    parser.add_argument('--portfolio_adjust_method', type=str, default='1D', choices=['min', '1D', '1M', '1Q'])
    parser.add_argument('--interest_method', type=str, default='compound', choices=['simple', 'compound'])
    parser.add_argument('--fc_freq', type=str, default='1d', choices=['1m', '5m', '1d'])

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

    parser.add_argument('--net_ret_threshold', type=float, default=0.0)
    parser.add_argument('--sharpe_threshold', type=float, default=0.5)
    parser.add_argument('--require_all_row', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--require_all_instruments', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--only', type=str, default='llm_prompt', choices=['llm_prompt'])
    return parser


def main(argv: Optional[Sequence[str]] = None):
    # In notebooks, default argparse on sys.argv may include kernel args.
    if argv is None and 'ipykernel' in sys.modules:
        argv = []
    args = build_parser().parse_args(argv)
    instrument_id = _parse_instrument_id(args.instrument_id)
    version = args.version or datetime.now().strftime('%Y%m%d')
    llm_result = None
    llm_result = run_llm_prompt_factor_generate(
        instrument_id=instrument_id,
        start_time=args.start_time,
        end_time=args.end_time,
        version=version,
        fc_freq=args.fc_freq,
        portfolio_adjust_method=args.portfolio_adjust_method,
        interest_method=args.interest_method,
        n_jobs=args.n_jobs,
        min_window_size=args.min_window_size,
        max_factor_count=args.llm_max_factor_count,
        apply_rolling_norm=args.apply_rolling_norm,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
        check_relative=args.check_relative,
        relative_threshold=args.relative_threshold,
        relative_check_version_list=[x.strip() for x in args.relative_check_version_list.split(',') if x.strip()]
        if args.relative_check_version_list else None,
        model_name=args.model_name,
        llm_temperature=args.llm_temperature,
        llm_factor_count=args.llm_factor_count,
        llm_early_stopping_round=args.llm_early_stopping_round,
        llm_user_requirement=args.llm_user_requirement,
        net_ret_threshold=args.net_ret_threshold,
        sharpe_threshold=args.sharpe_threshold,
        require_all_row=args.require_all_row,
        require_all_instruments=args.require_all_instruments,
    )
    print(f"[daily][llm_prompt] config_path={llm_result.get('config_path')}")
    print(f"[daily][llm_prompt] selected_factor_count={len(llm_result.get('selected_fc_name_list', []))}")

    return {
        'version': version,
        'llm_prompt': llm_result,
    }


if __name__ == '__main__':
    main()


