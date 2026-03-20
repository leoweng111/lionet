"""Daily entry for tsfresh factor generation.

Example:
    python daily/factor_generate/tsfresh_factor_generate.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import TsfreshFactorGenerator


def _parse_instrument_id(value: str):
    ids = [item.strip() for item in value.split(',') if item.strip()]
    return ids if len(ids) > 1 else ids[0]


def run_tsfresh_factor_generate(
    instrument_id,
    start_time: str,
    end_time: str,
    version: Optional[str] = None,
    fc_freq: str = '1d',
    portfolio_adjust_method: str = '1D',
    interest_method: str = 'compound',
    n_jobs: int = 5,
    min_window_size: int = 30,
    max_factor_count: int = 20000,
    tsfresh_profile: str = 'minimal',
    apply_rolling_norm: bool = True,
    rolling_norm_window: int = 30,
    rolling_norm_min_periods: int = 20,
    rolling_norm_eps: float = 1e-8,
    rolling_norm_clip: float = 10.0,
    net_ret_threshold: float = 0.0,
    sharpe_threshold: float = 0.5,
    require_all_row: bool = True,
    require_all_instruments: bool = True,
):
    version = version or datetime.now().strftime('%Y%m%d')
    fg = TsfreshFactorGenerator(
        instrument_id_list=instrument_id,
        fc_freq=fc_freq,
        start_time=start_time,
        end_time=end_time,
        portfolio_adjust_method=portfolio_adjust_method,
        interest_method=interest_method,
        n_jobs=n_jobs,
        min_window_size=min_window_size,
        max_factor_count=max_factor_count,
        tsfresh_profile=tsfresh_profile,
        apply_rolling_norm=apply_rolling_norm,
        rolling_norm_window=rolling_norm_window,
        rolling_norm_min_periods=rolling_norm_min_periods,
        rolling_norm_eps=rolling_norm_eps,
        rolling_norm_clip=rolling_norm_clip,
        version=version,
    )
    result = fg.auto_mine_select_and_save_fc(
        filter_indicator_dict={
            'Net Return': (net_ret_threshold, net_ret_threshold, 1),
            'Net Sharpe': (sharpe_threshold, sharpe_threshold, 1),
        },
        n_jobs=n_jobs,
        require_all_row=require_all_row,
        require_all_instruments=require_all_instruments,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate factors with tsfresh and run one-step mine/select/save.')
    parser.add_argument('--instrument_id', type=str, default='C0',
                        help='One instrument id (C0) or comma-separated ids (C0,FG0).')
    parser.add_argument('--start_time', type=str, default='20200101', help='Backtest start time in YYYYMMDD.')
    parser.add_argument('--end_time', type=str, default='20241231', help='Backtest end time in YYYYMMDD.')
    parser.add_argument('--version', type=str, default=None,
                        help='Config version suffix. Default: today YYYYMMDD.')

    parser.add_argument('--fc_freq', type=str, default='1d', choices=['1m', '5m', '1d'])
    parser.add_argument('--portfolio_adjust_method', type=str, default='1D', choices=['min', '1D', '1M', '1Q'])
    parser.add_argument('--interest_method', type=str, default='compound', choices=['simple', 'compound'])
    parser.add_argument('--n_jobs', type=int, default=5)

    parser.add_argument('--min_window_size', type=int, default=30)
    parser.add_argument('--max_factor_count', type=int, default=20000)
    parser.add_argument('--tsfresh_profile', type=str, default='minimal',
                        choices=['minimal', 'efficient', 'comprehensive'])

    parser.add_argument('--apply_rolling_norm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rolling_norm_window', type=int, default=30)
    parser.add_argument('--rolling_norm_min_periods', type=int, default=20)
    parser.add_argument('--rolling_norm_eps', type=float, default=1e-8)
    parser.add_argument('--rolling_norm_clip', type=float, default=10.0)

    parser.add_argument('--net_ret_threshold', type=float, default=0.05)
    parser.add_argument('--sharpe_threshold', type=float, default=0.7)
    parser.add_argument('--require_all_row', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--require_all_instruments', action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None):
    # In notebooks, default argparse on sys.argv may include kernel args.
    if argv is None and 'ipykernel' in sys.modules:
        argv = []
    args = build_parser().parse_args(argv)
    result = run_tsfresh_factor_generate(
        instrument_id=_parse_instrument_id(args.instrument_id),
        start_time=args.start_time,
        end_time=args.end_time,
        version=args.version,
        fc_freq=args.fc_freq,
        portfolio_adjust_method=args.portfolio_adjust_method,
        interest_method=args.interest_method,
        n_jobs=args.n_jobs,
        min_window_size=args.min_window_size,
        max_factor_count=args.max_factor_count,
        tsfresh_profile=args.tsfresh_profile,
        apply_rolling_norm=args.apply_rolling_norm,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
        net_ret_threshold=args.net_ret_threshold,
        sharpe_threshold=args.sharpe_threshold,
        require_all_row=args.require_all_row,
        require_all_instruments=args.require_all_instruments,
    )
    print(f"[tsfresh] config_path={result.get('config_path')}")
    print(f"[tsfresh] selected_factor_count={len(result.get('selected_fc_name_list', []))}")
    return result


if __name__ == '__main__':
    main()

