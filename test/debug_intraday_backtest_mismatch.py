"""Diagnose intraday-feature factor backtest mismatch.

When a GP-mined factor references intraday features (e.g. ``yz``, ``rkurt``,
``rs_vol``), the factor value is computed twice:
  1. During mining: `cand.node.calc(df_eval)` where df_eval has intraday columns
     merged by `_merge_intraday_features`.
  2. During standalone backtest: `calc_formula_series(df, formula)` where df has
     intraday columns merged by `ensure_intraday_features_in_df`.

This script recomputes the factor value the SAME way the backtest page does, and
prints yearly stats so you can compare against the mining log.  If the numbers
differ from mining, the intraday features (or the daily price frame) differ
between the two paths.

Usage:
    python -u test/debug_intraday_backtest_mismatch.py --version 20260901_gp_test_intra \
        --factor fac_gp_0003 --instrument C0 --source joinquant \
        --start 20200101 --end 20241231
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import get_futures_continuous_contract_price, get_factor_formula_map_by_version  # noqa: E402
from data.futures import get_futures_continuous_contract_price_1min  # noqa: E402
from factors.factor_ops import calc_formula_series  # noqa: E402
from factors.factor_utils import get_weighted_price, get_future_ret  # noqa: E402
from factors.factor_intraday_features import ensure_intraday_features_in_df  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--factor', default='fac_gp_0003')
    parser.add_argument('--instrument', default='C0')
    parser.add_argument('--source', default='joinquant')
    parser.add_argument('--start', default='20200101')
    parser.add_argument('--end', default='20241231')
    parser.add_argument('--collection', default='genetic_programming')
    args = parser.parse_args()

    # 1) Load formula from DB
    formula_map = get_factor_formula_map_by_version(
        fc_name_list=[args.factor],
        version=args.version,
        collections=[args.collection],
        database='factors',
    )
    formula = formula_map.get(args.factor, '')
    if not formula:
        print(f'!! Formula not found: {args.version} / {args.factor}')
        sys.exit(1)
    print(f'Formula: {formula}\n')

    # 2) Load daily price (same as backtest page)
    daily = get_futures_continuous_contract_price(
        instrument_id=args.instrument,
        start_date=args.start,
        end_date=args.end,
        from_database=True,
        source=args.source,
    )
    print(f'Daily rows: {len(daily)} ({args.source}), range {daily["time"].min()} ~ {daily["time"].max()}')
    daily = get_weighted_price(daily)
    daily = daily.sort_values(['instrument_id', 'time']).reset_index(drop=True)

    # 3) Load minute data (same as backtest page ensure path)
    minute = get_futures_continuous_contract_price_1min(
        instrument_id=args.instrument,
        start_date=args.start,
        end_date=args.end,
        source=args.source,
    )
    print(f'Minute rows: {len(minute)} ({args.source}), range {minute["time"].min()} ~ {minute["time"].max()}\n')

    # 4) Ensure intraday features (backtest-page path)
    daily2 = ensure_intraday_features_in_df(
        df=daily,
        formulas=[formula],
        instrument_id_list=args.instrument,
        source=args.source,
        start_date=args.start,
        end_date=args.end,
    )
    intra_cols = [c for c in daily2.columns if c not in daily.columns]
    print(f'Intraday feature cols added: {intra_cols}')

    # 5) Recompute factor value exactly as backtest does
    daily2[args.factor] = pd.to_numeric(calc_formula_series(daily2, formula=formula), errors='coerce')
    daily2 = get_future_ret(daily2, portfolio_adjust_method='1D', rfr=False)

    # 6) Yearly stats (same convention as get_performance)
    df = daily2.dropna(subset=['future_ret']).copy()
    df['year'] = pd.to_datetime(df['time']).dt.year
    print('\n=== Yearly stats (backtest-page path) ===')
    for year, g in df.groupby('year'):
        sig = pd.to_numeric(g[args.factor], errors='coerce').ffill().fillna(0)
        fut = pd.to_numeric(g['future_ret'], errors='coerce').fillna(0)
        gross = (sig * fut)
        # annualized = daily mean * 252
        ann = float(gross.mean()) * 252
        sharpe = ann / (float(gross.std()) * np.sqrt(252)) if gross.std() > 0 else float('nan')
        print(f'  {year}: gross_ret={ann:.4f}, sharpe={sharpe:.4f}, '
              f'factor_nan={int(pd.to_numeric(g[args.factor], errors="coerce").isna().sum())}/{len(g)}')

    # 7) Factor value sample (last 5 rows)
    print('\n=== Last 5 factor values (backtest-page path) ===')
    tail = daily2[['time', args.factor]].dropna().tail(5)
    print(tail.to_string(index=False))

    print('\nCompare the yearly gross_ret above against the mining log.')
    print('If mining shows positive but this shows negative, the intraday features')
    print('(or daily price frame) differ between mining and backtest paths.')


if __name__ == '__main__':
    main()
