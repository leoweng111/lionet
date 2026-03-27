"""One-click demo for Strategy.

Run:
    python strategy/demo_strategy.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy.strategy import Strategy


def main():
    strategy = Strategy(
        database='factors',
        collection='genetic_programming',
        version='20260323_gp_test',
        factor_name='fac_gp_0023',
        instrument_id='C0',
        start_time='20200101',
        end_time='20241231',
        initial_capital=1_000_000,
        margin_rate=0.1,
        fee_per_lot=2.0,
        slippage=1.0,
        apply_rolling_norm=True,
        rolling_norm_window=30,
        rolling_norm_min_periods=20,
        rolling_norm_eps=1e-8,
        rolling_norm_clip=5.0,
        signal_delay_days=1,
    )
    detail = strategy.backtest()
    print(detail[['time', 'factor_value', 'position_lots', 'equity', 'nav']].tail(10).to_string(index=False))
    strategy.plot_nav(show_baseline=True)


if __name__ == '__main__':
    main()

