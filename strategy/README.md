【# Strategy Module

This module provides a realistic daily open-to-open futures simulation based on one stored factor formula.

## Quick Start

```python
from strategy import Strategy

strategy = Strategy(
    database='factors',
    collection='genetic_programming',
    version='20260323_gp_test',
    factor_name='fac_gp_0023',
    instrument_id='C0',
    start_time='20200101',
    end_time='20241231',
)

strategy.backtest()
strategy.plot_nav(show_baseline=True)
```

## Demo

```bash
python strategy/demo_strategy.py
```

## Notes

- Factor value is computed on adjusted prices (`raw * weighted_factor`).
- Trading, margin check, fee, slippage, and PnL are computed on raw prices.
- Daily logs include gap PnL, intraday PnL, position change, margin usage, and equity.

