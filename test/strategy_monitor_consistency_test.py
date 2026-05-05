"""Consistency smoke test for strategy monitor metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from strategy.strategy import Strategy
from web.backend.main import StrategyMonitorParams, _run_strategy_monitor

EPS = 1e-9


def _assert_close(name: str, left: float, right: float, tol: float = EPS):
	if not np.isfinite(left) or not np.isfinite(right):
		raise AssertionError(f"{name} has non-finite value: left={left}, right={right}")
	if abs(left - right) > tol:
		raise AssertionError(f"{name} mismatch: left={left}, right={right}, diff={abs(left-right)}")


def main() -> int:
	params = StrategyMonitorParams(
		version="20260417_gp_test",
		factor_name="fac_gp_0041",
		instrument_id="C0",
		trading_start_time="20200101",
		collection="genetic_programming",
	)

	monitor = _run_strategy_monitor(params)

	strat = Strategy(
		version=params.version,
		factor_name=params.factor_name,
		instrument_id=params.instrument_id,
		start_time=params.trading_start_time,
		end_time=monitor["latest_price_date"],
		database=params.database,
		collection=params.collection,
		initial_capital=params.initial_capital,
		margin_rate=params.margin_rate,
		fee_per_lot=params.fee_per_lot,
		slippage=params.slippage,
		apply_rolling_norm=params.apply_rolling_norm,
		rolling_norm_window=params.rolling_norm_window,
		rolling_norm_min_periods=params.rolling_norm_min_periods,
		rolling_norm_eps=params.rolling_norm_eps,
		rolling_norm_clip=params.rolling_norm_clip,
		signal_delay_days=params.signal_delay_days,
		min_open_ratio=params.min_open_ratio,
	)
	strat.backtest()
	detail = strat.performance_detail.copy().sort_values("time").reset_index(drop=True)

	scoped = detail.loc[detail["time"] >= pd.Timestamp(params.trading_start_time)].copy()
	if scoped.empty:
		scoped = detail

	daily_net_ret = pd.to_numeric(scoped["daily_net_ret"], errors="coerce").fillna(0.0)
	expected_cum = float(daily_net_ret.sum())
	expected_ann = float(daily_net_ret.mean() * 252)
	latest = detail.iloc[-1]

	_assert_close("cumulative_return", float(monitor["cumulative_return"]), expected_cum)
	_assert_close("annualized_return", float(monitor["annualized_return"]), expected_ann)
	_assert_close("account_equity", float(monitor["account_equity"]), float(latest["equity"]))
	_assert_close("factor_value_t", float(monitor["factor_value_t"]), float(latest["factor_value"]))

	if int(monitor["current_position_lots"]) != int(latest["position_lots"]):
		raise AssertionError("current_position_lots mismatch")

	print("[PASS] strategy monitor consistency test passed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

