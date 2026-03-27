"""Analyze why BackTester and Strategy produce very different performance curves.

Run example:
    python test/performance_diff_analysis.py \
      --version 20260323_gp_test \
      --factor-name fac_gp_0023 \
      --instrument-id C0 \
      --start-time 20200101 \
      --end-time 20241231
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.backtest import BackTester
from strategy.strategy import Strategy
from utils.params import FUTURES_CONTRACT_MULTIPLIER


def _root_instrument(instrument_id: str) -> str:
    ins = str(instrument_id).upper().strip()
    if not ins:
        raise ValueError("instrument_id is empty.")
    return ins[:-1] if ins.endswith("0") else ins


def run_backtester(args: argparse.Namespace) -> BackTester:
    bt = BackTester(
        fc_name_list=[args.factor_name],
        version=args.version,
        instrument_type="futures_continuous_contract",
        instrument_id_list=[args.instrument_id],
        fc_freq="1d",
        start_time=args.start_time,
        end_time=args.end_time,
        portfolio_adjust_method="1D",
        interest_method="simple",
        risk_free_rate=False,
        calculate_baseline=True,
        apply_weighted_price=True,
        apply_rolling_norm=args.apply_rolling_norm,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
        n_jobs=args.n_jobs,
    )
    bt.backtest()
    return bt


def run_strategy(args: argparse.Namespace) -> Strategy:
    strategy = Strategy(
        database=args.database,
        collection=args.collection,
        version=args.version,
        factor_name=args.factor_name,
        instrument_id=args.instrument_id,
        start_time=args.start_time,
        end_time=args.end_time,
        initial_capital=args.initial_capital,
        margin_rate=args.margin_rate,
        fee_per_lot=args.fee_per_lot,
        slippage=args.slippage,
        apply_rolling_norm=args.apply_rolling_norm,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
        signal_delay_days=args.signal_delay_days,
    )
    strategy.backtest()
    return strategy


def _prepare_bt_detail(bt: BackTester, instrument_id: str, factor_name: str) -> pd.DataFrame:
    detail = bt.performance_detail.copy()
    detail = detail[
        (detail["instrument_id"] == instrument_id) & (detail["factor_name"] == factor_name)
    ].copy()
    if detail.empty:
        raise ValueError("BackTester performance_detail is empty for target instrument/factor.")

    out = detail[[
        "time",
        "factor_value",
        "daily_gross_ret",
        "daily_net_ret",
        "daily_turnover",
        "daily_gross_nav",
        "daily_net_nav",
        "future_ret",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "position",
    ]].copy()
    out = out.rename(columns={
        "factor_value": "bt_factor_value",
        "daily_gross_ret": "bt_daily_gross_ret",
        "daily_net_ret": "bt_daily_net_ret",
        "daily_turnover": "bt_daily_turnover",
        "daily_gross_nav": "bt_daily_gross_nav",
        "daily_net_nav": "bt_daily_net_nav",
        "future_ret": "bt_future_ret",
        "open": "bt_open",
        "high": "bt_high",
        "low": "bt_low",
        "close": "bt_close",
        "volume": "bt_volume",
        "position": "bt_position",
    })
    out["time"] = pd.to_datetime(out["time"])
    return out.sort_values("time").reset_index(drop=True)


def _prepare_strategy_detail(strategy: Strategy) -> pd.DataFrame:
    detail = strategy.performance_detail.copy()
    out = detail[[
        "time",
        "factor_value",
        "daily_gross_ret",
        "daily_net_ret",
        "daily_turnover",
        "daily_gross_nav",
        "daily_net_nav",
        "position_lots",
        "target_lots",
        "delta_lots",
        "fee",
        "slippage_cost",
        "warning",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "position",
        "equity",
        "required_margin",
        "available_cash",
        "terminated",
    ]].copy()
    out = out.rename(columns={
        "factor_value": "st_factor_value",
        "daily_gross_ret": "st_daily_gross_ret",
        "daily_net_ret": "st_daily_net_ret",
        "daily_turnover": "st_daily_turnover",
        "daily_gross_nav": "st_daily_gross_nav",
        "daily_net_nav": "st_daily_net_nav",
        "open": "st_open",
        "high": "st_high",
        "low": "st_low",
        "close": "st_close",
        "volume": "st_volume",
        "position": "st_position",
    })
    out["time"] = pd.to_datetime(out["time"])
    return out.sort_values("time").reset_index(drop=True)


def _build_daily_comparison(bt_detail: pd.DataFrame,
                            st_detail: pd.DataFrame,
                            instrument_id: str,
                            initial_capital: float,
                            signal_delay_days: int) -> pd.DataFrame:
    merged = bt_detail.merge(st_detail, on="time", how="inner", validate="1:1")

    root = _root_instrument(instrument_id)
    multiplier = FUTURES_CONTRACT_MULTIPLIER.get(root)
    if multiplier is None:
        raise ValueError(f"Missing multiplier for instrument root {root}.")

    # Keep raw Strategy return columns for diagnostics before timing alignment.
    merged["st_daily_gross_ret_raw"] = merged["st_daily_gross_ret"]
    merged["st_daily_net_ret_raw"] = merged["st_daily_net_ret"]
    merged["st_daily_turnover_raw"] = merged["st_daily_turnover"]

    merged["st_notional_exposure"] = (
        merged["position_lots"] * merged["st_open"] * float(multiplier) / float(initial_capital)
    )

    # When Strategy uses delayed execution (e.g. signal_delay_days=1),
    # st_factor_value/st_notional_exposure on day t correspond to BT signal from day t-1.
    # Shift Strategy series backward so day-t rows compare the same underlying signal timing.
    delay = int(signal_delay_days)
    if delay < 0:
        raise ValueError("signal_delay_days must be non-negative.")

    if delay > 0:
        if len(merged) <= delay:
            raise ValueError(
                f"Not enough rows ({len(merged)}) for signal_delay_days={delay}."
            )
        merged["st_factor_value_aligned"] = merged["st_factor_value"].shift(-delay)
        merged["st_notional_exposure_aligned"] = merged["st_notional_exposure"].shift(-delay)
        merged["st_daily_gross_ret_aligned"] = merged["st_daily_gross_ret"].shift(-delay)
        merged["st_daily_net_ret_aligned"] = merged["st_daily_net_ret"].shift(-delay)
        merged["st_daily_turnover_aligned"] = merged["st_daily_turnover"].shift(-delay)
        merged = merged.iloc[:-delay].copy().reset_index(drop=True)
    else:
        merged["st_factor_value_aligned"] = merged["st_factor_value"]
        merged["st_notional_exposure_aligned"] = merged["st_notional_exposure"]
        merged["st_daily_gross_ret_aligned"] = merged["st_daily_gross_ret"]
        merged["st_daily_net_ret_aligned"] = merged["st_daily_net_ret"]
        merged["st_daily_turnover_aligned"] = merged["st_daily_turnover"]

    # Use delay-aligned Strategy returns as the canonical comparison series.
    merged["st_daily_gross_ret"] = merged["st_daily_gross_ret_aligned"]
    merged["st_daily_net_ret"] = merged["st_daily_net_ret_aligned"]
    merged["st_daily_turnover"] = merged["st_daily_turnover_aligned"]

    merged["daily_net_ret_diff"] = merged["bt_daily_net_ret"] - merged["st_daily_net_ret"]
    merged["daily_gross_ret_diff"] = merged["bt_daily_gross_ret"] - merged["st_daily_gross_ret"]

    merged["bt_cost_proxy"] = merged["bt_daily_gross_ret"] - merged["bt_daily_net_ret"]
    merged["st_cost_proxy"] = merged["st_daily_gross_ret"] - merged["st_daily_net_ret"]
    merged["cost_proxy_diff"] = merged["bt_cost_proxy"] - merged["st_cost_proxy"]

    merged["factor_diff"] = merged["bt_factor_value"] - merged["st_factor_value_aligned"]
    merged["exposure_diff_vs_bt_factor"] = merged["bt_factor_value"] - merged["st_notional_exposure_aligned"]

    merged["sign_mismatch_net_ret"] = (
        np.sign(merged["bt_daily_net_ret"].fillna(0.0))
        != np.sign(merged["st_daily_net_ret"].fillna(0.0))
    )
    merged["sign_mismatch_factor"] = (
        np.sign(merged["bt_factor_value"].fillna(0.0))
        != np.sign(merged["st_factor_value_aligned"].fillna(0.0))
    )
    merged["sign_mismatch_exposure"] = (
        np.sign(merged["bt_factor_value"].fillna(0.0))
        != np.sign(merged["st_notional_exposure_aligned"].fillna(0.0))
    )

    merged["bt_win_st_lose"] = (merged["bt_daily_net_ret"] > 0) & (merged["st_daily_net_ret"] < 0)
    merged["st_win_bt_lose"] = (merged["st_daily_net_ret"] > 0) & (merged["bt_daily_net_ret"] < 0)
    merged["abs_ret_diff"] = merged["daily_net_ret_diff"].abs()

    return merged.sort_values("time").reset_index(drop=True)


def _classify_reason_tags(row: pd.Series,
                          factor_tol: float,
                          exposure_tol: float,
                          cost_tol: float) -> str:
    """对单日差异打标签，用于快速定位“为什么两种回测结果差很大”。

    reason_tags 含义（会用 "|" 拼接多个标签）：
    1) factor_value_mismatch
       - 含义：同一天 BackTester 与 Strategy（已按 signal_delay_days 对齐后）的因子值差异过大。
       - 判定：|bt_factor_value - st_factor_value_aligned| > factor_tol。
       - 可能原因：滚动归一化参数不一致、因子输入列不一致、缺失值处理口径不同。

    2) exposure_mismatch
       - 含义：Strategy 实际持仓敞口（按手数换算）与 BackTester 理想因子敞口差异大。
       - 判定：|bt_factor_value - st_notional_exposure_aligned| > exposure_tol。
       - 可能原因：整数手约束、保证金约束导致降杠杆、资金不足无法完全按信号建仓。

    3) cost_model_gap
       - 含义：两种回测对“交易成本影响”的估计差异大。
       - 判定：|cost_proxy_diff| > cost_tol。
       - 说明：这里是 proxy 对比（gross_ret - net_ret），用于快速报警，不代表逐笔精确归因。

    4) ret_sign_mismatch
       - 含义：同一天净收益方向相反（一个赚钱一个亏钱）。
       - 判定：sign(bt_daily_net_ret) != sign(st_daily_net_ret)。
       - 作用：这是最关键的 bad-case 信号之一。

    5) position_direction_mismatch
       - 含义：BackTester 信号方向与 Strategy 实际持仓方向相反。
       - 判定：sign(bt_factor_value) != sign(st_notional_exposure)。
       - 可能原因：换仓当日冲击、约束触发、仓位截断后符号翻转（极少数场景）。

    6) margin_constraint
       - 含义：Strategy 触发保证金约束，目标手数被下调。
       - 判定：warning 含 "target_lots_downgraded_by_margin"。
       - 影响：会直接降低实际敞口，导致收益/回撤路径偏离 BackTester。

    7) lot_size_floor
       - 含义：最小交易单位约束触发，信号虽非0但无法开出至少1手。
       - 判定：warning 含 "cannot_open_min_one_lot"。
       - 影响：Strategy 当天可能“应有信号但无仓位”。

    8) strategy_terminated
       - 含义：策略在该日触发终止条件（例如权益<=0）。
       - 判定：warning 含 "terminated="。
       - 影响：后续路径与 BackTester 不再可比，需要优先排查该触发日。

    9) no_strong_signal
       - 含义：当天没有命中任何显著差异标签。
       - 用途：说明该日的偏差不大，或已在阈值内。
    """
    tags: List[str] = []

    warning_text = str(row.get("warning", "") or "")

    # 因子值口径差异：两边的 factor_value 本身不一致。
    if abs(float(row["factor_diff"])) > factor_tol:
        tags.append("factor_value_mismatch")
    # 敞口差异：Strategy 实际仓位换算的敞口和 BackTester 理想敞口偏差过大。
    if abs(float(row["exposure_diff_vs_bt_factor"])) > exposure_tol:
        tags.append("exposure_mismatch")
    # 成本差异：两边 net/gross 的差（成本代理）明显不同。
    if abs(float(row["cost_proxy_diff"])) > cost_tol:
        tags.append("cost_model_gap")
    # 收益方向冲突：同日一个正收益、一个负收益。
    if bool(row["sign_mismatch_net_ret"]):
        tags.append("ret_sign_mismatch")
    # 方向冲突：BackTester 信号方向与 Strategy 实际持仓方向不一致。
    if bool(row["sign_mismatch_exposure"]):
        tags.append("position_direction_mismatch")
    # 保证金约束触发：目标手数被系统下调。
    if "target_lots_downgraded_by_margin" in warning_text:
        tags.append("margin_constraint")
    # 最小手数约束触发：信号存在但无法开出1手。
    if "cannot_open_min_one_lot" in warning_text:
        tags.append("lot_size_floor")
    # 回测提前终止：通常意味着权益风险事件，需要优先排查。
    if "terminated=" in warning_text:
        tags.append("strategy_terminated")

    # 未命中任何显著标签，标记为“无强信号差异日”。
    if not tags:
        tags.append("no_strong_signal")
    return "|".join(tags)


def _summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    bt_total_ret = float(df["bt_daily_net_nav"].iloc[-1] - 1.0)
    st_total_ret = float(df["st_daily_net_nav"].iloc[-1] - 1.0)
    nav_gap = bt_total_ret - st_total_ret

    return {
        "sample_days": int(len(df)),
        "bt_total_net_return": bt_total_ret,
        "strategy_total_net_return": st_total_ret,
        "total_return_gap_bt_minus_strategy": nav_gap,
        "daily_net_ret_corr": float(df[["bt_daily_net_ret", "st_daily_net_ret"]].corr().iloc[0, 1]),
        "mean_abs_daily_net_ret_diff": float(df["abs_ret_diff"].mean()),
        "p95_abs_daily_net_ret_diff": float(df["abs_ret_diff"].quantile(0.95)),
        "sign_mismatch_net_ret_ratio": float(df["sign_mismatch_net_ret"].mean()),
        "bt_win_st_lose_days": int(df["bt_win_st_lose"].sum()),
        "st_win_bt_lose_days": int(df["st_win_bt_lose"].sum()),
        "factor_sign_mismatch_ratio": float(df["sign_mismatch_factor"].mean()),
        "exposure_sign_mismatch_ratio": float(df["sign_mismatch_exposure"].mean()),
        "mean_abs_factor_diff": float(df["factor_diff"].abs().mean()),
        "mean_abs_exposure_diff": float(df["exposure_diff_vs_bt_factor"].abs().mean()),
        "mean_cost_proxy_gap": float((df["st_cost_proxy"] - df["bt_cost_proxy"]).mean()),
    }


def _reason_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    scoreboard = pd.DataFrame({
        "metric": [
            "factor_value_mismatch_ratio(|bt_factor-st_factor_aligned|>0.1)",
            "exposure_mismatch_ratio(|bt_factor-st_exposure_aligned|>0.1)",
            "margin_constraint_ratio",
            "lot_size_floor_ratio",
            "ret_sign_mismatch_ratio",
            "cost_model_gap_ratio(|bt_cost-st_cost|>0.001)",
        ],
        "value": [
            float((df["factor_diff"].abs() > 0.1).mean()),
            float((df["exposure_diff_vs_bt_factor"].abs() > 0.1).mean()),
            float(df["warning"].astype(str).str.contains("target_lots_downgraded_by_margin", regex=False).mean()),
            float(df["warning"].astype(str).str.contains("cannot_open_min_one_lot", regex=False).mean()),
            float(df["sign_mismatch_net_ret"].mean()),
            float((df["cost_proxy_diff"].abs() > 0.001).mean()),
        ],
    })
    return scoreboard.sort_values("value", ascending=False).reset_index(drop=True)


def _extract_bad_cases(df: pd.DataFrame,
                       top_n: int,
                       factor_tol: float,
                       exposure_tol: float,
                       cost_tol: float) -> Dict[str, pd.DataFrame]:
    case_cols = [
        "time",
        "bt_daily_net_ret",
        "st_daily_net_ret",
        "st_daily_net_ret_raw",
        "daily_net_ret_diff",
        "bt_daily_net_nav",
        "st_daily_net_nav",
        "bt_factor_value",
        "st_factor_value",
        "st_factor_value_aligned",
        "factor_diff",
        "st_notional_exposure",
        "st_notional_exposure_aligned",
        "exposure_diff_vs_bt_factor",
        "bt_daily_turnover",
        "st_daily_turnover",
        "st_daily_turnover_raw",
        "bt_cost_proxy",
        "st_cost_proxy",
        "cost_proxy_diff",
        "position_lots",
        "target_lots",
        "delta_lots",
        "fee",
        "slippage_cost",
        "warning",
        "abs_ret_diff",
    ]

    bt_win = df[df["bt_win_st_lose"]].copy().sort_values("abs_ret_diff", ascending=False).head(top_n)
    st_win = df[df["st_win_bt_lose"]].copy().sort_values("abs_ret_diff", ascending=False).head(top_n)
    top_abs = df.sort_values("abs_ret_diff", ascending=False).head(top_n).copy()

    for item in (bt_win, st_win, top_abs):
        item["reason_tags"] = item.apply(
            lambda r: _classify_reason_tags(r, factor_tol=factor_tol, exposure_tol=exposure_tol, cost_tol=cost_tol),
            axis=1,
        )

    return {
        "bad_cases_bt_win_strategy_lose": bt_win[case_cols + ["reason_tags"]],
        "bad_cases_strategy_win_bt_lose": st_win[case_cols + ["reason_tags"]],
        "top_abs_diff_days": top_abs[case_cols + ["reason_tags"]],
    }


def _save_outputs(output_dir: Path,
                  merged_daily: pd.DataFrame,
                  summary: Dict[str, float],
                  scoreboard: pd.DataFrame,
                  bad_cases: Dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_daily.to_csv(output_dir / "daily_comparison.csv", index=False)
    scoreboard.to_csv(output_dir / "reason_scoreboard.csv", index=False)

    for name, df in bad_cases.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# Performance Diff Analysis",
        "",
        "## Headline",
        (
            f"- BT total return={summary['bt_total_net_return']:.4f}, "
            f"Strategy total return={summary['strategy_total_net_return']:.4f}, "
            f"gap(BT-Strategy)={summary['total_return_gap_bt_minus_strategy']:.4f}"
        ),
        (
            f"- daily corr={summary['daily_net_ret_corr']:.4f}, "
            f"sign mismatch ratio={summary['sign_mismatch_net_ret_ratio']:.2%}"
        ),
        (
            f"- bad days: bt_win_st_lose={summary['bt_win_st_lose_days']}, "
            f"st_win_bt_lose={summary['st_win_bt_lose_days']}"
        ),
        "",
        "## Files",
        "- daily_comparison.csv",
        "- reason_scoreboard.csv",
        "- bad_cases_bt_win_strategy_lose.csv",
        "- bad_cases_strategy_win_bt_lose.csv",
        "- top_abs_diff_days.csv",
        "- summary.json",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze diff between BackTester and Strategy performance.")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--factor-name", type=str, required=True)
    parser.add_argument("--instrument-id", type=str, default="C0")
    parser.add_argument("--start-time", type=str, default="20200101")
    parser.add_argument("--end-time", type=str, default="20241231")

    parser.add_argument("--database", type=str, default="factors")
    parser.add_argument("--collection", type=str, default="genetic_programming")

    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--margin-rate", type=float, default=0.1)
    parser.add_argument("--fee-per-lot", type=float, default=2.0)
    parser.add_argument("--slippage", type=float, default=1.0)

    parser.add_argument("--apply-rolling-norm", dest="apply_rolling_norm", action="store_true")
    parser.add_argument("--no-apply-rolling-norm", dest="apply_rolling_norm", action="store_false")
    parser.set_defaults(apply_rolling_norm=True)
    parser.add_argument("--rolling-norm-window", type=int, default=30)
    parser.add_argument("--rolling-norm-min-periods", type=int, default=20)
    parser.add_argument("--rolling-norm-eps", type=float, default=1e-8)
    parser.add_argument("--rolling-norm-clip", type=float, default=5.0)
    parser.add_argument("--signal-delay-days", type=int, default=1)

    parser.add_argument("--factor-diff-threshold", type=float, default=0.1)
    parser.add_argument("--exposure-diff-threshold", type=float, default=0.1)
    parser.add_argument("--cost-diff-threshold", type=float, default=0.001)

    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Default: test/artifacts/performance_diff_analysis_<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bt = run_backtester(args)
    strategy = run_strategy(args)

    bt_detail = _prepare_bt_detail(bt, instrument_id=args.instrument_id, factor_name=args.factor_name)
    st_detail = _prepare_strategy_detail(strategy)
    merged_daily = _build_daily_comparison(
        bt_detail=bt_detail,
        st_detail=st_detail,
        instrument_id=args.instrument_id,
        initial_capital=args.initial_capital,
        signal_delay_days=args.signal_delay_days,
    )

    merged_daily["reason_tags"] = merged_daily.apply(
        lambda r: _classify_reason_tags(
            r,
            factor_tol=args.factor_diff_threshold,
            exposure_tol=args.exposure_diff_threshold,
            cost_tol=args.cost_diff_threshold,
        ),
        axis=1,
    )

    summary = _summary_metrics(merged_daily)
    scoreboard = _reason_scoreboard(merged_daily)
    bad_cases = _extract_bad_cases(
        merged_daily,
        top_n=args.top_n,
        factor_tol=args.factor_diff_threshold,
        exposure_tol=args.exposure_diff_threshold,
        cost_tol=args.cost_diff_threshold,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("test") / "artifacts" / f"performance_diff_analysis_{ts}"

    _save_outputs(
        output_dir=output_dir,
        merged_daily=merged_daily,
        summary=summary,
        scoreboard=scoreboard,
        bad_cases=bad_cases,
    )

    print("=== Performance Diff Analysis Done ===")
    print(f"Output dir: {output_dir.resolve()}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop reason signals:")
    print(scoreboard.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



