"""
验证日频(futures.continuous_contract_price_daily)与分钟(futures.continuous_contract_price_1min)
数据是否完全匹配。

验证内容
--------
1. 由分钟数据按「交易日」聚合出的 open/high/low/close，是否等于日频库的 open/high/low/close；
2. 每天 is_rollover / weighted_factor / cur_weighted_factor / symbol 是否一致；
3. 其它：volume 之和、position(收盘持仓量)、每天分钟 bar 数是否正常；
   settle 日频是真实结算价、分钟库中取的是 close，因此预期不同(单独报告)。

交易日落日说明
--------
中国商品期货夜盘(21:00 之后)归属下一个自然日的交易日，聚合分钟数据时按此规则归组，
与 import_c0_1min_to_db.py 的 assign_trading_day 保持一致。

用法
----
python -u test/data/verify_daily_vs_1min.py                     # 全部交易日对比
python -u test/data/verify_daily_vs_1min.py --instrument C0     # 指定品种
python -u test/data/verify_daily_vs_1min.py --sample 30         # 随机抽 30 天对比
python -u test/data/verify_daily_vs_1min.py --atol 0.02         # 价格差异阈值(元)

说明
----
- 只读不写，不会改动数据库。
- 数据量较大时建议先 --sample 抽样；默认全量对比(输出摘要)。
- 判定"完全一致" = 所有对比日中 OHLC/volume/position/factor/is_rollover/symbol 全部匹配。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mongo.mongify import get_data  # noqa: E402
from import_c0_1min_to_db import assign_trading_day  # noqa: E402  与导入脚本保持同一归属规则

DATABASE = "futures"
DAILY_COLLECTION = "continuous_contract_price_daily"
MIN_COLLECTION = "continuous_contract_price_1min"


def _clean_time(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def aggregate_minute_to_daily(df_min: pd.DataFrame, trading_days=None) -> pd.DataFrame:
    """把分钟数据按交易日聚合为每日 OHLCV / position / factor。

    trading_days: 交易日列表(用于夜盘归属, 默认从分钟数据的日盘日期推断)。
    """
    df = df_min.copy()
    df["time"] = _clean_time(df["time"])
    df = df.dropna(subset=["time"])
    df["td"] = assign_trading_day(df["time"], trading_days=trading_days)

    for c in ["open", "high", "low", "close", "volume", "position", "money"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["weighted_factor"] = pd.to_numeric(df.get("weighted_factor"), errors="coerce").fillna(1.0)
    df["cur_weighted_factor"] = pd.to_numeric(df.get("cur_weighted_factor"), errors="coerce").fillna(1.0)
    if "is_rollover" in df.columns:
        df["is_rollover"] = df["is_rollover"].astype(bool)
    else:
        df["is_rollover"] = False

    g = df.groupby("td")
    agg = pd.DataFrame({
        "time": g["time"].first(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "position": g["position"].last(),
        "money": g["money"].sum(),
        "weighted_factor": g["weighted_factor"].first(),
        "cur_weighted_factor": g["cur_weighted_factor"].first(),
        "is_rollover": g["is_rollover"].any(),
        "symbol": g["symbol"].first(),
        "bar_count": g.size(),
    })
    agg["td"] = agg.index
    return agg.reset_index(drop=True)


def _diff_report(name: str, daily: pd.Series, mini: pd.Series, ok: pd.Series,
                 unit: str = "", relative: bool = False):
    """统计某字段的匹配情况。"""
    d = pd.to_numeric(daily, errors="coerce") if not pd.api.types.is_bool_dtype(daily) else daily.astype(bool)
    m = pd.to_numeric(mini, errors="coerce") if not pd.api.types.is_bool_dtype(mini) else mini.astype(bool)
    matched = ok.sum()
    total = len(ok)
    if total == 0:
        return None
    if pd.api.types.is_bool_dtype(d):
        pct = 100.0 * matched / total
        print(f"  {name:<18} 匹配 {matched}/{total} ({pct:.2f}%)")
        return None
    if d.isna().all() or m.isna().all():  # 非数值列(如 symbol), 仅报匹配率
        pct = 100.0 * matched / total
        print(f"  {name:<18} 匹配 {matched}/{total} ({pct:.2f}%)")
        return None
    diff = (d - m).abs()
    if relative:
        denom = d.abs().replace(0, np.nan)
        diff_rel = (diff / denom).dropna()
        summary = diff_rel.describe(percentiles=[.5, .9, .99])
        print(f"  {name:<18} 匹配 {matched}/{total} ({100.0*matched/total:.2f}%)  | 相对差异: "
              f"中位 {summary['50%']:.6f}, P90 {summary['90%']:.6f}, P99 {summary['99%']:.6f}, 最大 {summary['max']:.6f}")
    else:
        summary = diff.describe(percentiles=[.5, .9, .99])
        print(f"  {name:<18} 匹配 {matched}/{total} ({100.0*matched/total:.2f}%)  | 绝对差异{unit}: "
              f"中位 {summary['50%']:.4g}, P90 {summary['90%']:.4g}, P99 {summary['99%']:.4g}, 最大 {summary['max']:.4g}")
    return diff


def main():
    parser = argparse.ArgumentParser(description="验证日频与分钟数据是否匹配")
    parser.add_argument("--instrument", default="C0")
    parser.add_argument("--database", default=DATABASE)
    parser.add_argument("--daily-col", default=DAILY_COLLECTION)
    parser.add_argument("--min-col", default=MIN_COLLECTION)
    parser.add_argument("--sample", type=int, default=None, help="随机抽样对比 N 个交易日(默认全部)")
    parser.add_argument("--atol", type=float, default=0.02, help="OHLC 价格差异阈值(元), 默认 0.02")
    parser.add_argument("--vol-rtol", type=float, default=0.01, help="volume 相对差异阈值, 默认 1%")
    parser.add_argument("--factor-rtol", type=float, default=1e-9, help="factor 相对差异阈值")
    args = parser.parse_args()

    print("=" * 78)
    print(f"验证日频 vs 分钟数据 | 品种 {args.instrument}")
    print("=" * 78)

    # ---- 读日频 ----
    df_daily = get_data(args.database, args.daily_col,
                        {"instrument_id": args.instrument})
    if df_daily is None or df_daily.empty:
        print(f"[错误] 日频库 {args.database}.{args.daily_col} 无 {args.instrument} 数据")
        sys.exit(1)
    df_daily = df_daily.copy()
    df_daily["time"] = _clean_time(df_daily["time"])
    df_daily["td"] = df_daily["time"].dt.normalize()
    df_daily = df_daily.dropna(subset=["td"]).drop_duplicates(subset=["td"], keep="last")
    for c in ["open", "high", "low", "close", "volume", "position", "weighted_factor", "cur_weighted_factor"]:
        if c in df_daily.columns:
            df_daily[c] = pd.to_numeric(df_daily[c], errors="coerce")
    if "is_rollover" in df_daily.columns:
        df_daily["is_rollover"] = df_daily["is_rollover"].astype(bool)
    else:
        df_daily["is_rollover"] = False
    print(f"日频: {len(df_daily)} 行, 范围 {df_daily['td'].min().date()} ~ {df_daily['td'].max().date()}")

    # ---- 读分钟 ----
    df_min = get_data(args.database, args.min_col,
                      {"instrument_id": args.instrument})
    if df_min is None or df_min.empty:
        print(f"[错误] 分钟库 {args.database}.{args.min_col} 无 {args.instrument} 数据")
        sys.exit(1)
    print(f"分钟: {len(df_min)} 行")

    # 交易日列表以日频库为准(用于夜盘归属: 周五夜盘 -> 下周一)
    trading_days = sorted(pd.unique(df_daily["td"]).tolist())
    if not trading_days:
        print("[警告] 日频库无有效交易日, 改用分钟数据日盘日期推断")
        trading_days = None

    # ---- 聚合分钟 ----
    min_daily = aggregate_minute_to_daily(df_min, trading_days=trading_days)
    print(f"分钟聚合: {len(min_daily)} 个交易日, 范围 {min_daily['td'].min().date()} ~ {min_daily['td'].max().date()}")

    # ---- 抽样 ----
    if args.sample and 0 < args.sample < len(min_daily):
        sample_days = min_daily["td"].sample(args.sample, random_state=42).tolist()
        min_daily = min_daily[min_daily["td"].isin(sample_days)].copy()
        df_daily = df_daily[df_daily["td"].isin(sample_days)].copy()
        print(f"[抽样] 只对比 {args.sample} 个交易日")

    # ---- 合并 ----
    merged = pd.merge(df_daily, min_daily, on="td", how="outer", suffixes=("_daily", "_min"), indicator=True)
    only_daily = merged[merged["_merge"] == "left_only"]
    only_min = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()
    both = both.sort_values("td").reset_index(drop=True)

    print(f"\n交易日: 两边都有 {len(both)} 天, 仅日频 {len(only_daily)} 天, 仅分钟 {len(only_min)} 天")
    if len(only_daily):
        print("  仅日频存在的日子: " + ", ".join(pd.Timestamp(t).date().isoformat() for t in only_daily["td"]))
    if len(only_min):
        print("  仅分钟存在的日子: " + ", ".join(pd.Timestamp(t).date().isoformat() for t in only_min["td"]))

    if both.empty:
        print("\n[结论] 两边没有共同交易日, 无法对比!")
        sys.exit(1)

    # ---- 字段对比 ----
    atol = args.atol
    print("\n--- 对比(两边都有 %d 天) ---" % len(both))

    def _close_ok(a, b, thresh):
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return (a - b).abs() <= thresh

    ok_open = _close_ok(both["open_daily"], both["open_min"], atol)
    ok_high = _close_ok(both["high_daily"], both["high_min"], atol)
    ok_low = _close_ok(both["low_daily"], both["low_min"], atol)
    ok_close = _close_ok(both["close_daily"], both["close_min"], atol)

    vol_d = pd.to_numeric(both.get("volume_daily"), errors="coerce")
    vol_m = pd.to_numeric(both.get("volume_min"), errors="coerce")
    vol_denom = vol_d.abs().replace(0, np.nan)
    ok_vol = ((vol_d - vol_m).abs() / vol_denom <= args.vol_rtol) | (vol_d.isna() & vol_m.isna())

    pos_d = pd.to_numeric(both.get("position_daily"), errors="coerce")
    pos_m = pd.to_numeric(both.get("position_min"), errors="coerce")
    pos_denom = pos_d.abs().replace(0, np.nan)
    ok_pos = ((pos_d - pos_m).abs() / pos_denom <= 0.02) | (pos_d.isna() & pos_m.isna())

    wf_d = pd.to_numeric(both.get("weighted_factor_daily"), errors="coerce")
    wf_m = pd.to_numeric(both.get("weighted_factor_min"), errors="coerce")
    wf_denom = wf_d.abs().replace(0, np.nan)
    ok_wf = ((wf_d - wf_m).abs() / wf_denom <= args.factor_rtol) | (wf_d.isna() & wf_m.isna())

    cwf_d = pd.to_numeric(both.get("cur_weighted_factor_daily"), errors="coerce")
    cwf_m = pd.to_numeric(both.get("cur_weighted_factor_min"), errors="coerce")
    cwf_denom = cwf_d.abs().replace(0, np.nan)
    ok_cwf = ((cwf_d - cwf_m).abs() / cwf_denom <= args.factor_rtol) | (cwf_d.isna() & cwf_m.isna())

    ok_roll = both["is_rollover_daily"].astype(bool) == both["is_rollover_min"].astype(bool)
    ok_symbol = both.get("symbol_daily", pd.Series("", index=both.index)).astype(str) == \
                both.get("symbol_min", pd.Series("", index=both.index)).astype(str)

    _diff_report("open", both["open_daily"], both["open_min"], ok_open, unit="元")
    _diff_report("high", both["high_daily"], both["high_min"], ok_high, unit="元")
    _diff_report("low", both["low_daily"], both["low_min"], ok_low, unit="元")
    _diff_report("close", both["close_daily"], both["close_min"], ok_close, unit="元")
    _diff_report("volume", both["volume_daily"], both["volume_min"], ok_vol, relative=True)
    _diff_report("position", both["position_daily"], both["position_min"], ok_pos, relative=True)
    _diff_report("weighted_factor", both["weighted_factor_daily"], both["weighted_factor_min"], ok_wf, relative=True)
    _diff_report("cur_weighted_factor", both["cur_weighted_factor_daily"], both["cur_weighted_factor_min"], ok_cwf, relative=True)
    _diff_report("is_rollover", both["is_rollover_daily"], both["is_rollover_min"], ok_roll)
    _diff_report("symbol", both["symbol_daily"], both["symbol_min"], ok_symbol)

    # ---- 汇总 ----
    all_ok = pd.Series(np.ones(len(both), dtype=bool))
    for mask in [ok_open, ok_high, ok_low, ok_close, ok_vol, ok_pos, ok_wf, ok_cwf, ok_roll, ok_symbol]:
        all_ok &= mask.fillna(False) if hasattr(mask, "fillna") else mask

    print("\n--- 不匹配明细(至少一个字段不一致的日子) ---")
    bad = both[~all_ok]
    if bad.empty:
        print("  全部一致 ✅")
    else:
        for _, r in bad.iterrows():
            issues = []
            if not bool(ok_open.loc[r.name]): issues.append(f"open {r['open_daily']} vs {r['open_min']}")
            if not bool(ok_high.loc[r.name]): issues.append(f"high {r['high_daily']} vs {r['high_min']}")
            if not bool(ok_low.loc[r.name]): issues.append(f"low {r['low_daily']} vs {r['low_min']}")
            if not bool(ok_close.loc[r.name]): issues.append(f"close {r['close_daily']} vs {r['close_min']}")
            if not bool(ok_vol.loc[r.name]): issues.append(f"volume {r['volume_daily']} vs {r['volume_min']}")
            if not bool(ok_pos.loc[r.name]): issues.append(f"position {r['position_daily']} vs {r['position_min']}")
            if not bool(ok_wf.loc[r.name]): issues.append(f"wf {r['weighted_factor_daily']} vs {r['weighted_factor_min']}")
            if not bool(ok_cwf.loc[r.name]): issues.append(f"cwf {r['cur_weighted_factor_daily']} vs {r['cur_weighted_factor_min']}")
            if not bool(ok_roll.loc[r.name]): issues.append(f"is_rollover {r['is_rollover_daily']} vs {r['is_rollover_min']}")
            if not bool(ok_symbol.loc[r.name]): issues.append(f"symbol {r['symbol_daily']} vs {r['symbol_min']}")
            print(f"  {pd.Timestamp(r['td']).date()}: " + "; ".join(issues))
        print(f"\n  共 {len(bad)}/{len(both)} 天不完全一致")

    # ---- 每天分钟 bar 数异常检查(全量) ----
    print("\n--- 每天分钟 bar 数检查(全量) ---")
    bar_df = aggregate_minute_to_daily(df_min, trading_days=trading_days)[["td", "bar_count"]]
    abnormal = bar_df[(bar_df["bar_count"] < 200) | (bar_df["bar_count"] > 400)]
    if abnormal.empty:
        print(f"  全部交易日 bar 数在正常区间(中位 {int(bar_df['bar_count'].median())}, "
              f"范围 {int(bar_df['bar_count'].min())}~{int(bar_df['bar_count'].max())}) ✅")
    else:
        print(f"  {len(abnormal)} 天 bar 数异常(可能数据缺失):")
        for _, r in abnormal.iterrows():
            print(f"    {pd.Timestamp(r['td']).date()}: {int(r['bar_count'])} 根")

    # ---- settle 说明 ----
    print("\n--- settle 说明 ---")
    print("  日频 settle 为交易所真实结算价; 分钟库中 settle 在导入时取 close, 因此两者必然不同(属预期)。")

    # ---- 结论 ----
    match = both[~all_ok].empty and only_daily.empty and only_min.empty
    print("\n" + "=" * 78)
    if match:
        print("结论: ✅ 日频与分钟数据在所有对比维度上完全一致, 两个数据源匹配。")
    else:
        print("结论: ❌ 存在不一致, 请查看上方不匹配明细。")
    print("=" * 78)


if __name__ == "__main__":
    main()
