"""
检查 futures.continuous_contract_price_1min 中是否有重复的 (time, instrument_id)。

背景: 若某日分钟库 bar 重复, verify 聚合时 volume 会被重复累加、open 取到重复行,
导致"分钟"侧与 CSV/日频对不上(如 volume 偏大好几倍)。

用法:
    python -u test/data/check_duplicate_1min.py                     # 全库查重
    python -u test/data/check_duplicate_1min.py --date 2026-05-12   # 只看某日期
    python -u test/data/check_duplicate_1min.py --instrument C0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mongo.mongify import get_data  # noqa: E402

DATABASE = "futures"
MIN_COLLECTION = "continuous_contract_price_1min"


def main():
    parser = argparse.ArgumentParser(description="检查分钟库重复数据")
    parser.add_argument("--instrument", default="C0")
    parser.add_argument("--date", default=None, help="只看该时间戳日期(如 2026-05-12)")
    args = parser.parse_args()

    df = get_data(DATABASE, MIN_COLLECTION, {"instrument_id": args.instrument})
    if df is None or df.empty:
        print(f"[错误] {DATABASE}.{MIN_COLLECTION} 无 {args.instrument} 数据")
        sys.exit(1)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    total = len(df)
    unique_time = df["time"].nunique()
    dup = df[df.duplicated(subset=["time", "instrument_id"], keep=False)]
    print(f"分钟库 {args.instrument}: 总条数 {total}, 唯一 time 数 {unique_time}, "
          f"重复条数 {len(dup)} (重复 time 数 {dup['time'].nunique()})")

    if len(dup):
        dup["date"] = dup["time"].dt.strftime("%Y-%m-%d")
        grp = dup.groupby("date").size().sort_values(ascending=False)
        print("\n有重复的日期(按条数排序):")
        print(grp.head(20).to_string())

    if args.date:
        day = df[df["time"].dt.strftime("%Y-%m-%d") == args.date].copy()
        print(f"\n=== 时间戳日期 {args.date} ===")
        print(f"  条数: {len(day)}")
        if "volume" in day.columns:
            print(f"  volume 合计: {day['volume'].sum():.0f}")
        if "open" in day.columns:
            print(f"  首行 open: {day['open'].iloc[0] if not day.empty else 'N/A'}")
        dup_day = day[day.duplicated(subset=["time"], keep=False)].sort_values("time")
        if not dup_day.empty:
            print(f"  重复 time 条数: {len(dup_day)}")
            cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in dup_day.columns]
            print(dup_day[cols].head(30).to_string())
        else:
            print("  该时间戳日期内无重复 time")

    print("\n若发现重复: 建议清空该 collection 后重新全量导入(upsert 会去重覆盖):")
    print("  python -c \"from mongo.mongify import delete_data; "
          "delete_data('futures','continuous_contract_price_1min',{'instrument_id':'C0'})\"")


if __name__ == "__main__":
    main()
