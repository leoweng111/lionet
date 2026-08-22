"""
对比数据库 minutes 与聚宽 CSV 在指定"交易日"的分钟 bar, 定位差异(重复/缺失/值不一致)。

背景
----
verify 用「交易日」口径聚合(td = 前一交易日夜盘 + 当日日盘)。
当你查 time=某日历日 有 345 条无重复时, 不代表 td=该日交易日没问题——
因为 td=2026-05-12 需要的是「2026-05-11 21:00~23:00 夜盘 + 2026-05-12 09:00~15:00 日盘」。

用法
----
python -u test/data/compare_db_vs_csv_1min.py --date 2026-05-12
    - 对比数据库 vs CSV 在 td=2026-05-12 交易日(05-11夜盘+05-12日盘) 的 bar
    - 打印缺失 bar、数据库多余 bar、值不一致的 bar
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.futures import assign_trading_day_1min  # noqa: E402
from mongo.mongify import get_data  # noqa: E402

CSV_PATH = "/Users/wenglongao/Downloads/C9999.XDCE.csv"
DATABASE = "futures"
MIN_COLLECTION = "continuous_contract_price_1min"
INSTRUMENT = "C0"
FIELDS = ["open", "high", "low", "close", "volume", "position"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="交易日(td), 如 2026-05-12")
    parser.add_argument("--csv", default=CSV_PATH)
    args = parser.parse_args()

    td = pd.Timestamp(args.date)

    # ---- CSV ----
    csv = pd.read_csv(args.csv, parse_dates=["datetime"])
    csv = csv.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    csv["td"] = assign_trading_day_1min(csv["datetime"])
    csv_td = csv[csv["td"] == td].copy()
    print(f"CSV td={td.date()} (前一交易日20:59~当日15:00): {len(csv_td)} 根")

    # ---- 数据库 ----
    db = get_data(DATABASE, MIN_COLLECTION, {"instrument_id": INSTRUMENT})
    db = db.copy()
    db["time"] = pd.to_datetime(db["time"], errors="coerce")
    db = db.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    db["td"] = assign_trading_day_1min(db["time"])
    db_td = db[db["td"] == td].copy()
    print(f"数据库 td={td.date()}: {len(db_td)} 根")

    # ---- 按 time 对齐 ----
    csv_k = csv_td.set_index("datetime")[["open", "high", "low", "close", "volume", "open_interest"]]
    csv_k = csv_k.rename(columns={"open_interest": "position"})
    db_k = db_td.set_index("time")[["open", "high", "low", "close", "volume", "position"]]
    # 统一 index 格式
    csv_k.index = pd.to_datetime(csv_k.index)
    db_k.index = pd.to_datetime(db_k.index)

    common = csv_k.index.intersection(db_k.index)
    only_csv = csv_k.index.difference(db_k.index)
    only_db = db_k.index.difference(csv_k.index)
    print(f"\n两边共同 bar: {len(common)} | 仅CSV有(数据库缺): {len(only_csv)} | 仅数据库有(CSV缺): {len(only_db)}")

    if len(only_csv):
        print(f"\n[数据库缺失] 这些 time 在 CSV 有但数据库没有 ({len(only_csv)} 根):")
        print(csv_k.loc[only_csv][["open", "close", "volume"]].to_string())
    if len(only_db):
        print(f"\n[数据库多余] 这些 time 在数据库有但 CSV 没有 ({len(only_db)} 根):")
        print(db_k.loc[only_db][["open", "close", "volume"]].to_string())

    # ---- 值不一致 ----
    if len(common):
        merged = db_k.loc[common].merge(csv_k.loc[common], left_index=True, right_index=True,
                                        suffixes=("_db", "_csv"))
        diffs = []
        for f in FIELDS:
            a = pd.to_numeric(merged[f"{f}_db"], errors="coerce")
            b = pd.to_numeric(merged[f"{f}_csv"], errors="coerce")
            m = (a - b).abs() > 0.01 if f != "volume" else ((a - b).abs() / b.replace(0, pd.NA) > 0.01)
            diffs.append(merged[m].index)
        bad_idx = set()
        for d in diffs:
            bad_idx |= set(d)
        if bad_idx:
            print(f"\n[值不一致] 共同 bar 中 {len(bad_idx)} 根 DB 与 CSV 不一致:")
            show = merged.loc[sorted(bad_idx)]
            show["DB_vol"] = pd.to_numeric(show["volume_db"], errors="coerce")
            show["CSV_vol"] = pd.to_numeric(show["volume_csv"], errors="coerce")
            print(show[["open_db", "open_csv", "close_db", "close_csv",
                        "volume_db", "volume_csv", "position_db", "position_csv"]].to_string())
        else:
            print("\n共同 bar 的 open/high/low/close/volume/position 全部一致 ✅")

    # ---- 合计 ----
    print("\n--- 交易日合计 ---")
    print(f"CSV:     bar={len(csv_td)}, volume={pd.to_numeric(csv_td['volume'], errors='coerce').sum():.0f}, "
          f"position(收盘)={pd.to_numeric(csv_td['open_interest'], errors='coerce').iloc[-1]:.0f}")
    vol_col = "volume" if "volume" in db_td.columns else None
    pos_col = "position" if "position" in db_td.columns else None
    db_vol = pd.to_numeric(db_td[vol_col], errors="coerce").sum() if vol_col else float("nan")
    db_pos = pd.to_numeric(db_td[pos_col], errors="coerce").iloc[-1] if pos_col else float("nan")
    print(f"数据库:  bar={len(db_td)}, volume={db_vol:.0f}, position(收盘)={db_pos:.0f}")


if __name__ == "__main__":
    main()
