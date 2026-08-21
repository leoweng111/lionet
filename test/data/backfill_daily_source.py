"""
为日频库 futures.continuous_contract_price_daily 中「没有 source 字段」的记录回填 source='akshare'。

背景
----
日频数据原本由 akshare 接口写入, 但没有 source 字段。为统一口径,
本脚本把日频库中所有缺失/为空 source 的记录补上 source='akshare'。
(joinquant 聚合来的日频会带 source='joinquant', 不会受影响)

用法
----
python -u test/data/backfill_daily_source.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mongo.mongify import get_data, update_one_data  # noqa: E402

DATABASE = "futures"
DAILY_COLLECTION = "continuous_contract_price_daily"
SOURCE = "akshare"


def main():
    df = get_data(DATABASE, DAILY_COLLECTION, None, idx=True)  # 含 _id
    if df is None or df.empty:
        print(f"{DATABASE}.{DAILY_COLLECTION} 无数据")
        return

    if "source" not in df.columns:
        df["source"] = ""
    missing = df[df["source"].isna() | (df["source"].astype(str).str.strip() == "")]
    print(f"日频库共 {len(df)} 条, 其中缺 source 的 {len(missing)} 条")

    done = 0
    for _, row in missing.iterrows():
        update_one_data(DATABASE, DAILY_COLLECTION,
                        {"_id": row["_id"]}, {"source": SOURCE}, upsert=False)
        done += 1
        if done % 5000 == 0:
            print(f"  已回填 {done} 条")

    print(f"完成: 回填 {done} 条 source={SOURCE}")


if __name__ == "__main__":
    main()
