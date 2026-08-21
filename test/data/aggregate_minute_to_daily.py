"""
把分钟频数据库中的 joinquant 数据聚合为日频, 写入日频库(source=joinquant, 与 akshare 并存)。

- 换月日/weighted_factor/cur_weighted_factor/is_rollover 直接沿用分钟数据(由聚宽数据确定)。
- 写入唯一键含 source, 不覆盖 akshare 日频数据。
- 对「无分钟数据 / 缺夜盘 / 节假日无夜盘」的交易日输出 warning。

用法
----
python -u test/data/aggregate_minute_to_daily.py                    # 全部品种, 全历史(自 RESEARCH_START_DATE)
python -u test/data/aggregate_minute_to_daily.py --instrument C0
python -u test/data/aggregate_minute_to_daily.py --start 2026-01-01 --end 2026-08-20
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.futures import (  # noqa: E402
    update_futures_continuous_contract_price_from_minute,
    SOURCE_JOINQUANT,
)


def main():
    parser = argparse.ArgumentParser(description="分钟聚合成日频写入日频库")
    parser.add_argument("--instrument", default=None, help="品种, 留空=全部")
    parser.add_argument("--start", default=None, help="开始日期 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--source", default=SOURCE_JOINQUANT, help="分钟数据来源, 默认 joinquant")
    parser.add_argument("--method", default="bulk_write_update")
    args = parser.parse_args()

    update_futures_continuous_contract_price_from_minute(
        instrument_id=args.instrument,
        start_date=args.start,
        end_date=args.end,
        method=args.method,
        source=args.source,
    )
    print("完成。")


if __name__ == "__main__":
    main()
