"""检查 C0 在数据库中的重复主键数据（instrument_id + time）。

用法：
    python -u test/check_c0_duplicates.py
    python -u test/check_c0_duplicates.py --start-date 20200101 --end-date 20241231 --max-keys 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.futures import get_futures_continuous_contract_price


def _to_text(value) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _print_duplicate_block(df_dup: pd.DataFrame, key_cols: List[str], max_keys: int) -> None:
    if df_dup.empty:
        print("未发现重复主键记录。")
        return

    grouped = df_dup.groupby(key_cols, dropna=False, sort=True)
    keys = list(grouped.groups.keys())

    print(f"重复主键组数: {len(keys)}")
    if len(keys) > max_keys:
        print(f"仅展示前 {max_keys} 组，剩余 {len(keys) - max_keys} 组未展开。")

    for i, key in enumerate(keys[:max_keys], 1):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_text = ", ".join(f"{k}={_to_text(v)}" for k, v in zip(key_cols, key_tuple))

        block = grouped.get_group(key).copy()
        print("\n" + "=" * 100)
        print(f"[{i}] 重复键: {key_text} | 记录数={len(block)}")

        # 找出同一键下取值不一致的字段，帮助定位“什么数据”重复且不一致
        varying_cols = [
            c for c in block.columns
            if c not in key_cols and block[c].nunique(dropna=False) > 1
        ]
        if varying_cols:
            print(f"字段不一致（同一键下值不同）: {varying_cols}")
        else:
            print("字段不一致: 无（同一键下记录完全一致）")

        show_cols = key_cols + [c for c in [
            "symbol", "open", "high", "low", "close", "settle", "volume", "position",
            "weighted_factor", "cur_weighted_factor", "is_rollover"
        ] if c in block.columns]
        show_cols = list(dict.fromkeys(show_cols))
        print(block[show_cols].sort_values(key_cols).to_string(index=False))


def _print_duplicate_rows_by_date(df_dup: pd.DataFrame,
                                  key_cols: List[str],
                                  max_days: int,
                                  show_cols: List[str]) -> None:
    """按日期展开打印重复行明细，每个日期用 '=' * 100 分隔。"""
    if df_dup.empty:
        return

    if "time" not in df_dup.columns:
        print("无法按日期展开：缺少 time 字段。")
        return

    by_day = df_dup.groupby("time", dropna=False, sort=True)
    days = list(by_day.groups.keys())

    print(f"\n按日期展开重复明细：共 {len(days)} 天")
    if len(days) > max_days:
        print(f"仅展示前 {max_days} 天，剩余 {len(days) - max_days} 天未展开。")

    for i, day in enumerate(days[:max_days], 1):
        block = by_day.get_group(day).copy()

        # 同一天里可能存在多个重复键，进一步按 key 分组看看每组重复条数
        key_group_stat = (
            block.groupby(key_cols, dropna=False)
            .size()
            .reset_index(name="dup_count")
            .sort_values(key_cols)
        )

        print("\n" + "=" * 100)
        print(f"[日期 {i}] {pd.Timestamp(day).strftime('%Y-%m-%d')} | 重复总行数={len(block)}")
        print("同日重复键统计:")
        print(key_group_stat.to_string(index=False))

        out_cols = [c for c in show_cols if c in block.columns]
        print("同日重复数据明细:")
        print(block[out_cols].sort_values(key_cols).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="检查连续合约价格数据中的重复主键")
    parser.add_argument("--instrument-id", default="C0", help="合约ID，默认 C0")
    parser.add_argument("--start-date", default="20010101", help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期，格式 YYYYMMDD；默认使用函数内部默认值")
    parser.add_argument("--max-keys", type=int, default=50, help="最多展开多少个重复键组")
    parser.add_argument("--max-days", type=int, default=50, help="按日期展开时最多展示多少天")
    args = parser.parse_args()

    print("开始读取数据...")
    df = get_futures_continuous_contract_price(
        instrument_id=args.instrument_id,
        start_date=args.start_date,
        end_date=args.end_date,
        from_database=True,
    )

    if df is None or df.empty:
        print("未读到任何数据。")
        return

    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values([c for c in ["instrument_id", "time"] if c in df.columns]).reset_index(drop=True)

    print(f"\n数据总行数: {len(df)}")
    print(f"字段列表: {list(df.columns)}")

    key_cols = ["instrument_id", "time"]
    missing_key_cols = [c for c in key_cols if c not in df.columns]
    if missing_key_cols:
        print(f"缺少主键字段，无法检查重复: {missing_key_cols}")
        return

    total_dup_rows = int(df.duplicated(subset=key_cols, keep=False).sum())
    dup_key_df = df[df.duplicated(subset=key_cols, keep=False)].copy()

    unique_key_count = int(df.drop_duplicates(subset=key_cols).shape[0])
    print(f"主键唯一行数(instrument_id,time): {unique_key_count}")
    print(f"主键重复行数(instrument_id,time): {total_dup_rows}")

    # 先打印按日期聚合的重复统计
    if not dup_key_df.empty:
        by_day = (
            dup_key_df.groupby("time", dropna=False)
            .size()
            .reset_index(name="duplicate_row_count")
            .sort_values("time")
        )
        print("\n按日期统计重复行数（同一天可能包含多个重复键组）:")
        print(by_day.to_string(index=False))

    # 再展开每个重复键详情（兼容旧输出）
    _print_duplicate_block(dup_key_df, key_cols=key_cols, max_keys=max(1, args.max_keys))

    # 新增：按日期展开重复明细，每个日期用 '=' * 100 分隔
    detail_show_cols = key_cols + [
        c for c in [
            "symbol", "open", "high", "low", "close", "settle", "volume", "position",
            "weighted_factor", "cur_weighted_factor", "is_rollover"
        ] if c in dup_key_df.columns
    ]
    detail_show_cols = list(dict.fromkeys(detail_show_cols))
    _print_duplicate_rows_by_date(
        dup_key_df,
        key_cols=key_cols,
        max_days=max(1, args.max_days),
        show_cols=detail_show_cols,
    )


if __name__ == "__main__":
    main()

