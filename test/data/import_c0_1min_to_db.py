"""
一键导入聚宽导出的玉米(C0) 1 分钟连续合约数据到 MongoDB: futures.continuous_contract_price_1min

说明
----
- 聚宽 9999 主力连续合约是「未后复权、未做换月拼接」的原始价格序列, 换月处有跳空。
- 本脚本**基于聚宽数据自身检测换月日**(隔夜跳空 + 持仓量跳变), 计算 weighted_factor 链,
  因此 cur_weighted_factor / weighted_factor / is_rollover 均直接由聚宽数据确定(与日频库无关)。
- 每条记录带 source='joinquant'。
- 会**覆盖更新**分钟库中该品种的所有分钟数据(按 time+instrument_id upsert)。

CSV 合并
----
默认合并两个文件:
  1. 主 CSV:      /Users/wenglongao/Downloads/C9999.XDCE.csv
  2. 补夜盘 CSV:  /Users/wenglongao/Downloads/C9999.XDCE_fix_night.csv
两者可能重复(后者含前者缺失的部分夜盘), 按时间戳 datetime 合并去重。

用法
----
python -u test/data/import_c0_1min_to_db.py                  # 默认导入
python -u test/data/import_c0_1min_to_db.py --preview        # 仅预览, 不写库
python -u test/data/import_c0_1min_to_db.py --csv 主CSV --csv-fix 补CSV
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复用 data.futures 中基于聚宽分钟数据的交易日归属 / 换月检测 / 构建逻辑
from data.futures import (  # noqa: E402
    assign_trading_day_1min,
    detect_rollover_from_minute_df,
    build_minute_continuous_df_from_edb,
)

# ================== 配置区 ==================
CSV_PATH = os.environ.get("C0_1MIN_CSV", "/Users/wenglongao/Downloads/C9999.XDCE.csv")
CSV_FIX_PATH = os.environ.get("C0_1MIN_CSV_FIX", "/Users/wenglongao/Downloads/C9999.XDCE_fix_night.csv")
INSTRUMENT_ID = "C0"
DATABASE = "futures"
MIN_COLLECTION = "continuous_contract_price_1min"
SOURCE = "joinquant"   # 数据来源

# 分批写入大小
BATCH_SIZE = 5000
# 换月日: 只把「第一根分钟 bar」标记 is_rollover=True(True), 或整日都标记(False)
MARK_ROLLOVER_FIRST_MINUTE_ONLY = True
# 换月检测阈值(与 data.futures.detect_rollover_from_minute_df 一致)
GAP_THRESHOLD = 0.01
OI_CHG_THRESHOLD = 0.15
METHOD = "bulk_write_update"


def read_and_merge_csvs(main_csv: str, fix_csv: str) -> pd.DataFrame:
    """读取主 CSV 与补夜盘 CSV, 按时间戳 datetime 合并去重。"""
    frames = []
    for path in [main_csv, fix_csv]:
        if path and os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["datetime"])
            df = df.dropna(subset=["datetime"])
            frames.append(df)
            print(f"  读取 {path}: {len(df)} 行")
        else:
            print(f"  [跳过] CSV 不存在: {path}")
    if not frames:
        print("[错误] 没有可用的 CSV")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  合并后: {len(df)} 行, 范围 {df['datetime'].min()} ~ {df['datetime'].max()}")
    return df


def write_in_batches(out: pd.DataFrame) -> int:
    """分批写入数据库, 支持 Ctrl+C 安全中断。返回已写入条数。"""
    from mongo.mongify import update_data

    total = len(out)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    written = 0
    print(f"\n开始分批写入 {DATABASE}.{MIN_COLLECTION}: 共 {total} 条, "
          f"每批 {BATCH_SIZE} 条, 共 {n_batches} 批。按 Ctrl+C 可安全中断。")
    try:
        for i, start in enumerate(range(0, total, BATCH_SIZE), 1):
            chunk = out.iloc[start:start + BATCH_SIZE]
            update_data(database=DATABASE,
                        collection=MIN_COLLECTION,
                        df=chunk,
                        method=METHOD,
                        filter_column=["time", "instrument_id"])
            written += len(chunk)
            print(f"  批次 {i}/{n_batches}: 已写入 {written}/{total} 条")
    except KeyboardInterrupt:
        print(f"\n[中断] 已写入约 {written}/{total} 条。")
        print("  已写入的批次是完整数据; 当前批次可能只写了部分(upsert 幂等, 重跑会覆盖补齐)。")
        print("  重新运行本脚本即可继续/补齐, 无需清库。")
        sys.exit(130)
    return written


def main():
    preview = "--preview" in sys.argv
    if "--csv" in sys.argv:
        main_csv = sys.argv[sys.argv.index("--csv") + 1]
    else:
        main_csv = CSV_PATH
    if "--csv-fix" in sys.argv:
        fix_csv = sys.argv[sys.argv.index("--csv-fix") + 1]
    else:
        fix_csv = CSV_FIX_PATH

    print("=" * 70)
    print(f"导入 {INSTRUMENT_ID} 1 分钟数据 -> {DATABASE}.{MIN_COLLECTION}  (source={SOURCE})")
    print("=" * 70)

    # 1) 读取并合并两个 CSV
    df = read_and_merge_csvs(main_csv, fix_csv)

    # 2) 交易日归属
    df["td"] = assign_trading_day_1min(df["datetime"])
    print(f"{df['td'].nunique()} 个交易日")

    # 3) 换月字段: 优先用 CSV 中已有的(聚宽 get_dominant_future 导出); 否则回退到分钟数据自身检测
    if "is_rollover" in df.columns and "weighted_factor" in df.columns:
        print("[信息] 使用 CSV 中已有的换月字段(symbol/is_rollover/weighted_factor)")
        out = df.rename(columns={"datetime": "time", "open_interest": "position"})
        out["instrument_id"] = INSTRUMENT_ID
        out["settle"] = out["close"]
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        # 按交易日填充换月字段: 主 CSV 提供每个交易日的值; fix 补充的 NaN bar 也按交易日取值
        out["td"] = assign_trading_day_1min(out["time"])
        _valid = out.dropna(subset=["weighted_factor"])
        if _valid.empty:
            _map = pd.DataFrame(columns=["td", "symbol", "is_rollover", "weighted_factor", "cur_weighted_factor"])
        else:
            _map = _valid.groupby("td").agg({
                "symbol": "last",
                "is_rollover": "max",
                "weighted_factor": "last",
                "cur_weighted_factor": "last",
            }).reset_index()
        out = out.merge(_map, on="td", how="left", suffixes=("", "_m"))
        for _c in ["symbol", "is_rollover", "weighted_factor", "cur_weighted_factor"]:
            _mc = f"{_c}_m"
            if _mc in out.columns:
                out[_c] = out[_c].fillna(out[_mc])
                out = out.drop(columns=[_mc])
        out = out.drop(columns=["td"])
        out["is_rollover"] = out["is_rollover"].fillna(False).astype(bool)
        out["weighted_factor"] = pd.to_numeric(out["weighted_factor"], errors="coerce").fillna(1.0)
        out["cur_weighted_factor"] = pd.to_numeric(out["cur_weighted_factor"], errors="coerce").fillna(1.0)
        cols = ["time", "instrument_id", "symbol", "open", "high", "low", "close", "settle",
                "volume", "position", "money", "weighted_factor", "cur_weighted_factor", "is_rollover"]
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0 if c in ("volume", "position", "money") else ("" if c == "symbol" else np.nan)
        for c in ["open", "high", "low", "close", "settle", "volume", "position", "money"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out[cols].dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
        out["settle"] = out["settle"].fillna(out["close"])
        out["volume"] = out["volume"].fillna(0.0)
        out["position"] = out["position"].fillna(0.0)
        out["money"] = out["money"].fillna(0.0)
        out["source"] = SOURCE
        out = out.drop_duplicates(subset=["time", "instrument_id"], keep="last").reset_index(drop=True)
    else:
        print("[信息] CSV 无换月字段, 回退到基于分钟数据自身检测换月(gap+持仓量跳变)")
        schedule = detect_rollover_from_minute_df(
            df,
            gap_threshold=GAP_THRESHOLD,
            oi_chg_threshold=OI_CHG_THRESHOLD,
        )
        schedule["symbol"] = f"KQ.m@{INSTRUMENT_ID[0]}"  # 聚宽主连, symbol 用主连标识
        roll_days = schedule.loc[schedule["is_rollover"], "td"].tolist()
        print(f"检测到换月日: {len(roll_days)} 个")
        if roll_days:
            print(f"  换月日示例: {[pd.Timestamp(t).date().isoformat() for t in roll_days[:10]]} ...")
        out = build_minute_continuous_df_from_edb(
            df, schedule, INSTRUMENT_ID,
            symbol=f"KQ.m@{INSTRUMENT_ID[0]}",
            mark_first_only=MARK_ROLLOVER_FIRST_MINUTE_ONLY,
        )
        out["source"] = SOURCE
    print(f"\n构建完成: {len(out)} 行 x {len(out.columns)} 列")
    print(f"  时间范围: {out['time'].min()} ~ {out['time'].max()}")
    print(f"  换月点分钟数: {int(out['is_rollover'].sum())}")

    print("\n--- 预览 ---")
    print(out.head(5).to_string())
    print("...")
    print(out.tail(3).to_string())

    if preview:
        print("\n[预览模式] 未写入数据库。去掉 --preview 即可正式导入。")
        return

    # 5) 分批写入(覆盖更新该品种全部分钟数据)
    write_in_batches(out)
    print(f"\n完成: 已写入 {len(out)} 条记录(source={SOURCE})到 {DATABASE}.{MIN_COLLECTION}。")


if __name__ == "__main__":
    main()
