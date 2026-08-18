"""
一键导入聚宽导出的玉米(C0) 1 分钟连续合约数据到 MongoDB: futures.continuous_contract_price_1min

背景
----
聚宽 9999 主力连续合约 (如 C9999.XDCE) 是「未做后复权、未做换月拼接」的原始价格序列:
主力合约切换时, 序列会直接切换到新合约的原始价格, 因此存在换月跳空。
本脚本复用日频库 futures.continuous_contract_price_daily 中已经算好的换月安排
(symbol / weighted_factor / cur_weighted_factor / is_rollover), 按「交易日」把每天的
分钟 bar 归到对应的日频行, 从而让分钟数据与日频数据在换月调整上完全一致。

列说明(与日频一致, 每分钟一行)
----
instrument_id : 固定 "C0"
time          : 分钟 bar 时间戳(精确到分钟, 聚宽时间戳为时间段结束)
open/high/low/close : 开高低收
settle        : 分钟数据无结算价, 取 close(与日频缺省处理一致)
volume        : 成交量(手)
position      : 持仓量(手, 取自 open_interest)
money         : 成交额(元) —— 日频无此列, 分钟数据保留
symbol        : 该交易日对应的主力合约(来自日频库)
weighted_factor / cur_weighted_factor : 后复权因子(来自日频库, 与日频一致)
is_rollover   : 是否换月点。默认仅标记换月日的「第一根分钟 bar」为 True

用法
----
python -u test/data/import_c0_1min_to_db.py                  # 默认导入
python -u test/data/import_c0_1min_to_db.py --preview        # 仅预览, 不写库
python -u test/data/import_c0_1min_to_db.py --csv 路径       # 指定 CSV

注意
----
- 需在本机(能连到 127.0.0.1:27017 MongoDB)运行。
- 日频库没有覆盖的交易日: 会沿用上一个已知 factor/symbol 并打印警告;
  若整个日频换月安排都为空, 则回退为「基于分钟数据隔夜跳空」检测换月。
- 可重复运行(bulk_write_update 按 time+instrument_id upsert, 会覆盖旧数据)。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ================== 配置区 ==================
CSV_PATH = os.environ.get("C0_1MIN_CSV", "/Users/wenglongao/Downloads/C9999.XDCE.csv")
INSTRUMENT_ID = "C0"
DATABASE = "futures"
DAILY_COLLECTION = "continuous_contract_price_daily"
MIN_COLLECTION = "continuous_contract_price_1min"

# 是否优先复用日频库的换月安排(True); False 或日频库为空时用分钟 gap 检测回退
USE_DB_ROLLOVER_SCHEDULE = True
# 换月日: 只把「第一根分钟 bar」标记 is_rollover=True(True), 或整日都标记(False)
MARK_ROLLOVER_FIRST_MINUTE_ONLY = True
# 隔夜跳空告警阈值(1%): 分钟数据跳空超过该值但日频库未标换月 → 打印告警
GAP_WARNING_THRESHOLD = 0.01
METHOD = "bulk_write_update"


# ================== 核心逻辑(可独立测试) ==================
def assign_trading_day(datetime_series: pd.Series, trading_days=None) -> pd.Series:
    """给分钟 bar 打「交易日」标签。

    中国商品期货夜盘(21:00-23:00 等, 小时 >= 20)属于「下一个交易日」;
    日盘(09:00-15:00)属于当天。

    特别注意: 周五夜盘属于「下周一」的交易日(而非周六)。
    因此夜盘归属需要基于交易日列表, 而不是简单地 +1 天。

    Parameters
    ----------
    datetime_series : pd.Series
        分钟 bar 的时间戳。
    trading_days : list-like, optional
        交易日列表(排好序)。默认从 datetime_series 中「日盘 bar(小时<20)」的日期推断。
    """
    if trading_days is None:
        day_mask = datetime_series.dt.hour < 20
        tds = np.array(sorted(pd.unique(datetime_series[day_mask].dt.normalize())),
                       dtype="datetime64[D]")
    else:
        tds = np.array(pd.to_datetime(list(trading_days)).normalize(), dtype="datetime64[D]")
        tds = np.unique(tds)

    night = datetime_series.dt.hour >= 20
    cal = datetime_series.dt.normalize().values.astype("datetime64[D]")
    out = pd.Series(pd.NaT, index=datetime_series.index, dtype="datetime64[ns]")
    if tds.size == 0:
        return out

    # 日盘 -> 当天(正常当天就是交易日)
    day_pos = np.searchsorted(tds, cal[~night], side="left")
    day_pos = np.minimum(day_pos, tds.size - 1)
    out.loc[~night] = pd.to_datetime(tds[day_pos])
    # 夜盘 -> 下一个交易日(严格大于夜盘日历日)
    night_pos = np.searchsorted(tds, cal[night], side="right")
    night_pos = np.minimum(night_pos, tds.size - 1)
    out.loc[night] = pd.to_datetime(tds[night_pos])
    return out


def load_daily_schedule(instrument_id: str, database: str, collection: str) -> pd.DataFrame:
    """从日频库读取换月安排, 返回列: td, symbol, weighted_factor, cur_weighted_factor, is_rollover"""
    from mongo.mongify import get_data

    df_d = get_data(database, collection, {"instrument_id": instrument_id})
    if df_d is None or df_d.empty:
        return pd.DataFrame()
    df_d = df_d.copy()
    df_d["time"] = pd.to_datetime(df_d["time"], errors="coerce")
    df_d = df_d.dropna(subset=["time"])
    if df_d.empty:
        return pd.DataFrame()
    for c in ["weighted_factor", "cur_weighted_factor"]:
        if c not in df_d.columns:
            df_d[c] = 1.0
    if "is_rollover" not in df_d.columns:
        df_d["is_rollover"] = False
    if "symbol" not in df_d.columns:
        df_d["symbol"] = ""
    df_d["weighted_factor"] = pd.to_numeric(df_d["weighted_factor"], errors="coerce").fillna(1.0)
    df_d["cur_weighted_factor"] = pd.to_numeric(df_d["cur_weighted_factor"], errors="coerce").fillna(1.0)
    df_d["is_rollover"] = df_d["is_rollover"].astype(bool)
    # 交易日取日期部分; 若为带时区(UTC)则先去掉时区再 normalize
    if getattr(df_d["time"].dt, "tz", None) is not None:
        df_d["td"] = df_d["time"].dt.tz_localize(None).dt.normalize()
    else:
        df_d["td"] = df_d["time"].dt.normalize()
    df_d = df_d.drop_duplicates(subset=["td"], keep="last")
    return df_d[["td", "symbol", "weighted_factor", "cur_weighted_factor", "is_rollover"]]


def detect_rollover_schedule_from_gaps(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """回退方案: 用分钟数据隔夜跳空检测换月, 并计算 weighted_factor 链。

    注意: 连续 9999 序列只有「新合约开盘」, 没有「旧合约当日开盘」,
    因此换月比例近似用 close_prev / open_new, 与日频库的 old_open/new_open 略有差异。
    仅当日频库无数据时使用。
    """
    first = df.groupby("td").first()
    last = df.groupby("td").last()
    daily = pd.DataFrame({"open": first["open"], "close": last["close"]}).sort_index()
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_ret"] = daily["open"] / daily["prev_close"] - 1.0
    noise = daily["gap_ret"].abs().rolling(60, min_periods=20).median()
    daily["is_rollover"] = (daily["gap_ret"].abs() > np.maximum(noise * 8.0, threshold))

    wf = 1.0
    wf_list = []
    for td, row in daily.iterrows():
        if bool(row["is_rollover"]) and pd.notna(row["prev_close"]) and abs(row["open"]) > 1e-12:
            wf *= float(row["prev_close"] / row["open"])
        wf_list.append(wf)
    daily["weighted_factor"] = wf_list
    daily["cur_weighted_factor"] = 1.0
    daily["symbol"] = "UNKNOWN"
    return daily.reset_index()[["td", "symbol", "weighted_factor", "cur_weighted_factor", "is_rollover"]]


def build_minute_continuous_df(df: pd.DataFrame,
                               schedule: pd.DataFrame,
                               instrument_id: str,
                               mark_first_only: bool = True) -> pd.DataFrame:
    """将分钟 bar 与日频换月安排合并, 生成待入库 DataFrame。

    df 需含: datetime, open, high, low, close, volume, open_interest, td
    """
    if "td" not in df.columns:
        df = df.copy()
        df["td"] = assign_trading_day(df["datetime"])

    sched = schedule.copy()
    if "td" not in sched.columns:
        sched["td"] = pd.to_datetime(sched.index)
    sched = sched[["td", "symbol", "weighted_factor", "cur_weighted_factor", "is_rollover"]]

    out = df.merge(sched, on="td", how="left")

    # 日频库缺失的交易日: 沿用上一个已知 symbol/factor
    out["symbol"] = out["symbol"].ffill()
    out["weighted_factor"] = pd.to_numeric(out["weighted_factor"], errors="coerce").ffill().fillna(1.0)
    out["cur_weighted_factor"] = pd.to_numeric(out["cur_weighted_factor"], errors="coerce").ffill().fillna(1.0)
    daily_is_rollover = out["is_rollover"].fillna(False).astype(bool)

    first_of_day = ~out.duplicated(subset="td", keep="first")
    out["is_rollover"] = daily_is_rollover & first_of_day if mark_first_only else daily_is_rollover

    out["instrument_id"] = instrument_id
    out["settle"] = out["close"]
    out = out.rename(columns={"datetime": "time", "open_interest": "position"})

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    price_cols = ["open", "high", "low", "close", "settle", "volume", "position", "money",
                  "weighted_factor", "cur_weighted_factor"]
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "money" not in out.columns:
        out["money"] = np.nan

    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    for c, fill_val in [("settle", None), ("volume", 0.0), ("position", 0.0),
                        ("money", 0.0), ("weighted_factor", 1.0), ("cur_weighted_factor", 1.0)]:
        if c == "settle":
            out["settle"] = out["settle"].fillna(out["close"])
        else:
            out[c] = out[c].fillna(fill_val)

    cols = ["time", "instrument_id", "symbol", "open", "high", "low", "close", "settle",
            "volume", "position", "money", "weighted_factor", "cur_weighted_factor", "is_rollover"]
    out = out[cols].sort_values("time").reset_index(drop=True)
    # 安全去重(同一分钟只保留一条), 避免 bulk_write 因重复键报错
    out = out.drop_duplicates(subset=["time", "instrument_id"], keep="last").reset_index(drop=True)
    return out


def cross_check_gaps(df: pd.DataFrame, schedule: pd.DataFrame, threshold: float) -> None:
    """交叉校验: 分钟数据存在大跳空但日频库未标换月的交易日 → 打印告警。"""
    first = df.groupby("td").first()
    last = df.groupby("td").last()
    daily = pd.DataFrame({"open": first["open"], "close": last["close"]}).sort_index()
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_ret"] = daily["open"] / daily["prev_close"] - 1.0

    roll_days = set()
    if schedule is not None and not schedule.empty and "is_rollover" in schedule.columns:
        roll_days = set(schedule.loc[schedule["is_rollover"], "td"])

    big = daily[daily["gap_ret"].abs() > threshold]
    if big.empty:
        return
    for td, r in big.iterrows():
        if td not in roll_days:
            print(f"  [警告] {pd.Timestamp(td).date()} 隔夜跳空 {r['gap_ret']:.4f}, "
                  f"但不在日频换月日列表(可能聚宽换月与本地日频不一致, 请核对)")
    n_miss = len([td for td in roll_days if td not in daily.index])
    if n_miss:
        print(f"  [提示] 日频换月日中有 {n_miss} 天不在分钟数据中(可能分钟数据缺失或已到期)")


def main():
    preview = "--preview" in sys.argv
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]
    else:
        csv_path = CSV_PATH

    print("=" * 70)
    print(f"导入 {INSTRUMENT_ID} 1 分钟数据 -> {DATABASE}.{MIN_COLLECTION}")
    print(f"CSV: {csv_path}")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"[错误] CSV 不存在: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["td"] = assign_trading_day(df["datetime"])
    print(f"读取 {len(df)} 行, 时间范围 {df['datetime'].min()} ~ {df['datetime'].max()}, "
          f"{df['td'].nunique()} 个交易日")

    # 1) 换月安排
    schedule = pd.DataFrame()
    if USE_DB_ROLLOVER_SCHEDULE:
        try:
            schedule = load_daily_schedule(INSTRUMENT_ID, DATABASE, DAILY_COLLECTION)
            if schedule.empty:
                print("[提示] 日频库 futures.continuous_contract_price_daily 为空, 回退到 gap 检测")
        except Exception as e:
            print(f"[警告] 读取日频换月安排失败({e}), 回退到 gap 检测")
            schedule = pd.DataFrame()
    if schedule.empty:
        schedule = detect_rollover_schedule_from_gaps(df, threshold=GAP_WARNING_THRESHOLD)
        print(f"[回退] 基于隔夜跳空检测到 {int(schedule['is_rollover'].sum())} 个换月日")

    roll_days = schedule.loc[schedule["is_rollover"], "td"].tolist()
    print(f"换月安排: 共 {len(schedule)} 个交易日, 其中换月日 {len(roll_days)} 个")
    if roll_days:
        print(f"  换月日示例: {[pd.Timestamp(t).date().isoformat() for t in roll_days[:10]]} ...")

    # 2) 构建分钟 DataFrame
    out = build_minute_continuous_df(df, schedule, INSTRUMENT_ID,
                                     mark_first_only=MARK_ROLLOVER_FIRST_MINUTE_ONLY)
    print(f"\n构建完成: {len(out)} 行 x {len(out.columns)} 列")
    print(f"  时间范围: {out['time'].min()} ~ {out['time'].max()}")

    # 3) 交叉校验(仅当使用日频换月安排时)
    if USE_DB_ROLLOVER_SCHEDULE and not schedule.empty:
        cross_check_gaps(df, schedule, GAP_WARNING_THRESHOLD)

    # 4) 摘要
    print("\n--- 预览 ---")
    print(out.head(5).to_string())
    print("...")
    print(out.tail(3).to_string())
    print(f"\n换月点分钟数: {int(out['is_rollover'].sum())}")

    if preview:
        print("\n[预览模式] 未写入数据库。去掉 --preview 即可正式导入。")
        return

    # 5) 写入数据库
    from mongo.mongify import update_data
    update_data(database=DATABASE,
                collection=MIN_COLLECTION,
                df=out,
                method=METHOD,
                filter_column=["time", "instrument_id"])
    print(f"\n完成: 已写入 {len(out)} 条记录到 {DATABASE}.{MIN_COLLECTION}。")


if __name__ == "__main__":
    main()
