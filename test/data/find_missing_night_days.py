"""
找出聚宽 CSV 中「缺夜盘(bar 数不足)」的交易日, 并给出每个交易日完整的拉取时间范围。

背景
----
玉米每个交易日正常分钟 bar 数约 345 = 夜盘(21:00-23:00, 120根) + 日盘(约225根)。
若某交易日夜盘缺失, 只剩日盘约 225 根。本脚本把这些「缺夜盘交易日」找出来,
并给出在聚宽里重新拉取该交易日完整数据的起止时间(前一交易日 21:00 ~ 当日 15:00)。

用法
----
python -u test/data/find_missing_night_days.py
    - 打印缺夜盘交易日明细
    - 保存到 项目根目录/missing_night_days.csv
    - 同时输出可直接粘贴到 joinquant_fix_missing_night.py 的 MISSING_RANGES 列表
"""

import datetime
import sys
from pathlib import Path

import chinese_calendar as cc
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.futures import assign_trading_day_1min  # noqa: E402 复用同一归属规则

CSV_PATH = "/Users/wenglongao/Downloads/C9999.XDCE.csv"
BAR_THRESHOLD = 340   # 正常 345; 低于此视为缺夜盘/数据不完整
OUT_CSV = PROJECT_ROOT / "missing_night_days.csv"


def has_stat_holiday_between(d1, d2) -> bool:
    """判断 (d1, d2) 之间是否有法定节假日(非周末)。若 true, 说明 d1 是节前最后交易日, 当晚无夜盘。"""
    cur = d1 + datetime.timedelta(days=1)
    while cur < d2:
        if cur.weekday() < 5 and cc.is_holiday(cur.date()):
            return True
        cur += datetime.timedelta(days=1)
    return False


def main():
    import sys as _sys
    csv_path = CSV_PATH
    if "--csv" in _sys.argv:
        csv_path = _sys.argv[_sys.argv.index("--csv") + 1]
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["td"] = assign_trading_day_1min(df["datetime"])

    cnt = df.groupby("td").agg(bar_count=("open", "count")).sort_index()
    tds = list(cnt.index)
    missing = cnt[cnt["bar_count"] < BAR_THRESHOLD].copy()

    print(f"全量交易日: {len(cnt)} 个 | 缺夜盘(bar<{BAR_THRESHOLD}): {len(missing)} 个\n")

    # 第一个交易日(无前一交易日, 如 CSV 首日 2020-01-02)通常"没有前一夜盘"是正常的,
    # 不属于需要补的缺夜盘交易日, 单独提示。
    first_td = tds[0] if tds else None

    holiday_cnt = 0
    rows = []
    for i, t in enumerate(tds):
        if t not in missing.index:
            continue
        prev_td = tds[i - 1] if i > 0 else None
        if prev_td is None:
            continue  # 第一个交易日, 跳过(其前一夜盘不在 CSV 范围内)
        # 前一交易日是节前最后交易日(中间有法定节假日)时, 当晚无夜盘属正常, 无需补
        if has_stat_holiday_between(prev_td, t):
            holiday_cnt += 1
            continue
        # 该交易日完整时段 = 前一交易日 21:00(夜盘开始) ~ 当日 15:00(日盘收盘)
        fetch_start = prev_td.normalize().replace(hour=20, minute=59)
        fetch_end = t.normalize().replace(hour=15, minute=0)
        rows.append({
            "td": t.date().isoformat(),            # 缺夜盘的交易日
            "bar_count": int(missing.loc[t, "bar_count"]),
            "prev_td": prev_td.date().isoformat(), # 缺夜盘的日历日(前一交易日)
            "fetch_start": str(fetch_start),       # 聚宽拉取起点(前一交易日 20:59)
            "fetch_end": str(fetch_end),           # 聚宽拉取终点(当日 15:00)
        })
    if holiday_cnt:
        print(f"[提示] 另有 {holiday_cnt} 个交易日 bar 数偏少, 但其前一交易日是节前最后交易日"
              f"(法定节假日前夜盘暂停), 属正常现象, 无需补。\n")
    if first_td is not None and first_td in missing.index:
        print(f"[提示] CSV 首日 {first_td.date()} bar 数也偏少({int(missing.loc[first_td, 'bar_count'])}), "
              f"但它是导出起点(前一夜盘属于更早日期, 不在 CSV 内), 未计入补数列表; "
              f"如需补它, 请在聚宽单独拉取更早日期的夜盘。\n")
    out = pd.DataFrame(rows)
    if out.empty:
        print("没有需要补的缺夜盘交易日 ✅")
        return
    print(out.to_string(index=False))
    out.to_csv(OUT_CSV, index=False)
    print(f"\n已保存到 {OUT_CSV}")

    # 输出可直接粘贴到聚宽脚本的 MISSING_RANGES
    print("\n--- 粘贴到 joinquant_fix_missing_night.py 的 MISSING_RANGES ---")
    print("MISSING_RANGES = [")
    for _, r in out.iterrows():
        print(f"    ('{r['td']}', '{r['fetch_start']}', '{r['fetch_end']}'),")
    print("]")


if __name__ == "__main__":
    main()
