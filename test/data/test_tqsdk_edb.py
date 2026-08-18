"""
数据源测试: 天勤量化 EDB 行情历史服务（信易科技，注册免费，分钟线无需 token）

RESTful HTTP 接口, 返回 CSV 流式数据:
    GET https://edb.shinnytech.com/md/kline
        period=60      # 60 秒 = 1 分钟线
        period=86400   # 日线(按交易日)
        symbol         # 合约标识, 如 SHFE.rb2401 / KQ.m@SHFE.rb(主连) / KQ.i@SHFE.rb(指数)
        start_time / end_time  # YYYY-MM-DD HH:MM:SS
        col            # 可选列: open,high,low,close,volume,open_oi,close_oi

免费访问: 可获取**最近 1 年**的分钟线 + 任意历史区间的日线。
专业版(付费): 全部历史分钟线与日线 (https://www.shinnytech.com 用户中心购买)

用法:
    python -u test/data/test_tqsdk_edb.py
    python -u test/data/test_tqsdk_edb.py "KQ.m@DCE.m"    # 指定其它品种主连
    python -u test/data/test_tqsdk_edb.py "SHFE.rb2601"   # 指定具体合约
"""

import sys
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "https://edb.shinnytech.com/md/kline"

# 默认测试品种: 螺纹钢主连 (KQ.m@交易所.品种)
DEFAULT_SYMBOL = "KQ.m@SHFE.rb"


def fetch_kline(symbol: str, period: int, start_time: str, end_time: str,
                cols: str = None, token: str = None) -> pd.DataFrame:
    """调用 EDB /kline 接口, 返回 pandas DataFrame"""
    params = {
        "period": period,
        "symbol": symbol,
        "start_time": start_time,
        "end_time": end_time,
    }
    if cols:
        params["col"] = cols
    if token:
        params["token"] = token

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", "ignore")

    if not raw.strip():
        raise RuntimeError(f"接口返回为空, 请检查 symbol/时间范围: {symbol}")

    df = pd.read_csv(StringIO(raw))
    # 纳秒时间戳 -> 可读时间
    if "datetime_nano" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_nano"], unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return df


def show(df: pd.DataFrame, title: str):
    print(f"\n>>> {title}")
    print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"    列名: {list(df.columns)}")
    print("\n    前 3 行:")
    print(df.head(3).to_string())
    print("\n    后 2 行:")
    print(df.tail(2).to_string())


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYMBOL

    print("=" * 70)
    print("数据源: 天勤 EDB 行情历史服务 (免费, 无需 token)")
    print(f"品种: {symbol}")
    print("=" * 70)

    # 免费可访问: 近 1 年分钟线; 这里取最近 3 个交易日做演示
    minute_start, minute_end = "2026-08-11 21:00:00", "2026-08-17 15:00:00"

    # 日线: 任意历史区间均可访问
    daily_start, daily_end = "2025-01-01 00:00:00", "2026-08-17 00:00:00"

    try:
        # 1) 分钟线
        df_min = fetch_kline(symbol, period=60, start_time=minute_start, end_time=minute_end)
        show(df_min, f"1 分钟线 (period=60)  [{minute_start} ~ {minute_end}]")
        print("\n    字段说明: open/high/low/close=开高低收, volume=成交量(手), open_oi/close_oi=开盘/收盘持仓量(手)")

        # 2) 日线
        df_day = fetch_kline(symbol, period=86400, start_time=daily_start, end_time=daily_end)
        show(df_day, f"日线 (period=86400, 任意历史)  [{daily_start} ~ {daily_end}]")

        # 3) 指定列
        df_cols = fetch_kline(symbol, period=60, start_time=minute_start, end_time=minute_end,
                              cols="open,close,volume")
        show(df_cols, "仅指定列 (col=open,close,volume)")

    except Exception as e:
        print(f"\n[失败] {type(e).__name__}: {e}")
        print("可能原因: 网络无法访问 edb.shinnytech.com / 品种代码错误 / 免费分钟额度超出近 1 年")
        sys.exit(1)

    print("\n完成。若需早于近 1 年的分钟数据, 请购买专业版 (https://www.shinnytech.com)。")


if __name__ == "__main__":
    main()
