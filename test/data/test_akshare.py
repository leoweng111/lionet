"""
数据源测试: AkShare（免费、开源，无需注册）

AkShare 封装了新浪/东方财富/腾讯等网页接口，本脚本重点演示**新浪期货分钟数据**:
    futures_zh_minute_sina(symbol, period)   # 分钟 K 线, period: 1/5/15/30/60

用法:
    pip install akshare
    python -u test/data/test_akshare.py

说明:
    - 免费、免注册, pip 安装即用
    - 新浪接口通常只返回最近几百根分钟线, 非全历史
    - 底层为网页接口, 可能限流/失效, 仅适合快速取数测试
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 新浪期货主力合约代码示例: RB0=螺纹钢主力, AU0=沪金主力, CU0=沪铜主力
DEFAULT_SYMBOLS = ["RB0", "CU0", "AU0"]
DEFAULT_PERIOD = "1"  # 1/5/15/30/60 分钟


def test_minute_sina(symbol: str, period: str = DEFAULT_PERIOD):
    """测试新浪期货分钟数据"""
    import akshare as ak

    df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    if df is None or df.empty:
        print(f"    [{symbol}] 返回为空")
        return None
    return df


def test_main_sina():
    """附带演示: 新浪主力/连续合约全历史日线(可用于生成品种清单)"""
    import akshare as ak

    df = ak.futures_main_sina()
    return df


def main():
    print("=" * 70)
    print("数据源: AkShare (免费开源, 新浪期货分钟数据)")
    print("=" * 70)

    try:
        import akshare  # noqa: F401
    except ImportError:
        print("\n[错误] 未安装 akshare, 请先执行:  pip install akshare")
        sys.exit(1)

    # ---- 1) 新浪期货分钟数据 ----
    got_any = False
    for symbol in DEFAULT_SYMBOLS:
        print(f"\n>>> 新浪期货分钟数据 futures_zh_minute_sina(symbol='{symbol}', period='{DEFAULT_PERIOD}')")
        try:
            df = test_minute_sina(symbol)
            if df is None:
                continue
            got_any = True
            print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
            print(f"    列名: {list(df.columns)}")
            print(f"    起始时间: {df.iloc[0, 0]}  ->  结束时间: {df.iloc[-1, 0]}")
            print("\n    前 3 行:")
            print(df.head(3).to_string())
            print("\n    后 2 行:")
            print(df.tail(2).to_string())
            # 数据字段含义
            print("\n    字段说明: datetime=时间, open/high/low/close=开高低收, volume=成交量(手), hold=持仓量(手)")
        except Exception as e:
            print(f"    [失败] {type(e).__name__}: {e}")

    if not got_any:
        print("\n[提示] 上述新浪分钟接口均失败。可能是新浪接口变更或网络受限。")

    # ---- 2) 附带: 新浪主力连续合约(日线, 品种清单) ----
    print("\n>>> (附带) 新浪主力/连续合约日线 futures_main_sina()  —— 用于获取品种/主连清单")
    try:
        df = test_main_sina()
        if df is not None and not df.empty:
            print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
            print(f"    列名: {list(df.columns)}")
            print(df.head(3).to_string())
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")

    print("\n完成。若需更稳定的分钟历史数据, 可参考 test_tqsdk_edb.py / test_tushare.py。")


if __name__ == "__main__":
    main()
