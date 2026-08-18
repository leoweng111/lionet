"""
数据源测试: 掘金量化 MyQuant（免费注册，分钟数据最早 2017-01-01）

⚠️ 平台限制（重要）:
    gm SDK 在 PyPI 上只有 win32 / win_amd64 / manylinux1_x86_64 三种轮子,
    **没有 macOS 轮子, 也没有 ARM 轮子**。掘金量化终端官方也仅支持 Windows 系统。
    因此在 macOS 上无法 pip 安装 gm —— 这是官方限制, 不是命令问题。

    仅在 **Windows x86_64** 或 **Linux x86_64** 环境可安装:
        pip install gm -i https://pypi.org/simple
        # 注意: 清华/阿里/百度等国内镜像未同步 gm 包, 必须用官方 PyPI 源。
        # 手动下载安装: pip install https://files.pythonhosted.org/packages/29/c8/0c31f148c74766fd4513ce968cebfe22f8a845884e0dfd906a784e494d42/gm-3.0.112-py3-none-manylinux1_x86_64.whl

掘金量化提供 Python SDK (gm), 通过本地运行的「掘金终端」取数:
    - 分钟行情最早支持 2017-01-01 (60s/300s/900s/1800s/3600s)
    - tick 最早支持 2022-08-10
    - 支持 8 大交易所(含广期所), 免费版实时订阅上限 50 个标的(回测无限制)

用法(在 Windows / Linux x86_64 环境):
    pip install gm -i https://pypi.org/simple
    # 1. 先到官网注册并下载掘金终端: https://www.myquant.cn
    # 2. 打开掘金终端并登录 (取数需本地终端在线)
    # 3. 在终端「帮助->账户信息」复制 token
    GM_TOKEN=你的token python -u test/data/test_myquant.py

说明:
    - 若未安装 gm SDK 或终端未启动/未登录, 脚本会给出明确提示, 不会崩溃。
    - macOS 用户建议改用 test_tqsdk_edb.py / test_akshare.py / test_tushare.py
      (这些源跨平台可用, 分钟数据同样覆盖)。
    - 默认取螺纹钢主力连续合约 SHFE.RB 的 60 秒 bar 做演示。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SYMBOL = "SHFE.RB"  # 螺纹钢主力/连续合约 (掘金用大写品种代码)
DEFAULT_FREQ = "60s"
DEFAULT_START = "2026-08-10 09:00:00"
DEFAULT_END = "2026-08-14 15:00:00"


def main():
    print("=" * 70)
    print("数据源: 掘金量化 MyQuant (免费, 分钟历史自 2017-01-01)")
    print("=" * 70)

    try:
        from gm.api import set_token, history
    except ImportError:
        print("\n[错误] 未安装 gm SDK, 请先执行:  pip install gm")
        print("  掘金量化官网: https://www.myquant.cn")
        sys.exit(1)

    token = os.environ.get("GM_TOKEN", "")
    if not token:
        print("\n[提示] 未设置 GM_TOKEN 环境变量。")
        print("  取数步骤:")
        print("    1) 打开掘金终端并登录 (https://www.myquant.cn 下载)")
        print("    2) 在终端「帮助 -> 账户信息」复制 token")
        print("    3) 运行:  GM_TOKEN=你的token python -u test/data/test_myquant.py")
        print("  若终端尚未启动/登录, 取数会失败, 这是正常现象。")
        sys.exit(1)

    try:
        set_token(token)
    except Exception as e:
        print(f"\n[失败] set_token 失败: {e}")
        sys.exit(1)

    print(f"\n>>> history(symbol='{DEFAULT_SYMBOL}', frequency='{DEFAULT_FREQ}', 起止={DEFAULT_START} ~ {DEFAULT_END})")
    try:
        df = history(symbol=DEFAULT_SYMBOL, frequency=DEFAULT_FREQ,
                     start_time=DEFAULT_START, end_time=DEFAULT_END,
                     fields="symbol,frequency,open,high,low,close,volume,open_oi,close_oi",
                     df=True)
        if df is None or df.empty:
            print("    返回为空。请检查终端是否在线 / 合约代码 / 时间范围。")
            return
        print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        print(f"    列名: {list(df.columns)}")
        print("\n    前 3 行:")
        print(df.head(3).to_string())
        print("\n    后 2 行:")
        print(df.tail(2).to_string())
        print("\n    字段说明: volume=成交量(手), open_oi/close_oi=开盘/收盘持仓量(手)")
    except Exception as e:
        msg = str(e)
        print(f"    [失败] {type(e).__name__}: {msg}")
        if "connect" in msg.lower() or "60010" in msg or "终端" in msg:
            print("\n[提示] 掘金终端未连接。请先在本地启动并登录掘金终端, 再运行本脚本。")
        else:
            print("\n[提示] 若为合约代码问题, 请确认该品种当前是否有连续合约(如 SHFE.RB)。")

    print("\n完成。")


if __name__ == "__main__":
    main()
