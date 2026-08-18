"""
数据源测试: 聚宽 JQData（有免费体验额度，正式使用付费）

JQData 提供期货分钟数据, 支持周期 1m/5m/15m/30m/60m/120m:
    get_price(security, frequency='1m', ...)
    期货合约代码后缀: .XSGE=上期所, .XDCE=大商所, .XZCE=郑商所, .XINE=能源, .XGFEX=广期所
    主力连续示例: 'RB9999.XSGE' (螺纹钢主连)

用法:
    pip install jqdatasdk
    JQ_USER=你的手机号 JQ_PASS=你的密码 python -u test/data/test_jqdata.py
    # 可用环境变量 JQ_FUTURE_CODE 指定其它合约, 默认取螺纹钢主连 RB9999.XSGE

说明:
    - 需要聚宽账号 (https://www.joinquant.com 注册)
    - 首次需在聚宽研究环境/官网绑定手机号作为账号
    - 免费额度有限, 超额需购买 JQData 数据服务
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONTRACT = os.environ.get("JQ_FUTURE_CODE", "RB9999.XSGE")
DEFAULT_FREQ = "1m"
DEFAULT_COUNT = 10  # 最近 10 根 1 分钟 bar


def main():
    print("=" * 70)
    print("数据源: 聚宽 JQData (付费, 有免费体验)")
    print(f"合约: {DEFAULT_CONTRACT}, 频率: {DEFAULT_FREQ}, 取最近 {DEFAULT_COUNT} 根")
    print("=" * 70)

    user = os.environ.get("JQ_USER", "")
    password = os.environ.get("JQ_PASS", "")
    if not user or not password:
        print("\n[错误] 未设置聚宽账号密码。请先执行:")
        print("    JQ_USER=你的手机号 JQ_PASS=你的密码 python -u test/data/test_jqdata.py")
        print("  注册地址: https://www.joinquant.com")
        sys.exit(1)

    try:
        from jqdatasdk import auth, get_price, get_future_contracts
    except ImportError:
        print("\n[错误] 未安装 jqdatasdk, 请先执行:  pip install jqdatasdk")
        sys.exit(1)

    print("\n>>> 登录认证 auth()")
    try:
        auth(user, password)
    except Exception as e:
        print(f"    [失败] 认证失败: {e}")
        print("    请确认账号/密码正确, 并已在聚宽官网绑定该手机号。")
        sys.exit(1)
    print("    认证成功")

    # ---- 1) 拉取分钟 bar ----
    print(f"\n>>> get_price('{DEFAULT_CONTRACT}', frequency='{DEFAULT_FREQ}')")
    try:
        df = get_price(DEFAULT_CONTRACT, frequency=DEFAULT_FREQ,
                       count=DEFAULT_COUNT, end_date="2026-08-17", fields=None)
        if df is None or df.empty:
            print("    返回为空。请检查合约代码是否有效 (可用下方合约列表函数确认)。")
            return
        print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        print(f"    列名: {list(df.columns)}")
        print("\n    数据(最近 %d 根):" % DEFAULT_COUNT)
        print(df.to_string())
        print("\n    字段说明: open/high/low/close=开高低收, volume=成交量(手), money=成交额, open_interest=持仓量")
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        print("    常见原因: 合约代码无效 / 该合约无 1m 数据权限 / 免费额度用尽。")

    # ---- 2) 附带: 查询某品种当前可交易合约, 帮助确认合约代码 ----
    print("\n>>> (附带) get_future_contracts('RB')  —— 查看螺纹钢当前存续合约代码")
    try:
        contracts = get_future_contracts("RB")
        print(f"    共 {len(contracts)} 个合约: {contracts[:8]}{'...' if len(contracts) > 8 else ''}")
    except Exception as e:
        print(f"    [失败] {e}")

    print("\n完成。")


if __name__ == "__main__":
    main()
