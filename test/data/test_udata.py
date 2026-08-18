"""
数据源测试: 恒有数 UData（恒生电子，免费注册，跨平台，Python 接口）

这是目前**macOS 上获取多年期货分钟数据的最佳免费方案**:
    - 免费注册 + 免费体验套餐, 不限次、不限量调用
    - 官方 Python SDK `hs_udata` 为纯 Python 包(py3-none-any), macOS/Linux/Windows 通用
    - 提供期货 1 分钟切片 `fut_quote_minute`, 含 OHLC + 成交量 + 成交额 + 持仓量
    - 历史数据(官方宣传含 30 年), 盘后更新(交易日 16:30 后), 适合因子挖掘与回测

用法:
    pip install hs_udata
    # 1. 注册恒有数并领取免费体验套餐: https://udata.hs.net
    # 2. 登录后进入「总览」页面获取个人 Token
    UDATA_TOKEN=你的token python -u test/data/test_udata.py
    # 可选: UDATA_FUTURE_CODE=RB2610.SHF 指定其它期货合约

说明:
    - 无 Token 时脚本会给出指引, 不会崩溃
    - 需要真实有效的期货合约代码, 可用脚本内的 fut_list() 查看全部在册合约
    - 若访问 https://udata.hs.net 被墙/超时, 请确认网络(国内访问正常)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CODE = os.environ.get("UDATA_FUTURE_CODE", "RB2610.SHF")
DEFAULT_BEGIN = "2025-08-01"   # 近 1 年演示; 免费套餐可按需拉取多年
DEFAULT_END = "2026-08-17"


def main():
    print("=" * 70)
    print("数据源: 恒有数 UData (免费, 跨平台, 期货分钟)")
    print("=" * 70)

    try:
        import hs_udata as hs
    except ImportError:
        print("\n[错误] 未安装 hs_udata, 请先执行:  pip install hs_udata")
        sys.exit(1)

    token = os.environ.get("UDATA_TOKEN", "")
    if not token:
        print("\n[提示] 未设置 UDATA_TOKEN 环境变量。")
        print("  1) 注册恒有数并订阅免费体验套餐: https://udata.hs.net")
        print("  2) 登录后进入「总览」页面复制 Token")
        print("  3) 运行:  UDATA_TOKEN=你的token python -u test/data/test_udata.py")
        sys.exit(1)

    print("\n>>> 设置 Token 并初始化")
    try:
        hs.set_token(token)
        print("    set_token 完成")
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        sys.exit(1)

    # ---- 1) 获取期货合约列表 (5 大交易所) ----
    print("\n>>> fut_list()  —— 列出全部在册期货合约")
    try:
        df_list = hs.fut_list()
        if df_list is not None and len(df_list) > 0:
            print(f"    共 {len(df_list)} 条记录, 列: {list(df_list.columns)}")
            print(df_list.head(5).to_string())
            # 按输入合约过滤展示
            if "secu_code" in df_list.columns:
                hit = df_list[df_list["secu_code"].astype(str).str.upper() == DEFAULT_CODE.upper()]
                if hit.empty:
                    print(f"\n    [提示] 默认合约 {DEFAULT_CODE} 不在列表中, 请参考上面列表改用一个有效合约。")
                else:
                    print(f"\n    命中合约 {DEFAULT_CODE}:")
                    print(hit.head(1).to_string())
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        print("    可能原因: Token 无效 / 未订阅免费套餐 / 网络无法访问 udata.hs.net")

    # ---- 2) 获取期货 1 分钟切片 ----
    print(f"\n>>> fut_quote_minute(en_prod_code='{DEFAULT_CODE}', {DEFAULT_BEGIN} ~ {DEFAULT_END})")
    try:
        df = hs.fut_quote_minute(en_prod_code=DEFAULT_CODE,
                                 begin_date=DEFAULT_BEGIN, end_date=DEFAULT_END)
        if df is None or df.empty:
            print("    返回为空。请确认合约代码有效(参考 fut_list)或该时段有数据。")
            return
        print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        print(f"    列名: {list(df.columns)}")
        print("\n    前 3 行:")
        print(df.head(3).to_string())
        print("\n    后 2 行:")
        print(df.tail(2).to_string())
        print("\n    字段说明: open/high/low/close=开高低收, turnover_volume=成交量(手),")
        print("            turnover_value=成交额(元), amount=持仓量(手)")
    except Exception as e:
        print(f"    [失败] {type(e).__name__}: {e}")
        print("    可能原因: 合约代码无效 / Token 无期货分钟权限 / 网络问题")

    print("\n完成。免费套餐拉取多年分钟数据后, 建议落盘存储(parquet/MongoDB)再用于因子挖掘。")


if __name__ == "__main__":
    main()
