"""
数据源测试: Tushare Pro（积分制，需注册 + token）

重点演示期货历史分钟行情:
    pro.ft_mins(ts_code, freq, start_date, end_date)
        - 1min/5min/15min/30min/60min
        - 超 10 年历史, 单次最大 8000 行
        - 字段: open/close/high/low/vol/amount/oi
        - 需单独开通权限 (120 积分可试调 2 次)
另附带免费接口 trade_cal(交易日历) 用于验证 token 连通性。

用法:
    pip install tushare
    # 方式一: 环境变量
    TUSHARE_TOKEN=你的token python -u test/data/test_tushare.py
    # 方式二: 已在本机执行过 ts.set_token() 保存 token
    python -u test/data/test_tushare.py

说明:
    - 合约代码形如 RB2609.SHF / CU2610.SHF, 请改成当前存续的有效合约;
      可通过环境变量 TUSHARE_FUTURE_CODE 指定。
    - 分钟接口未开通权限时会返回权限错误, 脚本会给出提示。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONTRACT = os.environ.get("TUSHARE_FUTURE_CODE", "RB2609.SHF")
DEFAULT_FREQ = "1min"


def load_token() -> str:
    """按优先级取 token: 环境变量 -> 本机缓存文件 -> tushare 内置存储"""
    env_token = os.environ.get("TUSHARE_TOKEN")
    if env_token:
        return env_token

    # tushare 支持把 token 存到本机 (ts.set_token 保存的位置)
    cache_file = Path.home() / ".tushare" / "token"
    if cache_file.exists():
        tok = cache_file.read_text().strip()
        if tok:
            return tok

    # 最后尝试 tushare 内置 (若之前运行过 ts.set_token)
    try:
        import tushare as ts

        tok = ts.get_token()
        if tok:
            return tok
    except Exception:
        pass
    return ""


def main():
    print("=" * 70)
    print("数据源: Tushare Pro (积分制)")
    print("=" * 70)

    token = load_token()
    if not token:
        print("\n[错误] 未找到 Tushare token。请任选一种方式提供:")
        print("  1) 设置环境变量:  export TUSHARE_TOKEN=你的token")
        print("  2) 先执行:  python -c \"import tushare as ts; ts.set_token('你的token')\"")
        print("  注册/获取 token: https://tushare.pro  (注册后个人主页有 token)")
        sys.exit(1)

    try:
        import tushare as ts
    except ImportError:
        print("\n[错误] 未安装 tushare, 请先执行:  pip install tushare")
        sys.exit(1)

    try:
        pro = ts.pro_api(token)
    except Exception as e:
        print(f"\n[失败] 初始化 token 失败: {e}")
        sys.exit(1)

    # ---- 0) 免费接口验证连通性 ----
    print("\n>>> (连通性验证) 交易日历 trade_cal (免费接口)")
    try:
        df_cal = pro.trade_cal(exchange="SHFE", start_date="20260801", end_date="20260817")
        print(f"    数据形状: {df_cal.shape}")
        print(df_cal.head(3).to_string())
    except Exception as e:
        print(f"    [失败] {e}")

    # ---- 1) 期货历史分钟行情 ----
    print(f"\n>>> 期货历史分钟行情 ft_mins(ts_code='{DEFAULT_CONTRACT}', freq='{DEFAULT_FREQ}')")
    try:
        df = pro.ft_mins(ts_code=DEFAULT_CONTRACT, freq=DEFAULT_FREQ,
                         start_date="2026-08-10 09:00:00", end_date="2026-08-14 15:00:00")
        if df is None or df.empty:
            print("    返回为空, 可能该合约在该时段无数据或代码不正确。")
            return
        print(f"    数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        print(f"    列名: {list(df.columns)}")
        print("\n    前 3 行:")
        print(df.head(3).to_string())
        print("\n    后 2 行:")
        print(df.tail(2).to_string())
        print("\n    字段说明: vol=成交量(手), amount=成交额(元), oi=持仓量(手)")
        print("\n[提示] 如果返回 '权限不足/无权限' 错误, 说明你的账号尚未开通 ft_mins 接口权限,")
        print("      请参考 https://tushare.pro/document/2?doc_id=313 的开通说明。")
    except Exception as e:
        msg = str(e)
        print(f"    [失败] {type(e).__name__}: {msg}")
        if "权限" in msg or "permission" in msg.lower():
            print("\n[提示] 该报错通常是 ft_mins 接口权限未开通。")
            print("       - 先到权限中心确认是否已开通「期货分钟行情」")
            print("       - 120 积分可试调 2 次, 正式使用需按官方说明开通")
            print("       - 文档: https://tushare.pro/document/2?doc_id=313")
        else:
            print(f"\n[提示] 若为合约代码无效, 可通过环境变量 TUSHARE_FUTURE_CODE 指定有效合约, 例如:")
            print("       TUSHARE_FUTURE_CODE=CU2610.SHF python -u test/data/test_tushare.py")

    print("\n完成。")


if __name__ == "__main__":
    main()
