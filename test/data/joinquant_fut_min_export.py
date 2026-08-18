"""
聚宽研究环境: 一次性导出 5 年期货分钟量价数据 (主力连续合约)

在聚宽研究环境(网页版 Jupyter)中运行, 把多个商品期货品种的分钟数据
分批拉取、去重、排序后保存为 CSV, 供本地下载用于因子挖掘/回测。

=================== 使用方法 ===================
1. 打开 https://www.joinquant.com 研究环境, 新建一个 notebook;
2. 把本文件全部内容粘贴到一个单元格(或上传本 .py 后执行 %run joinquant_fut_min_export.py);
3. 修改下方【配置区】, 然后运行;
4. 运行结束后, 在研究环境左侧文件树中找到 data/fut_min/ 目录, 右键下载 CSV。

=================== 说明 ===================
- 聚宽研究环境已内置 jqdata, 无需登录/授权, 直接调用 get_price。
- 主力连续合约代码格式: 品种+9999+交易所后缀 (如 RB9999.XSGE);
  指数合约(持仓额加权)格式: 品种+8888+交易所后缀 (如 RB8888.XSGE)。
- 分钟数据时间戳为「时间段结束」(1m 的 bar 时间戳从 09:31 到 15:00, 夜盘延续)。
- 脚本按 14 天一个窗口分批拉取, 避免单次请求数据量过大; 每品种独立保存,
  并以「窗口标记文件」做断点续传——中断后重新运行, 已完成的窗口会跳过。
- 若免费权限拉不到 5 年(某时段无数据/报错), 脚本会打印警告并继续,
  可适当缩短 START_DATE 或开通研究版。
"""

import os
import time
import datetime as dt

import pandas as pd

# ---- 解析 get_price (兼容不同运行环境) ----
# 聚宽研究环境: get_price 等数据函数已被预置为「全局函数」, 直接使用即可, 无需 import;
#               from jqdata import get_price 会报 ImportError, 因为 jqdata 模块不直接导出它。
# 本地 jqdatasdk: 需 from jqdatasdk import * 或 from jqdatasdk import get_price。
try:
    get_price  # noqa: F821  研究环境全局已存在
except NameError:
    try:
        from jqdata import get_price
    except ImportError:
        try:
            from jqdata import *
        except ImportError:
            from jqdatasdk import get_price

# ================== 配置区 ==================
FREQ = "1m"                # 分钟周期: 1m / 5m / 15m / 30m / 60m
START_DATE = "2021-08-01"   # 5 年前(按需修改)
END_DATE = "2026-08-17"     # 截止日(含当天)
FIELDS = ["open", "high", "low", "close", "volume", "money", "open_interest"]
CHUNK_DAYS = 14             # 每窗口天数(1m 下 14 天约 2800 根, 稳妥)
SLEEP_SEC = 0.3             # 每次请求间隔, 避免触发限频
OUT_DIR = "data/fut_min"    # 输出目录(研究环境文件系统, 相对当前工作目录)

# 主力连续合约代码(从聚宽官方「商品期货数据」页确认的格式)
# 上期所 .XSGE
FUTURE_CODES = [
    "RB9999.XSGE",  # 螺纹钢
    "CU9999.XSGE",  # 铜
    "AU9999.XSGE",  # 黄金
    "AG9999.XSGE",  # 白银
    "M9999.XDCE",   # 豆粕(大商所)
    "I9999.XDCE",   # 铁矿石(大商所)
    "TA9999.XZCE",  # PTA(郑商所)
    "SA9999.XZCE",  # 纯碱(郑商所)
    "SC9999.XINE",  # 原油(能源中心)
]

# 更多品种(按需取消注释启用)
# 上期所: AL9999.XSGE 铝, ZN9999.XSGE 锌, NI9999.XSGE 镍, SN9999.XSGE 锡,
#         PB9999.XSGE 铅, FU9999.XSGE 燃料油, BU9999.XSGE 沥青, RU9999.XSGE 橡胶,
#         SP9999.XSGE 纸浆, SS9999.XSGE 不锈钢, HC9999.XSGE 热轧卷板, AO9999.XSGE 氧化铝
# 郑商所: SR9999.XZCE 白糖, CF9999.XZCE 棉花, MA9999.XZCE 甲醇, FG9999.XZCE 玻璃,
#         UR9999.XZCE 尿素, AP9999.XZCE 苹果, CJ9999.XZCE 红枣, OI9999.XZCE 菜油,
#         RM9999.XZCE 菜粕, PF9999.XZCE 短纤, PK9999.XZCE 花生, SF9999.XZCE 硅铁,
#         SM9999.XZCE 锰硅, PX9999.XZCE 对二甲苯
# 大商所: Y9999.XDCE 豆油, A9999.XDCE 豆一, C9999.XDCE 玉米, CS9999.XDCE 淀粉,
#         JD9999.XDCE 鸡蛋, L9999.XDCE 聚乙烯, V9999.XDCE PVC, PP9999.XDCE 聚丙烯,
#         J9999.XDCE 焦炭, JM9999.XDCE 焦煤, EG9999.XDCE 乙二醇, EB9999.XDCE 苯乙烯,
#         PG9999.XDCE 液化气, LH9999.XDCE 生猪, P9999.XDCE 棕榈油
# 能源:  NR9999.XINE 20号胶, LU9999.XINE 低硫燃料油, BC9999.XINE 国际铜
# 广期所 .XGFEX: SI9999.XGFEX 工业硅, LC9999.XGFEX 碳酸锂


# ================== 以下无需修改 ==================
END_DT = dt.datetime.strptime(END_DATE, "%Y-%m-%d").replace(hour=15)
FLAG_ROOT = os.path.join(OUT_DIR, "_flags")


def gen_windows(start_dt, end_dt, chunk_days):
    """按 chunk_days 生成 [start, end] 窗口列表, end 一律取到 15:00"""
    windows = []
    cur = start_dt
    while cur < end_dt:
        win_end = min(cur + dt.timedelta(days=chunk_days), end_dt).replace(hour=15)
        windows.append((cur, win_end))
        cur = win_end.replace(hour=0) + dt.timedelta(days=1)
    return windows


def fetch_chunk(code, start, end, depth=0):
    """拉取一个窗口, 若因数据量过大/超时失败则折半重试"""
    try:
        df = get_price(code, start_date=str(start), end_date=str(end),
                       frequency=FREQ, fields=FIELDS)
        return df
    except Exception as e:
        days = (end - start).days
        if days > 1 and depth < 4:
            mid = start + dt.timedelta(days=days // 2)
            mid = mid.replace(hour=15)
            left = fetch_chunk(code, start, mid, depth + 1)
            right = fetch_chunk(code, mid.replace(hour=0) + dt.timedelta(days=1),
                                end, depth + 1)
            if left is None or right is None:
                return None
            return pd.concat([left, right])
        print(f"      [警告] 窗口 {start}~{end} 拉取失败: {type(e).__name__}: {e}")
        return None


def normalize(df, code):
    """get_price 返回 MultiIndex(time, code) DataFrame, 转成规整 DataFrame"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    cols = list(df.columns)
    # 找时间列与代码列
    time_col, code_col = None, None
    for c in cols:
        if str(c).lower() in ("time", "datetime", "index"):
            time_col = c
        if str(c).lower() == "code":
            code_col = c
    if time_col is None:
        time_col = cols[0]
    df = df.rename(columns={time_col: "datetime"})
    if code_col is None:
        second = df.columns[1]
        if str(second) not in FIELDS:
            df = df.rename(columns={second: "code"})
        else:
            df.insert(1, "code", code)
    df["datetime"] = df["datetime"].astype(str)
    return df


def load_existing(code):
    """读取该品种已保存的 CSV(续传时与新拉数据合并)"""
    path = os.path.join(OUT_DIR, f"{code}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FLAG_ROOT, exist_ok=True)
    start_dt = dt.datetime.strptime(START_DATE, "%Y-%m-%d")
    windows = gen_windows(start_dt, END_DT, CHUNK_DAYS)
    print("=" * 70)
    print(f"聚宽研究环境 | 导出 {len(FUTURE_CODES)} 个品种 {FREQ} 分钟数据")
    print(f"时间范围: {START_DATE} ~ {END_DATE} | 总窗口: {len(windows)}")
    print("=" * 70)

    for code in FUTURE_CODES:
        flag_dir = os.path.join(FLAG_ROOT, code)
        os.makedirs(flag_dir, exist_ok=True)
        done = set(f.split(".")[0] for f in os.listdir(flag_dir))
        todo = [w for w in windows if w[0].strftime("%Y%m%d") not in done]

        if not todo:
            print(f"[跳过] {code}: 全部窗口已完成")
            continue

        print(f"\n>>> 正在导出 {code} ... 待处理 {len(todo)}/{len(windows)} 窗口")
        parts = []
        existing = load_existing(code)
        if len(existing) > 0:
            parts.append(existing)

        for i, (ws, we) in enumerate(todo, 1):
            flag = os.path.join(flag_dir, ws.strftime("%Y%m%d") + ".done")
            df = fetch_chunk(code, ws, we)
            time.sleep(SLEEP_SEC)
            if df is not None:
                nd = normalize(df, code)
                if not nd.empty:
                    parts.append(nd)
                open(flag, "w").close()  # 成功(含该时段无数据)即标记完成
            else:
                print(f"      [警告] 窗口 {ws}~{we} 拉取失败, 未标记, 重跑会重试")
            if i % 10 == 0 or i == len(todo):
                print(f"    {code} 进度 {i}/{len(todo)} 窗口")

        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.drop_duplicates(subset=["datetime", "code"], keep="last")
            merged = merged.sort_values("datetime").reset_index(drop=True)
            merged.to_csv(os.path.join(OUT_DIR, f"{code}.csv"), index=False)
            print(f"    完成: {code} | {len(merged)} 行 "
                  f"| {merged['datetime'].min()} ~ {merged['datetime'].max()}")
        else:
            print(f"    [警告] {code} 未拉到任何数据。可能免费权限无该品种分钟数据, "
                  f"或合约代码无效。")

    print("\n" + "=" * 70)
    print("全部完成。请到研究环境左侧文件树中打开 data/fut_min/ 目录下载 CSV。")
    print("提示: 若某品种数据缺失, 通常是免费研究版权限限制, 可缩短 START_DATE 或开通研究版。")


if __name__ == "__main__":
    main()
