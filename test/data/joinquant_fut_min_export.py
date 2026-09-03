"""
聚宽研究环境: 一次性导出 5 年期货分钟量价数据 (主力连续合约)

在聚宽研究环境(网页版 Jupyter)中运行, 把多个商品期货品种的分钟数据
分批拉取、去重、排序后保存为 CSV, 供本地下载用于因子挖掘/回测。

换月日判断
----
使用聚宽官方接口 get_dominant_future 获取每个交易日的主力合约:
    get_dominant_future(root, start_date, end_date=end_date)   # 注意 end_date 是关键字参数!
主力合约切换的日期即为换月日, 并用换月日当天新旧合约的开盘价计算后复权因子:
    cur_ratio = 旧合约开盘 / 新合约开盘,  weighted_factor 累乘。
CSV 中会直接导出 symbol / is_rollover / weighted_factor / cur_weighted_factor 字段,
本地导入时直接使用, 不再自行检测换月。

=================== 使用方法 ===================
1. 打开 https://www.joinquant.com 研究环境, 新建一个 notebook;
2. 把本文件全部内容粘贴到一个单元格(或上传本 .py 后执行 %run joinquant_fut_min_export.py);
3. 修改下方【配置区】, 然后运行;
4. 运行结束后, 在研究环境左侧文件树中找到 data/fut_min/ 目录, 右键下载 CSV。

=================== 说明 ===================
- 聚宽研究环境已内置 jqdata, get_price / get_dominant_future 均为全局函数, 无需 import。
- 分钟数据时间戳为「时间段结束」(1m 的 bar 时间戳从 09:31 到 15:00, 夜盘延续)。
- 夜盘(21:00 后)归下一个交易日; 周五夜盘归下周一。
- 若某时段无数据/报错, 脚本打印警告并继续。
"""

import os
import time
import datetime as dt

import numpy as np
import pandas as pd

# ---- 解析 get_price / get_dominant_future (兼容不同运行环境) ----
try:
    get_price  # noqa: F821
except NameError:
    try:
        from jqdata import get_price, get_dominant_future
    except ImportError:
        try:
            from jqdata import *
        except ImportError:
            from jqdatasdk import get_price, get_dominant_future
try:
    get_dominant_future  # noqa: F821
except NameError:
    try:
        from jqdata import get_dominant_future
    except ImportError:
        from jqdatasdk import get_dominant_future

# ================== 配置区 ==================
FREQ = "1m"                # 分钟周期: 1m / 5m / 15m / 30m / 60m
START_DATE = "2020-01-01"   # 5 年前(按需修改)
END_DATE = "2026-08-17"     # 截止日(含当天)
FIELDS = ["open", "high", "low", "close", "volume", "money", "open_interest"]
CHUNK_DAYS = 14             # 每窗口天数(1m 下 14 天约 2800 根, 稳妥)
SLEEP_SEC = 0.3             # 每次请求间隔, 避免触发限频
OUT_DIR = "data/fut_min"    # 输出目录(研究环境文件系统, 相对当前工作目录)

# 主力连续合约代码(从聚宽官方「商品期货数据」页确认的格式)
FUTURE_CODES = [
    "C9999.XDCE",  # 玉米
    # "RB9999.XSGE",  # 螺纹钢
    # "CU9999.XSGE",  # 铜
    # "AU9999.XSGE",  # 黄金
    # "AG9999.XSGE",  # 白银
    # "M9999.XDCE",   # 豆粕(大商所)
    # "I9999.XDCE",   # 铁矿石(大商所)
    # "TA9999.XZCE",  # PTA(郑商所)
    # "SA9999.XZCE",  # 纯碱(郑商所)
    # "SC9999.XINE",  # 原油(能源中心)
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


def _root_from_code(code: str) -> str:
    """从主力连续代码提取品种 root, 如 C9999.XDCE -> C ; TA9999.XZCE -> TA"""
    sym = str(code).split(".")[0]   # C9999
    root = sym.rstrip("0123456789")  # C
    return root


def _get_day_open(contract: str, date) -> float:
    """获取某合约在某交易日的日线开盘价。失败返回 None。"""
    try:
        df = get_price(contract, start_date=str(date), end_date=str(date),
                       frequency="daily", fields=["open"])
        if df is not None and not df.empty:
            df = df.reset_index()
            return float(pd.to_numeric(df["open"], errors="coerce").iloc[-1])
    except Exception:
        pass
    return None


def get_dominant_series_compat(root: str, trade_days):
    """获取每日主力合约 Series(交易日 -> 合约)。

    兼容不同版本:
      1) 优先尝试批量: get_dominant_future(root, start, end_date=end);
      2) 若版本不支持 end_date(TypeError), 回退为逐交易日查询 get_dominant_future(root, date)。
    """
    trade_days = sorted(set(pd.Timestamp(d) for d in trade_days))
    if not trade_days:
        return pd.Series(dtype=object)

    try:
        dom = get_dominant_future(root, str(trade_days[0].date()),
                                  end_date=str(trade_days[-1].date()))
        if dom is not None and len(dom) > 0:
            return dom
    except TypeError:
        pass
    except Exception:
        pass

    print("    [信息] get_dominant_future 不支持 end_date, 改为逐交易日查询")
    series = {}
    for d in trade_days:
        try:
            series[pd.Timestamp(d)] = str(get_dominant_future(root, str(d.date())))
        except Exception as e:
            print(f"    [警告] {d.date()} 主力获取失败: {e}")
    return pd.Series(series).sort_index()


def build_dominant_schedule(root: str, trade_days):
    """用 get_dominant_future 计算每日主力合约 / 换月日 / 后复权因子链。

    返回 DataFrame: td(交易日), symbol, is_rollover, weighted_factor, cur_weighted_factor
    """
    dom = get_dominant_series_compat(root, trade_days)
    if dom is None or len(dom) == 0:
        return pd.DataFrame()
    if isinstance(dom, pd.DataFrame):
        # 兼容返回 DataFrame 的版本: 取主力合约代码列
        col = None
        for c in dom.columns:
            if any(k in str(c).lower() for k in ("symbol", "dominant", "code")):
                col = c
                break
        dom = dom[col if col is not None else dom.columns[0]]
    dom = pd.Series(dom).astype(str)
    dom.index = pd.to_datetime(dom.index).normalize()
    # 交易日唯一化 + 排序: 防 get_dominant_future 返回重复/乱序日期导致 wf 链错乱
    dom = dom[~dom.index.duplicated(keep="last")].sort_index()
    dom = dom[dom.str.len() > 0]

    rows = []
    wf = 1.0
    cur_cwf = 1.0
    prev_symbol = None
    for d, symbol in dom.items():
        symbol = str(symbol)
        is_roll = (prev_symbol is not None and symbol != prev_symbol)
        if is_roll:
            old_open = _get_day_open(prev_symbol, d)
            new_open = _get_day_open(symbol, d)
            if old_open and new_open and abs(new_open) > 1e-12:
                ratio = old_open / new_open
                if 0.7 < ratio < 1.3:
                    cur_cwf = ratio
                    wf *= cur_cwf
                else:
                    # 开盘比异常(取价失败/合约错配)时忽略本次换月, 保持 wf 链连续
                    print(f"    [警告] {d.date()} 换月 {prev_symbol}->{symbol} 开盘比 {ratio:.4f} 异常, 忽略本次换月")
                    symbol = prev_symbol
                    is_roll = False
            else:
                print(f"    [警告] {d.date()} 换月 {prev_symbol}->{symbol} 开盘价取价失败, 忽略本次换月")
                symbol = prev_symbol
                is_roll = False
        rows.append({
            "td": pd.Timestamp(d).normalize(),
            "symbol": symbol,
            "is_rollover": is_roll,
            "weighted_factor": wf,
            "cur_weighted_factor": cur_cwf,
        })
        prev_symbol = symbol

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="td", keep="last").sort_values("td")
    if df.empty:
        return df
    # 自检: 非换月日的 wf 突变 = schedule 不一致, 打印告警便于发现
    chg = df["weighted_factor"].pct_change().abs()
    bad = (chg > 1e-6) & (~df["is_rollover"].astype(bool))
    if bad.any():
        for _, r in df[bad].iterrows():
            print(f"    [警告] {r['td'].date()} 非换月日 wf 异常变化: {r['weighted_factor']:.6f}")
    return df


def assign_trading_day_local(datetime_series, trading_days):
    """给分钟 bar 打交易日标签(夜盘归下一交易日, 周五夜盘归下周一)。"""
    tds = np.array(sorted(pd.to_datetime(list(trading_days)).normalize()), dtype="datetime64[D]")
    tds = np.unique(tds)
    night = datetime_series.dt.hour >= 20
    cal = datetime_series.dt.normalize().values.astype("datetime64[D]")
    out = pd.Series(pd.NaT, index=datetime_series.index, dtype="datetime64[ns]")
    if tds.size == 0:
        return out
    day_pos = np.minimum(np.searchsorted(tds, cal[~night], side="left"), tds.size - 1)
    out.loc[~night] = pd.to_datetime(tds[day_pos])
    night_pos = np.minimum(np.searchsorted(tds, cal[night], side="right"), tds.size - 1)
    out.loc[night] = pd.to_datetime(tds[night_pos])
    return out


def gen_windows(start_dt, end_dt, chunk_days):
    windows = []
    cur = start_dt
    while cur < end_dt:
        win_end = min(cur + dt.timedelta(days=chunk_days), end_dt).replace(hour=15)
        windows.append((cur, win_end))
        cur = win_end.replace(hour=0) + dt.timedelta(days=1)
    return windows


def fetch_chunk(code, start, end, depth=0):
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
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    cols = list(df.columns)
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
    print(f"聚宽研究环境 | 导出 {len(FUTURE_CODES)} 个品种 {FREQ} 分钟数据 (换月用 get_dominant_future)")
    print(f"时间范围: {START_DATE} ~ {END_DATE} | 总窗口: {len(windows)}")
    print("=" * 70)

    for code in FUTURE_CODES:
        root = _root_from_code(code)
        print(f"\n>>> 处理 {code} (root={root})")

        flag_dir = os.path.join(FLAG_ROOT, code)
        os.makedirs(flag_dir, exist_ok=True)
        done = set(f.split(".")[0] for f in os.listdir(flag_dir))
        todo = [w for w in windows if w[0].strftime("%Y%m%d") not in done]
        if not todo:
            print(f"[跳过] {code}: 全部窗口已完成")
            continue

        print(f"    {code} 待处理 {len(todo)}/{len(windows)} 窗口")
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
                open(flag, "w").close()
            else:
                print(f"      [警告] 窗口 {ws}~{we} 拉取失败, 未标记, 重跑会重试")
            if i % 10 == 0 or i == len(todo):
                print(f"    {code} 进度 {i}/{len(todo)} 窗口")

        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.drop_duplicates(subset=["datetime", "code"], keep="last")
            merged = merged.sort_values("datetime").reset_index(drop=True)

            # 1) 用 get_dominant_future 计算换月 schedule(交易日从分钟数据日盘推断)
            merged["_dt"] = pd.to_datetime(merged["datetime"])
            day_mask = merged["_dt"].dt.hour < 20
            trade_days = sorted(pd.unique(merged.loc[day_mask, "_dt"].dt.normalize()))
            merged = merged.drop(columns=["_dt"])
            sched = build_dominant_schedule(root, trade_days)
            if sched.empty:
                print(f"    [警告] {code} get_dominant_future 返回空, 换月字段缺失")
            else:
                n_roll = int(sched["is_rollover"].sum())
                print(f"    换月日 {n_roll} 个")

            # 2) 打交易日标签并 merge 换月字段
            if not sched.empty:
                merged["datetime"] = pd.to_datetime(merged["datetime"])
                # 剥离旧 CSV 自带的换月字段: 每轮以最新 schedule 全区间重建, 避免新旧列冲突/幽灵残留
                _drop = [c for c in merged.columns
                         if c in ("symbol", "is_rollover", "weighted_factor", "cur_weighted_factor", "td")
                         or c.endswith(("_x", "_y"))]
                if _drop:
                    merged = merged.drop(columns=_drop)
                merged["td"] = assign_trading_day_local(merged["datetime"], sched["td"])
                merged = merged.merge(sched, on="td", how="left")
                # schedule 缺交易日(数据洞)时沿用前后交易日值并告警
                if merged["weighted_factor"].isna().any():
                    n = int(merged["weighted_factor"].isna().sum())
                    print(f"    [警告] {n} 行交易日无换月 schedule(数据洞), 已沿用前后交易日值")
                for c in ["symbol", "weighted_factor", "cur_weighted_factor"]:
                    merged[c] = merged[c].ffill().bfill()
                merged["is_rollover"] = merged["is_rollover"].fillna(False).astype(bool)
                # 移除 td 辅助列
                merged = merged.drop(columns=["td"], errors="ignore")

            merged.to_csv(os.path.join(OUT_DIR, f"{code}.csv"), index=False)
            print(f"    完成: {code} | {len(merged)} 行 | 列: {list(merged.columns)}")
        else:
            print(f"    [警告] {code} 未拉到任何数据。")

    print("\n" + "=" * 70)
    print("全部完成。请到研究环境左侧文件树中打开 data/fut_min/ 目录下载 CSV。")
    print("CSV 含换月字段: symbol / is_rollover / weighted_factor / cur_weighted_factor")


if __name__ == "__main__":
    main()
