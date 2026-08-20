# 国内商品期货分钟频率量价数据源调研

> 调研日期：2026-08-17
> 目标：为本项目（日频因子挖掘框架）补充国内商品期货各品种的**分钟频率量价数据**（OHLCV + 持仓量）。
> 覆盖交易所：上期所(SHFE)、大商所(DCE)、郑商所(CZCE)、上期能源(INE)、广期所(GFEX)。
> 说明：本文所有链接均在调研时通过实际访问或官方文档确认，**未包含任何编造的链接**。付费价格以官方最新报价为准。

---

## 0. 结论先行（TL;DR）

| 方案 | 费用 | 分钟历史深度 | 适合场景 |
|---|---|---|---|
| **AkShare**（新浪/东财封装） | 免费、开源 | 仅最近几天~数百根 | 快速测试、临时取数 |
| **天勤 TqSdk / EDB** | 注册免费（专业版付费） | 免费近 **1 年**分钟线；付费全历史 | 免费主力首选、实时行情 |
| **Tushare Pro `ft_mins`** | 积分/会员制 | **超 10 年**历史分钟 | 历史因子回测首选 |
| **掘金量化 MyQuant** | 免费版 | 分钟 **2017-01-01 起** | 免费全历史分钟 |
| **聚宽 JQData** | 付费（有免费体验） | 期货分钟完整历史 | 数据质量要求高 |
| **米筐 RQData** | 付费（有试用） | 分钟线 + tick | 机构级、一站式 |
| **恒有数 UData** | **免费**注册+体验套餐 | 期货 1 分钟（多年历史，盘后） | **macOS 免费 5 年分钟首选** |
| **迅投研 / QMT** | 券商开通（免费/资金门槛） | 分钟线 + tick | 实盘 + 数据一体 |
| **Wind / iFinD / Choice** | 机构级、较贵 | 分钟线 | 金融机构从业者 |

**针对本项目的推荐组合**：
1. **纯免费（macOS 可用，推荐）**：**恒有数 UData**（免费，可拉多年期货 1 分钟，盘后）+ 天勤 EDB（近 1 年分钟，含持仓量，可做交叉校验）+ AkShare（随手测试）。
2. **小成本**：Tushare Pro `ft_mins`（10 年历史分钟，但**单独收费 2000 元/年**，非免费）。
3. **需要实时行情**：天勤 TqSdk 免费实时行情，或 QMT/迅投研。
4. **不推荐在 macOS 上折腾掘金 gm**：`gm` 无 macOS 轮子，掘金终端仅支持 Windows。

---

## 1. 免费数据源

### 1.1 AkShare（开源免费，封装新浪/东财/腾讯接口）

- **类型**：开源 Python 库，聚合新浪财经、东方财富、腾讯等多处网页接口，无需注册。
- **分钟相关函数**（实测文档确认存在）：
  - `futures_zh_minute_sina(symbol, period)` —— 新浪期货分钟数据，`period` 支持 1/5/15/30/60；
  - `futures_zh_minute_sina_df(symbol)` —— 新浪期货分钟数据（按合约代码）；
  - `futures_zh_realtime()` / `futures_zh_realtime_df()` —— 东方财富期货实时行情快照。
- **历史深度**：新浪分时/分钟接口通常只返回最近一段时间（几百根左右），**不是全历史**。
- **优点**：免费、免注册、pip 安装即用、品种覆盖全。
- **缺点**：底层是爬取的网页接口，无官方保障，可能限流/失效；部分接口不含持仓量；历史深度有限。
- **链接**：
  - 官方文档（期货数据）：https://akshare.akfamily.xyz/data/futures/futures.html
  - GitHub 仓库：https://github.com/akfamily/akshare
  - 期货数据文档源码（md）：https://github.com/akfamily/akshare/blob/main/docs/data/futures/futures.md

### 1.2 天勤 TqSdk / EDB 历史数据服务（信易科技，免费注册）

- **类型**：天勤量化（信易科技/上海快期）提供，注册即用。
- **EDB 行情历史服务（RESTful HTTP，返回 CSV）**，官方文档明确：
  - **免费访问**：无需 token 即可获取 **最近 1 年** 的分钟线（`period=60`）+ **任意历史区间** 的日线；
  - **专业版（付费）**：可获取全部历史分钟线与日线，购买入口在官网用户中心；
  - 历史数据范围预计覆盖 **2021-01-01 以来**，支持 1 分钟线（`period=60`）；
  - 服务地址：`https://edb.shinnytech.com/md`，接口示例：
    `https://edb.shinnytech.com/md/kline?period=60&symbol=SHFE.rb2401&start_time=2023-08-01 09:00:00&end_time=2023-08-01 15:00:00`
  - 返回字段：`datetime_nano, open, high, low, close, volume, open_oi, close_oi`（**含开盘/收盘持仓量**）；
  - 支持合约标识：`SHFE.rb2401`、`DCE.m1901`、`CZCE.SR901`、`INE.sc2109`、`GFEX.si2301` 及主连 `KQ.m@CFFEX.IF`、指数 `KQ.i@SHFE.bu`。
- **TqSdk Python 库**：实时行情 / 历史数据 / 实盘交易一体化，分钟线实时更新（vnpy 官方数据服务文档确认）。
- **优点**：免费额度对“近 1 年分钟 + 任意日线”完全够用；含持仓量；支持主连/指数合约；文档规范。
- **缺点**：免费版分钟线仅最近 1 年，更早历史需付费专业版。
- **链接**：
  - EDB 行情历史服务文档：https://doc.shinnytech.com/edb/latest/md_server.html
  - TqSdk GitHub：https://github.com/shinnytech/tqsdk-python
  - 天勤官网：https://www.shinnytech.com
  - 天勤量化注册（vnpy 文档给出）：https://www.shinnytech.com/tianqin

### 1.3 Tushare Pro（积分制，注册即用，分钟需单独开通权限）

- **类型**：国内最知名的开源 Python 金融数据接口平台，积分制。
- **期货历史分钟行情 `ft_mins`**（官方文档确认）：
  - 支持 1min / 5min / 15min / 30min / 60min；
  - 提供**超过 10 年**历史分钟数据；单次最大 8000 行（可按合约+时间循环取）；
  - 输出字段：`open, close, high, low, vol, amount, oi`（**含成交额与持仓量**）；
  - 需单独开权限，120 积分可试调 2 次；正式权限见权限说明。
- **期货实时分钟行情 `rt_fut_min`** + `rt_fut_min_daily`（当日开市以来分钟快照回放）：支持 1/5/15/30/60MIN，字段含 `vol/amount/oi`。
- **主力合约**：需先通过主力合约映射接口（`doc_id=189`，需至少 2000 积分）获取合约代码。
- **优点**：接口规范、历史深、含持仓量与成交额、Python SDK 完善。
- **缺点**：分钟接口需单独开通权限并付费/高积分；免费额度有限。
- **链接**：
  - 期货数据总览：https://tushare.pro/document/2?doc_id=134
  - 期货历史分钟行情 `ft_mins`：https://tushare.pro/document/2?doc_id=313
  - 期货实时分钟行情 `rt_fut_min`：https://tushare.pro/document/2?doc_id=340
  - 主力/连续合约映射：https://tushare.pro/document/2?doc_id=189
  - 权限说明：https://tushare.pro/document/1?doc_id=290

### 1.4 掘金量化 MyQuant（免费注册，需装终端）

- **类型**：国产量化投研/交易平台，Python/C++/C#/Matlab SDK，需下载终端并登录。
- **数据支持**（官方 FAQ 确认）：
  - 支持 8 大交易所，含全部期货交易所（含广期所）；
  - **分钟行情最早支持 2017-01-01**（60s / 300s / 900s / 1800s / 3600s）；
  - **tick 行情最早支持 2022-08-10**；
  - 期货实时行情频率：tick、15s、30s、60s、300s、900s、1800s、3600s；
  - 1 分钟 bar 历史一次最多 33000 根。
- **免费版限制**：实时订阅最多 50 个标的（回测无限制）；tick 可提取最近 3 个月。
- **优点**：免费版就能拿到 2017 年以来的全历史分钟数据，性价比高。
- **缺点**：需要 Windows 终端（Mac 支持较弱）；有流控限制；导出需自行 `to_csv/to_excel`。
- **链接**：
  - 官网：https://www.myquant.cn
  - 数据问题 FAQ（含历史时间范围）：https://myquant.cn/docs2/faq/数据问题.html
  - 数据文档-期货：https://myquant.cn/docs2/docs/期货.html

### 1.5 聚宽 JQData（有免费体验，正式使用付费）

- **类型**：聚宽推出的数据服务，支持 HTTP API 与 Python SDK（`jqdatasdk`）。
- **期货分钟数据**：支持 `1m / 5m / 15m / 30m / 60m / 120m` 周期（JQData 官方说明确认）。
- 聚宽研究平台本身支持期货回测/模拟（含商品期货新品种）。
- **优点**：数据质量好、含持仓量、支持主连合约、文档清晰。
- **缺点**：正式批量取数需要付费（有免费试用额度）。
- **链接**：
  - 聚宽官网：https://www.joinquant.com
  - JQData 使用说明：https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9979

### 1.6 恒有数 UData（恒生电子，免费注册）——macOS 免费多年分钟首选

- **类型**：恒生电子出品的免费金融数据社区/云端数据服务，号称"用数自由"；Python SDK 为纯 Python 包 `hs_udata`（`py3-none-any`），**macOS / Linux / Windows 通用**。
- **期货分钟数据**：`fut_quote_minute(en_prod_code, begin_date, end_date, fields)` —— 期货盘后 1 分钟切片（交易日 16:30 后提供），字段含 `date, time, open, high, low, close, turnover_volume, turnover_value, change, change_pct, amount(持仓量)`。
- **合约列表**：`fut_list()` 返回 5 大期货交易所（上期所/大商所/郑商所/中金所/上期能源）全部在册合约代码。
- **费用**：免费注册 + 免费体验套餐，官方宣传"不限次、不限量"；历史数据覆盖多年（宣传含 30 年）。
- **优点**：免费、跨平台（macOS 可用）、Python 接口、含持仓量、盘后数据适合因子挖掘与回测。
- **缺点**：K 线为盘后更新（非实时，当日盘中数据取不到）；单次下载上限约 10000 根，多年数据需分次拉取；官网 `udata.hs.net` 海外访问可能超时（国内正常）；免费套餐对期货分钟的具体覆盖范围以注册后实际权限为准。
- **用法**：
  ```bash
  pip install hs_udata
  UDATA_TOKEN=你的token python -u test/data/test_udata.py
  ```
  ```python
  import hs_udata as hs
  hs.set_token(token="...")            # 总览页获取 Token
  df = hs.fut_quote_minute(en_prod_code="RB2610.SHF",
                           begin_date="2021-08-01", end_date="2026-08-17")
  ```
- **链接**：
  - 恒有数官网：https://udata.hs.net
  - 订阅/免费套餐：https://udata.hs.net/subscribe
  - vnpy 适配仓库：https://github.com/vnpy/vnpy_udata
  - vnpy 社区介绍文章：https://www.vnpy.com/forum/topic/7995

### 1.7 新浪 / 腾讯 / 东方财富 网页接口（非官方，免费）

- 以上三家行情网站本身有期货分钟/分时数据接口，被 AkShare 等库封装，也可直接 HTTP 调用。
  - 新浪期货中心：https://finance.sina.com.cn/futures/
  - 东方财富期货频道：https://futures.eastmoney.com/
  - 腾讯证券（期货行情）：https://gu.qq.com/
- **注意**：这些接口无官方文档、字段不稳定、可能缺持仓量、有反爬风险，**不建议作为回测主数据源**。

---

## 2. 低成本付费数据源

### 2.1 米筐 RQData

- **类型**：米筐科技云端数据服务，Python SDK，覆盖股票/期货/期权/基金。
- **期货数据周期**（vnpy 官方文档确认）：日线、小时线、分钟线、**TICK（实时更新）**。
- **费用**：付费制，有免费试用；具体价格需询价/登录查看（定价页为 SPA，价格动态展示）。
- **vnpy 生态**：有官方适配接口。
- **链接**：
  - RQData 介绍页：https://m.ricequant.com/welcome/rq-data
  - 产品定价页：https://rqopen.ricequant.com/welcome/pricing
  - 米筐官网：https://www.ricequant.com
  - vnpy 适配仓库：https://github.com/vnpy/vnpy_rqdata

### 2.2 恒有数 UData（恒生电子）

- **类型**：恒生电子推出的云端数据服务，宣称不限次、不限量。
- **数据周期**（vnpy 官方文档确认）：**分钟线（盘后更新）**，覆盖股票、期货。
- **链接**：
  - vnpy 适配仓库：https://github.com/vnpy/vnpy_udata
  - 官网：https://udata.hs.net/home （注：调研时海外网络访问失败，国内网络请自行验证）

### 2.3 迅投研 / QMT（睿智融科）

- **类型**：迅投（睿智融科）推出的量化数据服务与交易终端。
- **数据周期**（vnpy 官方文档确认）：日线、小时线、分钟线、**TICK（实时更新）**，覆盖股票/期货/期权/基金。
- **接入方式**：QMT/miniQMT 终端通常通过券商免费开通（一般有资金门槛，如 50 万）；迅投研专业数据服务付费。
- **链接**：
  - vnpy 适配仓库：https://github.com/vnpy/vnpy_xt
  - 迅投 QMT 社区：https://www.xuntou.net
  - 迅投研注册（vnpy 文档给出）：https://xuntou.net/#/signup

### 2.4 文华财经

- **类型**：国内期货行情软件龙头（WH6 / WH9 / WT9 等），提供分钟数据下载，行情为付费制。
- **链接**：
  - 官网：https://www.wenhua.com.cn
  - WH6 赢顺：https://wh6.wenhua.com.cn
  - WT9 量化交易终端：https://wt9.wenhua.com.cn

---

## 3. 机构级付费数据源（参考）

| 数据源 | 说明 | 链接 |
|---|---|---|
| **Wind** | 金融机构标配，分钟线实时更新 | https://www.wind.com.cn/newsite/wft.html |
| **同花顺 iFinD** | 面向机构，分钟线实时更新 | http://www.51ifind.com/ |
| **东方财富 Choice** | 付费终端，含期货分钟 | http://choice.eastmoney.com/ |
| **天软 TinySoft** | 券商研报常用，分钟线 | http://www.tinysoft.com.cn/ |

> 以上四个数据源的期货分钟能力均出自 vnpy 官方「数据服务」文档：https://www.vnpy.com/docs/en/community/info/datafeed.html

---

## 4. Tick / 高频数据商（如需要更细粒度）

- **财富通数据中心**：提供商品期货 tick / 分笔 / 五档 level2 历史数据（按年/月售卖）。
  - 官网：https://caifushuju.cn
  - 示例：https://www.caifushuju.cn/goods.php?id=199（中金所 Level2 高频历史数据）
- 其他 CSDN/付费下载渠道（如 CMES 金融数据库）提供的期货分钟/五档数据，质量需自行甄别。

---

## 5. 与本项目对接的参考建议

1. **数据入库**：分钟数据量远大于日频，建议只保存**主力连续合约**的分钟线，MongoDB 中按「品种 + 日期」分区存储，字段统一为 `open/high/low/close/volume/amount/open_interest`。
2. **主力/连续合约**：不同数据源的主连标识不同（Tushare 需 mapping；EDB 用 `KQ.m@交易所.品种`；掘金用大写品种代码如 `SHFE.HC`），需要建立统一的品种映射表。
3. **交易时段**：国内商品期货含夜盘（21:00–次日 02:30），分钟数据需保留完整时间戳，避免与日线拼接错位。
4. **免费源校验**：网页类接口（AkShare/新浪）数据可能缺行或字段不稳，入库前做完整性校验；多源交叉验证（如 EDB 与掘金对比收盘价）。
5. **Python 接入**：优先选有 Python SDK 的数据源（Tushare/TqSdk/JQData/RQData/掘金），与现有 FastAPI 后端集成成本最低。

---

## 6. 附：本次调研实际验证过的链接清单

以下链接均在 2026-08-17 通过直接访问或官方文档交叉确认存在（HTTP 200 或来自 vnpy 官方文档）：

- Tushare 期货历史分钟 `ft_mins`：http://tushare.pro/document/2?doc_id=313
- Tushare 期货实时分钟 `rt_fut_min`：http://tushare.pro/document/2?doc_id=340
- 天勤 EDB 行情历史服务：https://doc.shinnytech.com/edb/latest/md_server.html
- 天勤 TqSdk GitHub：https://github.com/shinnytech/tqsdk-python
- AkShare 期货数据文档：https://akshare.akfamily.xyz/data/futures/futures.html
- AkShare GitHub：https://github.com/akfamily/akshare
- 掘金量化数据 FAQ：https://myquant.cn/docs2/faq/数据问题.html
- 掘金量化官网：https://www.myquant.cn
- 聚宽官网：https://www.joinquant.com
- 迅投 QMT 社区：https://www.xuntou.net
- 文华财经官网：https://www.wenhua.com.cn
- 财富通数据中心：https://caifushuju.cn
- 米筐 RQData 定价页：https://rqopen.ricequant.com/welcome/pricing
- vnpy 官方数据服务文档：https://www.vnpy.com/docs/en/community/info/datafeed.html
- 恒有数 UData 官网：https://udata.hs.net （本次从海外网络未能直接访问，国内网络正常；请自行验证）
- 恒有数 UData 免费套餐订阅页：https://udata.hs.net/subscribe
- vnpy 社区 UData 介绍文章：https://www.vnpy.com/forum/topic/7995 （已验证）
- vnpy_udata 适配仓库：https://github.com/vnpy/vnpy_udata （已验证）
- hs_udata Python 包：https://pypi.org/project/hs-udata/ （已验证可 pip 安装，含 `fut_quote_minute` 接口）
