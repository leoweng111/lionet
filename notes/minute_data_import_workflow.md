# 期货分钟频率数据导入指南（通用）

> 从聚宽(JoinQuant)导出分钟数据 → 导入 MongoDB → 校验与日频一致性 → 缺失夜盘补全，完整流程。
> 以玉米 C0 为例，适用于任意品种（只需替换品种代码与路径）。
> 涉及脚本均位于 `test/data/`；数据库 `futures`；分钟 collection `continuous_contract_price_1min`，日频 collection `continuous_contract_price_daily`。

---

## 1. 总体流程

```
聚宽研究环境导出 CSV            test/data/joinquant_fut_min_export.py
   ↓
本地 CSV（raw 分钟，未后复权，换月处跳空）
   ↓ 导入（复用日频换月安排）    test/data/import_c0_1min_to_db.py
MongoDB futures.continuous_contract_price_1min
   ↓ 验证一致性                   test/data/verify_daily_vs_1min.py
发现缺夜盘？ → 识别+补全         find_missing_night_days.py / joinquant_fix_missing_night.py / check_duplicate_1min.py
   ↓
最终验证 ✅
```

---

## 2. 数据语义与口径（必须先理解）

### 2.1 聚宽 9999 是什么

- `C9999.XDCE`（9999）= **聚宽自己按主力规则选出的主力合约的原始价格**拼接，**未做后复权，换月处有跳空**。
- `C8888.XDCE`（8888）= 指数（按持仓量加权），更平滑。
- 聚宽主力切换日由聚宽规则决定，**与你本地日频库的换月日不一定相同**（见第 4.2 节）。

### 2.2 数据库 time 的存储口径（重要，易混淆）

- **数据库 `time` 存的是聚宽导出的原始时间戳**：夜盘 bar 的 `time` 就是当晚 21:00–23:00，**不做"归交易日"处理**，与 CSV 的 `datetime` 完全一致。
- "夜盘归下一个交易日"**只是计算口径**，发生在两处且不改变存储的 `time`：
  1. 导入时 `assign_trading_day()` 算 `td`，**仅用于把日频库的 factor/symbol 对到正确的天**；
  2. 验证时按交易日重新聚合分钟数据。
- 所以**查数据库按 `time=某日期` 是"日历日"口径；verify 按 `td=某日期` 是"交易日"口径**（= 前一交易日夜盘 + 当日日盘）。两者指的不是同一批 bar，对比时务必区分。

### 2.3 交易日归属规则（关键）

- **日盘**（09:00–15:00，小时 < 20）归当天；
- **夜盘**（21:00 之后，小时 ≥ 20）归**下一个交易日**；
- **周五夜盘归下周一**（不能简单 +1 天，否则会归到周六）；
- 归属必须基于**交易日列表**（`assign_trading_day(datetime_series, trading_days)`），交易日列表默认从"分钟数据日盘日期"推断，也可显式传入（如日频库 ∪ 分钟数据）。

---

## 3. 第一步：聚宽研究环境导出分钟数据

脚本：`test/data/joinquant_fut_min_export.py`

- 在聚宽研究环境（https://www.joinquant.com → 研究环境）新建 notebook，把脚本内容粘贴进一个单元格运行（或上传后 `%run`）。
- 配置：`FREQ`（1m/5m/…）、`START_DATE`、`END_DATE`、`FUTURE_CODES`。
- 主力连续代码格式：`品种代码 + 9999 + 交易所后缀`。各交易所后缀：上期所 `.XSGE`、大商所 `.XDCE`、郑商所 `.XZCE`、能源 `.XINE`、广期所 `.XGFEX`。
- 运行后从研究环境左侧文件树 `data/fut_min/` 下载 CSV。

### CSV 格式

```
datetime,code,open,high,low,close,volume,money,open_interest
2020-01-02 09:01:00,C9999.XDCE,1911.0,1914.0,1910.0,1912.0,9863.0,188589160.0,720926.0
```

字段：`datetime`（分钟时间戳，为时间段结束）、`code`、`open/high/low/close`、`volume`（手）、`money`（元）、`open_interest`（持仓量·手）。

> 聚宽环境里 `get_price` 是**全局函数**，直接调用；**不要** `from jqdata import get_price`（会 `ImportError`）。

---

## 4. 第二步：确认日频库换月安排

分钟数据的换月因子完全复用日频库，导入前确认：

- MongoDB 已启动可连；
- `futures.continuous_contract_price_daily` 已包含该品种日频数据（覆盖分钟日期范围），由 `data/futures.py` 的 `update_futures_continuous_contract_price()` 生成。

### 4.1 日频换月日是怎么算的

核心在 `data/futures.py` 的 `_build_roll_adjusted_continuous_from_panel()`：

1. 每天选**成交量最大**的合约为主力（持仓量仅诊断，不一致打 `[DominantMismatch]` 警告，仍以成交量为准）；
2. **防未来函数**：`dominant_used = dominant_today.shift(1)`——第 t 天用的合约是第 t-1 天的主力；
3. **换月日**：当天用合约 ≠ 前一天用合约 → `is_rollover=True`（因 shift(1)，比"成交量主力易主"晚一天）；
4. **换月比例**：`cur_ratio = old_open / new_open`，`weighted_factor` 自 `RESEARCH_START_DATE`（2020-01-01）累乘；
5. 库中存**原始价**，后复权价 = `原始价 × weighted_factor`。

### 4.2 聚宽换月日 vs 日频换月日（重要）

- 两者**切换日不同**，导致在"换月窗口"（切换点前后若干交易日）内两数据源指向**不同合约**：
  - OHLC 差异可达合约间价差（玉米几十到上百元）；
  - volume/position 差异可达数倍（不同合约成交量/持仓量不同）。
- **聚宽侧换月日**的识别信号：① 隔夜价格跳空 `|gap| > 1%`；② **持仓量单日大幅跳变** `|oi_chg| > 15%`（价格接近但持仓量跳变，如 2026-04-15 持仓 +44%）。
- **过渡期差异**：聚宽先切、日频后切（或反之），中间一段两者指向不同合约；此时价格接近（gap<1%）但 volume/position 差异大，靠跳变检测识别不出来。

---

## 5. 第三步：导入分钟数据到 MongoDB

脚本：`test/data/import_c0_1min_to_db.py`

```bash
cd /Users/wenglongao/work_repo/lionet
python -u test/data/import_c0_1min_to_db.py --preview   # 预览，不写库
python -u test/data/import_c0_1min_to_db.py             # 正式导入
python -u test/data/import_c0_1min_to_db.py --csv 路径  # 指定 CSV
```

预览重点：
- `symbol` 应为真实合约（如 `C2607`），不是 `UNKNOWN`（`UNKNOWN`=日频库没读到，先别写）；
- 无 `[警告] 分钟数据有 N 个交易日不在日频换月安排中`；
- `换月点分钟数` 合理。

关键行为：
- 交易日归属以「日频库交易日 ∪ 分钟数据」为基准；
- 字段：`instrument_id`、`time`(分钟)、`open/high/low/close`、`settle=close`、`volume`、`position`(=open_interest)、`money`、`symbol`、`weighted_factor`、`cur_weighted_factor`、`is_rollover`；
- `is_rollover` 默认只标换月日**第一根分钟 bar**；
- **分批写入**：每批 `BATCH_SIZE=5000`，Ctrl+C 可安全中断（打印进度，退出码 130）；
- **幂等**：按 `time + instrument_id` upsert，可重复运行覆盖。

文档示例：

```json
{ "time": "2020-01-02 09:01:00", "instrument_id": "C0", "symbol": "C2001",
  "open": 1911.0, "high": 1914.0, "low": 1910.0, "close": 1912.0, "settle": 1912.0,
  "volume": 9863.0, "position": 720926.0, "money": 188589160.0,
  "weighted_factor": 1.0, "cur_weighted_factor": 1.0, "is_rollover": false }
```

---

## 6. 第四步：验证日频与分钟一致性

脚本：`test/data/verify_daily_vs_1min.py`

```bash
python -u test/data/verify_daily_vs_1min.py                  # 全量
python -u test/data/verify_daily_vs_1min.py --sample 60      # 随机抽样
python -u test/data/verify_daily_vs_1min.py --atol 0.02      # 价格阈值(元)
python -u test/data/verify_daily_vs_1min.py --roll-window 5  # 换月窗口天数
python -u test/data/verify_daily_vs_1min.py --no-roll-window # 不打印 [换月窗口,预期差异] 的日子, 只打印非换月窗口的日子
```

### 验证内容

- OHLC：分钟按交易日聚合 vs 日频，报告匹配率 + 差异分布（中位/P90/P99/最大）；
- volume（分钟求和）、position（收盘持仓）；
- `is_rollover`、`weighted_factor`、`cur_weighted_factor`、`symbol`；
- 每天 bar 数是否正常（玉米 225~345；低于阈值视为缺夜盘）。

### 输出里三类差异如何区分

1. **`[换月窗口,预期差异]`**：该日落在任一换月日 ± N 天（默认 5）。换月日 = 日频侧 `is_rollover` ∪ 分钟侧（gap>1% 或 oi_chg>15%）。属**两数据源换月规则不同**，预期差异，不算错。
2. **`[数据缺失:bar数=N]`**：该交易日 bar 数 < 300（缺夜盘/数据不完整），vol 偏低。需补数据（见第 7 节）。
3. **无标记**：真不一致，需排查。

若只想看"真不一致"，用 `--no-roll-window` 跳过换月窗口日子（默认打印全部）；统计行仍反映全量（换月窗口/非换月各多少天）。

### 一个重要修复：交易日列表用并集

若 verify 里某天分钟 `volume` 被异常放大（如 491113 → 1232761），常因**日频库缺了某些交易日**，导致仅用日频库 `trading_days` 时夜盘 `searchsorted` 归属错位、某天被塞入过多 bar。修复：verify 用「日频库 ∪ 分钟数据日盘日期」的并集，并打印 `[提示] 分钟数据有 N 个交易日不在日频库`。

---

## 7. 缺夜盘数据的识别与补全

### 7.1 识别缺夜盘的交易日

脚本：`test/data/find_missing_night_days.py`

```bash
python -u test/data/find_missing_night_days.py --csv 你的CSV
```

- 正常交易日 bar 数约 345（夜盘 120 + 日盘 225）；`bar < 340` 视为缺夜盘。
- 会自动排除两类**正常**情况，不算缺失：
  - **节假日正常**：前一交易日是节前最后交易日（法定节假日前夜盘暂停），当晚无夜盘，用 `chinese_calendar` 判断；
  - **CSV 首日**（导出起点，前一夜盘属于更早日期）。
- 输出真缺失交易日 + 每个交易日完整的拉取范围（前一交易日 20:59 ~ 当日 15:00），并生成可粘贴到补数脚本的 `MISSING_RANGES`。

### 7.2 聚宽补拉缺失交易日

脚本：`test/data/joinquant_fix_missing_night.py`（聚宽研究环境运行）

- 把 7.1 生成的 `MISSING_RANGES` 粘贴进配置区；
- 对每个缺失交易日拉 `[前一交易日 20:59 ~ 当日 15:00]` 完整数据（夜盘+日盘）；
- 结果合并去重后存为 `data/fix_night/C9999.XDCE_fix_night.csv`，列格式与原始 CSV 一致。

> 兼容性注意：聚宽 `get_price` 返回值在单标的/老 pandas 下索引可能只有 `time` 一层，`reset_index()` 后需判断是否有 `code` 列（否则补常量 `code`），并对缺失字段补 `NaN`——否则会 `KeyError`。

### 7.3 导入补数据

```bash
python -u test/data/import_c0_1min_to_db.py --csv /path/to/C9999.XDCE_fix_night.csv
```

复用导入脚本，按 `time+instrument_id` upsert，只覆盖这些交易日，其它天不动。

### 7.4 查重与逐根对比

- `test/data/check_duplicate_1min.py`：检查分钟库是否有重复 `(time, instrument_id)`（重复会导致 volume 被重复累加、open 取到重复行）。`--date 某日` 可看单日。
- `test/data/compare_db_vs_csv_1min.py --date 某交易日`：把数据库分钟库与 CSV 在指定**交易日**逐根对比，输出缺失/多余/值不一致的 bar，用于定位"数据库 vs CSV 到底哪里不同"。

---

## 8. 运维：停止导入 / MongoDB 启停

### 为什么 Ctrl+C 后数据还在涨

`bulk_write` 提交给 MongoDB 后，**服务端收到命令会完整执行**；Ctrl+C 只中断客户端，**不能回滚已接收的写入**。分批写入可把影响控制在当前一批，重跑 upsert 覆盖补齐。

### 一键停止

```bash
python -u test/data/stop_c0_import.py
```

- `pkill` 终止导入进程；`currentOp` + `killOp` 终止 MongoDB 端针对该 collection 的写操作；3 秒后二次采样判断是否停。

### MongoDB 服务启停（macOS）

- 停止：`brew services stop mongodb-community` 或 `sudo kill <mongod PID>`；
- 启动：`brew services start mongodb-community`，或 `mongod --config /opt/homebrew/etc/mongod.conf --fork`；
- 启动失败：去掉 `--fork` 前台看报错；常见为数据目录权限：`sudo chown -R $(whoami) /opt/homebrew/var/mongodb`；
- 验证：`mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'`。

---

## 9. 完整重导流程（推荐）

```bash
cd /Users/wenglongao/work_repo/lionet

# 1) 确认 Mongo 已启动
mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'

# 2) 清空旧分钟数据
python -c "from mongo.mongify import delete_data; delete_data('futures','continuous_contract_price_1min',{'instrument_id':'C0'})"

# 3) 预览
python -u test/data/import_c0_1min_to_db.py --preview

# 4) 正式导入（分批，可中断）
python -u test/data/import_c0_1min_to_db.py

# 5) 验证
python -u test/data/verify_daily_vs_1min.py
# 若报 [数据缺失] → 走第 7 节补夜盘
```

---

## 10. 常见错误与排查（完整清单）

| 现象 | 原因 | 解决 |
|---|---|---|
| 聚宽研究环境 `from jqdata import get_price` 报 `ImportError` | 数据函数是全局预置，`jqdata` 不导出 | 直接调用 `get_price(...)` |
| macOS 无法 `pip install gm` | `gm` 无 macOS 轮子，终端仅支持 Windows | 用聚宽/天勤/akshare 等跨平台源 |
| 分钟数据出现周六伪交易日、bar 只有 225 | 旧版归属把周五夜盘归周六 | 用修复后 `assign_trading_day`（基于交易日列表）重导 |
| Ctrl+C 后数据仍在涨 | 服务端继续执行已提交 bulk | `stop_c0_import.py` 杀进程 + killOp |
| mongosh/Compass `ECONNREFUSED` | mongod 未启动 | 按第 8 节重启 |
| 预览 `symbol=UNKNOWN` | 日频库没读到，触发 gap 回退 | 先确认 Mongo/日频库有数据 |
| 验证报 `[换月窗口,预期差异]` | 两数据源换月日不同 | 预期差异；若要消除需统一换月口径（见第 11 节） |
| 验证报 `[数据缺失:bar数=225]` | 该交易日缺夜盘 | 补数据（第 7 节）；若是节假日正常则无需补 |
| 验证某天分钟 volume 异常放大（如 491113→1232761） | 日频库缺交易日导致夜盘归属错位 | 用并集交易日列表修复 verify，并补日频库缺的天 |
| 聚宽补数脚本 `KeyError: code` | `get_price` 返回单层索引，`code` 列缺失 | 判断无 `code` 则插入常量，缺失字段补 NaN |
| 查库 `time=某日` 与 verify `td=某日` 对不上 | 日历日 vs 交易日口径不同 | 理解 2.2 节，用 `compare_db_vs_csv_1min.py` 按交易日对比 |

---

## 11. 换月口径对策略研究的影响

- **夜盘缺失（如 225 根的日子）**：
  - 影响夜盘相关因子（夜盘动量、隔夜跳空、开盘 gap、夜盘成交量占比）和日频聚合因子（high/low/volume）；
  - 对纯日盘策略影响较小（但日频聚合仍会因 high/low/volume 缺失失真）；
  - 处理：补齐（第 7 节）；补不齐则建议在分钟库加 `is_night_missing` 标记并过滤。
- **换月日不一致（日频 vs 聚宽）**：
  - 分钟研究**独立用聚宽口径**（聚宽换月日 + 聚宽后复权）：内部自洽，可独立做分钟策略；
  - 日频、分钟**混用/对齐**（组合因子、跨频率验证）：必须统一换月日，最好日频也改用聚宽 9999（同源同规则）。

### 分钟数据的换月口径（务必区分）

1. **历史全量导入**（`import_c0_1min_to_db.py`）：**优先使用 CSV 中由聚宽 `get_dominant_future` 导出的换月字段**（聚宽官方口径，最准）；若 CSV 无这些字段则回退到"分钟数据自身 gap 检测"。
2. **分钟增量更新**（`data.futures.update_futures_continuous_contract_price_1min`，天勤 EDB）：**基于分钟数据自身检测换月日**（隔夜跳空 + 持仓量跳变），自己算 `weighted_factor` 链；天勤无聚宽接口，故用此近似。

> 聚宽口径是官方主力切换日（最准）；天勤口径是自洽近似。两者写入同一分钟库但 source 不同（joinquant / tqsdk_edb），可并存；下游按 source 过滤。

---

## 12. 脚本清单

| 脚本 | 作用 |
|---|---|
| `test/data/joinquant_fut_min_export.py` | 聚宽研究环境导出主力连续分钟 CSV（用 `get_dominant_future` 计算换月，CSV 含 `symbol/is_rollover/weighted_factor/cur_weighted_factor`） |
| `test/data/joinquant_fix_missing_night.py` | 聚宽环境补拉缺夜盘交易日的完整数据（同样带换月字段） |
| `test/data/import_c0_1min_to_db.py` | 合并两个 CSV → **优先用 CSV 换月字段** → `source='joinquant'` → 覆盖更新分钟库 |
| `test/data/backfill_daily_source.py` | 给日频库已有记录回填 `source='akshare'` |
| `test/data/aggregate_minute_to_daily.py` | 把 joinquant 分钟聚合为日频写入日频库（不覆盖 akshare） |
| `test/data/verify_daily_vs_1min.py` | 验证日频 vs 分钟一致性（区分换月窗口/数据缺失/真不一致） |
| `test/data/find_missing_night_days.py` | 找出缺夜盘交易日（排除节假日正常、CSV 首日）并生成补数列表 |
| `test/data/check_duplicate_1min.py` | 检查分钟库重复 bar |
| `test/data/compare_db_vs_csv_1min.py` | 按交易日逐根对比数据库与 CSV |
| `test/data/stop_c0_import.py` | 一键停止导入（杀进程 + killOp） |
| `data/futures.py` | 日频构建 + 分钟(天勤)/聚合日频等更新逻辑 |
| `mongo/mongify.py` | MongoDB 读写封装（get_data/update_data/delete_data） |

---

## 13. 导入其它品种的步骤

1. **确认日频库**已包含该品种（跑 `data/futures.py` 的 `update_futures_continuous_contract_price(instrument_id='RB0')` 等）；
2. **聚宽导出**：修改 `joinquant_fut_min_export.py` 的 `FUTURE_CODES`（如 `RB9999.XSGE`）与日期，导出并下载 CSV；
3. **预览导入**：修改 `import_c0_1min_to_db.py` 的 `CSV_PATH`、`INSTRUMENT_ID`（如 `RB0`），先 `--preview` 确认 `symbol` 正常、无大量"不在日频换月安排"警告；
4. **正式导入**；
5. **验证**：`verify_daily_vs_1min.py --instrument RB0`；
6. **若有缺夜盘**：`find_missing_night_days.py --csv 该品种CSV` → 聚宽补拉 → `import --csv 补数文件` → 再验证；
7. **若日频库缺交易日**：先补日频数据，再重跑验证。

> 关键提醒：每种品种的**夜盘时段**（如 21:00–23:00、或 21:00–次日 01:00）不同，但"夜盘归下一交易日"规则一致；`assign_trading_day` 按小时 ≥ 20 判定夜盘，对多数品种适用（若有品种夜盘早于 20:00 或特殊时段，需调整阈值）。

---

## 14. 数据源(source)管理与运行顺序指南

### 14.1 各库的 source 字段

| 库 | 来源 | 标识 | 说明 |
|---|---|---|---|
| 分钟 `continuous_contract_price_1min` | 聚宽（多年） | `joinquant` | 全量历史，截止 2026-08-17 |
| 分钟 `continuous_contract_price_1min` | 天勤 EDB | `tqsdk_edb` | 近期（免费近1年分钟线） |
| 日频 `continuous_contract_price_daily` | akshare 接口 | `akshare` | 原日频更新 |
| 日频 `continuous_contract_price_daily` | 分钟聚合 | `joinquant` | 由 joinquant 分钟聚合而来 |

> 分钟/日频库的唯一键都**含 source**，因此同一 (time, instrument_id) 可能同时存在 akshare 与 joinquant 两条日频、或 joinquant 与 tqsdk_edb 两条分钟。**下游读取/回测时如需单一口径，请按 source 过滤。**

### 14.2 完整操作步骤

#### A. 首次全量初始化（一次性，用聚宽历史数据）

```bash
cd /Users/wenglongao/work_repo/lionet

# 0) 确认 Mongo 已启动
mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'

# 1) 给日频库已有数据回填 source='akshare'（只补缺 source 的记录）
python -u test/data/backfill_daily_source.py

# 2) 聚宽研究环境运行导出脚本（用 get_dominant_future 计算换月, CSV 带换月字段）：
#    test/data/joinquant_fut_min_export.py   -> 导出主 CSV
#    下载到本地: /Users/wenglongao/Downloads/C9999.XDCE.csv

# 3) 本地根据主 CSV 找出缺夜盘交易日, 生成补数列表(基于 CSV, 不需等落库):
python -u test/data/find_missing_night_days.py --csv /Users/wenglongao/Downloads/C9999.XDCE.csv
#    -> 把输出的 MISSING_RANGES 整段粘贴到 joinquant_fix_missing_night.py 配置区

# 4) 聚宽研究环境运行补夜盘脚本(同样带换月字段):
#    test/data/joinquant_fix_missing_night.py
#    下载到本地: /Users/wenglongao/Downloads/C9999.XDCE_fix_night.csv

# 5) 全量导入聚宽分钟（自动合并主CSV+补夜盘CSV, source=joinquant, 覆盖该品种旧分钟数据）
python -u test/data/import_c0_1min_to_db.py

# 6) 把 joinquant 分钟聚合为日频写入日频库（source=joinquant，不覆盖 akshare）
python -u test/data/aggregate_minute_to_daily.py --instrument C0

# 7) 验证（会区分换月窗口/数据缺失/真不一致）
python -u test/data/verify_daily_vs_1min.py
```

#### B. 日常增量更新

- **分钟频（近期数据）**：前端「行情数据 → 分钟频价格数据更新」，来源选 `tqsdk_edb`（天勤），默认取最近 90 天；等价后端 `POST /api/market-data/update-price-1min`。
- **日频**：前端「行情数据 → 日频价格数据更新」，来源二选一：
  - `akshare`：走 akshare 接口更新日频（source='akshare'）；
  - `joinquant`：从分钟库聚合 joinquant 日频（source='joinquant'，不覆盖 akshare），对"无分钟数据/缺夜盘/节假日"输出 warning 并在前端日志显示。
- 手动与定时自动更新都支持选择来源。

#### C. 前端功能位置

- 行情数据 → **日频价格数据更新**：手动 + 自动，都有「来源」下拉框（akshare / joinquant）
- 行情数据 → **分钟频价格数据更新**：「来源」下拉框（tqsdk_edb，预留扩展）
- 数据概览：**日频 / 分钟频** 两个子页
- 数据查看：**频率**下拉（日频/分钟频），分钟频支持 K 线

### 14.3 代码实现要点

- **聚宽换月日（官方口径）**：`joinquant_fut_min_export.py` / `joinquant_fix_missing_night.py` 使用聚宽 `get_dominant_future` 获取每个交易日的主力合约，主力切换日即换月日，并用换月日新旧合约开盘价计算 `cur_ratio`，`weighted_factor` 累乘。**CSV 中直接导出 `symbol / is_rollover / weighted_factor / cur_weighted_factor`**。
  - 用法注意：部分聚宽环境支持批量 `get_dominant_future(root, 开始, end_date=结束)`（`end_date` 是关键字参数）；**但研究环境部分版本不支持 `end_date`**，此时脚本会自动回退为**逐交易日查询** `get_dominant_future(root, 日期)`。交易日从已拉取的分钟数据日盘自动推断，无需额外接口。
- `test/data/import_c0_1min_to_db.py`：合并两个 CSV → **优先使用 CSV 中已有的换月字段**；若 CSV 无这些字段则回退到"分钟数据自身检测换月(gap+持仓量跳变)" → `source='joinquant'` → 覆盖更新分钟库。
- `data/futures.py`：
  - `update_futures_continuous_contract_price`（日频 akshare）写库前加 `source='akshare'`，唯一键含 source；
  - `update_futures_continuous_contract_price_1min`（天勤 EDB）：**天勤无聚宽接口，保留基于分钟数据自身检测换月日**（gap + 持仓量跳变），`source='tqsdk_edb'`；
  - `update_futures_continuous_contract_price_from_minute`：把分钟库 joinquant 聚合成日频写入日频库（不覆盖 akshare），并输出缺失/缺夜盘/节假日的 warning；
  - `aggregate_minute_to_daily_df` / `assign_trading_day_1min` / `detect_rollover_from_minute_df` / `build_minute_continuous_df_from_edb` 均在 data/futures.py 内实现，**不再从 test 文件夹 import**。
- **换月口径小结**：聚宽分钟数据用**聚宽官方换月日**（最准）；天勤近期数据用**分钟数据自身 gap 检测**（天勤无聚宽接口）。两者都是自洽的，但主力规则可能不同，跨 source 混用时注意。

### 14.4 注意事项

- joinquant 全量导入按 `(time, instrument_id)` upsert，会覆盖该品种旧的分钟记录（含历史无 source 的），但不会删除/覆盖 `tqsdk_edb` 独有的时间段（如 08-18 以后）。
- 天勤分钟更新按 `(time, instrument_id, source)` upsert，与 joinquant 分钟并存。
- 日频库 joinquant 与 akshare 并存；若某天 joinquant 分钟缺失/缺夜盘，`update_futures_continuous_contract_price_from_minute` 会输出 `[min2daily] ... 无分钟数据 / 疑似缺夜盘 / 节假日正常` 的 warning，前端日志面板会展示。

---

## 附：相关调研

数据源选择与对比见 `notes/minute_data.md`（免费/付费分钟数据源、天勤 EDB、恒有数 UData、聚宽等）。
