# 期货分钟数据导入全流程（以玉米 C0 为例）

> 记录从聚宽(JoinQuant)手动导出分钟数据，到导入 MongoDB，再到验证与日频数据一致性的完整流程。
> 涉及脚本均位于 `test/data/`，数据库为 `futures`，collection 为 `continuous_contract_price_1min`（分钟）与 `continuous_contract_price_daily`（日频）。

---

## 1. 整体流程

```
聚宽研究环境导出 CSV
   ↓  test/data/joinquant_fut_min_export.py
本地 CSV（raw 分钟，未后复权，换月处有跳空）
   ↓  test/data/import_c0_1min_to_db.py
MongoDB futures.continuous_contract_price_1min
（复用日频库换月安排，weighted_factor 与日频一致）
   ↓  test/data/verify_daily_vs_1min.py
验证：OHLC / volume / position / factor / is_rollover / symbol 与日频完全一致
```

---

## 2. 数据语义（重要）

聚宽 `9999` 主力连续合约（如 `C9999.XDCE`）的分钟数据是**「换月拼接但未做后复权」的原始价格序列**：

- 主力合约切换时会直接切换到新合约的原始价格，因此**换月点存在价格跳空**；
- 不是后复权数据，不能直接用于连续序列的因子计算，必须做换月调整。

换月调整方案：**复用日频库 `continuous_contract_price_daily` 中已经算好的换月安排**（`symbol`、`weighted_factor`、`cur_weighted_factor`、`is_rollover`），把每分钟 bar 按其「交易日」归到对应日频行，从而保证分钟数据与日频数据在换月调整上**完全一致**。

### 交易日归属规则（关键）

中国商品期货：
- **日盘**（09:00–15:00）属于当天；
- **夜盘**（21:00 之后，小时 ≥ 20）属于**下一个交易日**；
- **周五夜盘属于下周一**（不能简单 +1 天，否则会归到周六）。

导入脚本 `import_c0_1min_to_db.py` 中的 `assign_trading_day()` 基于交易日列表处理上述规则，并以**日频库的交易日列表为基准**来归属分钟数据。

---

## 3. 第一步：聚宽研究环境导出分钟数据

脚本：`test/data/joinquant_fut_min_export.py`

### 用法

在聚宽研究环境（https://www.joinquant.com → 研究环境）新建 notebook，把脚本内容粘贴进一个单元格，或上传后 `%run joinquant_fut_min_export.py`，修改顶部配置后运行。

```python
# 配置区
FREQ = "1m"                # 1m/5m/15m/30m/60m
START_DATE = "2021-08-01"   # 5 年起点
END_DATE   = "2026-08-17"
FUTURE_CODES = ["RB9999.XSGE", "C9999.XDCE", ...]   # 主力连续代码
```

运行结束后，到研究环境左侧文件树 `data/fut_min/` 目录下载 CSV 到本地。

### 主力连续合约代码格式

`品种代码 + 9999 + 交易所后缀`，例如：
- 上期所 `.XSGE`：`RB9999.XSGE`（螺纹）、`CU9999.XSGE`（铜）
- 大商所 `.XDCE`：`C9999.XDCE`（玉米）、`M9999.XDCE`（豆粕）
- 郑商所 `.XZCE`：`TA9999.XZCE`（PTA）
- 能源 `.XINE`：`SC9999.XINE`（原油）

脚本注释里已列全各交易所品种代码。

### 导出的 CSV 格式

```
datetime,code,open,high,low,close,volume,money,open_interest
2020-01-02 09:01:00,C9999.XDCE,1911.0,1914.0,1910.0,1912.0,9863.0,188589160.0,720926.0
...
```

字段：`datetime`（分钟时间戳，为时间段结束）、`code`、`open/high/low/close`、`volume`（成交量·手）、`money`（成交额·元）、`open_interest`（持仓量·手）。

---

## 4. 第二步：确认日频库换月安排已就绪

分钟数据的换月调整完全复用日频库，因此导入前需确认：

- MongoDB 已启动、能连上（`mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'`）；
- `futures.continuous_contract_price_daily` 中已有 `C0` 的日频数据（覆盖分钟数据的日期范围）；
- 日频换月安排由 `data/futures.py` 的 `build_roll_adjusted_continuous_contract_price()` / `update_futures_continuous_contract_price()` 生成。

### 日频换月日是怎么算出来的

核心在 `data/futures.py` 的 `_build_roll_adjusted_continuous_from_panel()`：

1. 每天从所有上市合约中选**成交量最大**的作为当日主力（持仓量仅用于诊断，若成交量最大与持仓量最大不一致会打 `[DominantMismatch]` 警告，仍以成交量为准）；
2. **防未来函数**：`dominant_used = dominant_today.shift(1)` —— 第 t 天实际使用的合约是「第 t-1 天的主力」，第一天用当天主力，缺失向前填充；
3. **换月日判定**：当天实际使用的合约 ≠ 前一天使用的合约，则 `is_rollover = True`（由于 shift(1)，通常比成交量主力真正易主晚一天）；
4. **换月比例**：`cur_ratio = old_open / new_open`（换月日旧合约开盘价 ÷ 新合约开盘价），`weighted_factor` 自 `RESEARCH_START_DATE`（2020-01-01）起累乘所有换月比例；
5. 库中存**原始价**，后复权价 = `原始价 × weighted_factor`。

---

## 5. 第三步：导入分钟数据到 MongoDB

脚本：`test/data/import_c0_1min_to_db.py`

### 用法

```bash
cd /Users/wenglongao/work_repo/lionet

# 先预览（不写库）
python -u test/data/import_c0_1min_to_db.py --preview

# 正式导入
python -u test/data/import_c0_1min_to_db.py
```

预览时重点确认：
- `symbol` 是**真实合约**（如 `C2607`），不是 `UNKNOWN`（若 `UNKNOWN` 说明日频库没读到，先别写）；
- 没有 `[警告] 分钟数据有 N 个交易日不在日频换月安排中`；
- `换月点分钟数` 合理（玉米约 20~30 个）。

### 关键行为

- **交易日归属**：以日频库交易日列表为基准，`assign_trading_day()` 把夜盘归到下一交易日（周五夜盘归周一）；
- **字段对齐日频**：`instrument_id="C0"`、`time` 精确到分钟、`open/high/low/close`、`settle=close`（分钟无结算价）、`volume`、`position`（=open_interest）、`money`、`symbol`、`weighted_factor`、`cur_weighted_factor`、`is_rollover`；
- **is_rollover**：默认只标记换月日「第一根分钟 bar」为 `True`（配置 `MARK_ROLLOVER_FIRST_MINUTE_ONLY` 可改为整日标记）；
- **换月因子**：每天所有分钟 bar 沿用该交易日日频行的 `weighted_factor`，与日频完全一致；
- **分批写入**：每批 5000 条（`BATCH_SIZE`），Ctrl+C 可安全中断，中断时打印已写入进度，退出码 130；
- **幂等**：按 `time + instrument_id` upsert，可重复运行、覆盖旧数据。

### 写入的文档示例

```json
{
  "time": "2020-01-02 09:01:00",
  "instrument_id": "C0",
  "symbol": "C2001",
  "open": 1911.0, "high": 1914.0, "low": 1910.0, "close": 1912.0,
  "settle": 1912.0,
  "volume": 9863.0, "position": 720926.0, "money": 188589160.0,
  "weighted_factor": 1.0, "cur_weighted_factor": 1.0, "is_rollover": false
}
```

---

## 6. 第四步：验证日频与分钟数据一致性

脚本：`test/data/verify_daily_vs_1min.py`

### 用法

```bash
cd /Users/wenglongao/work_repo/lionet
python -u test/data/verify_daily_vs_1min.py                 # 全量对比
python -u test/data/verify_daily_vs_1min.py --sample 60     # 随机抽 60 天
python -u test/data/verify_daily_vs_1min.py --atol 0.02     # 调整价格差异阈值(元)
```

### 验证内容

1. **OHLC**：把分钟数据按交易日聚合出每日 `open/high/low/close`，与日频逐日对比，报告匹配率与绝对差异分布（中位/P90/P99/最大）；
2. **volume / position**：分钟 `volume` 求和、`position`（收盘持仓，取每日最后一根）与日频对比；
3. **换月一致性**：每天 `is_rollover`、`weighted_factor`、`cur_weighted_factor`、`symbol` 是否与日频一致；
4. **bar 数检查**：每天分钟 bar 数是否正常（玉米约 225–345 根），异常说明数据缺失；
5. **settle 说明**：日频为真实结算价，分钟库取 `close`，预期不同（不计入"完全一致"判定）。

### 输出解读

- 每个字段输出：`匹配 N/M (xx%)` + 差异分布；
- 不匹配明细：逐日列出哪个字段两边各是多少；
- 结论行：`✅ 完全一致` 或 `❌ 存在不一致`。

> 只有验证输出 `✅` 后，才说明两个数据源真正对齐。

---

## 7. 运维：停止导入 / 中断处理

### 为什么 Ctrl+C 后数据还在涨

导入用 `bulk_write` 提交给 MongoDB 后，**服务端一旦收到命令会完整执行**；Ctrl+C 只能中断客户端等待/发送，**不能回滚服务端已接收并执行的写入**。分批写入（每批 5000）能把中断影响控制在当前一批，且重跑 upsert 可覆盖补齐。

### 一键停止

脚本：`test/data/stop_c0_import.py`

```bash
cd /Users/wenglongao/work_repo/lionet
python -u test/data/stop_c0_import.py
```

它会：
1. `pkill` 终止本机所有 `import_c0_1min_to_db` 相关进程；
2. 用 `currentOp` 找到 MongoDB 端正在对 `continuous_contract_price_1min` 执行的写操作并 `killOp`；
3. 3 秒后二次采样，判断条数是否已停止增长。

### MongoDB 服务的启停（macOS）

- 停止：`brew services stop mongodb-community`，或 `sudo kill <mongod PID>`；
- 启动：`brew services start mongodb-community`，或 `mongod --config /opt/homebrew/etc/mongod.conf --fork`；
- 启动失败排查：去掉 `--fork` 前台运行看报错；常见是数据目录权限问题，修复：`sudo chown -R $(whoami) /opt/homebrew/var/mongodb`；
- 验证：`mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'`。

---

## 8. 完整重导流程（推荐）

```bash
cd /Users/wenglongao/work_repo/lionet

# 1) 确认 Mongo 已启动
mongosh "mongodb://leo:密码@127.0.0.1:27017" --eval 'db.runCommand({ping:1})'

# 2) 清空旧分钟数据（含之前可能错误的半截数据）
python -c "
from mongo.mongify import delete_data
delete_data('futures', 'continuous_contract_price_1min', {'instrument_id': 'C0'})
"

# 3) 预览
python -u test/data/import_c0_1min_to_db.py --preview

# 4) 正式导入（分批，可 Ctrl+C 中断）
python -u test/data/import_c0_1min_to_db.py

# 5) 验证
python -u test/data/verify_daily_vs_1min.py
```

---

## 9. 常见问题与排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 聚宽研究环境 `from jqdata import get_price` 报 `ImportError` | 研究环境把数据函数预置为全局函数，`jqdata` 不直接导出 | 直接调用 `get_price(...)`，或按导出脚本中的多环境兼容写法 |
| macOS 无法 `pip install gm` | 掘金 `gm` 无 macOS 轮子，终端仅支持 Windows | 改用聚宽/天勤/akshare 等跨平台源 |
| 分钟数据出现周六伪交易日、bar 数只有 120 | 旧版交易日归属把周五夜盘归到周六 | 用修复后的 `assign_trading_day` 重新导入 |
| Ctrl+C 后数据仍在涨 | MongoDB 服务端继续执行已提交批次 | 用 `stop_c0_import.py` 杀进程 + `killOp` |
| `mongosh`/Compass 连不上 `ECONNREFUSED` | mongod 未启动（可能被 shutdown） | 按第 7 节重启 MongoDB |
| 预览时 `symbol=UNKNOWN` | 日频库没读到（Mongo 未启动 / 无 C0 数据），触发 gap 回退 | 先启动 Mongo、确认日频库有数据，再导入 |
| 验证有 `[警告] 隔夜跳空不在日频换月日` | 聚宽换月与本地日频换月日不一致 | 核对换月日，必要时以分钟数据跳空为准调整 |

---

## 10. 脚本清单

| 脚本 | 作用 |
|---|---|
| `test/data/joinquant_fut_min_export.py` | 在聚宽研究环境导出主力连续合约分钟数据 CSV |
| `test/data/import_c0_1min_to_db.py` | 读 CSV → 复用日频换月安排 → 分批写入 `continuous_contract_price_1min` |
| `test/data/verify_daily_vs_1min.py` | 验证日频与分钟数据完全一致 |
| `test/data/stop_c0_import.py` | 一键停止导入（杀进程 + MongoDB `killOp`） |
| `data/futures.py` | 日频连续合约构建与换月调整逻辑（`build_roll_adjusted_continuous_contract_price` 等） |
| `mongo/mongify.py` | MongoDB 读写封装（`get_data` / `update_data` / `delete_data`） |

---

## 附：相关调研

数据源选择与对比见 `notes/minute_data.md`（免费/付费分钟数据源、天勤 EDB、恒有数 UData、聚宽等）。
