# 利用分钟频数据增强日频期货因子——方法调研

> 调研日期：2026-08-27
> 目标场景：**日频调仓不变**（每天出一个仓位值/因子值，回测与资金逻辑完全不变），在此基础上利用**分钟频 OHLCV+持仓量**生成**更准确的日频因子值**。
> 研究范式：单品种期货时序策略；现有管线为 GP+梯度下降权重调整挖掘单品种时序因子 → 直接加和融合因子 → 历史数据回测。
> 数据约束：分钟 bar 仅含 `open, high, low, close, volume, position(open_interest)` 六字段（无 tick、无 L1/L2 盘口）。

---

## 0. 问题定义

你目前每个交易日产生一个因子值（信号/仓位），回测按"T 信号 → T+1 开盘执行"的 open-to-open 口径进行（见 `strategy/strategy.py`）。因子表达式基于日频 OHLCV+持仓量，由遗传编程（GP）在日频行上挖掘、`OpRollNorm` 标准化、`FactorFusioner` 直接加和融合。

本调研回答的问题是：**如何在不改变"日频调仓"这一低频属性的前提下，把分钟频数据用起来，让每天那一个因子值更准确。** 这与"分钟级调仓的高频策略"是两回事——我们不要 HFT 执行，要的是更细粒度信息浓缩成一个日频标量。

核心洞察先放在前面：**你的因子算子是频率无关的**（见第 1 节），因此"用分钟数据"在工程上有两条清晰路径——

- **路径 A（预聚合日频特征工程，推荐起步）**：把分钟 bar 聚合成**更丰富的日频列**（不止 OHLCV，还有已实现波动率、跳跃、日内动量、订单流失衡等），日频 df 多出若干列 → GP 把它们当叶子挖 → 融合与回测**完全不动**。
- **路径 B（原生分钟因子，更激进）**：把分钟 df 直接喂给 `calc_formula_series`，算子在分钟行上滚动，再把分钟级因子序列**物化**成日频值喂回 BackTester。

第 2 节综述方法，第 3 节给出与你的代码结合的落地分析，第 4 节给推荐路径。

---

## 1. 你现有管线概览（基于代码，落地分析的锚点）

### 1.1 数据层（`data/futures.py`）

你的分钟数据基础设施**已经存在**，只是还没被因子管线消费：

- MongoDB 集合 `futures.continuous_contract_price_1min`：分钟 bar，字段 `time, instrument_id, symbol, open, high, low, close, settle, volume, position, money, weighted_factor, cur_weighted_factor, is_rollover, source`。来源标识 `SOURCE_AKSHARE`（日频）、`SOURCE_JOINQUANT`（聚宽分钟）、`SOURCE_EDB`（天勤 EDB 免费近 1 年）。
- `update_futures_continuous_contract_price_1min()`：从天勤 EDB 拉主连分钟线（`period=60` 秒=1 分钟），免费近 1 年。
- `assign_trading_day_1min()`：给分钟 bar 打交易日标签——**夜盘（hour>=20）归次日**，日盘归当天，周五夜盘归下周一。这是国内商品期货正确的时间归属逻辑。
- `detect_rollover_from_minute_df()`：基于分钟数据自身检测主力切换（隔夜跳空>阈值 或 持仓量单日变化>阈值），并计算后复权因子链 `weighted_factor`。
- `aggregate_minute_to_daily_df(df_min)`：**当前唯一的分钟→日频聚合**，做的是最朴素的 OHLCV 聚合——`open=first, high=max, low=min, close=last, volume=sum, position=last, money=sum`。**这一步把日内结构全丢了**，正是本调研要改进的点。
- `get_futures_continuous_contract_price()`：读日频库（默认 `source='akshare'`，可传列表同时读 joinquant 聚合的日频）。

**关键结论**：分钟数据已入库、交易日归属已处理、复权因子链已建。你要做的不是"重新搭数据管道"，而是"在 `aggregate_minute_to_daily_df` 之外，多产出几列日频特征列"。

### 1.2 因子算子（`factors/factor_ops.py`）——频率无关

`calc_formula_series(df, formula, data_fields)` 把公式字符串解析成 AST，在 `df` 上求值返回一条 `pd.Series`。**所有算子的"窗口 N"指的是 N 行**，与行是日还是分钟无关：

- 叶子 `DataNode(field)`：可读 `open, high, low, close, volume, position`（`oi` 为 `position` 别名）。
- `OpRollNorm(child, window, min_periods, eps, clip)`：滚动 z-score，`hist=s.shift(1)` 保证**无前视泄露**。
- 时序算子：`OpReturn/LogReturn, OpVolatility, OpTsMean/Std/Delta/Sum/Max/Min/Argmax/Argmin/Rank, OpEma, OpTsTimeWeightedMean, OpTsPctDelta, OpBias, OpRangePosition, OpPriceAcceleration, OpTrueAmplitude`。
- 二元时序：`OpTsCorr, OpTsRankCorr, OpTsCov, OpTsBeta, OpVpDivergence`（量价背离）, `OpAmihud`（非流动性）, `OpTsResidual`（滚动回归残差）。
- 已有的微观结构相关算子：`OpOiTrendConviction(close, oi)=sign(close.diff(1))*oi.diff(1)`（价-OI 趋势信念）、`OpAmihud(close, volume)`、`OpVpDivergence`、`OpBodyRatio/UpperShadowRatio/LowerShadowRatio/StochasticK`（K 线形态）、`OpTsSkew/TsKurt/TsEntropy`。
- 类型系统 `FactorDataType`（PRICE/VOLUME/OI/RETURN/VOLATILITY/RATIO/BOOLEAN/GENERIC）防止无意义组合（如 `Price*Volume` 被拒）。

**关键结论**：算子集合已经相当丰富，且**频率无关**。若把日频 df 换成分钟 df，`OpReturn(close, 5)` 就是 5 分钟收益而非 5 日收益。但所有"已实现波动率/跳跃/订单流失衡"这类需要日内全样本求和的特征，**日频行上的算子无法还原**——它们必须在新特征工程里从分钟 bar 直接算成日频列。

### 1.3 GP 引擎与融合（`factors/factor_auto_search.py`、`factors/gp_factor_engine.py`）

- `GeneticFactorGenerator` / `FactorFusioner` 都继承同一基类，构造时收 `fc_freq∈{'1m','5m','1d'}` 与 `portfolio_adjust_method∈{'min','1D','1M','1Q'}`，断言已通过（`assert self.fc_freq in [...]`）。
- GP 生成随机树时：`DataNode(rng.choice(list(data_fields)))`——**叶子字段由 `base_col_list` 决定，可配置**。这是路径 A 的核心钩子：只要把新日频特征列加进 `base_col_list`，GP 就能把它们当叶子挖。
- 适应度评估：`calc_formula_df(df, formula_map, data_fields=self.base_col_list)` → `get_future_ret` → `BackTester(fc_freq=self.fc_freq, portfolio_adjust_method=self.portfolio_adjust_method, data=eval_df)`。`BackTester` 期望 `data` 含 `['time','instrument_id','future_ret', <因子列>]`，且**因子列与行对齐**（日频调仓就是日频行）。
- `FactorFusioner.fuse()`：加载多个因子（`use_version_dict={collection:[version,...]}`），加权/直接加和融合，`fusion_indicator_dict` 做加权多指标打分，支持 outsample 混合。融合因子最终也是一条日频 `pd.Series`，喂 `BackTester`。
- `LLMPromptFactorGenerator`：LLM 生成公式，`apply_rolling_norm=True` 时包 `OpRollNorm`，同样落回日频 series。

### 1.4 回测与策略（`factors/backtest.py`、`strategy/strategy.py`）

- `BackTester` 代码注释（`backtest.py` 第 39-42 行）已**明确预想**了你的场景："对数据频率 1min，调仓频率为 1day 情况，计算 IC 需要先将一分钟的数据聚合成一天的，然后计算日频收益率，再计算日频因子值和日频收益率之间的相关系数"。即框架已为"分钟数据+日频调仓"留了概念位置，只是当前 IC 计算路径仍按行对齐。
- `get_performance`：`gross_ret = signal_t * future_ret_t`；`get_ts_ret_and_turnover` 处理换月 turnover（`is_rollover` 标志位）。**信号必须是日频对齐的 series**。
- `Strategy.backtest()`：`signal_delay_days=1`，`open_to_open_pnl = position_lots * (next_open_px - open_px) * multiplier`。因子在 `calc_formula_series(df=factor_input, formula=factor_formula)` 后 `groupby(instrument_id).shift(signal_delay_days)`。**输入 df 是日频**。

**落地锚点小结**：信号必须是日频 series。所以无论方法多复杂，最后都要产出"每个交易日一个值"。路径 A 在特征工程层完成这一聚合；路径 B 在 GP 适应度循环里加一个"分钟→日频物化器"。

---

## 2. 方法综述

> 下面按"信息来源"分五大族。每条给：核心思想、日频值怎么算、参考、对单品种期货时序的相关性、与你管线的契合点。URL 核验状态见第 6 节说明。

### A. 已实现波动率与日内收益矩族

这一族把日内收益序列当作"日度波动率的样本"，把不可观测的日度波动率变成可观测的日频特征。它们本身大多**不带方向**，但作"状态/调制器"极有价值（与方向信号 `OpMul` 组合、或做仓位缩放）。

**A1. 已实现波动率 Realized Volatility (RV)**
- 思想：日内收益平方和作为日度波动率的一致估计，使波动率从隐变量变可观测量；对 RV 建模预测次日波动显著优于 GARCH。
- 日频值：`RV_d = Σ_{i=1..N} r_i²`，`r_i = ln(C_i/C_{i-1})`（首根用 `ln(C_1/O_1)` 或接前收盘）。可进一步做长记忆（ARFIMA/HAR）预测次日波动，用于波动目标/仓位定权。
- 参考：Andersen, Bollerslev, Diebold, Labys (2003), *Econometrica* 71(2):579-625；早期 NBER WP 8160。
- 相关性：**高**。期货日内连续、5 分钟采样噪声小，RV 是商品/外汇期货日度波动与定权最直接的特征。
- 契合：作 `DataNode('rv')` 叶子，GP 可演化 `OpMul(direction_leaf, OpRollNorm(OpInv(DataNode('rv'))))` 实现"波动率缩放仓位"。

**A2. 双幂变差与跳跃变差 Bipower Variation & Jump Variation**
- 思想：`BPV` 用相邻绝对收益乘积之和，对跳跃稳健，从而把日内总变差分解为"连续扩散"+"跳跃"。
- 日频值：`BPV_d = (π/2)·Σ_{i=2..N}|r_{i-1}||r_i|`；`Jump_d = max(RV_d - BPV_d, 0)`。`Jump` 分量对应宏观/库存数据冲击（原油 EIA、农产品 WASDE）。
- 参考：Barndorff-Nielsen & Shephard (2004), *Journal of Financial Econometrics* 2(1):1-48。
- 相关性：**高**。期货受数据公布驱动跳跃明显，跳跃分量与连续分量对次日预测意义不同。
- 契合：`rv, bpv, jump` 三个日频列，GP 可分别组合。

**A3. 已实现半方差与有符号跳跃 Realized Semivariance & Signed Jump**
- 思想：把 RV 按收益符号拆为上行 `RV⁺` 与下行 `RV⁻`，下行半方差对次日波动预测更强（杠杆/下行风险）。
- 日频值：`RV_neg_d = Σ r_i²·1{r_i<0}`，`RV_pos_d = Σ r_i²·1{r_i>0}`，`SJV = RV_pos - RV_neg`。
- 参考：Barndorff-Nielsen, Kinnebrock, Shephard (2010), "Measuring downside risk—realized semivariance"。
- 相关性：**高**。期货多空对称、下行跳跃风险溢价明显，半方差对趋势/反转择时有用。

**A4. 已实现偏度与峰度 Realized Skewness & Kurtosis**
- 思想：日内收益直接算高阶矩；负已实现偏度预示后续收益更低（横截面已验证），峰度对应尾部风险状态。
- 日频值：`RSkew_d = √N·Σ r_i³ / RV_d^{3/2}`，`RKurt_d = N·Σ r_i⁴ / RV_d²`。
- 参考：Amaya, Christoffersen, Jacobs, Vasquez (2015), *Journal of Financial Economics* 118(1):135-167。
- 相关性：中-高。原研究为横截面，但日度高阶矩对单品种尾部状态（挤兑、限板）刻画直接。

**A5. 跳跃稳健变差 medRV / 已实现核 Realized Kernel**
- 思想：`medRV` 用三邻域中位数截断，对跳跃与零收益更稳；`Realized Kernel` 用核加权的自协方差修正微结构噪声。
- 日频值：`medRV_d = [π/(6-4√3)]·[N/(N-2)]·Σ med(|r_{i-1}|,|r_i|,|r_{i+1}|)²`；`RK_d = γ_0 + 2·Σ_{h=1..H} k(h/H)·γ_h`（`γ_h` 为 h 阶自协方差，H 为带宽）。
- 参考：Andersen, Dobrev & Schaumburg, medRV（*Journal of Econometrics*）；Barndorff-Nielsen, Hansen, Lunde & Shephard (2008), *Econometrica* 76(6):1481-1536。
- 相关性：中。分钟 bar 已部分平滑噪声，`medRV` 适合作 `RV` 的稳健替代，`RK` 在低流动性品种更有价值。

### B. 区间型波动率估计族（仅用 OHLC）

这一族性价比最高——只需日 OHLC（甚至每分钟 OHLC 求和）就能比收盘价估计量高效数倍地估日度波动。

**B1. Parkinson (1980)**
- 日频值：`σ²_P = (ln H_d - ln L_d)² / (4 ln2)`，或逐分钟 `Σ_i (ln H_i - ln L_i)²/(4 ln2)`。约 5× 效率，零漂移假设。

**B2. Garman-Klass (1980)**
- 日频值：`σ²_GK = ½(ln(H/O))² - (2ln2-1)(ln(C/O))²`。约 7.4× 效率，零漂移。

**B3. Rogers-Satchell (1994)**
- 日频值：`σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)`。**容忍非零漂移**，趋势型期货更适用。约 8× 效率。
- 参考：Rogers, Satchell & Yoon (1994), *Applied Financial Economics* 4(3):241-247。

**B4. Yang-Zhang (2000)**
- 日频值：组合隔夜方差 + RS 项 + 开收漂移项，方差最小化权重 `k`。专为跳空设计。
- 参考：Yang & Zhang (2000), *Journal of Business* 73(3):477-491。
- 契合（B 族整体）：直接用日 OHLC 算成日频列 `pk, gk, rs, yz`，GP 可作波动调制器。你已有 `OpTrueAmplitude` 等算子但那是日频行上的近似；从分钟 bar 逐根求和再聚合更准。

### C. 日内动量与时段结构族

这一族**直接带方向**，对择时增益最直接。

**C1. 市场日内动量 Market Intraday Momentum（首半小时 vs 末半小时）**
- 思想：首半小时收益可预测末半小时收益，在波动高、成交大、宏观公告日更强；已验证存在于商品 ETF 与商品/外汇期货。
- 日频值：`r_open = ln P_{30min} - ln P_open`（开盘后 30 分钟收益），`r_close = ln P_close - ln P_{close-30min}`（收盘前 30 分钟）。回归 `r_close = α+β·r_open`，β 显著正。用 `r_open`（或 `r_open × hist_β`）作日频信号预测当日剩余/次日方向。
- 参考：Gao, Han, Li, Zhou (2018), *Journal of Financial Economics* 129(2):394-414。DOI: `10.1016/j.jfineco.2018.05.009`。
- 相关性：**高**。直接面向期货/ETF，首末半小时结构清晰。
- 契合：需要"日内时段切片"算子（取开盘后 30 分钟、收盘前 30 分钟）。可在特征工程层产 `intraday_mom_open`、`intraday_mom_close` 两列。

**C2. 隔夜 vs 日内收益分解（"拔河"）**
- 思想：收益分 `overnight=ln Open_t - ln Close_{t-1}`（收盘→开盘）与 `intraday=ln Close_t - ln Open_t`（开盘→收盘）。动量溢价几乎全发生在隔夜段；机构日内、散户/新闻流隔夜主导。
- 日频值：`R_on`、`R_intra` 及其平滑差 `R_on - R_intra` 作日频分解因子。
- 参考：Lou, Polk, Skouras (2019), *Journal of Financial Economics* 134(1):192-213。
- 相关性：中-高。期货有夜盘/电子盘，夜盘→日盘分解对金融/有色/原油夜盘活跃品种直接可用。

**C3. 日内时段周期性 Intraday Periodicity**
- 思想：半小时收益在"交易日整数倍"的相同时点存在延续性，持续 40 个交易日以上，是日内固定时点的可预测结构。
- 日频值：把交易日切 M 个等长时段，估计各时段收益的历史延续性，汇总为"日内动量强度"日频值。
- 参考：Heston, Korajczyk, Sadka (2010), *Journal of Finance* 65(4):1369-1407。arXiv: `1005.3535`。
- 相关性：中。原为横截面，但"固定时点收益延续"在期货（开盘/收盘集中）也存在。

**C4. 尾盘反转 / 零售注意力**
- 思想：按日内收益排序有动量、按隔夜收益排序无；尾盘错价反转；隔夜收益+关注度预示日内反转。
- 参考：Bogousslavsky (2021), *Journal of Financial Economics*；Berkman, Koch, Tuttle, Zhang (2012), *JFQA*。
- 相关性：中。横截面结论，但"尾盘收益反转"对单品种持仓过夜 vs 平仓有借鉴。

### D. 微观结构与订单流族（仅用 OHLCV+OI，无 tick/盘口）

这是你数据约束下**最独特、最可能出 alpha 的一族**——`position(open_interest)` 是期货原生字段，股票策略用不上。

**D1. 持仓量-价格流（OI/OFI 代理，期货原生）** ★最高优先
- 思想：价涨+OI 增=新多入场；价跌+OI 增=新空入场；OI 变化×价格方向刻画资金建仓强度。经典 OFI 需盘口队列，但期货可**直接用分钟 OI**做代理。
- 日频值（多个变体）：
  - `MoneyFlow_d = Σ_i ΔOI_i · r_i`（或 `Σ_i ΔOI_i · sign(C_i - C_{i-1})`）
  - `OI-weighted return`：`r^OI_d = Σ_i r_i·OI_i / mean(OI_i)`（高持仓时段收益权重更大，"知情定位"代理）
  - 日 `ΔOI_d` 与日收益 `r_d` 的组合规则（升价+升 OI=多头延续等）
- 参考：Cont, Kukanov, Stoikov (2014), *Journal of Financial Econometrics* 12(1):47-88（经典 OFI）；Bessembinder & Seguin (1992) 期货 OI-价格微观结构。
- 相关性：**高**。OI 是本约束下独有、最直接的 alpha 源。
- 契合：你已有 `OpOiTrendConviction(close, oi)=sign(close.diff)*oi.diff` 算子（日频行版）。**从分钟 bar 算成日频列 `oi_flow`、`oi_weighted_ret` 后，信息量远大于日频版的 `OpOiTrendConviction`**（日频 diff 把日内建仓方向全抹了）。

**D2. 累计成交量差 CVD（BVC 代理）**
- 思想：日内净主动性买卖压力，直接带方向；与价格背离是经典反转信号。无 tick 时用 Bulk Volume Classification：`buyV_i = V_i·Φ(r_i/σ_r)`，`delta_i = 2·buyV_i - V_i`。
- 日频值：`CVD_d = Σ_i delta_i`，标准化 `OFI_bar = CVD_d / Σ_i V_i`。或用 tick-rule 代理 `delta_i = V_i·sign(C_i - C_{i-1})` 做对比。
- 参考：Easley, López de Prado, O'Hara (2012), *Review of Financial Studies* 25(5):1457-1493（VPIN/BVC 同源）。
- 相关性：**高**。1 分钟 BVC 在 E-mini S&P 报告约 86% 分类准确。
- 契合：产日频列 `cvd`，与 `oi_flow` 互补（CVD 描成交主动性，`oi_flow` 描持仓建仓）。

**D3. 成交量时钟 / 成交量时间 Volume Clock**
- 思想：以"每固定成交量"而非"每固定时间"采样，消除日内季节性、部分恢复正态。
- 日频值：按成交量累计到阈值 ΔV 记一次"事件时间"价格，在此序列上算 RV/动量；或用"成交量时钟速度"（完成单位成交量所需时间）作日频活跃度。
- 参考：Ané & Geman (2000)；Easley, López de Prado, O'Hara (2012), *Journal of Portfolio Management*。
- 相关性：中。适合作 RV/动量的预处理增强。

**D4. VPIN（成交量同步知情交易概率）**
- 思想：在成交量时钟下把交易切等量桶，用买卖量失衡估知情交易概率；高 VPIN 预示流动性枯竭与短期波动峰值（闪崩前兆）。
- 日频值：日内桶均值。可用分钟 bar + BVC 实现。
- 参考：Easley, López de Prado, O'Hara (2012), *RFS* 25(5):1457-1493。
- 相关性：中-高。作 toxicity/regime 预警，"风险开关"（高 VPIN 降杠杆/暂停动量）。

**D5. Amihud 非流动性（日内增强）**
- 思想：单位成交额对应的绝对收益，度量价格冲击/非流动性。
- 日频值：经典 `|r_d|/DVOL_d`；日内增强 `AmihudR_d = (1/N)·Σ_i |r_i|/DV_i`。
- 参考：Amihud (2002), *Journal of Financial Markets* 5:31-56。
- 相关性：中-高。期货主力非流动性随近月/移仓变化明显。
- 契合：你已有 `OpAmihud(close, volume)` 算子（日频行版）；从分钟 bar 算 `amihud_intraday` 日频列更细。

**D6. Kyle λ 价格冲击斜率代理**
- 思想：对分钟回归 `r_i = α + λ·SV_i + ε`，`SV_i` 为带符号成交量，`λ` 即价格冲击系数。
- 日频值：回归斜率（或滚动窗口）。带符号量代理用 tick-rule `SV_i=V_i·sign(C_i-C_{i-1})` 或 BVC。
- 参考：Kyle (1985), *Econometrica* 53:1315-1335；Hasbrouck (2009), *Journal of Finance* 64:1445-1477。
- 相关性：中-高。λ 抬升=信息驱动流增强/流动性恶化。

**D7. VWAP 偏离 / 成交量分布 POC**
- 思想：`VWAP_d = Σ P_i·V_i / Σ V_i`，收盘在 VWAP 之上=日内净买盘占优；`POC`=成交量最大的价格档，收盘远离 POC=价值拒绝（动量）。
- 日频值：`dev_d = (C_N - VWAP_d)/VWAP_d`，`distPOC_d = (C_N - POC_d)/POC_d`。
- 相关性：中-高。方向偏置清晰，易与动量 `OpAdd/OpMul` 叠加。

**D8. 日内方差比 Variance Ratio（Lo-MacKinlay）**
- 思想：日内 k 分钟聚合收益的方差比 `VR(k)=Var(r^(k))/(k·Var(r^(1)))`。`VR>1` 正自相关（动量/惯性），`VR<1` 负自相关（均值回复）。
- 日频值：日内序列算 `VR(k)_d`。
- 参考：Lo & MacKinlay (1988), *Review of Financial Studies* 1(1):41-66。
- 相关性：中-高。作"动量 vs 反转"状态判别，调制信号：`VR>1` 启用动量 leaf，`VR<1` 启用反转 leaf（用 `OpIfElse`）。

**D9. 首达时间 / 区间漂移**
- 思想：记录日内首达日高时刻 `t_H` 与首达日低 `t_L`，`sign(t_H-t_L)` 反映日内趋势节奏。
- 参考：Magdon-Ismail, Atiya & Pratama (2003/2004)。
- 相关性：中。较冷门但首达不对称可反映节奏。

> 备注：CFTC Commitment of Traders (COT) 报告为**周频**，不能做日频信号，但可作长周期方向背景过滤。

### E. 深度学习范式族

这一族把日内分钟序列整体喂给模型，输出一个日频标量。**与你 GP 管线是互补关系**：产出"NN 日频因子"，与 GP 因子并列进 `FactorFusioner` 加权融合。

**E1. 自监督表征学习 + 轻量预测头（最推荐互补）**
- 思想：先用海量无标注分钟历史自监督预训练表征，再加线性/MLP 头微调输出次日收益预测，该标量即"日频 NN 因子"。
- 代表：**TS2Vec**（层次对比学习，时间戳级表征，可聚合为序列向量）；**PatchTST 自监督分支**（把序列切 patch 当 token，掩码重建预训练）；**TimeMAE**（子序列掩码自编码器）；**TF-C**（时频一致性对比）。
- 输入→输出：分钟序列 → 编码器 → 表征 → 池化/取末步 → 线性头 → 日频标量。
- 参考：Yue et al. (2022), TS2Vec, AAAI 2022, arXiv: `2106.10466`；Nie et al. (2023), PatchTST, ICLR 2023, arXiv: `2211.14730`。
- 互补性：**最佳互补候选**。GP 因子是日频 OHLCV 的符号回归表达式，NN 因子编码日内分钟微观结构，两者信息正交，融合分散度高。预训练一次（单 GPU 数小时）→ 推理每日毫秒级。

**E2. 一维卷积/TCN（高性价比监督式）**
- 思想：因果膨胀一维卷积，感受野随深度指数增长，并行计算，梯度稳定。
- 代表：**TCN**（Bai, Kolter, Koltun, arXiv: `1803.01271`）；**WaveNet**（门控膨胀因果卷积，arXiv: `1609.03499`）；**InceptionTime**（多尺度并行一维卷积集成，arXiv: `1909.04939`）。
- 输入→输出：分钟序列 → 卷积编码 → 全局平均池化/取末步 → FC → 日频标量。
- 互补性：因果卷积天然无前视泄露，计算量与 GP 同量级或更低，适合作"第二意见"因子。

**E3. Transformer 长序列家族**
- 代表：**Informer**（ProbSparse 自注意力，AAAI 2021，arXiv: `2012.07436`）；**Autoformer**（趋势/季节分解+自相关，NeurIPS 2021，arXiv: `2106.13008`）；**TimesNet**（1D→2D 周期建模，ICLR 2023，arXiv: `2210.02186`）；**PatchTST**（patch 化通道独立，ICLR 2023，arXiv: `2211.14730`）。
- 互补性：patch 化（PatchTST）使计算量与 GP 可比；适合日内序列很长（夜盘+日盘数百根）的场景。

**E4. TFT（有元数据时的可解释多步预测）**
- 思想：变量选择网络 + 静态协变量 GRN + seq2seq + temporal self-attention，显式区分已知未来输入/历史观测/静态元数据。
- 代表：Lim et al. (2021), *International Journal of Forecasting*，arXiv: `1912.09363`。
- 互补性：若有合约到期、交割日、季节性日历等已知未来输入，TFT 的变量选择+attention 最匹配；attention 权重本身也可作辅助特征。

**E5. 纯前馈基扩展 N-BEATS / N-HiTS**
- 代表：N-BEATS（ICLR 2020，arXiv: `1905.10437`）；N-HiTS（AAAI 2023，arXiv: `2201.12886`，多速率分层插值）。
- 互补性：多频率分解对"分钟跳动+日间趋势"有意义；计算极轻量。但纯 FC 对长日内序列不如卷积/Transformer。

**E6. 表格化 + 树模型（最简基线）** ★最低成本起步
- 思想：把日内分钟序列手工聚合为日频特征向量（D 族的统计量），再 XGBoost/MLP 预测次日收益。
- 互补性：零 GPU、秒级训练、可解释（特征重要性）。**建议作为第一步**：若 XGBoost 日内因子在回测中有 IC 增益，再上 TS2Vec/PatchTST 提升。

**E7. 强化学习（范式不匹配，不推荐直接融合）**
- FinRL 等 DRL 输出的是"动作/头寸"而非可加因子值，与你的"因子加和融合"范式不兼容。仅当愿把 GP 因子作为 RL 状态特征、整体改为"GP 因子→RL 头寸"时考虑，偏离互补目标。

---

## 3. 与现有 GP+融合+回测管线的结合分析

### 3.1 路径 A：日频特征工程（预聚合，推荐起步）

**核心改动只有一处**：在 `aggregate_minute_to_daily_df`（或新模块 `factors/factor_intraday_features.py`）里，除了现有的 OHLCV 聚合，**多产出若干日频特征列**。每列是一条 `pd.Series`，index=交易日。

落地步骤：

1. **新增特征工程模块**（建议 `factors/factor_intraday_features.py`），输入分钟 df（`continuous_contract_price_1min`，按 `assign_trading_day_1min` 打好 `td`），输出日频 df，列含：
   - 已实现族：`rv, bpv, jump, rv_neg, rv_pos, rskew, rkurt, medrv`
   - 区间族：`pk, gk, rs, yz`（逐分钟求和更细）
   - 时段族：`ret_open30`（开盘后 30 分钟）、`ret_close30`（收盘前 30 分钟）、`ret_overnight`（夜盘→日盘）、`ret_intraday`（开盘→收盘）
   - 订单流族：`oi_flow`（`Σ ΔOI·sign(r)`）、`oi_weighted_ret`、`cvd`（BVC）、`vpin`、`amihud_intraday`、`kyle_lambda`、`vwap_dev`、`vr_k`（方差比）
2. **把新列注册为 GP 叶子**：在 `GeneticFactorGenerator`/`FactorFusioner` 构造时，把 `base_col_list` 从默认 `['open','high','low','close','volume','position']` 扩展为含上述新字段；同时更新 `factor_ops.py` 的 `_BASE_FIELD_TYPE_MAP` 与 `infer_field_type`（如 `rv`→VOLATILITY、`cvd`→RETURN、`vpin`→RATIO）。
3. **融合与回测完全不动**：GP 在扩展后的日频 df 上挖因子（新叶子可和旧 OHLCV 叶子组合），`FactorFusioner` 仍直接加和融合，`BackTester`/`Strategy` 仍日频调仓。`fc_freq='1d'`、`portfolio_adjust_method='1D'` 不变。
4. **数据流**：`get_futures_continuous_contract_price` 读日频库；新增一个 `get_intraday_enriched_daily_df(instrument_id, start, end)` 读 `continuous_contract_price_1min` → 特征工程 → 返回日频 df（含新列）。GP 基类的 `df = get_futures_continuous_contract_price(...)` 处加一个开关走增强路径。

**路径 A 的优势**：改动集中在特征工程层，GP/融合/回测代码零改动；可逐条加特征、可回退；每条特征都有学术依据；GP 的可解释性（符号回归）保留。**局限**：日频列是手工设计的，表达能力受限于你选的特征集；无法学习分钟序列的非线性时序模式（那要靠路径 B 或 E 族 NN）。

### 3.2 路径 B：原生分钟因子（更激进）

把分钟 df 直接喂 `calc_formula_series`：`OpReturn(close, 5)` 变成 5 分钟收益，`OpRollNorm(close, 240, ...)` 在 240 分钟窗口标准化。GP 在分钟行上挖表达式，产出分钟级因子 series。

**关键问题**：你的 GP 适应度评估最终调用 `BackTester`，而 `BackTester` 的 `get_performance` 要求信号与 `future_ret` 在**日频行**对齐算 IC/夏普。所以必须在 GP 适应度循环里加一个"分钟→日频物化器"：

- 物化方式（任选其一）：每日取**收盘前固定时刻**的因子值（如 14:55）；或每日取**最后分钟值**；或每日对分钟因子值做 `OpRollNorm` 后再聚合（均值/末值）。
- 实现位置：在 `GeneticFactorGenerator` 的 `generate`/评估方法里，`calc_formula_df` 在分钟 df 上算出分钟 series 后，按 `assign_trading_day_1min` groupby 取日频值，再与 `get_future_ret`（日频）对齐喂 `BackTester(fc_freq='1d', portfolio_adjust_method='1D')`。
- **前视泄露风险**：物化必须取"当日收盘前已知"的值，不能用到当日收盘后信息。取"收盘前 N 分钟值"或"当日最后已知值"安全；取日均值需谨慎（含全日）。`OpRollNorm` 已用 `shift(1)`，分钟级也安全。

**路径 B 的优势**：GP 能直接演化"5 分钟动量×持仓量变化"这类高频微观结构表达式，表达力远超路径 A 的手工列。**局限**：适应度评估成本×240（分钟行数）；GP 树更深更慢；物化器设计不当易泄露；与现有日频 `BackTester` 接口需适配。建议**只在路径 A 验证有效后**，对少数品种试。

### 3.3 NN 因子融合（E 族）

E 族（尤其 TS2Vec/PatchTST 自监督 + 轻量头）产出的日频标量，作为"NN 因子"纳入 `FactorFusioner`：
- 训练：分钟历史 → 自监督预训练 → 微调头输出次日收益预测（一个日频 series）。
- 入库：作为 `factor_fusion` 集合的一个因子版本，或新建 `factors.llm_or_nn` 类。
- 融合：`FactorFusioner` 的 `use_version_dict` 加入 NN 因子版本；`fusion_indicator_dict` 评估其与 GP 因子的相似度与泄漏，加权融合。**注意 outsample**：NN 必须严格训练集/测试集切分，`FactorFusioner` 的 `outsample_ratio` 混合可防过拟合。

### 3.4 各方法与管线的映射表

| 方法族 | 落地形态 | 代码改动点 | 与 GP 关系 |
|---|---|---|---|
| A 已实现族 | 日频特征列 | `factor_intraday_features.py` 新模块 | GP 叶子（扩展 `base_col_list`） |
| B 区间族 | 日频特征列 | 同上 | GP 叶子（波动调制器） |
| C 时段族 | 日频特征列（需时段切片） | 同上 + `assign_trading_day_1min` 切片 | GP 叶子（方向信号） |
| D 订单流族 | 日频特征列 | 同上 | GP 叶子（方向/调制）；`OpOiTrendConviction` 已有日频版，分钟列更强 |
| E 深度学习 | NN 日频因子 | 独立训练管线 → 入库 | 进 `FactorFusioner` 与 GP 因子加权融合 |
| 路径 B | 原生分钟 GP | GP 评估循环加物化器 | 替换/扩展 GP 的 df 粒度 |

---

## 4. 落地可行方法分析与推荐路径

按"性价比/与管线契合度/可回退性"排序，建议三阶段渐进。

### Phase 1：低成本日频特征 + 表格基线（1-2 周，零 GPU）

**做**：在 `factors/factor_intraday_features.py` 实现下列日频列（从 `continuous_contract_price_1min`），并扩展 `base_col_list`：
- 第一梯队（低门槛、学术扎实）：**RV、BPV、Jump、已实现半方差 RV⁻**（A1-A3）、**区间波动 Parkinson/GK/RS**（B1-B3）、**OI/OFI 流**（D1）、**CVD/BVC**（D2）、**VWAP 偏离**（D7）、**日内方差比**（D8）、**Amihud 日内**（D5）。
- 时段切片：**首末半小时收益**（C1）、**隔夜 vs 日内**（C2）——需要按 `assign_trading_day_1min` + 时间戳切片，但 C1/C2 对择时增益最直接。

**验证**：先用 E6（表格化+XGBoost）跑一个基线，看这些日内特征对次日收益有无 IC/夏普增益。若有，再让 GP 把它们当叶子挖——GP 会自动发现"方向 leaf × 1/RV leaf"这类组合。

**为什么先做这步**：所有特征都仅用 OHLCV+OI 从分钟 bar 直接算成日频列，与现有 `BackTester`/`Strategy` 零冲突；GP/融合代码零改动；可逐条加、可回退。你已有的 `OpOiTrendConviction`、`OpAmihud`、`OpVpDivergence` 是日频行版，分钟列版会显著更强（日内建仓方向不被日频 diff 抹掉）。

### Phase 2：NN 表征因子 + 融合（2-4 周，需 GPU）

**做**：用 TS2Vec 或 PatchTST 自监督在分钟历史上预训练 → 加轻量回归头微调 → 输出日频 NN 因子 → 纳入 `FactorFusioner` 与 GP 因子加权融合（`outsample_ratio` 防过拟合）。

**为什么**：Phase 1 的手工特征受限于设计；NN 能学习分钟序列的非线性时序模式，与 GP 符号回归信息正交。自监督无需标注日，可用全部分钟历史。

**风险**：训练集/测试集泄露是 NN 因子最大风险——必须严格按交易日切分，`FactorFusioner` 的 outsample 机制务必启用。

### Phase 3：原生分钟 GP（可选，4+ 周）

**做**：把分钟 df 直接喂 GP，评估循环加"分钟→日频物化器"（取收盘前固定时刻值，防泄露）。仅当 Phase 1/2 验证日内信息确有增益、且计算预算允许时推进。

**为什么留到最后**：表达力最强但成本最高（评估成本×240）、泄露风险最大、与现有日频 `BackTester` 接口需适配。Phase 1 的手工特征已能捕获大部分日内信息，Phase 3 的边际收益需评估。

### 实施细节与风险控制

- **交易日归属**：复用 `assign_trading_day_1min`（夜盘归次日）。所有日频特征 groupby `td` 聚合。
- **复权**：分钟 df 已带 `weighted_factor`（`detect_rollover_from_minute_df` 生成）。收益类特征用复权价 `C_i * weighted_factor`；OI/量类用原始值。与日频 `get_weighted_price` 口径一致。
- **前视泄露**：日频特征 groupby `td` 后，确保不跨日（如 `RV_d` 只用当日分钟，不用次日）。`OpRollNorm` 已 `shift(1)`。NN 因子严格按交易日切分训练/验证/测试。
- **数据深度**：天勤 EDB 免费仅近 1 年分钟，回测历史短。长历史需聚宽/Tushare Pro `ft_mins`（见 `notes/minute_data.md` 调研）。Phase 2 的 NN 预训练尤其需要长历史。
- **换月 turnover**：`get_ts_ret_and_turnover` 已处理 `is_rollover`；分钟特征入库时务必带 `is_rollover`，融合/回测沿用。
- **类型约束**：新日频列在 `factor_ops.py` 的 `infer_field_type` 注册类型（如 `rv`→VOLATILITY、`cvd`→RETURN、`vpin`→RATIO），否则 GP 类型系统会拒绝组合。

---

## 5. 关键结论一句话

你的管线已经为"分钟数据+日频调仓"预留了概念位置（`BackTester` 注释、频率无关算子、`base_col_list` 可配置叶子、分钟库已建）。**最划算的落地是路径 A**：写一个日频特征工程模块，把分钟 bar 浓缩成 8-15 条带学术依据的日频列，扩进 GP 的 `base_col_list`，GP/融合/回测代码零改动即可挖出利用日内信息的因子；再用 NN 自监督表征（TS2Vec/PatchTST）产一个日频 NN 因子进融合器作正交补充。原生分钟 GP（路径 B）留作可选的高阶路径。

---

## 6. 参考文献汇总

> **链接核验说明**：本调研的 WebSearch 工具在当前环境下多数只返回 Google 检索包装链接，而非论文落地页直链。下表中：(1) arXiv 编号与 DOI 为检索文本**显式确认**的稳定标识，按规范 URL 渲染（高置信）；(2) NBER 工作论文编号经检索确认；(3) 部分仅有书目的条目附 Google Scholar 检索链接以便核验，未杜撰 DOI。建议读者用 DOI/题目在出版社站点或 SSRN 取全文。

### A. 已实现波动率与日内收益矩
1. Andersen, Bollerslev, Diebold, Labys (2003). *Modeling and Forecasting Realized Volatility.* Econometrica 71(2):579-625. NBER WP 8160 — https://www.nber.org/papers/w8160
2. Barndorff-Nielsen & Shephard (2004). *Power and Bipower Variation with Stochastic Volatility and Jumps.* Journal of Financial Econometrics 2(1):1-48. 检索: https://scholar.google.com/scholar?q=Power+and+Bipower+Variation+Barndorff-Nielsen+Shephard
3. Barndorff-Nielsen, Kinnebrock, Shephard (2010). *Measuring downside risk—realized semivariance.* 检索: https://scholar.google.com/scholar?q=Measuring+downside+risk+realized+semivariance
4. Amaya, Christoffersen, Jacobs, Vasquez (2015). *Does Realized Skewness Predict the Cross-Section of Equity Returns?* Journal of Financial Economics 118(1):135-167. 检索: https://scholar.google.com/scholar?q=Amaya+Christoffersen+Jacobs+Vasquez+realized+skewness
5. Andersen, Dobrev, Schaumburg. *MinRV/MedRV jump-robust realized volatility.* Journal of Econometrics. 检索: https://scholar.google.com/scholar?q=Andersen+Dobrev+Schaumburg+medRV+jump+robust
6. Barndorff-Nielsen, Hansen, Lunde, Shephard (2008). *Designing Realised Kernels...* Econometrica 76(6):1481-1536. 检索: https://scholar.google.com/scholar?q=Designing+realised+kernels+Barndorff-Nielsen+Hansen+Lunde+Shephard

### B. 区间型波动率
7. Parkinson (1980). *The Extreme Value Method for Estimating the Variance of the Rate of Return.* Journal of Business 53(1):67-78. 检索: https://scholar.google.com/scholar?q=Parkinson+1980+extreme+value+variance
8. Garman & Klass (1980). *On the Estimation of Security Price Volatilities from Historical Data.* Journal of Business 53(1):67-78. 检索: https://scholar.google.com/scholar?q=Garman+Klass+estimation+security+price+volatilities
9. Rogers, Satchell & Yoon (1994). *Estimating the Volatility of Stock Prices...* Applied Financial Economics 4(3):241-247. 检索: https://scholar.google.com/scholar?q=Rogers+Satchell+Yoon+volatility+high+low+prices
10. Yang & Zhang (2000). *Drift-Independent Volatility Estimation...* Journal of Business 73(3):477-491. 检索: https://scholar.google.com/scholar?q=Yang+Zhang+drift+independent+volatility

### C. 日内动量与时段结构
11. Gao, Han, Li, Zhou (2018). *Market intraday momentum.* Journal of Financial Economics 129(2):394-414. DOI: 10.1016/j.jfineco.2018.05.009 — https://doi.org/10.1016/j.jfineco.2018.05.009
12. Lou, Polk, Skouras (2019). *A tug of war: Overnight versus intraday expected returns.* Journal of Financial Economics 134(1):192-213. 检索: https://scholar.google.com/scholar?q=Lou+Polk+Skouras+tug+of+war+overnight+intraday
13. Heston, Korajczyk, Sadka (2010). *Intraday Patterns in the Cross-section of Stock Returns.* Journal of Finance 65(4):1369-1407. arXiv:1005.3535 — https://arxiv.org/abs/1005.3535
14. Berkman, Koch, Tuttle, Zhang (2012). *Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open.* JFQA. 检索: https://scholar.google.com/scholar?q=Berkman+Koch+Tuttle+Zhang+overnight+returns+buying+at+the+open
15. Bogousslavsky (2021). *The cross-section of intraday and overnight returns.* Journal of Financial Economics. 检索: https://scholar.google.com/scholar?q=Bogousslavsky+cross-section+intraday+overnight+returns

### D. 微观结构与订单流
16. Cont, Kukanov, Stoikov (2014). *The Price Impact of Order Book Events.* Journal of Financial Econometrics 12(1):47-88. 检索: https://scholar.google.com/scholar?q=Cont+Kukanov+Stoikov+price+impact+order+book+events
17. Easley, López de Prado, O'Hara (2012). *Flow Toxicity and Liquidity in a High-Frequency World.* Review of Financial Studies 25(5):1457-1493. 检索: https://scholar.google.com/scholar?q=Easley+Lopez+de+Prado+O%27Hara+flow+toxicity+liquidity+high+frequency
18. Amihud (2002). *Illiquidity and stock returns: cross-section and time-series effects.* Journal of Financial Markets 5:31-56. 检索: https://scholar.google.com/scholar?q=Amihud+2002+illiquidity+stock+returns
19. Kyle (1985). *Continuous Auctions and Insider Trading.* Econometrica 53:1315-1335. 检索: https://scholar.google.com/scholar?q=Kyle+1985+continuous+auctions+insider+trading
20. Hasbrouck (2009). *Trading Costs and Returns for U.S. Equities...* Journal of Finance 64:1445-1477. 检索: https://scholar.google.com/scholar?q=Hasbrouck+2009+trading+costs+returns+effective+costs
21. Lo & MacKinlay (1988). *Stock Market Prices Do Not Follow Random Walks...* Review of Financial Studies 1(1):41-66. 检索: https://scholar.google.com/scholar?q=Lo+MacKinlay+stock+market+prices+random+walks+variance+ratio
22. Ané & Geman (2000). *Order Flow, Transaction Clock, and Normality of Asset Returns.* 检索: https://scholar.google.com/scholar?q=Ane+Geman+order+flow+transaction+clock+normality
23. Bessembinder & Seguin (1992). *Price-Volatility, Volume and Speculative Activity in Futures Markets.* 检索: https://scholar.google.com/scholar?q=Bessembinder+Seguin+price-volatility+volume+speculative+futures
24. Magdon-Ismail, Atiya, Pratama (2003/2004). *On the Maximum Drawdown of a Brownian Motion.* 检索: https://scholar.google.com/scholar?q=Magdon-Ismail+Atiya+maximum+drawdown+Brownian+motion

### E. 深度学习
25. Yue et al. (2022). *TS2Vec: Towards Universal Representation of Time Series.* AAAI 2022 — https://arxiv.org/abs/2106.10466
26. Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR 2023 — https://arxiv.org/abs/2211.14730
27. Bai, Kolter, Koltun (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (TCN).* — https://arxiv.org/abs/1803.01271
28. van den Oord et al. (2016). *WaveNet: A Generative Model for Raw Audio.* — https://arxiv.org/abs/1609.03499
29. Fawaz et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* Data Mining and Knowledge Discovery — https://arxiv.org/abs/1909.04939
30. Zhou et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021 — https://arxiv.org/abs/2012.07436
31. Wu et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 2021 — https://arxiv.org/abs/2106.13008
32. Wu et al. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* ICLR 2023 — https://arxiv.org/abs/2210.02186
33. Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting — https://arxiv.org/abs/1912.09363
34. Oreshkin et al. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.* ICLR 2020 — https://arxiv.org/abs/1905.10437
35. Challu et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 2023 — https://arxiv.org/abs/2201.12886
36. Zhang et al. (2022). *Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency (TF-C).* — https://arxiv.org/abs/2206.08496
37. Eldele et al. (2021). *Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC).* IJCAI 2021 — https://arxiv.org/abs/2106.14112
38. *TimeMAE: Self-supervised representations of time series with decoupled masked autoencoders.* — https://arxiv.org/abs/2303.00320
39. Liu et al. (2020). *FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance.* — https://arxiv.org/abs/2011.09384

### 项目内已有资料
40. 项目内分钟数据源调研: `notes/minute_data.md`（天勤 EDB / 聚宽 / Tushare Pro / 恒有数 UData 等，含链接核验记录）
41. 分钟入库与日频聚合脚本: `test/data/aggregate_minute_to_daily.py`、`test/data/import_c0_1min_to_db.py`、`data/futures.py` 中 `update_futures_continuous_contract_price_1min` / `aggregate_minute_to_daily_df` / `assign_trading_day_1min` / `detect_rollover_from_minute_df`
