<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Cpu /></el-icon> 遗传算法因子挖掘</h2>
      <p>配置 GP 遗传算法超参数，一键启动因子挖掘任务，挖掘完成后自动展示回测结果和净值曲线</p>
    </div>

    <el-tabs v-model="activeMiningTab" type="border-card" style="margin-bottom:12px;">
      <el-tab-pane label="开始挖掘" name="start" />
      <el-tab-pane label="自动挖掘" name="auto" />
    </el-tabs>

    <el-row :gutter="20" class="responsive-row top-panel-row">
      <el-col :span="24">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">超参数配置</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>
          <div class="param-scroll-panel">
          <el-form :model="params" label-width="auto" label-position="right" size="small">
            <el-row :gutter="16" class="param-section-grid">
              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">基础参数</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="合约"><el-input v-model="params.instrument_id_list" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="版本号"><el-input v-model="params.version" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="开始日期"><el-input v-model="params.start_time" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="结束日期"><el-input v-model="params.end_time" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="最大因子数"><el-input-number v-model="params.max_factor_count" :min="1" :max="500" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="最小窗口"><el-input-number v-model="params.min_window_size" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="因子频率"><el-select v-model="params.fc_freq" style="width:100%"><el-option label="1d" value="1d" /><el-option label="5m" value="5m" /><el-option label="1m" value="1m" /></el-select></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="调仓频率"><el-select v-model="params.portfolio_adjust_method" style="width:100%"><el-option label="1D" value="1D" /><el-option label="1M" value="1M" /><el-option label="1Q" value="1Q" /><el-option label="min" value="min" /></el-select></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="利息方式"><el-select v-model="params.interest_method" style="width:100%"><el-option label="simple" value="simple" /><el-option label="compound" value="compound" /></el-select></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="params.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="8"><el-form-item label="基准"><el-switch v-model="params.calculate_baseline" /></el-form-item></el-col>
                <el-col :span="8"><el-form-item label="无风险"><el-switch v-model="params.risk_free_rate" /></el-form-item></el-col>
                <el-col :span="8"><el-form-item label="复权"><el-switch v-model="params.apply_weighted_price" /></el-form-item></el-col>
              </el-row>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">适应度权重</el-divider>
              <div v-for="indicator in supportedIndicators" :key="`fitness-${indicator}`" class="indicator-row">
                <span class="indicator-label">{{ indicator }}</span>
                <el-input
                  v-model="params.fitness_indicator_dict[indicator]"
                  placeholder="0"
                  clearable
                  size="small"
                />
              </div>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">滚动标准化</el-divider>
              <el-form-item label="启用"><el-switch v-model="params.apply_rolling_norm" /></el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="窗口"><el-input-number v-model="params.rolling_norm_window" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="最小样本"><el-input-number v-model="params.rolling_norm_min_periods" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="Eps"><el-input-number v-model="params.rolling_norm_eps" :min="0" :step="1e-9" :precision="10" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="Clip"><el-input-number v-model="params.rolling_norm_clip" :min="0.5" :max="20" :step="0.5" :precision="1" style="width:100%" /></el-form-item></el-col>
              </el-row>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">去重 / 泄露检查</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="泄露抽样"><el-input-number v-model="params.check_leakage_count" :min="0" :max="200" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="相关阈值"><el-input-number v-model="params.relative_threshold" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-form-item label="相关去重"><el-switch v-model="params.check_relative" /></el-form-item>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">GP 核心参数</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="演化代数"><el-input-number v-model="params.gp_generations" :min="1" :max="500" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="种群规模"><el-input-number v-model="params.gp_population_size" :min="10" :max="5000" :step="50" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="最大深度"><el-input-number v-model="params.gp_max_depth" :min="2" :max="12" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="精英数量"><el-input-number v-model="params.gp_elite_size" :min="1" :max="500" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="交叉概率"><el-input-number v-model="params.gp_crossover_prob" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="变异概率"><el-input-number v-model="params.gp_mutation_prob" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="叶子概率"><el-input-number v-model="params.gp_leaf_prob" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="常数概率"><el-input-number v-model="params.gp_const_prob" :min="0" :max="0.5" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="锦标赛"><el-input-number v-model="params.gp_tournament_size" :min="2" :max="20" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="精英相关"><el-input-number v-model="params.gp_elite_relative_threshold" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-form-item label="窗口集合"><el-select v-model="params.gp_window_choices" multiple style="width:100%"><el-option v-for="w in [3,5,10,15,20,30,60]" :key="w" :label="w" :value="w" /></el-select></el-form-item>
              <el-form-item label="随机种子"><el-input v-model.number="params.random_seed" placeholder="留空=随机" clearable /></el-form-item>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">GP 高级参数</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="早停代数"><el-input-number v-model="params.gp_early_stopping_generation_count" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="日志间隔"><el-input-number v-model="params.gp_log_interval" :min="1" :max="100" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="深度惩罚"><el-input-number v-model="params.gp_depth_penalty_coef" :min="0" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="惩罚起始"><el-input-number v-model="params.gp_depth_penalty_start_depth" :min="2" :max="12" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="线性惩罚"><el-input-number v-model="params.gp_depth_penalty_linear_coef" :min="0" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="二次惩罚"><el-input-number v-model="params.gp_depth_penalty_quadratic_coef" :min="0" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="小因子惩罚"><el-input-number v-model="params.gp_small_factor_penalty_coef" :min="0" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="假设资金"><el-input-number v-model="params.gp_assumed_initial_capital" :min="1000" :max="10000000" :step="10000" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="停滞代数"><el-input-number v-model="params.gp_elite_stagnation_generation_count" :min="1" :max="50" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="Shock代数"><el-input-number v-model="params.gp_max_shock_generation" :min="0" :max="20" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="一致性惩罚"><el-switch v-model="params.consistency_penalty_enabled" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="惩罚系数"><el-input-number v-model="params.consistency_penalty_coef" :min="0" :max="10" :step="0.1" :precision="2" :disabled="!params.consistency_penalty_enabled" style="width:100%" /></el-form-item></el-col>
              </el-row>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">筛选阈值</el-divider>
              <div v-for="indicator in supportedIndicators" :key="`filter-${indicator}`" class="filter-row">
                <div class="filter-row-title">{{ indicator }}（方向: {{ (indicatorDirection[indicator] || 1) === 1 ? '越大越好' : '越小越好' }}）</div>
                <el-row :gutter="12">
                  <el-col :span="12"><el-input v-model="params.filter_indicator_dict[indicator].mean_threshold" placeholder="均值阈值（空=不筛选）" clearable size="small" /></el-col>
                  <el-col :span="12"><el-input v-model="params.filter_indicator_dict[indicator].yearly_threshold" placeholder="年度阈值（空=不筛选）" clearable size="small" /></el-col>
                </el-row>
              </div>
            </div>
              </el-col>
            </el-row>

              </el-form>
          </div>
        </el-card>
      </el-col>

    </el-row>

    <el-row :gutter="20" class="responsive-row bottom-panel-row" style="margin-top:16px;">
      <el-col :xs="24" :sm="24" :md="10" :lg="9" :xl="9" class="bottom-panel-col">
        <el-card shadow="hover" class="action-panel-card" style="margin-bottom:16px;">
          <template #header><span style="font-weight:600;">开始挖掘</span></template>
          <el-button type="primary" size="large" :loading="mining" @click="handleStartMining" style="width:100%;">
            <el-icon v-if="!mining"><VideoPlay /></el-icon>
            {{ mining ? '挖掘中...' : '🚀 启动因子挖掘' }}
          </el-button>
          <div style="margin-top:10px;color:#909399;font-size:12px;line-height:1.6;">
            提示：先在上方调整超参数，再点击启动；任务状态与结果在右侧实时刷新。
          </div>
        </el-card>

        <el-card v-if="activeMiningTab === 'auto'" shadow="hover" class="action-panel-card" style="margin-bottom:16px;">
          <template #header><span style="font-weight:600;">自动挖掘配置</span></template>
          <el-form :model="autoMiningSettings" label-position="top" size="small">
            <el-form-item label="自动挖掘">
              <el-switch v-model="autoMiningSettings.enabled" />
            </el-form-item>
            <el-form-item label="自动挖掘时间">
              <el-time-picker
                v-model="autoMiningSettings.scheduleTime"
                value-format="HH:mm"
                format="HH:mm"
                :disabled="!autoMiningSettings.enabled"
                style="width:100%;"
              />
            </el-form-item>
            <el-form-item label="任务数量">
              <el-input-number
                v-model="autoMiningSettings.taskCount"
                :min="1"
                :max="20"
                :disabled="!autoMiningSettings.enabled"
                style="width:100%;"
              />
            </el-form-item>
            <div style="font-size:12px;color:#909399;line-height:1.5;">
              版本号预览：{{ autoVersionPreview }}
            </div>
          </el-form>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="14" :lg="15" :xl="15" class="bottom-panel-col">
        <el-card shadow="hover" style="margin-bottom:16px;" v-if="taskId">
          <template #header><span style="font-weight:600;">任务状态</span></template>
          <el-descriptions :column="2" size="small" border>
            <el-descriptions-item label="任务ID">{{ taskId }}</el-descriptions-item>
            <el-descriptions-item label="状态"><el-tag :type="statusTag" size="small">{{ taskStatus }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="进度" :span="2">{{ taskProgress }}</el-descriptions-item>
          </el-descriptions>
          <div v-if="taskError" style="margin-top:12px;"><el-alert :title="taskError" type="error" show-icon :closable="false" style="white-space:pre-wrap;font-size:12px;" /></div>
        </el-card>
        <el-card shadow="hover" style="margin-bottom:16px;" v-if="miningResult">
          <template #header><span style="font-weight:600;">挖掘结果</span></template>
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="入选因子"><el-tag v-for="f in miningResult.selected_fc_name_list" :key="f" size="small" style="margin:2px;">{{ f }}</el-tag><span v-if="!miningResult.selected_fc_name_list?.length" style="color:#909399;">无因子通过筛选</span></el-descriptions-item>
            <el-descriptions-item label="版本">{{ miningResult.version }}</el-descriptions-item>
            <el-descriptions-item label="存储路径">{{ miningResult.config_path || '无' }}</el-descriptions-item>
            <el-descriptions-item label="消息" v-if="miningResult.message">{{ miningResult.message }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
        <el-card shadow="hover" style="margin-bottom:16px;" v-if="perfSummary.length">
          <template #header><span style="font-weight:600;">绩效概览</span></template>
          <el-table :data="perfSummary" stripe size="small" max-height="300" style="width:100%">
            <el-table-column v-for="col in perfColumns" :key="col" :prop="col" :label="col" min-width="100" show-overflow-tooltip />
          </el-table>
        </el-card>
        <div v-if="Object.keys(navCurves).length">
          <el-card v-for="(curve, fcName) in navCurves" :key="fcName" class="chart-card" shadow="hover">
            <NavChart :title="fcName + ' 净值曲线'" :curve-data="curve" height="350px" />
          </el-card>
        </div>

      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { startMining, getMiningStatus, getTasks, getMiningIndicatorOptions } from '../api'
import NavChart from '../components/NavChart.vue'

const MINING_TAB_KEY = 'GP_MINING_ACTIVE_TAB'
const AUTO_PARAMS_KEY = 'GP_AUTO_SEARCH_PARAMS'
const AUTO_WEIGHT_KEY = 'GP_AUTO_SEARCH_FITNESS_INDICATOR_WEIGHT'
const AUTO_SETTINGS_KEY = 'GP_AUTO_SEARCH_SETTINGS'
const AUTO_LAST_TRIGGER_DAY_KEY = 'GP_AUTO_SEARCH_LAST_TRIGGER_DAY'

const supportedIndicators = ref(['Net Return', 'Net Sharpe', 'TS IC'])
const indicatorDirection = ref({ 'Net Return': 1, 'Net Sharpe': 1, 'TS IC': 1 })

const _buildDefaultFitnessIndicatorWeight = (indicators) => {
  const out = {}
  indicators.forEach((indicator) => {
    out[indicator] = indicator === 'TS IC' ? 1 : 0
  })
  return out
}

const _buildDefaultFilterIndicatorDict = (indicators, directionMap) => {
  const out = {}
  indicators.forEach((indicator) => {
    out[indicator] = {
      mean_threshold: null,
      yearly_threshold: null,
      direction: directionMap[indicator] ?? 1,
    }
  })
  if (out['Net Return']) {
    out['Net Return'].mean_threshold = 0.05
    out['Net Return'].yearly_threshold = 0.03
  }
  if (out['Net Sharpe']) {
    out['Net Sharpe'].mean_threshold = 0.5
    out['Net Sharpe'].yearly_threshold = 0.3
  }
  return out
}

const defaultParams = () => ({
  instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0', fc_freq: '1d',
  start_time: '20200101', end_time: '20241231',
  version: new Date().toISOString().slice(0,10).replace(/-/g,'') + '_gp_test',
  portfolio_adjust_method: '1D', interest_method: 'simple', risk_free_rate: false,
  calculate_baseline: true, apply_weighted_price: true, n_jobs: 5, max_factor_count: 50,
  min_window_size: 30,
  fitness_metric: 'ic',
  fitness_indicator_dict: _buildDefaultFitnessIndicatorWeight(supportedIndicators.value),
  apply_rolling_norm: true, rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  check_leakage_count: 20, check_relative: true, relative_threshold: 0.7,
  gp_generations: 20, gp_population_size: 500, gp_max_depth: 6, gp_elite_size: 50,
  gp_elite_relative_threshold: 0.65, gp_crossover_prob: 0.3, gp_mutation_prob: 0.7,
  gp_leaf_prob: 0.2, gp_const_prob: 0.02, gp_tournament_size: 3,
  gp_window_choices: [3,5,10,20,30], random_seed: null,
  gp_early_stopping_generation_count: 20, gp_depth_penalty_coef: 0.0,
  gp_depth_penalty_start_depth: 6, gp_depth_penalty_linear_coef: 0.03,
  gp_depth_penalty_quadratic_coef: 0.0, gp_log_interval: 1,
  gp_small_factor_penalty_coef: 0.0, gp_assumed_initial_capital: 100000,
  gp_elite_stagnation_generation_count: 4, gp_max_shock_generation: 3,
  consistency_penalty_enabled: false, consistency_penalty_coef: 1.0,
  filter_indicator_dict: _buildDefaultFilterIndicatorDict(supportedIndicators.value, indicatorDirection.value),
  // Legacy fields kept for backward compatibility with older backend contracts.
  filter_net_return_mean: 0.05, filter_net_return_yearly: 0.03,
  filter_net_sharpe_mean: 0.5, filter_net_sharpe_yearly: 0.3,
})

const _clone = (obj) => JSON.parse(JSON.stringify(obj))
const _safeParse = (txt, fallback = null) => {
  if (!txt) return fallback
  try { return JSON.parse(txt) } catch { return fallback }
}

const activeMiningTab = ref(localStorage.getItem(MINING_TAB_KEY) || 'start')
const params = reactive(defaultParams())
const manualParamsSnapshot = ref(_clone(params))
const autoMiningSettings = reactive({
  enabled: false,
  scheduleTime: '18:00',
  taskCount: 1,
})

const _localDateText = () => {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}${m}${day}`
}

const _localTimeText = () => {
  const d = new Date()
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  return `${hh}:${mm}`
}

const autoVersionPreview = computed(() => {
  const datePart = _localDateText()
  const count = Number(autoMiningSettings.taskCount || 1)
  if (count <= 1) return `${datePart}_gp_test`
  const versions = Array.from({ length: count }, (_, idx) => idx === 0 ? `${datePart}_gp_test` : `${datePart}_gp_test_${idx}`)
  return versions.join(', ')
})

const _loadAutoParams = () => {
  const savedParams = _safeParse(localStorage.getItem(AUTO_PARAMS_KEY), {}) || {}
  const next = { ...defaultParams(), ...savedParams }
  const savedWeight = _safeParse(localStorage.getItem(AUTO_WEIGHT_KEY), {}) || {}
  next.fitness_indicator_dict = {
    ..._buildDefaultFitnessIndicatorWeight(supportedIndicators.value),
    ...(next.fitness_indicator_dict || {}),
    ...(savedWeight || {}),
  }
  Object.assign(params, next)
}

const _saveAutoParams = () => {
  localStorage.setItem(AUTO_PARAMS_KEY, JSON.stringify(_clone(params)))
  localStorage.setItem(AUTO_WEIGHT_KEY, JSON.stringify(_clone(params.fitness_indicator_dict || {})))
}

const _loadAutoSettings = () => {
  const saved = _safeParse(localStorage.getItem(AUTO_SETTINGS_KEY), {}) || {}
  autoMiningSettings.enabled = !!saved.enabled
  autoMiningSettings.scheduleTime = saved.scheduleTime || '18:00'
  autoMiningSettings.taskCount = Number(saved.taskCount) > 0 ? Number(saved.taskCount) : 1
}

const _saveAutoSettings = () => {
  localStorage.setItem(AUTO_SETTINGS_KEY, JSON.stringify(_clone(autoMiningSettings)))
}

const resetParams = () => {
  Object.assign(params, defaultParams())
  if (activeMiningTab.value === 'auto') {
    _saveAutoParams()
  } else {
    manualParamsSnapshot.value = _clone(params)
  }
}

onMounted(async () => {
  _loadAutoSettings()
  try {
    const { data } = await getMiningIndicatorOptions()
    supportedIndicators.value = data.supported_indicator || supportedIndicators.value
    indicatorDirection.value = data.indicator_direction || indicatorDirection.value

    Object.assign(params, defaultParams())
    manualParamsSnapshot.value = _clone(params)
    if (activeMiningTab.value === 'auto') {
      _loadAutoParams()
    }
  } catch {
    // fallback to local defaults when backend metadata endpoint is unavailable
    if (activeMiningTab.value === 'auto') {
      _loadAutoParams()
    }
  }
})

watch(activeMiningTab, (next, prev) => {
  localStorage.setItem(MINING_TAB_KEY, next)
  if (prev === 'auto') {
    _saveAutoParams()
  } else {
    manualParamsSnapshot.value = _clone(params)
  }
  if (next === 'auto') {
    _loadAutoParams()
  } else {
    Object.assign(params, _clone(manualParamsSnapshot.value || defaultParams()))
  }
})

watch(() => params, () => {
  if (activeMiningTab.value === 'auto') {
    _saveAutoParams()
  }
}, { deep: true })

watch(() => autoMiningSettings, () => {
  _saveAutoSettings()
}, { deep: true })

const mining = ref(false), taskId = ref(''), taskStatus = ref(''), taskProgress = ref(''), taskError = ref('')
const miningResult = ref(null), navCurves = ref({}), perfSummary = ref([]), perfColumns = ref([])
const statusTag = computed(() => taskStatus.value==='completed'?'success':taskStatus.value==='failed'?'danger':'warning')
let pollTimer = null
let autoScheduleTimer = null

const _toNullableNumber = (raw) => {
  if (raw === '' || raw === null || raw === undefined) return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

const _buildMiningPayload = () => {
  const fitnessIndicatorDict = {}
  supportedIndicators.value.forEach((indicator) => {
    const n = _toNullableNumber(params.fitness_indicator_dict?.[indicator])
    fitnessIndicatorDict[indicator] = n === null ? 0 : n
  })

  const filterIndicatorDict = {}
  supportedIndicators.value.forEach((indicator) => {
    const conf = params.filter_indicator_dict?.[indicator] || {}
    filterIndicatorDict[indicator] = {
      mean_threshold: _toNullableNumber(conf.mean_threshold),
      yearly_threshold: _toNullableNumber(conf.yearly_threshold),
      direction: Number(indicatorDirection.value?.[indicator] ?? 1),
    }
  })

  return {
    ...params,
    random_seed: params.random_seed || null,
    fitness_indicator_dict: fitnessIndicatorDict,
    filter_indicator_dict: filterIndicatorDict,
  }
}

const _buildAutoVersionList = () => {
  const datePart = _localDateText()
  const base = `${datePart}_gp_test`
  const count = Math.max(1, Number(autoMiningSettings.taskCount || 1))
  return Array.from({ length: count }, (_, idx) => (idx === 0 ? base : `${base}_${idx}`))
}

const _submitMiningByVersions = async (versionList, submitMode = 'manual') => {
  if (!versionList?.length) return

  try {
    const { data: taskData } = await getTasks()
    const runningVersions = new Set((taskData.tasks || [])
      .filter(t => t.status === 'running')
      .map(t => String(t.version || '').trim()))
    const duplicated = versionList.find(v => runningVersions.has(v))
    if (duplicated) {
      if (submitMode === 'auto') {
        taskProgress.value = `自动挖掘跳过：版本 ${duplicated} 正在运行`
      } else {
        ElMessage.warning(`禁止提交：版本号 ${duplicated} 已在运行中`)
      }
      return
    }
  } catch {
    // If pre-check fails, still try backend submission; backend has the final duplicate guard.
  }

  const payloadBase = _buildMiningPayload()
  const submittedTaskIds = []
  for (const version of versionList) {
    const { data } = await startMining({ ...payloadBase, version })
    submittedTaskIds.push(data.task_id)
  }
  if (submittedTaskIds.length) {
    taskId.value = submittedTaskIds[0]
    taskStatus.value = 'running'
    taskProgress.value = submittedTaskIds.length > 1
      ? `已提交 ${submittedTaskIds.length} 个任务，当前显示第1个任务进度`
      : '任务已提交...'
    startPolling()
    if (submitMode !== 'auto') {
      ElMessage.success(submittedTaskIds.length > 1
        ? `自动挖掘已启动，共 ${submittedTaskIds.length} 个任务`
        : `挖掘任务已启动: ${submittedTaskIds[0]}`)
    }
  }
}

const _autoScheduleTick = async () => {
  if (!autoMiningSettings.enabled || mining.value) return
  const target = String(autoMiningSettings.scheduleTime || '18:00').trim()
  if (!/^\d{2}:\d{2}$/.test(target)) return

  const today = _localDateText()
  const nowHm = _localTimeText()
  const lastTriggered = localStorage.getItem(AUTO_LAST_TRIGGER_DAY_KEY) || ''

  if (nowHm >= target && lastTriggered !== today) {
    localStorage.setItem(AUTO_LAST_TRIGGER_DAY_KEY, today)
    try {
      const versionList = _buildAutoVersionList()
      taskProgress.value = `自动挖掘触发：${today} ${target}`
      await _submitMiningByVersions(versionList, 'auto')
    } catch {
      localStorage.removeItem(AUTO_LAST_TRIGGER_DAY_KEY)
    }
  }
}

const handleStartMining = async () => {
  const manualVersion = String(params.version || '').trim()
  const versionList = activeMiningTab.value === 'auto'
    ? _buildAutoVersionList()
    : [manualVersion]
  if (!versionList[0]) {
    ElMessage.warning('版本号不能为空')
    return
  }

  mining.value = true
  taskError.value = ''
  miningResult.value = null
  navCurves.value = {}
  perfSummary.value = []
  try {
    await _submitMiningByVersions(versionList, activeMiningTab.value === 'auto' ? 'auto' : 'manual')
  } catch (err) {
    ElMessage.error('启动失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    // 仅在“提交请求”阶段显示 loading，提交成功后允许继续发起新任务。
    mining.value = false
  }
}
const startPolling = () => {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const { data } = await getMiningStatus(taskId.value)
      taskStatus.value = data.status; taskProgress.value = data.progress; taskError.value = data.error || ''
      if (data.status === 'completed' || data.status === 'terminated') {
        clearInterval(pollTimer); miningResult.value = data.result
        if (data.result?.nav_data) {
          navCurves.value = data.result.nav_data.nav_curves || {}
          const s = data.result.nav_data.performance_summary || []; perfSummary.value = s
          if (s.length) perfColumns.value = Object.keys(s[0])
        }
        ElMessage.success(data.status === 'completed' ? '因子挖掘完成!' : '任务已终止，已展示当前结果')
      } else if (data.status === 'failed') { clearInterval(pollTimer); ElMessage.error('因子挖掘失败') }
    } catch { /* keep polling */ }
  }, 3000)
}

onMounted(() => {
  if (autoScheduleTimer) clearInterval(autoScheduleTimer)
  autoScheduleTimer = setInterval(() => {
    _autoScheduleTick()
  }, 15000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (autoScheduleTimer) clearInterval(autoScheduleTimer)
})
</script>

<style scoped>
.param-section-grid .param-section {
  height: 100%;
}

.param-scroll-panel {
  max-height: 62vh;
  overflow: auto;
  padding-right: 4px;
}

.param-section {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  background: var(--el-fill-color-extra-light);
  padding: 6px 10px 2px;
  margin-bottom: 10px;
}

.param-section :deep(.el-divider) {
  margin: 6px 0 14px;
}

.param-section :deep(.el-divider__text) {
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.indicator-row {
  display: grid;
  grid-template-columns: 1fr 120px;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.indicator-label {
  font-size: 12px;
  color: var(--el-text-color-regular);
}

.filter-row {
  border: 1px dashed var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 8px;
  margin-bottom: 8px;
}

.filter-row-title {
  font-size: 12px;
  color: var(--el-text-color-regular);
  margin-bottom: 6px;
}

.param-scroll-panel :deep(.el-input),
.param-scroll-panel :deep(.el-select),
.param-scroll-panel :deep(.el-input-number) {
  width: 100%;
  max-width: 100%;
}

.bottom-panel-row {
  align-items: stretch;
}

.bottom-panel-col {
  display: flex;
}

.action-panel-card {
  width: 100%;
  height: 100%;
}

.action-panel-card :deep(.el-card__body) {
  min-height: 140px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
</style>

