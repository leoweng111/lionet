<template>
  <div>
    <div class="page-header">
      <h2><el-icon><MagicStick /></el-icon> LLM 因子挖掘</h2>
      <p>配置 LLM 参数，利用大语言模型生成因子公式并自动回测，支持实时进度展示和停止</p>
    </div>

    <el-tabs v-model="activeMiningTab" type="border-card" style="margin-bottom:12px;">
      <el-tab-pane label="开始挖掘" name="start" />
      <el-tab-pane label="自动挖掘" name="auto" />
    </el-tabs>

    <el-row v-if="activeMiningTab === 'auto'" :gutter="20" style="margin-bottom:12px;">
      <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
        <el-card shadow="hover" class="action-panel-card">
          <template #header><span style="font-weight:600;">自动挖掘配置</span></template>
          <el-form :model="autoMiningSettings" label-position="top" size="small">
            <el-form-item label="自动挖掘">
              <el-switch v-model="autoMiningSettings.enabled" />
            </el-form-item>
            <el-form-item label="自动挖掘时间">
              <el-time-picker v-model="autoMiningSettings.scheduleTime" value-format="HH:mm" format="HH:mm" :disabled="!autoMiningSettings.enabled" style="width:100%;" />
            </el-form-item>
            <el-form-item label="任务数量">
              <el-input-number v-model="autoMiningSettings.taskCount" :min="1" :max="20" :disabled="!autoMiningSettings.enabled" style="width:100%;" />
            </el-form-item>
            <div style="font-size:12px;color:#909399;line-height:1.5;">
              版本号预览：{{ autoVersionPreview }}
            </div>
          </el-form>
        </el-card>
      </el-col>
    </el-row>

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
            <el-row :gutter="16" class="param-section-grid" style="flex-wrap:wrap;">
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
              <el-row :gutter="12">
                <el-col :span="24">
                  <el-form-item label="样本外比例">
                    <el-input-number v-model="params.outsample_ratio" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" />
                  </el-form-item>
                </el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12">
                  <el-form-item label="样本外开始">
                    <el-input v-model="params.outsample_start_time" placeholder="20250101" :disabled="!params.outsample_ratio" />
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="样本外结束">
                    <el-input v-model="params.outsample_end_time" placeholder="20251231" :disabled="!params.outsample_ratio" />
                  </el-form-item>
                </el-col>
              </el-row>
            </div>
              </el-col>

              <el-col :xs="24" :sm="24" :md="12" :lg="12" :xl="12">
            <div class="param-section"><el-divider content-position="center">LLM 参数</el-divider>
              <el-form-item label="模型">
                <el-select v-model="params.llm_profile_name" style="width:100%" filterable>
                  <el-option v-for="p in llmProfiles" :key="p.name" :label="p.name" :value="p.name" />
                </el-select>
              </el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="每轮因子数"><el-input-number v-model="params.llm_factor_count" :min="1" :max="50" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="早停轮数"><el-input-number v-model="params.llm_early_stopping_round" :min="1" :max="100" style="width:100%" /></el-form-item></el-col>
              </el-row>
            </div>

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

    <!-- Prompt 输入区 + 操作按钮 -->
    <el-row :gutter="20" class="responsive-row" style="margin-top:16px;">
      <el-col :xs="24" :sm="24" :md="10" :lg="9" :xl="9">
        <el-card shadow="hover" style="margin-bottom:16px;">
          <template #header><span style="font-weight:600;">Prompt 设置</span></template>
          <el-input
            v-model="params.llm_user_requirement"
            type="textarea"
            :rows="5"
            placeholder="输入用户需求描述，用于指导 LLM 生成因子"
          />
          <div style="margin-top:12px;display:flex;gap:8px;">
            <el-button type="primary" size="large" :loading="mining" @click="handleStartMining" style="flex:1;">
              <el-icon v-if="!mining"><VideoPlay /></el-icon>
              {{ mining ? '挖掘中...' : '🚀 启动 LLM 因子挖掘' }}
            </el-button>
            <el-button v-if="mining || taskStatus === 'running'" type="danger" size="large" @click="handleStop">
              ⏹ 停止
            </el-button>
          </div>
          <div style="margin-top:10px;color:#909399;font-size:12px;line-height:1.6;">
            提示：LLM 将根据 Prompt 和算子集合自动生成因子公式，生成后自动回测。
          </div>
        </el-card>
      </el-col>

      <!-- 结果区 -->
      <el-col :xs="24" :sm="24" :md="14" :lg="15" :xl="15">
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
import { getLLMProfiles, startLLMMining, getLLMMiningStatus, terminateLLMMining, getMiningIndicatorOptions, getPageConfig, savePageConfig, resetPageConfig } from '../api'
import NavChart from '../components/NavChart.vue'

const supportedIndicators = ref(['Net Return', 'Net Sharpe', 'TS IC'])
const indicatorDirection = ref({ 'Net Return': 1, 'Net Sharpe': 1, 'TS IC': 1 })
const llmProfiles = ref([])
const _clone = (obj) => JSON.parse(JSON.stringify(obj))

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

const _localDateText = () => {
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}${m}${day}`
}

const defaultParams = () => ({
  instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0', fc_freq: '1d',
  start_time: '20200101', end_time: '20241231',
  version: _localDateText() + '_llm_test',
  portfolio_adjust_method: '1D', interest_method: 'simple', risk_free_rate: false,
  calculate_baseline: true, apply_weighted_price: true, n_jobs: 5, max_factor_count: 20,
  min_window_size: 30,
  apply_rolling_norm: true, rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  check_leakage_count: 20, check_relative: true, relative_threshold: 0.7,
  llm_factor_count: 5, llm_early_stopping_round: 20,
  llm_user_requirement: '生成期货的日频量价因子',
  llm_profile_name: 'MiniMax-M2.7-highspeed',
  outsample_ratio: 0.0, outsample_start_time: '20250101', outsample_end_time: '20251231',
  filter_indicator_dict: _buildDefaultFilterIndicatorDict(supportedIndicators.value, indicatorDirection.value),
})

const LLM_MINING_TAB_KEY = 'LLM_MINING_ACTIVE_TAB'
const activeMiningTab = ref(localStorage.getItem(LLM_MINING_TAB_KEY) || 'start')
const params = reactive(defaultParams())
const autoMiningSettings = reactive({
  enabled: false,
  scheduleTime: '18:00',
  taskCount: 1,
})
const configReady = ref(false)
let configSaveTimer = null

const autoVersionPreview = computed(() => {
  const datePart = _localDateText()
  const count = Number(autoMiningSettings.taskCount || 1)
  if (count <= 1) return `${datePart}_llm_test`
  return Array.from({ length: count }, (_, idx) => idx === 0 ? `${datePart}_llm_test` : `${datePart}_llm_test_${idx}`).join(', ')
})

const resetParams = async () => {
  try {
    const pageName = activeMiningTab.value === 'auto' ? 'llm_mining_auto' : 'llm_mining_start'
    const { data } = await resetPageConfig(pageName)
    const serverDefaults = data?.data || {}
    const merged = { ...defaultParams(), ...serverDefaults }
    merged.filter_indicator_dict = {
      ..._buildDefaultFilterIndicatorDict(supportedIndicators.value, indicatorDirection.value),
      ...(serverDefaults.filter_indicator_dict || {}),
    }
    Object.assign(params, merged)
    if (activeMiningTab.value === 'auto') {
      autoMiningSettings.enabled = !!serverDefaults.enabled
      autoMiningSettings.scheduleTime = serverDefaults.schedule_time || serverDefaults.scheduleTime || '18:00'
      autoMiningSettings.taskCount = Number(serverDefaults.task_count ?? serverDefaults.taskCount) > 0 ? Number(serverDefaults.task_count ?? serverDefaults.taskCount) : 1
    }
  } catch {
    Object.assign(params, defaultParams())
  }
}

const _applyPageConfig = (saved = {}) => {
  const merged = { ...defaultParams(), ...(saved || {}) }
  merged.filter_indicator_dict = {
    ..._buildDefaultFilterIndicatorDict(supportedIndicators.value, indicatorDirection.value),
    ...(merged.filter_indicator_dict || {}),
  }
  Object.assign(params, merged)
  if (activeMiningTab.value === 'auto') {
    autoMiningSettings.enabled = !!saved.enabled
    autoMiningSettings.scheduleTime = saved.schedule_time || saved.scheduleTime || '18:00'
    autoMiningSettings.taskCount = Number(saved.task_count ?? saved.taskCount) > 0 ? Number(saved.task_count ?? saved.taskCount) : 1
  }
}

const _savePageConfig = async () => {
  if (!configReady.value) return
  const pageName = activeMiningTab.value === 'auto' ? 'llm_mining_auto' : 'llm_mining_start'
  const payload = _clone(params)
  if (activeMiningTab.value === 'auto') {
    payload.enabled = autoMiningSettings.enabled
    payload.schedule_time = autoMiningSettings.scheduleTime
    payload.task_count = autoMiningSettings.taskCount
  }
  try { await savePageConfig(pageName, payload) } catch { /* silent */ }
}

const _queueSavePageConfig = () => {
  if (configSaveTimer) clearTimeout(configSaveTimer)
  configSaveTimer = setTimeout(_savePageConfig, 300)
}

const _loadPageConfig = async () => {
  const pageName = activeMiningTab.value === 'auto' ? 'llm_mining_auto' : 'llm_mining_start'
  try {
    const { data } = await getPageConfig(pageName)
    _applyPageConfig(data?.saved || {})
  } catch {
    _applyPageConfig({})
  }
}

const mining = ref(false), taskId = ref(''), taskStatus = ref(''), taskProgress = ref(''), taskError = ref('')
const miningResult = ref(null), navCurves = ref({}), perfSummary = ref([]), perfColumns = ref([])
const statusTag = computed(() => taskStatus.value === 'completed' ? 'success' : taskStatus.value === 'failed' ? 'danger' : 'warning')
let pollTimer = null

const _toNullableNumber = (raw) => {
  if (raw === '' || raw === null || raw === undefined) return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

const _buildPayload = () => {
  const filterIndicatorDict = {}
  supportedIndicators.value.forEach((indicator) => {
    const conf = params.filter_indicator_dict?.[indicator] || {}
    filterIndicatorDict[indicator] = {
      mean_threshold: _toNullableNumber(conf.mean_threshold),
      yearly_threshold: _toNullableNumber(conf.yearly_threshold),
      direction: Number(indicatorDirection.value?.[indicator] ?? 1),
    }
  })
  return { ...params, filter_indicator_dict: filterIndicatorDict }
}

const handleStartMining = async () => {
  const version = String(params.version || '').trim()
  if (!version) { ElMessage.warning('版本号不能为空'); return }
  mining.value = true
  taskError.value = ''
  miningResult.value = null
  navCurves.value = {}
  perfSummary.value = []
  try {
    const payload = _buildPayload()
    const { data } = await startLLMMining(payload)
    taskId.value = data.task_id
    taskStatus.value = 'running'
    taskProgress.value = '任务已提交...'
    startPolling()
    ElMessage.success(`LLM 挖掘任务已启动: ${data.task_id}`)
  } catch (err) {
    ElMessage.error('启动失败: ' + (err.response?.data?.detail || err.message))
  } finally { mining.value = false }
}

const handleStop = async () => {
  if (!taskId.value) return
  try {
    await terminateLLMMining(taskId.value)
    ElMessage.info('已发送停止请求')
  } catch (err) {
    ElMessage.error('停止失败: ' + (err.response?.data?.detail || err.message))
  }
}

const startPolling = () => {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const { data } = await getLLMMiningStatus(taskId.value)
      taskStatus.value = data.status
      taskProgress.value = data.progress
      taskError.value = data.error || ''
      if (data.status === 'completed' || data.status === 'terminated') {
        clearInterval(pollTimer)
        miningResult.value = data.result
        if (data.result?.nav_data) {
          navCurves.value = data.result.nav_data.nav_curves || {}
          const s = data.result.nav_data.performance_summary || []
          perfSummary.value = s
          if (s.length) perfColumns.value = Object.keys(s[0])
        }
        ElMessage.success(data.status === 'completed' ? 'LLM 因子挖掘完成!' : '任务已终止')
      } else if (data.status === 'failed') {
        clearInterval(pollTimer)
        ElMessage.error('LLM 因子挖掘失败')
      }
    } catch { /* keep polling */ }
  }, 3000)
}

onMounted(async () => {
  try {
    const [{ data: indData }, { data: profileData }] = await Promise.all([
      getMiningIndicatorOptions(),
      getLLMProfiles(),
    ])
    supportedIndicators.value = indData.supported_indicator || supportedIndicators.value
    indicatorDirection.value = indData.indicator_direction || indicatorDirection.value
    llmProfiles.value = profileData.profiles || []
    if (llmProfiles.value.length && !llmProfiles.value.find(p => p.name === params.llm_profile_name)) {
      params.llm_profile_name = llmProfiles.value[0].name
    }
    await _loadPageConfig()
    configReady.value = true
  } catch {
    configReady.value = true
  }
})

watch(activeMiningTab, async (next) => {
  localStorage.setItem(LLM_MINING_TAB_KEY, next)
  await _loadPageConfig()
})

watch(() => params, () => _queueSavePageConfig(), { deep: true })
watch(() => autoMiningSettings, () => _queueSavePageConfig(), { deep: true })

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (configSaveTimer) clearTimeout(configSaveTimer)
})
</script>

<style scoped>
.param-scroll-panel { max-height: 62vh; overflow: auto; padding-right: 4px; }
.param-section {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  background: var(--el-fill-color-extra-light);
  padding: 6px 10px 2px;
  margin-bottom: 10px;
}
.param-section :deep(.el-divider) { margin: 6px 0 14px; }
.param-section :deep(.el-divider__text) { font-weight: 600; color: var(--el-text-color-primary); }
.filter-row {
  border: 1px dashed var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 8px;
  margin-bottom: 8px;
}
.filter-row-title { font-size: 12px; color: var(--el-text-color-regular); margin-bottom: 6px; }
.param-scroll-panel :deep(.el-input),
.param-scroll-panel :deep(.el-select),
.param-scroll-panel :deep(.el-input-number) { width: 100%; max-width: 100%; }
.action-panel-card { width: 100%; height: 100%; }
</style>

