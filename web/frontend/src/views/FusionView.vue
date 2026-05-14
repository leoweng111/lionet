<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Connection /></el-icon> 因子融合</h2>
      <p>基于数据库因子做逐步融合，支持多指标加权评估和样本外比例控制，结果落库到 `factors.factor_fusion`</p>
    </div>

    <el-row :gutter="20" class="responsive-row">
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">融合参数</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>

          <div class="param-scroll-panel">
            <el-form :model="p" label-width="auto" size="small">
              <div class="param-section">
                <el-divider content-position="center">基础参数</el-divider>
                <el-form-item label="融合方法">
                  <el-select v-model="p.fusion_method" style="width:100%">
                    <el-option label="avg_weight" value="avg_weight" />
                  </el-select>
                </el-form-item>
                <el-form-item label="来源集合">
                  <el-select v-model="_selectedCollections" multiple collapse-tags collapse-tags-tooltip style="width:100%">
                    <el-option v-for="c in collections" :key="c" :label="c" :value="c" />
                  </el-select>
                </el-form-item>
                <el-form-item label="限定版本(可选)">
                  <el-select v-model="_selectedVersions" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%">
                    <el-option v-for="v in allVersions" :key="v" :label="v" :value="v" />
                  </el-select>
                </el-form-item>
                <el-form-item label="版本号">
                  <el-input v-model="p.version" />
                </el-form-item>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="合约"><el-input v-model="p.instrument_id_list" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="频率"><el-select v-model="p.fc_freq" style="width:100%"><el-option label="1d" value="1d" /><el-option label="5m" value="5m" /><el-option label="1m" value="1m" /></el-select></el-form-item></el-col>
                </el-row>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="开始日期"><el-input v-model="p.start_time" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="结束日期"><el-input v-model="p.end_time" /></el-form-item></el-col>
                </el-row>
                <el-form-item label="样本外比例">
                  <el-input-number v-model="p.outsample_ratio" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" />
                </el-form-item>
                <el-row :gutter="12">
                  <el-col :span="12">
                    <el-form-item label="样本外开始">
                      <el-input v-model="p.outsample_start_time" placeholder="20250101" :disabled="!p.outsample_ratio" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="12">
                    <el-form-item label="样本外结束">
                      <el-input v-model="p.outsample_end_time" placeholder="20251231" :disabled="!p.outsample_ratio" />
                    </el-form-item>
                  </el-col>
                </el-row>
              </div>

              <div class="param-section">
                <el-divider content-position="center">融合控制</el-divider>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="最大融合数"><el-input-number v-model="p.max_fusion_count" :min="1" :max="50" style="width:100%" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="p.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col>
                </el-row>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="相似度检查"><el-switch v-model="p.check_relative" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="相似度阈值"><el-input-number v-model="p.relative_threshold" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
                </el-row>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="泄露抽样"><el-input-number v-model="p.check_leakage_count" :min="0" :max="200" style="width:100%" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="复权"><el-switch v-model="p.apply_weighted_price" /></el-form-item></el-col>
                </el-row>
              </div>

              <div class="param-section">
                <el-divider content-position="center">融合评估指标权重</el-divider>
                <div v-for="indicator in supportedIndicators" :key="`fusion-${indicator}`" class="indicator-row">
                  <span class="indicator-label">{{ indicator }}</span>
                  <el-input
                    v-model="p.fusion_indicator_dict[indicator]"
                    placeholder="0"
                    clearable
                    size="small"
                  />
                </div>
              </div>

              <el-form-item style="margin-top:12px;">
                <el-button type="primary" size="large" :loading="loading" @click="run" style="width:100%;">
                  {{ loading ? '融合中...' : '🚀 开始融合' }}
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <el-card shadow="hover" style="margin-bottom:16px;" v-if="taskId">
          <template #header><span style="font-weight:600;">任务状态</span></template>
          <el-descriptions :column="2" size="small" border>
            <el-descriptions-item label="任务ID">{{ taskId }}</el-descriptions-item>
            <el-descriptions-item label="状态"><el-tag :type="taskStatus==='completed' ? 'success' : taskStatus==='failed' ? 'danger' : 'warning'" size="small">{{ taskStatus }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="进度" :span="2">{{ taskProgress }}</el-descriptions-item>
          </el-descriptions>
          <div v-if="taskError" style="margin-top:12px;"><el-alert :title="taskError" type="error" show-icon :closable="false" style="white-space:pre-wrap;font-size:12px;" /></div>
        </el-card>

        <el-card shadow="hover" style="margin-bottom:16px;" v-if="result">
          <template #header><span style="font-weight:600;">融合结果</span></template>
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="落库状态">
              <el-tag :type="result.persisted ? 'success' : 'warning'" size="small">{{ result.persisted ? '已落库' : '未落库(检查未通过或未形成有效融合)' }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="落库位置" v-if="result.persisted">factors.{{ result.collection }}@{{ result.version }}</el-descriptions-item>
            <el-descriptions-item label="融合因子名" v-if="result.fusion_factor_name">{{ result.fusion_factor_name }}</el-descriptions-item>
            <el-descriptions-item label="融合公式" v-if="result.fusion_formula"><span style="word-break:break-all;">{{ result.fusion_formula }}</span></el-descriptions-item>
            <el-descriptions-item label="最终指标">{{ formatMetrics(result.final_metrics) }}</el-descriptions-item>
            <el-descriptions-item label="样本外指标" v-if="result.final_metrics_outsample">{{ formatMetrics(result.final_metrics_outsample) }}</el-descriptions-item>
          </el-descriptions>
        </el-card>

        <el-card shadow="hover" style="margin-bottom:16px;" v-if="selectedFactors.length">
          <template #header><span style="font-weight:600;">入选因子明细</span></template>
          <el-table :data="selectedFactors" stripe size="small" max-height="280" style="width:100%">
            <el-table-column prop="factor_key" label="factor_key" min-width="180" show-overflow-tooltip />
            <el-table-column prop="collection" label="collection" min-width="120" show-overflow-tooltip />
            <el-table-column prop="version" label="version" min-width="120" show-overflow-tooltip />
            <el-table-column prop="factor_name" label="factor_name" min-width="120" show-overflow-tooltip />
          </el-table>
        </el-card>

        <div v-if="Object.keys(navCurves).length">
          <el-card v-for="(curve, fcName) in navCurves" :key="fcName" class="chart-card" shadow="hover">
            <NavChart
              :title="fcName + ' 净值曲线'"
              :title-tooltip="chartTitleTooltip(fcName)"
              :curve-data="curve"
              :split-date="result?.nav_split_date || ''"
              height="350px"
            />
          </el-card>
        </div>

        <el-card v-if="!result" shadow="hover">
          <el-empty description="配置参数后点击「开始融合」" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, reactive, ref, onMounted, onUnmounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, startFusion, getFusionStatus, getFusionIndicatorOptions, updateFusionIndicatorOptions, resetPageConfig } from '../api'
import NavChart from '../components/NavChart.vue'

const today = () => new Date().toISOString().slice(0, 10).replace(/-/g, '')

const supportedIndicators = ref(['Net Return', 'Net Sharpe', 'TS IC'])
const indicatorDirection = ref({})
const serverDefaultFusionIndicatorDict = ref({})

const _buildDefaultFusionIndicatorDict = (indicators) => {
  const out = {}
  indicators.forEach((indicator) => {
    out[indicator] = indicator === 'TS IC' ? 1 : 0
  })
  return out
}

const defaults = () => ({
  fusion_method: 'avg_weight',
  use_version_dict: { 'genetic_programming': [] },
  instrument_type: 'futures_continuous_contract',
  instrument_id_list: 'C0',
  fc_freq: '1d',
  start_time: '20200101',
  end_time: '20241231',
  portfolio_adjust_method: '1D',
  interest_method: 'simple',
  risk_free_rate: false,
  apply_weighted_price: true,
  check_leakage_count: 20,
  check_relative: true,
  relative_threshold: 0.7,
  relative_check_version_list: [],
  max_fusion_count: 5,
  fusion_indicator_dict: {
    ..._buildDefaultFusionIndicatorDict(supportedIndicators.value),
    ...(serverDefaultFusionIndicatorDict.value || {}),
  },
  version: `${today()}_factor_fusion_test`,
  n_jobs: 5,
  base_col_list: ['open', 'high', 'low', 'close', 'volume', 'position'],
  outsample_ratio: 0.0,
  outsample_start_time: '20250101',
  outsample_end_time: '20251231',
})

const p = reactive(defaults())
const _selectedCollections = ref(['genetic_programming'])
const _selectedVersions = ref([])
const loading = ref(false)
const result = ref(null)
const navCurves = ref({})
const taskId = ref('')
const taskStatus = ref('')
const taskProgress = ref('')
const taskError = ref('')
const collections = ref([])
const allVersions = ref([])
let pollTimer = null
let saveFusionDictTimer = null

const selectedFactors = computed(() => result.value?.selected_factors_detail || [])

const resetParams = async () => {
  try {
    const { data } = await resetPageConfig('fusion')
    const serverDefaults = data?.data || {}
    const merged = { ...defaults(), ...serverDefaults }
    merged.fusion_indicator_dict = {
      ..._buildDefaultFusionIndicatorDict(supportedIndicators.value),
      ...(serverDefaults.fusion_indicator_dict || {}),
    }
    Object.assign(p, merged)
    _selectedCollections.value = serverDefaults.selected_collections || ['genetic_programming']
    _selectedVersions.value = serverDefaults.selected_versions || []
  } catch {
    Object.assign(p, defaults())
    _selectedCollections.value = ['genetic_programming']
    _selectedVersions.value = []
  }
}

const formatMetrics = (m) => {
  if (!m) return ''
  return Object.keys(m).map(k => `${k}=${Number(m[k]).toFixed(6)}`).join(', ')
}

const chartTitleTooltip = (fcName) => {
  const fusionName = result.value?.fusion_curve_name || result.value?.fusion_factor_name
  if (fusionName && fcName === fusionName && result.value?.fusion_formula) {
    return `Fusion Formula: ${result.value.fusion_formula}`
  }
  return fcName
}

const _toNullableNumber = (raw) => {
  if (raw === '' || raw === null || raw === undefined) return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

const _saveFusionIndicatorDict = async () => {
  try {
    const dict = {}
    supportedIndicators.value.forEach((indicator) => {
      const n = _toNullableNumber(p.fusion_indicator_dict?.[indicator])
      dict[indicator] = n === null ? 0 : n
    })
    await updateFusionIndicatorOptions({ default_fusion_indicator_dict: dict })
  } catch {
    // silent
  }
}

const _queueSaveFusionDict = () => {
  if (saveFusionDictTimer) clearTimeout(saveFusionDictTimer)
  saveFusionDictTimer = setTimeout(_saveFusionIndicatorDict, 500)
}

watch(() => p.fusion_indicator_dict, () => {
  _queueSaveFusionDict()
}, { deep: true })

const loadMeta = async () => {
  try {
    const [{ data: verData }, { data: indData }] = await Promise.all([
      getVersions(),
      getFusionIndicatorOptions(),
    ])
    collections.value = verData.collections || []
    allVersions.value = verData.all_versions || []
    supportedIndicators.value = indData.supported_indicator || supportedIndicators.value
    indicatorDirection.value = indData.indicator_direction || {}
    serverDefaultFusionIndicatorDict.value = indData.default_fusion_indicator_dict || {}
    p.fusion_indicator_dict = {
      ..._buildDefaultFusionIndicatorDict(supportedIndicators.value),
      ...(serverDefaultFusionIndicatorDict.value || {}),
    }
  } catch {
    collections.value = ['genetic_programming', 'llm_prompt', 'factor_fusion']
    allVersions.value = []
  }
}

const run = async () => {
  if (p.outsample_ratio < 0 || p.outsample_ratio > 1) {
    ElMessage.error('样本外比例必须在 0 到 1 之间')
    return
  }

  loading.value = true
  try {
    taskError.value = ''
    result.value = null
    navCurves.value = {}

    const fusionIndicatorDict = {}
    supportedIndicators.value.forEach((indicator) => {
      const n = _toNullableNumber(p.fusion_indicator_dict?.[indicator])
      fusionIndicatorDict[indicator] = n === null ? 0 : n
    })

    // Build use_version_dict from _selectedCollections and _selectedVersions
    const useVersionDict = {}
    const cols = _selectedCollections.value.length ? _selectedCollections.value : ['genetic_programming']
    const vers = _selectedVersions.value || []
    cols.forEach((c) => {
      useVersionDict[c] = vers.length ? [...vers] : []
    })

    const payload = {
      ...p,
      use_version_dict: useVersionDict,
      fusion_indicator_dict: fusionIndicatorDict,
      relative_check_version_list: (p.relative_check_version_list || []).length ? p.relative_check_version_list : null,
      outsample_start_time: p.outsample_start_time || null,
      outsample_end_time: p.outsample_end_time || null,
    }
    const { data } = await startFusion(payload)
    taskId.value = data.task_id
    taskStatus.value = data.status || 'running'
    taskProgress.value = '融合任务已提交...'
    if (pollTimer) clearInterval(pollTimer)
    pollTimer = setInterval(async () => {
      try {
        const { data: statusData } = await getFusionStatus(taskId.value)
        taskStatus.value = statusData.status
        taskProgress.value = statusData.progress || ''
        taskError.value = statusData.error || ''
        if (statusData.status === 'completed' || statusData.status === 'terminated') {
          clearInterval(pollTimer)
          result.value = statusData.result || null
          navCurves.value = statusData.result?.nav_data?.nav_curves || {}
          ElMessage.success('融合完成')
        } else if (statusData.status === 'failed') {
          clearInterval(pollTimer)
          ElMessage.error('融合失败')
        }
      } catch {
        // keep polling
      }
    }, 3000)
  } catch (e) {
    ElMessage.error(`融合失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    loading.value = false
  }
}

onMounted(loadMeta)
onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (saveFusionDictTimer) clearTimeout(saveFusionDictTimer)
})
</script>

<style scoped>
.param-scroll-panel {
  max-height: 68vh;
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
</style>
