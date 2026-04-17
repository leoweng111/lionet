<template>
  <div>
    <div class="page-header">
      <h2><el-icon><TrendCharts /></el-icon> 回测分析</h2>
      <p>支持数据库因子或公式回测；若配置样本外区间，将在同一张净值图中展示样本内/外曲线</p>
    </div>
    <el-row :gutter="20" class="responsive-row">
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><span style="font-weight:600;">回测参数</span><el-button size="small" @click="resetParams">恢复默认</el-button></div></template>
          <div class="param-scroll-panel">
          <el-form :model="p" label-width="auto" size="small">
            <div class="param-section"><el-divider content-position="left">因子选择</el-divider>
              <el-form-item label="输入模式">
                <el-radio-group v-model="inputMode" size="small">
                  <el-radio-button value="db">数据库因子</el-radio-button>
                  <el-radio-button value="formula">公式输入</el-radio-button>
                </el-radio-group>
              </el-form-item>
              <el-form-item label="集合"><el-select v-model="p.collection" style="width:100%" :disabled="inputMode==='formula'" @change="onCollChange"><el-option v-for="c in collections" :key="c" :label="c" :value="c" /></el-select></el-form-item>
              <el-form-item label="版本"><el-select v-model="p.version" filterable style="width:100%" :disabled="inputMode==='formula'" @change="onVerChange"><el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" /></el-select></el-form-item>
              <el-form-item label="因子"><el-select v-model="p.fc_name_list" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%" :disabled="inputMode==='formula'" placeholder="可多选因子"><el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" /></el-select></el-form-item>
              <div style="text-align:right;margin-bottom:8px;"><el-button size="small" link type="primary" @click="p.fc_name_list=[...availableFactors]" :disabled="inputMode==='formula' || !availableFactors.length">全选</el-button><el-button size="small" link @click="p.fc_name_list=[]" :disabled="inputMode==='formula'">清空选择</el-button></div>
              <el-form-item label="公式"><el-input type="textarea" :autosize="{ minRows: 2, maxRows: 6 }" v-model="formulaInput" :disabled="inputMode==='db'" placeholder="输入公式，如 OpRollNorm(TsMean(close, 10), 30, 20, 1e-08, 5)" /></el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-form-item label="合约"><el-input v-model="p.instrument_id_list" /></el-form-item>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="因子频率"><el-select v-model="p.fc_freq" style="width:100%"><el-option label="1d" value="1d" /><el-option label="5m" value="5m" /><el-option label="1m" value="1m" /></el-select></el-form-item></el-col><el-col :span="12"><el-form-item label="调仓频率"><el-select v-model="p.portfolio_adjust_method" style="width:100%"><el-option label="1D" value="1D" /><el-option label="1M" value="1M" /><el-option label="1Q" value="1Q" /><el-option label="min" value="min" /></el-select></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="利息方式"><el-select v-model="p.interest_method" style="width:100%"><el-option label="simple" value="simple" /><el-option label="compound" value="compound" /></el-select></el-form-item></el-col><el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="p.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="8"><el-form-item label="基准"><el-switch v-model="p.calculate_baseline" /></el-form-item></el-col><el-col :span="8"><el-form-item label="无风险"><el-switch v-model="p.risk_free_rate" /></el-form-item></el-col><el-col :span="8"><el-form-item label="复权"><el-switch v-model="p.apply_weighted_price" /></el-form-item></el-col></el-row>
            </div>

            <div class="param-section"><el-divider content-position="left">样本内区间</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="p.start_time" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="p.end_time" /></el-form-item></el-col></el-row></div>
            <div class="param-section"><el-divider content-position="left">样本外区间</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="oosStart" placeholder="20250101" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="oosEnd" placeholder="20251231" /></el-form-item></el-col></el-row></div>

            <el-form-item><el-button type="primary" @click="handleBt" :loading="loading" style="width:100%"><el-icon v-if="!loading"><CaretRight /></el-icon>{{ loading ? '回测中...' : '运行回测' }}</el-button></el-form-item>
          </el-form></div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <div v-if="resultMap && Object.keys(resultMap).length" style="text-align:right;margin-bottom:12px;"><el-button size="small" type="danger" plain @click="clearResults"><el-icon><Delete /></el-icon> 清空结果</el-button></div>

        <template v-if="resultMap && Object.keys(resultMap).length">
          <template v-for="(item, fn) in resultMap" :key="fn">
            <el-card shadow="hover" style="margin-bottom:12px;">
              <template #header><span style="font-weight:600;">{{ fn }} 绩效</span></template>
              <el-tag type="success" size="small" style="margin-bottom:6px;">样本内</el-tag>
              <el-table :data="item.isSummary" stripe size="small" max-height="220" style="margin-bottom:10px;">
                <el-table-column v-for="c in item.columns" :key="`is_${fn}_${c}`" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
              </el-table>
              <template v-if="item.hasOos">
                <el-tag type="warning" size="small" style="margin-bottom:6px;">样本外</el-tag>
                <el-table :data="item.oosSummary" stripe size="small" max-height="220">
                  <el-table-column v-for="c in item.columns" :key="`oos_${fn}_${c}`" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
                </el-table>
              </template>
            </el-card>
            <el-card class="chart-card" shadow="hover" style="margin-bottom:24px;"><NavChart :title="fn+' 连续净值曲线'" :curve-data="item.curve" :split-date="item.hasOos ? oosStart : ''" height="360px" /></el-card>
          </template>
        </template>

        <el-card v-if="!resultMap || !Object.keys(resultMap).length" shadow="hover"><el-empty description="配置参数后点击运行回测" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, runBacktest } from '../api'
import NavChart from '../components/NavChart.vue'

const SK = 'lionet_backtest'

const collections = ref([]), versionMap = ref({}), allVersions = ref([]), availableFactors = ref([])
const loading = ref(false)
const inputMode = ref('db')
const formulaInput = ref('')
const oosStart = ref('20250101')
const oosEnd = ref('20251231')
const resultMap = ref(null)

const p = reactive({
  version: '', fc_name_list: [], collection: 'genetic_programming', instrument_type: 'futures_continuous_contract',
  instrument_id_list: 'C0', fc_freq: '1d', start_time: '20200101', end_time: '20241231', portfolio_adjust_method: '1D',
  interest_method: 'simple', risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5,
})

const filteredVersions = computed(() => p.collection && versionMap.value[p.collection] ? versionMap.value[p.collection] : allVersions.value)

const resetParams = () => {
  Object.assign(p, {
    version: '', fc_name_list: [], collection: 'genetic_programming', instrument_type: 'futures_continuous_contract',
    instrument_id_list: 'C0', fc_freq: '1d', start_time: '20200101', end_time: '20241231', portfolio_adjust_method: '1D',
    interest_method: 'simple', risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5,
  })
  oosStart.value = '20250101'
  oosEnd.value = '20251231'
  formulaInput.value = ''
  resultMap.value = null
}

const fetchVersions = async () => {
  try {
    const { data } = await getVersions()
    collections.value = data.collections || []
    versionMap.value = data.version_map || {}
    allVersions.value = data.all_versions || []
  } catch {
    collections.value = []
  }
}

const onCollChange = () => { p.version = ''; p.fc_name_list = []; availableFactors.value = [] }
const onVerChange = async () => {
  p.fc_name_list = []
  if (!p.version) { availableFactors.value = []; return }
  try {
    const q = { version: p.version }
    if (p.collection) q.collection = p.collection
    const { data } = await getFactors(q)
    availableFactors.value = (data.factors || []).map(f => f.factor_name)
  } catch {
    availableFactors.value = []
  }
}

const pickSummaryByFactor = (summary, fn) => {
  const s = summary || []
  if (!s.length) return []
  const nameCol = s[0]['Factor Name'] !== undefined ? 'Factor Name' : (s[0].factor_name !== undefined ? 'factor_name' : null)
  if (!nameCol) return s
  return s.filter(x => x[nameCol] === fn)
}

const handleBt = async () => {
  if (inputMode.value === 'db') {
    if (!p.version || !p.fc_name_list.length) {
      ElMessage.warning('请先选择版本和因子')
      return
    }
  } else if (!formulaInput.value.trim()) {
    ElMessage.warning('请输入因子公式')
    return
  }

  loading.value = true
  try {
    const basePayload = { ...p }
    if (inputMode.value === 'formula') {
      basePayload.formula = formulaInput.value.trim()
      basePayload.version = null
      basePayload.fc_name_list = []
    }

    const hasOos = Boolean(oosStart.value && oosEnd.value)
    const fullPayload = {
      ...basePayload,
      start_time: p.start_time,
      end_time: hasOos ? oosEnd.value : p.end_time,
    }
    const { data: fullData } = await runBacktest(fullPayload)

    const isData = hasOos ? (await runBacktest(basePayload)).data : fullData

    let oosData = null
    if (hasOos) {
      const oosPayload = { ...basePayload, start_time: oosStart.value, end_time: oosEnd.value }
      const { data } = await runBacktest(oosPayload)
      oosData = data
    }

    const fullCurves = fullData?.nav_data?.nav_curves || {}
    const allNames = Array.from(
      new Set([
        ...Object.keys(fullCurves),
        ...Object.keys(isData?.nav_data?.nav_curves || {}),
        ...Object.keys(oosData?.nav_data?.nav_curves || {}),
      ]),
    )

    const merged = {}
    allNames.forEach((fn) => {
      const isSummary = pickSummaryByFactor(isData?.nav_data?.performance_summary || [], fn)
      const oosSummary = pickSummaryByFactor(oosData?.nav_data?.performance_summary || [], fn)
      const cols = isSummary.length ? Object.keys(isSummary[0]) : (oosSummary.length ? Object.keys(oosSummary[0]) : [])
      merged[fn] = {
        isSummary,
        oosSummary,
        hasOos,
        columns: cols,
        curve: fullCurves[fn] || { time: [], gross_nav: [], net_nav: [] },
      }
    })
    resultMap.value = merged
    ElMessage.success('回测完成')
  } catch (e) {
    const detail = e.response?.data?.detail || e.message || '未知错误'
    ElMessage.error({ message: `回测失败: ${detail}`, duration: 10000, showClose: true })
  } finally {
    loading.value = false
  }
}

const clearResults = () => {
  resultMap.value = null
  sessionStorage.removeItem(SK)
}

watch(resultMap, () => {
  try {
    if (resultMap.value && Object.keys(resultMap.value).length) {
      sessionStorage.setItem(SK, JSON.stringify(resultMap.value))
    } else {
      sessionStorage.removeItem(SK)
    }
  } catch {
    // noop
  }
}, { deep: true })

onMounted(() => {
  fetchVersions()
  try {
    const s = sessionStorage.getItem(SK)
    if (s) resultMap.value = JSON.parse(s)
  } catch {
    // noop
  }
})
</script>
