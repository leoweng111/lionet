<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Coin /></el-icon> 策略分析</h2>
      <p>选择因子运行期货模拟交易策略（支持多因子），展示逐日持仓、损益、净值曲线</p>
    </div>
    <el-row :gutter="20" class="responsive-row">
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><span style="font-weight:600;">策略参数</span><el-button size="small" @click="resetParams">恢复默认</el-button></div></template>
          <div class="param-scroll-panel">
          <el-form :model="sp" label-width="auto" size="small">
            <div class="param-section"><el-divider content-position="left">因子选择</el-divider>
              <el-form-item label="集合"><el-select v-model="sp.collection" style="width:100%" @change="onCollChange"><el-option v-for="c in collections" :key="c" :label="c" :value="c" /></el-select></el-form-item>
              <el-form-item label="版本"><el-select v-model="sp.version" filterable style="width:100%" @change="onVerChange"><el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" /></el-select></el-form-item>
              <el-form-item label="因子"><el-select v-model="sp.factor_name_list" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%" placeholder="可多选因子"><el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" /></el-select></el-form-item>
              <div style="text-align:right;margin-bottom:8px;"><el-button size="small" link type="primary" @click="sp.factor_name_list=[...availableFactors]" :disabled="!availableFactors.length">全选</el-button><el-button size="small" link @click="sp.factor_name_list=[]">清空选择</el-button></div>
            </div>
            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-form-item label="合约"><el-input v-model="sp.instrument_id" /></el-form-item>
              <el-form-item label="数据库"><el-input v-model="sp.database" /></el-form-item>
            </div>
            <div class="param-section"><el-divider content-position="left">样本内区间</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="sp.start_time" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="sp.end_time" /></el-form-item></el-col></el-row></div>
            <div class="param-section"><el-divider content-position="left">样本外区间</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="oosStart" placeholder="20250101" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="oosEnd" placeholder="20251231" /></el-form-item></el-col></el-row></div>
            <div class="param-section"><el-divider content-position="left">实际样本外区间</el-divider>
              <el-row :gutter="8"><el-col :span="24"><el-form-item label="启用"><el-switch v-model="enableRealOos" /></el-form-item></el-col></el-row>
              <el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="realOosStart" placeholder="20260101" :disabled="!enableRealOos" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="realOosEnd" placeholder="20260330" :disabled="!enableRealOos" /></el-form-item></el-col></el-row></div>
            <div class="param-section"><el-divider content-position="left">资金与费用</el-divider>
              <el-form-item label="初始资金"><el-input-number v-model="sp.initial_capital" :min="1000" :max="100000000" :step="10000" style="width:100%" /></el-form-item>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="保证金率"><el-input-number v-model="sp.margin_rate" :min="0.01" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col><el-col :span="12"><el-form-item label="每手手续费"><el-input-number v-model="sp.fee_per_lot" :min="0" :max="100" :step="0.5" :precision="2" style="width:100%" /></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="滑点(点数)"><el-input-number v-model="sp.slippage" :min="0" :max="50" :step="0.5" :precision="2" style="width:100%" /></el-form-item></el-col><el-col :span="12"><el-form-item label="信号延迟天"><el-input-number v-model="sp.signal_delay_days" :min="0" :max="10" style="width:100%" /></el-form-item></el-col></el-row>
              <el-form-item label="最小开仓比"><el-input-number v-model="sp.min_open_ratio" :min="0" :max="1" :step="0.1" :precision="2" style="width:100%" /></el-form-item>
            </div>
            <div class="param-section"><el-divider content-position="left">滚动标准化</el-divider>
              <el-form-item label="启用"><el-switch v-model="sp.apply_rolling_norm" /></el-form-item>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="窗口"><el-input-number v-model="sp.rolling_norm_window" :min="1" :max="200" style="width:100%" /></el-form-item></el-col><el-col :span="12"><el-form-item label="最小样本"><el-input-number v-model="sp.rolling_norm_min_periods" :min="1" :max="200" style="width:100%" /></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="Eps"><el-input-number v-model="sp.rolling_norm_eps" :min="0" :step="1e-9" :precision="10" style="width:100%" /></el-form-item></el-col><el-col :span="12"><el-form-item label="Clip"><el-input-number v-model="sp.rolling_norm_clip" :min="0.5" :max="20" :step="0.5" :precision="1" style="width:100%" /></el-form-item></el-col></el-row>
            </div>
            <el-form-item><el-button type="primary" @click="handleRun" :loading="running" style="width:100%"><el-icon v-if="!running"><VideoPlay /></el-icon> {{ running ? '模拟中...' : '🚀 运行策略模拟' }}</el-button></el-form-item>
          </el-form></div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <div v-if="results.length" style="text-align:right;margin-bottom:12px;"><el-button size="small" type="danger" plain @click="clearResults"><el-icon><Delete /></el-icon> 清空结果</el-button></div>
        <template v-for="(item, idx) in results" :key="'strat_'+idx">
          <!-- 绩效概览：样本内 / 样本外 / 实际样本外 -->
          <el-card shadow="hover" style="margin-bottom:12px;">
            <template #header><span style="font-weight:600;"><el-tag type="primary" size="small" style="margin-right:6px;">策略</el-tag>{{ item.factor_name }} 绩效概览</span></template>
            <el-descriptions :column="1" size="small" border style="margin-bottom:10px;" v-if="item.factor_formula"><el-descriptions-item label="策略公式"><span style="word-break:break-all;">{{ item.factor_formula }}</span></el-descriptions-item></el-descriptions>

            <!-- 样本内绩效 -->
            <el-tag type="success" size="small" style="margin-bottom:6px;">样本内</el-tag>
            <el-table :data="item.isSummary" stripe size="small" max-height="200" style="margin-bottom:10px;">
              <el-table-column v-for="c in (item.columns || [])" :key="'is_ss_'+item.factor_name+c" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
            </el-table>

            <!-- 样本外绩效 -->
            <template v-if="item.hasOos">
              <el-tag type="warning" size="small" style="margin-bottom:6px;">样本外</el-tag>
              <el-table :data="item.oosSummary" stripe size="small" max-height="200" style="margin-bottom:10px;">
                <el-table-column v-for="c in (item.columns || [])" :key="'oos_ss_'+item.factor_name+c" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
              </el-table>
            </template>

            <!-- 实际样本外绩效 -->
            <template v-if="item.hasRealOos">
              <el-tag type="danger" size="small" style="margin-bottom:6px;">实际样本外</el-tag>
              <el-table :data="item.realOosSummary" stripe size="small" max-height="200">
                <el-table-column v-for="c in (item.columns || [])" :key="'roos_ss_'+item.factor_name+c" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
              </el-table>
            </template>
          </el-card>

          <!-- 净值曲线图（带所有分界点） -->
          <el-card v-for="(curve, name) in item.nav_data.nav_curves" :key="'sc_'+name" class="chart-card" shadow="hover" style="margin-bottom:16px;">
            <NavChart :title="name+' 策略净值'" :curve-data="curve" :split-dates="item.split_dates || []" height="350px" />
          </el-card>

          <!-- 逐日交易明细 -->
          <el-card shadow="hover" style="margin-bottom:24px;" v-if="item.nav_data.trade_detail?.length">
            <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><span style="font-weight:600;">{{ item.factor_name }} 逐日交易明细</span><el-tag size="small">{{ item.nav_data.trade_detail.length }} 条</el-tag></div></template>
            <el-table :data="item.nav_data.trade_detail" stripe border size="small" max-height="300" style="width:100%">
              <el-table-column prop="time" label="日期" width="110"><template #default="{row}">{{ row.time?.slice(0,10) }}</template></el-table-column>
              <el-table-column prop="factor_value" label="因子值" width="90"><template #default="{row}">{{ row.factor_value != null ? row.factor_value.toFixed(4) : '-' }}</template></el-table-column>
              <el-table-column prop="position_lots" label="持仓" width="70" /><el-table-column prop="delta_lots" label="变动" width="70" />
              <el-table-column prop="daily_net_pnl" label="净损益" width="100"><template #default="{row}"><span :style="{color:row.daily_net_pnl>0?'#67c23a':row.daily_net_pnl<0?'#f56c6c':''}">{{ row.daily_net_pnl?.toFixed(2) }}</span></template></el-table-column>
              <el-table-column prop="equity" label="净值" width="110"><template #default="{row}">{{ row.equity?.toFixed(2) }}</template></el-table-column>
            </el-table>
          </el-card>
        </template>
        <el-card v-if="!results.length" shadow="hover"><el-empty description="选择因子并配置参数后，点击「运行策略模拟」" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>
<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, runStrategy } from '../api'
import NavChart from '../components/NavChart.vue'

const SK = 'lionet_strat'
const collections = ref([]), versionMap = ref({}), allVersions = ref([]), availableFactors = ref([])
const running = ref(false), results = ref([])
const factorFormulaMap = ref({})

// IS / OOS / Real-OOS range params
const oosStart = ref('20250101')
const oosEnd = ref('20251231')
const enableRealOos = ref(false)
const realOosStart = ref('20260101')
const realOosEnd = ref('20260330')

const sp = reactive({
  version: '', factor_name_list: [], instrument_id: 'C0',
  start_time: '20200101', end_time: '20241231',
  database: 'factors', collection: 'genetic_programming',
  initial_capital: 1000000, margin_rate: 0.1, fee_per_lot: 2.0,
  slippage: 1.0, apply_rolling_norm: true,
  rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  signal_delay_days: 1, min_open_ratio: 1.0,
})

const filteredVersions = computed(() =>
  sp.collection && versionMap.value[sp.collection]
    ? versionMap.value[sp.collection]
    : allVersions.value
)

const resetParams = () => {
  const kv = sp.version, kf = [...sp.factor_name_list], kc = sp.collection
  Object.assign(sp, {
    version: kv, factor_name_list: kf, instrument_id: 'C0',
    start_time: '20200101', end_time: '20241231',
    database: 'factors', collection: kc,
    initial_capital: 1000000, margin_rate: 0.1, fee_per_lot: 2.0,
    slippage: 1.0, apply_rolling_norm: true,
    rolling_norm_window: 30, rolling_norm_min_periods: 20,
    rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
    signal_delay_days: 1, min_open_ratio: 1.0,
  })
  oosStart.value = '20250101'
  oosEnd.value = '20251231'
  enableRealOos.value = false
  realOosStart.value = '20260101'
  realOosEnd.value = '20260330'
}

const fetchVersions = async () => {
  try {
    const { data } = await getVersions()
    collections.value = data.collections || []
    versionMap.value = data.version_map || {}
    allVersions.value = data.all_versions || []
  } catch { /* */ }
}

const onCollChange = () => {
  sp.version = ''; sp.factor_name_list = []
  availableFactors.value = []; factorFormulaMap.value = {}
}

const pickFormulaFromRecord = (rec) =>
  String(rec?.formula || rec?.factor_formula || rec?.['Factor Formula'] || rec?.expr || '')

const onVerChange = async () => {
  sp.factor_name_list = []
  if (!sp.version) { availableFactors.value = []; factorFormulaMap.value = {}; return }
  try {
    const q = { version: sp.version }
    if (sp.collection) q.collection = sp.collection
    const { data } = await getFactors(q)
    const factors = data.factors || []
    availableFactors.value = factors.map(f => f.factor_name)
    const m = {}
    factors.forEach((f) => {
      if (!f?.factor_name) return
      const formula = pickFormulaFromRecord(f)
      if (formula) m[f.factor_name] = formula
    })
    factorFormulaMap.value = m
  } catch {
    availableFactors.value = []; factorFormulaMap.value = {}
  }
}

// ── Performance summary helpers ──────────────────────────────────────

const toYearNumber = (v) => {
  if (v == null) return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

const buildWindowSummary = (summary, startDay, endDay) => {
  if (!summary || !summary.length) return []
  const startYear = toYearNumber(String(startDay || '').slice(0, 4))
  const endYear = toYearNumber(String(endDay || '').slice(0, 4))
  if (!startYear || !endYear) return []

  const yearRows = summary.filter((r) => {
    const y = toYearNumber(r.year)
    return y != null && y >= startYear && y <= endYear
  })

  if (yearRows.length === 1) {
    return [...yearRows, { ...yearRows[0], year: 'all' }]
  }
  return yearRows
}

// ── Handle run ──────────────────────────────────────────────────────

const handleRun = async () => {
  if (!sp.version || !sp.factor_name_list.length) {
    ElMessage.warning('请选择版本和因子'); return
  }
  running.value = true
  try {
    const hasOos = Boolean(oosStart.value && oosEnd.value)
    const hasRealOos = enableRealOos.value && Boolean(realOosStart.value && realOosEnd.value)

    // 计算请求的结束时间：取样本内、样本外、实际样本外的最大结束时间
    const endTimes = [sp.end_time]
    if (hasOos) endTimes.push(oosEnd.value)
    if (hasRealOos) endTimes.push(realOosEnd.value)
    const finalEndTime = endTimes.reduce((max, t) => t > max ? t : max, '')

    const payload = {
      ...sp,
      start_time: sp.start_time,
      end_time: finalEndTime,
    }

    const { data } = await runStrategy(payload)

    // 构建分割点数组（传给 NavChart）
    const splitDates = []
    if (hasOos) splitDates.push(oosStart.value)
    if (hasRealOos) splitDates.push(realOosStart.value)

    results.value = (data.results || []).map((item) => {
      const summary = item.nav_data?.performance_summary || []
      const isSummary = buildWindowSummary(summary, sp.start_time, sp.end_time)
      const oosSummary = hasOos ? buildWindowSummary(summary, oosStart.value, oosEnd.value) : []
      const realOosSummary = hasRealOos ? buildWindowSummary(summary, realOosStart.value, realOosEnd.value) : []
      const cols = isSummary.length
        ? Object.keys(isSummary[0])
        : (oosSummary.length ? Object.keys(oosSummary[0]) : (realOosSummary.length ? Object.keys(realOosSummary[0]) : []))

      return {
        ...item,
        factor_formula: factorFormulaMap.value[item.factor_name] || '',
        isSummary,
        oosSummary,
        realOosSummary,
        hasOos,
        hasRealOos,
        columns: cols,
        split_dates: splitDates,
      }
    })
    ElMessage.success('策略模拟完成')
  } catch (e) {
    ElMessage.error('策略模拟失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    running.value = false
  }
}

const clearResults = () => { results.value = []; sessionStorage.removeItem(SK) }
watch(results, () => {
  try { sessionStorage.setItem(SK, JSON.stringify(results.value)) } catch { /* */ }
}, { deep: true })
onMounted(() => {
  fetchVersions()
  try {
    const s = sessionStorage.getItem(SK)
    if (s) results.value = JSON.parse(s)
  } catch { /* */ }
})
</script>
