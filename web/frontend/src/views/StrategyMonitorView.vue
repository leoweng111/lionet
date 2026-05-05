<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Monitor /></el-icon> 策略监控</h2>
      <p>基于最新行情更新后，生成下一交易日（T+1 开盘）的交易建议与账户指标。</p>
    </div>

    <el-row :gutter="20" class="responsive-row">
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">监控参数</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>
          <div class="param-scroll-panel">
            <el-form :model="sp" label-width="auto" size="small">
              <div class="param-section"><el-divider content-position="left">因子选择</el-divider>
                <el-form-item label="集合"><el-select v-model="sp.collection" style="width:100%" @change="onCollChange"><el-option v-for="c in collections" :key="c" :label="c" :value="c" /></el-select></el-form-item>
                <el-form-item label="版本"><el-select v-model="sp.version" filterable style="width:100%" @change="onVerChange"><el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" /></el-select></el-form-item>
                <el-form-item label="因子"><el-select v-model="sp.factor_name" filterable style="width:100%" placeholder="选择1个因子"><el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" /></el-select></el-form-item>
              </div>

              <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
                <el-form-item label="合约"><el-input v-model="sp.instrument_id" /></el-form-item>
                <el-form-item label="数据库"><el-input v-model="sp.database" /></el-form-item>
              </div>

              <div class="param-section"><el-divider content-position="left">交易参数</el-divider>
                <el-form-item label="交易起始日"><el-input v-model="sp.trading_start_time" placeholder="YYYYMMDD" /></el-form-item>
              </div>

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

              <el-form-item>
                <el-button type="primary" @click="handleUpdatePrice" :loading="updatingPrice" style="width:100%;margin-bottom:8px;">
                  <el-icon v-if="!updatingPrice"><Refresh /></el-icon>
                  {{ updatingPrice ? '更新行情中...' : '更新行情数据' }}
                </el-button>
                <el-button type="success" @click="handleGenerateStrategy" :loading="generatingStrategy" style="width:100%;">
                  <el-icon v-if="!generatingStrategy"><Operation /></el-icon>
                  {{ generatingStrategy ? '生成中...' : '生成T+1交易策略' }}
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <el-card shadow="hover" v-if="monitor">
          <template #header><span style="font-weight:600;">监控结论</span></template>
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="最新价格日期">{{ monitor.latest_price_date_cn }}</el-descriptions-item>
            <el-descriptions-item label="下一交易日">{{ monitor.next_trade_date_cn || '-' }}</el-descriptions-item>
            <el-descriptions-item label="账户余额">{{ fmtMoney(monitor.account_equity) }}</el-descriptions-item>
            <el-descriptions-item label="当前持仓">{{ monitor.current_position_lots }} 手</el-descriptions-item>
            <el-descriptions-item label="T日因子值">{{ fmtNum(monitor.factor_value_t, 6) }}</el-descriptions-item>
            <el-descriptions-item label="T+1日因子值">{{ fmtNum(monitor.factor_value_t1, 6) }}</el-descriptions-item>
            <el-descriptions-item label="建议目标仓位">{{ monitor.target_position_lots }} 手</el-descriptions-item>
            <el-descriptions-item label="下日操作建议"><el-tag :type="actionTagType">{{ monitor.action?.action_text }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="累计收益(单利)">{{ fmtPct(monitor.cumulative_return) }}</el-descriptions-item>
            <el-descriptions-item label="年化收益">{{ fmtPct(monitor.annualized_return) }}</el-descriptions-item>
            <el-descriptions-item label="估算保证金">{{ fmtMoney(monitor.required_margin_est) }}</el-descriptions-item>
            <el-descriptions-item label="估算可用资金">{{ fmtMoney(monitor.available_cash_est) }}</el-descriptions-item>
          </el-descriptions>
          <el-alert style="margin-top:12px;" type="info" :title="monitor.reference_price_note" :closable="false" />
          <el-alert style="margin-top:12px;" :type="encouragementType" :title="monitor.encouragement" :closable="false" />
        </el-card>

        <el-card shadow="hover" v-if="summaryRows.length" style="margin-top:12px;">
          <template #header><span style="font-weight:600;">策略绩效（与策略分析同口径）</span></template>
          <el-table :data="summaryRows" stripe size="small" max-height="250">
            <el-table-column v-for="c in summaryColumns" :key="`sm_${c}`" :prop="c" :label="c" min-width="100" show-overflow-tooltip />
          </el-table>
        </el-card>

        <el-card shadow="hover" v-if="latestTrades.length" style="margin-top:12px;">
          <template #header><span style="font-weight:600;">最近交易明细</span></template>
          <el-table :data="latestTrades" stripe size="small" max-height="260">
            <el-table-column prop="time" label="日期" width="120"><template #default="{row}">{{ row.time?.slice(0,10) }}</template></el-table-column>
            <el-table-column prop="factor_value" label="因子值" width="90"><template #default="{row}">{{ fmtNum(row.factor_value, 4) }}</template></el-table-column>
            <el-table-column prop="position_lots" label="持仓" width="70" />
            <el-table-column prop="delta_lots" label="变动" width="70" />
            <el-table-column prop="daily_net_pnl" label="净损益" width="110"><template #default="{row}">{{ fmtMoney(row.daily_net_pnl) }}</template></el-table-column>
            <el-table-column prop="equity" label="净值" width="120"><template #default="{row}">{{ fmtMoney(row.equity) }}</template></el-table-column>
          </el-table>
        </el-card>

        <el-card v-if="!monitor" shadow="hover"><el-empty description="先更新行情数据，再生成T+1交易策略" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, updateStrategyMonitorPrice, generateStrategyMonitor } from '../api'

const SK = 'lionet_strategy_monitor'

const collections = ref([])
const versionMap = ref({})
const allVersions = ref([])
const availableFactors = ref([])
const updatingPrice = ref(false)
const generatingStrategy = ref(false)
const monitor = ref(null)
const summaryRows = ref([])
const summaryColumns = ref([])
const latestTrades = ref([])

const sp = reactive({
  version: '', factor_name: '', instrument_id: 'C0',
  trading_start_time: '20200101',
  database: 'factors', collection: 'genetic_programming',
  initial_capital: 1000000, margin_rate: 0.1, fee_per_lot: 2.0,
  slippage: 1.0, apply_rolling_norm: true,
  rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  signal_delay_days: 1, min_open_ratio: 1.0,
})

const _buildPersistState = () => ({
  sp: { ...sp },
  monitor: monitor.value,
  summaryRows: summaryRows.value,
  summaryColumns: summaryColumns.value,
  latestTrades: latestTrades.value,
})

const _restorePersistState = () => {
  try {
    const raw = sessionStorage.getItem(SK)
    if (!raw) return
    const parsed = JSON.parse(raw)
    if (parsed?.sp && typeof parsed.sp === 'object') {
      Object.assign(sp, parsed.sp)
    }
    monitor.value = parsed?.monitor || null
    summaryRows.value = Array.isArray(parsed?.summaryRows) ? parsed.summaryRows : []
    summaryColumns.value = Array.isArray(parsed?.summaryColumns) ? parsed.summaryColumns : []
    latestTrades.value = Array.isArray(parsed?.latestTrades) ? parsed.latestTrades : []
  } catch {
    // ignore broken session payload
  }
}

const filteredVersions = computed(() =>
  sp.collection && versionMap.value[sp.collection]
    ? versionMap.value[sp.collection]
    : allVersions.value
)

const actionTagType = computed(() => {
  const action = monitor.value?.action?.action || ''
  if (action.includes('LONG')) return 'danger'
  if (action.includes('SHORT')) return 'success'
  if (action === 'HOLD') return 'info'
  return 'warning'
})

const encouragementType = computed(() => {
  const ar = Number(monitor.value?.annualized_return || 0)
  if (ar > 0.15) return 'success'
  if (ar > 0) return 'primary'
  if (ar < 0) return 'warning'
  return 'info'
})

const fmtMoney = (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(2) : '-'
const fmtNum = (v, p = 4) => Number.isFinite(Number(v)) ? Number(v).toFixed(p) : '-'
const fmtPct = (v) => Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(2)}%` : '-'

const resetParams = () => {
  const keep = { version: sp.version, factor_name: sp.factor_name, collection: sp.collection }
  Object.assign(sp, {
    version: keep.version,
    factor_name: keep.factor_name,
    instrument_id: 'C0',
    trading_start_time: '20200101',
    database: 'factors',
    collection: keep.collection,
    initial_capital: 1000000,
    margin_rate: 0.1,
    fee_per_lot: 2.0,
    slippage: 1.0,
    apply_rolling_norm: true,
    rolling_norm_window: 30,
    rolling_norm_min_periods: 20,
    rolling_norm_eps: 1e-8,
    rolling_norm_clip: 5.0,
    signal_delay_days: 1,
    min_open_ratio: 1.0,
  })
  monitor.value = null
  summaryRows.value = []
  summaryColumns.value = []
  latestTrades.value = []
}

const fetchVersions = async () => {
  try {
    const { data } = await getVersions()
    collections.value = data.collections || []
    versionMap.value = data.version_map || {}
    allVersions.value = data.all_versions || []
  } catch {
    // ignore
  }
}

const onCollChange = () => {
  sp.version = ''
  sp.factor_name = ''
  availableFactors.value = []
}

const onVerChange = async (preserveSelected = false) => {
  if (!preserveSelected) {
    sp.factor_name = ''
  }
  if (!sp.version) {
    availableFactors.value = []
    return
  }
  try {
    const q = { version: sp.version }
    if (sp.collection) q.collection = sp.collection
    const { data } = await getFactors(q)
    const factors = data.factors || []
    availableFactors.value = factors.map(f => f.factor_name)
    if (preserveSelected && sp.factor_name && !availableFactors.value.includes(sp.factor_name)) {
      sp.factor_name = ''
    }
  } catch {
    availableFactors.value = []
    if (preserveSelected) {
      sp.factor_name = ''
    }
  }
}

const _buildPayload = () => ({
  ...sp,
  factor_name: sp.factor_name,
})

const _consumeMonitorResult = (result) => {
  monitor.value = result || null
  const summary = monitor.value?.nav_data?.performance_summary || []
  summaryRows.value = summary
  summaryColumns.value = summary.length ? Object.keys(summary[0]) : []
  const trades = monitor.value?.nav_data?.trade_detail || []
  latestTrades.value = trades.slice(Math.max(0, trades.length - 20))
}

const handleUpdatePrice = async () => {
  if (!sp.version || !sp.factor_name) {
    ElMessage.warning('请选择版本和因子')
    return
  }
  if (!String(sp.trading_start_time || '').trim()) {
    ElMessage.warning('请输入交易起始日')
    return
  }
  updatingPrice.value = true
  try {
    await updateStrategyMonitorPrice(_buildPayload())
    ElMessage.success('行情数据已更新到最新交易日')
  } catch (e) {
    ElMessage.error('行情更新失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    updatingPrice.value = false
  }
}

const handleGenerateStrategy = async () => {
  if (!sp.version || !sp.factor_name) {
    ElMessage.warning('请选择版本和因子')
    return
  }
  if (!String(sp.trading_start_time || '').trim()) {
    ElMessage.warning('请输入交易起始日')
    return
  }
  generatingStrategy.value = true
  try {
    const { data } = await generateStrategyMonitor(_buildPayload())
    _consumeMonitorResult(data.result)
    ElMessage.success('T+1交易策略已生成')
  } catch (e) {
    if (e.response?.status === 409) {
      ElMessage.warning(e.response?.data?.detail || '最新交易日数据未更新，请先更新行情数据')
    } else {
      ElMessage.error('策略生成失败: ' + (e.response?.data?.detail || e.message))
    }
  } finally {
    generatingStrategy.value = false
  }
}

onMounted(() => {
  _restorePersistState()
  fetchVersions()
  if (sp.version) {
    onVerChange(true)
  }
})

watch(
  () => _buildPersistState(),
  (val) => {
    try {
      sessionStorage.setItem(SK, JSON.stringify(val))
    } catch {
      // ignore storage quota / serialization issue
    }
  },
  { deep: true }
)
</script>

