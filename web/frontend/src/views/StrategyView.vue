<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Coin /></el-icon> 策略分析</h2>
      <p>选择一个因子运行期货模拟交易策略，展示逐日持仓、损益、净值曲线</p>
    </div>

    <el-row :gutter="20" class="responsive-row">
      <!-- Left: Config -->
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">策略参数</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>
          <div class="param-scroll-panel">
          <el-form :model="sp" label-width="auto" size="small">

            <div class="param-section"><el-divider content-position="left">因子选择</el-divider>
              <el-form-item label="集合">
                <el-select v-model="sp.collection" style="width:100%" @change="onCollChange">
                  <el-option v-for="c in collections" :key="c" :label="c" :value="c" />
                </el-select>
              </el-form-item>
              <el-form-item label="版本">
                <el-select v-model="sp.version" filterable style="width:100%" @change="onVerChange">
                  <el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" />
                </el-select>
              </el-form-item>
              <el-form-item label="因子">
                <el-select v-model="sp.factor_name" filterable style="width:100%" placeholder="选择因子">
                  <el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" />
                </el-select>
              </el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-form-item label="合约"><el-input v-model="sp.instrument_id" /></el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="开始"><el-input v-model="sp.start_time" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="结束"><el-input v-model="sp.end_time" /></el-form-item></el-col>
              </el-row>
              <el-form-item label="数据库"><el-input v-model="sp.database" /></el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">资金与费用</el-divider>
              <el-form-item label="初始资金"><el-input-number v-model="sp.initial_capital" :min="1000" :max="100000000" :step="10000" style="width:100%" /></el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="保证金率"><el-input-number v-model="sp.margin_rate" :min="0.01" :max="1" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="每手手续费"><el-input-number v-model="sp.fee_per_lot" :min="0" :max="100" :step="0.5" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="滑点(点数)"><el-input-number v-model="sp.slippage" :min="0" :max="50" :step="0.5" :precision="2" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="信号延迟天"><el-input-number v-model="sp.signal_delay_days" :min="0" :max="10" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-form-item label="最小开仓比">
                <el-input-number v-model="sp.min_open_ratio" :min="0" :max="1" :step="0.1" :precision="2" style="width:100%" />
              </el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">滚动标准化</el-divider>
              <el-form-item label="启用"><el-switch v-model="sp.apply_rolling_norm" /></el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="窗口"><el-input-number v-model="sp.rolling_norm_window" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="最小样本"><el-input-number v-model="sp.rolling_norm_min_periods" :min="1" :max="200" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="Eps"><el-input-number v-model="sp.rolling_norm_eps" :min="0" :step="1e-9" :precision="10" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="Clip"><el-input-number v-model="sp.rolling_norm_clip" :min="0.5" :max="20" :step="0.5" :precision="1" style="width:100%" /></el-form-item></el-col>
              </el-row>
            </div>

            <el-form-item>
              <el-button type="primary" @click="handleRun" :loading="running" style="width:100%">
                <el-icon v-if="!running"><VideoPlay /></el-icon> {{ running ? '模拟中...' : '🚀 运行策略模拟' }}
              </el-button>
            </el-form-item>
          </el-form>
          </div>
        </el-card>
      </el-col>

      <!-- Right: Results -->
      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <!-- Performance Summary -->
        <template v-if="result">
          <el-card shadow="hover" style="margin-bottom:16px;">
            <template #header><span style="font-weight:600;">策略绩效概览</span></template>
            <el-table :data="result.nav_data.performance_summary" stripe size="small" max-height="250">
              <el-table-column v-for="col in summaryCols" :key="col" :prop="col" :label="col" min-width="100" show-overflow-tooltip />
            </el-table>
          </el-card>

          <!-- NAV Chart -->
          <el-card v-for="(curve, name) in result.nav_data.nav_curves" :key="'sc_'+name" class="chart-card" shadow="hover">
            <NavChart :title="name + ' 策略净值曲线'" :curve-data="curve" height="380px" />
          </el-card>

          <!-- Trade Detail -->
          <el-card shadow="hover" style="margin-top:16px;" v-if="result.nav_data.trade_detail?.length">
            <template #header>
              <div style="display:flex;align-items:center;justify-content:space-between;">
                <span style="font-weight:600;">逐日交易明细</span>
                <el-tag size="small">{{ result.nav_data.trade_detail.length }} 条</el-tag>
              </div>
            </template>
            <el-table :data="result.nav_data.trade_detail" stripe border size="small" max-height="400" style="width:100%">
              <el-table-column prop="time" label="日期" width="110" show-overflow-tooltip>
                <template #default="{row}">{{ row.time?.slice(0,10) }}</template>
              </el-table-column>
              <el-table-column prop="factor_value" label="因子值" width="90" show-overflow-tooltip>
                <template #default="{row}">{{ row.factor_value != null ? row.factor_value.toFixed(4) : '-' }}</template>
              </el-table-column>
              <el-table-column prop="position_lots" label="持仓(手)" width="80" />
              <el-table-column prop="delta_lots" label="变动(手)" width="80" />
              <el-table-column prop="open" label="开盘" width="80" show-overflow-tooltip />
              <el-table-column prop="close" label="收盘" width="80" show-overflow-tooltip />
              <el-table-column prop="daily_gross_pnl" label="毛损益" width="100" show-overflow-tooltip>
                <template #default="{row}"><span :style="{color:row.daily_gross_pnl>0?'#67c23a':row.daily_gross_pnl<0?'#f56c6c':''}">{{ row.daily_gross_pnl?.toFixed(2) }}</span></template>
              </el-table-column>
              <el-table-column prop="daily_net_pnl" label="净损益" width="100" show-overflow-tooltip>
                <template #default="{row}"><span :style="{color:row.daily_net_pnl>0?'#67c23a':row.daily_net_pnl<0?'#f56c6c':''}">{{ row.daily_net_pnl?.toFixed(2) }}</span></template>
              </el-table-column>
              <el-table-column prop="fee" label="手续费" width="80" show-overflow-tooltip />
              <el-table-column prop="equity" label="净值" width="110" show-overflow-tooltip>
                <template #default="{row}">{{ row.equity?.toFixed(2) }}</template>
              </el-table-column>
              <el-table-column prop="is_rebalanced" label="调仓" width="60">
                <template #default="{row}"><el-tag v-if="row.is_rebalanced" type="warning" size="small">是</el-tag></template>
              </el-table-column>
              <el-table-column prop="warning" label="警告" min-width="150" show-overflow-tooltip />
            </el-table>
          </el-card>
        </template>

        <el-card v-if="!result" shadow="hover"><el-empty description="选择因子并配置参数后，点击「运行策略模拟」" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, runStrategy } from '../api'
import NavChart from '../components/NavChart.vue'

const collections = ref([]), versionMap = ref({}), allVersions = ref([]), availableFactors = ref([])
const running = ref(false), result = ref(null)

const sp = reactive({
  version: '', factor_name: '', instrument_id: 'C0',
  start_time: '20200101', end_time: '20241231',
  database: 'factors', collection: 'genetic_programming',
  initial_capital: 1000000, margin_rate: 0.1, fee_per_lot: 2.0, slippage: 1.0,
  apply_rolling_norm: true, rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  signal_delay_days: 1, min_open_ratio: 1.0,
})

const resetParams = () => {
  const keepVersion = sp.version
  const keepFactorName = sp.factor_name
  const keepCollection = sp.collection
  Object.assign(sp, {
    version: keepVersion, factor_name: keepFactorName, instrument_id: 'C0',
    start_time: '20200101', end_time: '20241231',
    database: 'factors', collection: keepCollection,
    initial_capital: 1000000, margin_rate: 0.1, fee_per_lot: 2.0, slippage: 1.0,
    apply_rolling_norm: true, rolling_norm_window: 30, rolling_norm_min_periods: 20,
    rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
    signal_delay_days: 1, min_open_ratio: 1.0,
  })
}

const summaryCols = computed(() => { const s = result.value?.nav_data?.performance_summary; return s?.length ? Object.keys(s[0]) : [] })
const filteredVersions = computed(() => sp.collection && versionMap.value[sp.collection] ? versionMap.value[sp.collection] : allVersions.value)

const fetchVersions = async () => { try { const { data } = await getVersions(); collections.value = data.collections||[]; versionMap.value = data.version_map||{}; allVersions.value = data.all_versions||[] } catch {} }
const onCollChange = () => { sp.version = ''; sp.factor_name = ''; availableFactors.value = [] }
const onVerChange = async () => {
  sp.factor_name = ''
  if (!sp.version) { availableFactors.value = []; return }
  try { const p = { version: sp.version }; if (sp.collection) p.collection = sp.collection; const { data } = await getFactors(p); availableFactors.value = (data.factors||[]).map(f=>f.factor_name) } catch { availableFactors.value = [] }
}

const handleRun = async () => {
  if (!sp.version || !sp.factor_name) { ElMessage.warning('请选择版本和因子'); return }
  running.value = true; result.value = null
  try { const { data } = await runStrategy(sp); result.value = data; ElMessage.success('策略模拟完成') }
  catch (err) { ElMessage.error('策略模拟失败: '+(err.response?.data?.detail||err.message)) }
  finally { running.value = false }
}

onMounted(() => fetchVersions())
</script>

