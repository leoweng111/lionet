<template>
  <div>
    <div class="page-header">
      <h2><el-icon><TrendCharts /></el-icon> 回测分析</h2>
      <p>对数据库中已有因子进行样本内/外回测，展示净值曲线与绩效指标</p>
    </div>

    <el-row :gutter="20" class="responsive-row">
      <!-- Left: Config -->
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">回测参数</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>
          <div class="param-scroll-panel">
          <el-form :model="btParams" label-width="auto" size="small">

            <div class="param-section"><el-divider content-position="left">因子选择</el-divider>
              <el-form-item label="集合">
                <el-select v-model="btParams.collection" style="width:100%" @change="onCollectionChange">
                  <el-option v-for="c in collections" :key="c" :label="c" :value="c" />
                </el-select>
              </el-form-item>
              <el-form-item label="版本">
                <el-select v-model="btParams.version" filterable style="width:100%" @change="onVersionChange">
                  <el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" />
                </el-select>
              </el-form-item>
              <el-form-item label="因子">
                <el-select v-model="btParams.fc_name_list" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%" placeholder="选择因子">
                  <el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" />
                </el-select>
              </el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-form-item label="合约"><el-input v-model="btParams.instrument_id_list" /></el-form-item>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="因子频率"><el-select v-model="btParams.fc_freq" style="width:100%"><el-option label="1d" value="1d" /><el-option label="5m" value="5m" /><el-option label="1m" value="1m" /></el-select></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="调仓频率"><el-select v-model="btParams.portfolio_adjust_method" style="width:100%"><el-option label="1D" value="1D" /><el-option label="1M" value="1M" /><el-option label="1Q" value="1Q" /><el-option label="min" value="min" /></el-select></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="利息方式"><el-select v-model="btParams.interest_method" style="width:100%"><el-option label="simple" value="simple" /><el-option label="compound" value="compound" /></el-select></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="btParams.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="8"><el-form-item label="基准"><el-switch v-model="btParams.calculate_baseline" /></el-form-item></el-col>
                <el-col :span="8"><el-form-item label="无风险"><el-switch v-model="btParams.risk_free_rate" /></el-form-item></el-col>
                <el-col :span="8"><el-form-item label="复权"><el-switch v-model="btParams.apply_weighted_price" /></el-form-item></el-col>
              </el-row>
            </div>

            <div class="param-section"><el-divider content-position="left">样本内区间</el-divider>
              <el-row :gutter="8">
                <el-col :span="12"><el-form-item label="开始"><el-input v-model="btParams.start_time" placeholder="20200101" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="结束"><el-input v-model="btParams.end_time" placeholder="20241231" /></el-form-item></el-col>
              </el-row>
            </div>

            <div class="param-section"><el-divider content-position="left">样本外区间 (可选)</el-divider>
              <el-row :gutter="8">
                <el-col :span="12"><el-form-item label="开始"><el-input v-model="oosStart" placeholder="20250101" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="结束"><el-input v-model="oosEnd" placeholder="20260101" /></el-form-item></el-col>
              </el-row>
            </div>

            <el-form-item>
              <el-button type="primary" @click="handleBacktest(false)" :loading="backtesting" style="width:100%">
                <el-icon v-if="!backtesting"><CaretRight /></el-icon> {{ backtesting ? '回测中...' : '运行样本内回测' }}
              </el-button>
            </el-form-item>
            <el-form-item v-if="oosStart && oosEnd">
              <el-button type="warning" @click="handleBacktest(true)" :loading="backtesting" style="width:100%">运行样本外回测</el-button>
            </el-form-item>
          </el-form>
          </div>
        </el-card>
      </el-col>

      <!-- Right: Results -->
      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <template v-if="isResult">
          <el-card shadow="hover" style="margin-bottom:16px;">
            <template #header><span style="font-weight:600;"><el-tag type="success" size="small" style="margin-right:6px;">样本内</el-tag>绩效概览</span></template>
            <el-table :data="isResult.nav_data.performance_summary" stripe size="small" max-height="250">
              <el-table-column v-for="col in isSummaryCols" :key="'is'+col" :prop="col" :label="col" min-width="100" show-overflow-tooltip />
            </el-table>
          </el-card>
          <el-card v-for="(curve, name) in isResult.nav_data.nav_curves" :key="'is_c_'+name" class="chart-card" shadow="hover">
            <NavChart :title="name + ' 样本内净值曲线'" :curve-data="curve" height="350px" />
          </el-card>
        </template>
        <template v-if="oosResult">
          <el-card shadow="hover" style="margin-bottom:16px;">
            <template #header><span style="font-weight:600;"><el-tag type="warning" size="small" style="margin-right:6px;">样本外</el-tag>绩效概览</span></template>
            <el-table :data="oosResult.nav_data.performance_summary" stripe size="small" max-height="250">
              <el-table-column v-for="col in oosSummaryCols" :key="'oos'+col" :prop="col" :label="col" min-width="100" show-overflow-tooltip />
            </el-table>
          </el-card>
          <el-card v-for="(curve, name) in oosResult.nav_data.nav_curves" :key="'oos_c_'+name" class="chart-card" shadow="hover">
            <NavChart :title="name + ' 样本外净值曲线'" :curve-data="curve" height="350px" />
          </el-card>
        </template>
        <el-card v-if="!isResult && !oosResult" shadow="hover"><el-empty description="选择版本和因子后，点击运行回测" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, runBacktest } from '../api'
import NavChart from '../components/NavChart.vue'

const collections = ref([]), versionMap = ref({}), allVersions = ref([]), availableFactors = ref([])
const backtesting = ref(false), oosStart = ref(''), oosEnd = ref('')

const btParams = reactive({
  version: '', fc_name_list: [], collection: 'genetic_programming',
  instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0',
  fc_freq: '1d', start_time: '20200101', end_time: '20241231',
  portfolio_adjust_method: '1D', interest_method: 'simple',
  risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5,
})

const resetParams = () => {
  const keepVersion = btParams.version
  const keepFcList = [...btParams.fc_name_list]
  const keepCollection = btParams.collection
  Object.assign(btParams, {
    version: keepVersion, fc_name_list: keepFcList, collection: keepCollection,
    instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0',
    fc_freq: '1d', start_time: '20200101', end_time: '20241231',
    portfolio_adjust_method: '1D', interest_method: 'simple',
    risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5,
  })
  oosStart.value = ''; oosEnd.value = ''
}

const isResult = ref(null), oosResult = ref(null)
const isSummaryCols = computed(() => { const s = isResult.value?.nav_data?.performance_summary; return s?.length ? Object.keys(s[0]) : [] })
const oosSummaryCols = computed(() => { const s = oosResult.value?.nav_data?.performance_summary; return s?.length ? Object.keys(s[0]) : [] })
const filteredVersions = computed(() => btParams.collection && versionMap.value[btParams.collection] ? versionMap.value[btParams.collection] : allVersions.value)

const fetchVersions = async () => { try { const { data } = await getVersions(); collections.value = data.collections||[]; versionMap.value = data.version_map||{}; allVersions.value = data.all_versions||[] } catch {} }
const onCollectionChange = () => { btParams.version = ''; btParams.fc_name_list = []; availableFactors.value = [] }
const onVersionChange = async () => {
  btParams.fc_name_list = []
  if (!btParams.version) { availableFactors.value = []; return }
  try { const p = { version: btParams.version }; if (btParams.collection) p.collection = btParams.collection; const { data } = await getFactors(p); availableFactors.value = (data.factors||[]).map(f=>f.factor_name) } catch { availableFactors.value = [] }
}
const handleBacktest = async (isOOS) => {
  if (!btParams.version || !btParams.fc_name_list.length) { ElMessage.warning('请先选择版本和因子'); return }
  backtesting.value = true
  const payload = { ...btParams }
  if (isOOS) { payload.start_time = oosStart.value; payload.end_time = oosEnd.value }
  try { const { data } = await runBacktest(payload); if (isOOS) oosResult.value = data; else isResult.value = data; ElMessage.success((isOOS?'样本外':'样本内')+'回测完成') }
  catch (err) { ElMessage.error('回测失败: '+(err.response?.data?.detail||err.message)) }
  finally { backtesting.value = false }
}
onMounted(() => {
  fetchVersions()
  const prefill = sessionStorage.getItem('backtest_prefill')
  if (prefill) { try { const p = JSON.parse(prefill); if(p.version) btParams.version=p.version; if(p.collection) btParams.collection=p.collection; if(p.fc_name_list){btParams.fc_name_list=p.fc_name_list; availableFactors.value=p.fc_name_list} } catch{}; sessionStorage.removeItem('backtest_prefill') }
})
</script>

