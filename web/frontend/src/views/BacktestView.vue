<template>
  <div>
    <div class="page-header">
      <h2><el-icon><TrendCharts /></el-icon> 回测分析</h2>
      <p>对数据库中已有因子进行样本内/外回测，支持多因子同时回测，逐因子展示绩效与净值曲线</p>
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

              <template v-if="inputMode === 'db'">
                <el-form-item label="集合"><el-select v-model="p.collection" style="width:100%" @change="onCollChange"><el-option v-for="c in collections" :key="c" :label="c" :value="c" /></el-select></el-form-item>
                <el-form-item label="版本"><el-select v-model="p.version" filterable style="width:100%" @change="onVerChange"><el-option v-for="v in filteredVersions" :key="v" :label="v" :value="v" /></el-select></el-form-item>
                <el-form-item label="因子"><el-select v-model="p.fc_name_list" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%" placeholder="可多选因子"><el-option v-for="f in availableFactors" :key="f" :label="f" :value="f" /></el-select></el-form-item>
                <div style="text-align:right;margin-bottom:8px;"><el-button size="small" link type="primary" @click="p.fc_name_list=[...availableFactors]" :disabled="!availableFactors.length">全选</el-button><el-button size="small" link @click="p.fc_name_list=[]">清空选择</el-button></div>
              </template>

              <template v-else>
                <el-form-item label="因子公式">
                  <el-input v-model="formulaInput" type="textarea" :autosize="{ minRows: 2, maxRows: 6 }" placeholder="输入因子公式，例如: OpRollNorm(TsMean(close, 10), 30, 20, 1e-08, 5)" />
                </el-form-item>
              </template>
            </div>
            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-form-item label="合约"><el-input v-model="p.instrument_id_list" /></el-form-item>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="因子频率"><el-select v-model="p.fc_freq" style="width:100%"><el-option label="1d" value="1d" /><el-option label="5m" value="5m" /><el-option label="1m" value="1m" /></el-select></el-form-item></el-col><el-col :span="12"><el-form-item label="调仓频率"><el-select v-model="p.portfolio_adjust_method" style="width:100%"><el-option label="1D" value="1D" /><el-option label="1M" value="1M" /><el-option label="1Q" value="1Q" /><el-option label="min" value="min" /></el-select></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="12"><el-form-item label="利息方式"><el-select v-model="p.interest_method" style="width:100%"><el-option label="simple" value="simple" /><el-option label="compound" value="compound" /></el-select></el-form-item></el-col><el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="p.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col></el-row>
              <el-row :gutter="12"><el-col :span="8"><el-form-item label="基准"><el-switch v-model="p.calculate_baseline" /></el-form-item></el-col><el-col :span="8"><el-form-item label="无风险"><el-switch v-model="p.risk_free_rate" /></el-form-item></el-col><el-col :span="8"><el-form-item label="复权"><el-switch v-model="p.apply_weighted_price" /></el-form-item></el-col></el-row>
            </div>
            <div class="param-section"><el-divider content-position="left">样本内区间</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="p.start_time" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="p.end_time" /></el-form-item></el-col></el-row></div>
            <div class="param-section"><el-divider content-position="left">样本外区间 (可选)</el-divider><el-row :gutter="8"><el-col :span="12"><el-form-item label="开始"><el-input v-model="oosStart" placeholder="20250101" /></el-form-item></el-col><el-col :span="12"><el-form-item label="结束"><el-input v-model="oosEnd" placeholder="20260101" /></el-form-item></el-col></el-row></div>
            <el-form-item><el-button type="primary" @click="handleBt(false)" :loading="loading" style="width:100%"><el-icon v-if="!loading"><CaretRight /></el-icon>{{ loading ? '回测中...' : '运行样本内回测' }}</el-button></el-form-item>
            <el-form-item v-if="oosStart && oosEnd"><el-button type="warning" @click="handleBt(true)" :loading="loading" style="width:100%">运行样本外回测</el-button></el-form-item>
          </el-form></div>
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
        <div v-if="isR || oosR" style="text-align:right;margin-bottom:12px;"><el-button size="small" type="danger" plain @click="clearResults"><el-icon><Delete /></el-icon> 清空结果</el-button></div>
        <template v-if="isR"><template v-for="(curve, fn) in isR.nav_data.nav_curves" :key="'is_'+fn">
          <el-card shadow="hover" style="margin-bottom:12px;"><template #header><span style="font-weight:600;"><el-tag type="success" size="small" style="margin-right:6px;">样本内</el-tag>{{ fn }} 绩效</span></template><el-table :data="fcSum(isR,fn)" stripe size="small" max-height="200"><el-table-column v-for="c in sCols(isR)" :key="'is_s_'+fn+c" :prop="c" :label="c" min-width="100" show-overflow-tooltip /></el-table></el-card>
          <el-card class="chart-card" shadow="hover" style="margin-bottom:24px;"><NavChart :title="fn+' 样本内净值'" :curve-data="curve" height="350px" /></el-card>
        </template></template>
        <template v-if="oosR"><template v-for="(curve, fn) in oosR.nav_data.nav_curves" :key="'oos_'+fn">
          <el-card shadow="hover" style="margin-bottom:12px;"><template #header><span style="font-weight:600;"><el-tag type="warning" size="small" style="margin-right:6px;">样本外</el-tag>{{ fn }} 绩效</span></template><el-table :data="fcSum(oosR,fn)" stripe size="small" max-height="200"><el-table-column v-for="c in sCols(oosR)" :key="'oos_s_'+fn+c" :prop="c" :label="c" min-width="100" show-overflow-tooltip /></el-table></el-card>
          <el-card class="chart-card" shadow="hover" style="margin-bottom:24px;"><NavChart :title="fn+' 样本外净值'" :curve-data="curve" height="350px" /></el-card>
        </template></template>
        <el-card v-if="!isR && !oosR" shadow="hover"><el-empty description="选择版本和因子后，点击运行回测" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>
<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, getFactors, runBacktest } from '../api'
import NavChart from '../components/NavChart.vue'

const SK = 'lionet_bt'
const collections = ref([]), versionMap = ref({}), allVersions = ref([]), availableFactors = ref([])
const loading = ref(false), oosStart = ref(''), oosEnd = ref('')
const inputMode = ref('db')
const formulaInput = ref('')
const p = reactive({ version: '', fc_name_list: [], collection: 'genetic_programming', instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0', fc_freq: '1d', start_time: '20200101', end_time: '20241231', portfolio_adjust_method: '1D', interest_method: 'simple', risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5 })
const resetParams = () => { const kv = p.version, kf = [...p.fc_name_list], kc = p.collection; Object.assign(p, { version: kv, fc_name_list: kf, collection: kc, instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0', fc_freq: '1d', start_time: '20200101', end_time: '20241231', portfolio_adjust_method: '1D', interest_method: 'simple', risk_free_rate: false, calculate_baseline: true, apply_weighted_price: true, n_jobs: 5 }); oosStart.value = ''; oosEnd.value = ''; formulaInput.value = '' }
const isR = ref(null), oosR = ref(null)
const filteredVersions = computed(() => p.collection && versionMap.value[p.collection] ? versionMap.value[p.collection] : allVersions.value)
const sCols = (r) => { const s = r?.nav_data?.performance_summary; return s?.length ? Object.keys(s[0]) : [] }
const fcSum = (r, fn) => { const s = r?.nav_data?.performance_summary || []; const nc = s.length && s[0]['factor_name'] !== undefined ? 'factor_name' : (s.length && s[0]['Factor Name'] !== undefined ? 'Factor Name' : null); return nc ? s.filter(x => x[nc] === fn) : s }
const fetchVersions = async () => { try { const { data } = await getVersions(); collections.value = data.collections || []; versionMap.value = data.version_map || {}; allVersions.value = data.all_versions || [] } catch { /* */ } }
const onCollChange = () => { p.version = ''; p.fc_name_list = []; availableFactors.value = [] }
const onVerChange = async () => { p.fc_name_list = []; if (!p.version) { availableFactors.value = []; return }; try { const q = { version: p.version }; if (p.collection) q.collection = p.collection; const { data } = await getFactors(q); availableFactors.value = (data.factors || []).map(f => f.factor_name) } catch { availableFactors.value = [] } }
const handleBt = async (isOOS) => {
  if (inputMode.value === 'db') {
    if (!p.version || !p.fc_name_list.length) { ElMessage.warning('请先选择版本和因子'); return }
  } else {
    if (!formulaInput.value.trim()) { ElMessage.warning('请输入因子公式'); return }
  }
  loading.value = true
  const pl = { ...p }
  if (inputMode.value === 'formula') {
    pl.formula = formulaInput.value.trim()
    pl.version = ''
    pl.fc_name_list = []
  }
  if (isOOS) { pl.start_time = oosStart.value; pl.end_time = oosEnd.value }
  try {
    const { data } = await runBacktest(pl)
    if (isOOS) oosR.value = data; else isR.value = data
    ElMessage.success((isOOS ? '样本外' : '样本内') + '回测完成')
  } catch (e) {
    const detail = e.response?.data?.detail || e.message || '未知错误'
    ElMessage.error({ message: '回测失败: ' + detail, duration: 8000, showClose: true })
  } finally { loading.value = false }
}
const clearResults = () => { isR.value = null; oosR.value = null; sessionStorage.removeItem(SK) }
watch([isR, oosR], () => { try { sessionStorage.setItem(SK, JSON.stringify({ is: isR.value, oos: oosR.value })) } catch { /* */ } }, { deep: true })
onMounted(() => { fetchVersions(); try { const s = sessionStorage.getItem(SK); if (s) { const d = JSON.parse(s); isR.value = d.is; oosR.value = d.oos } } catch { /* */ }; const pf = sessionStorage.getItem('backtest_prefill'); if (pf) { try { const d = JSON.parse(pf); if (d.version) p.version = d.version; if (d.collection) p.collection = d.collection; if (d.fc_name_list) { p.fc_name_list = d.fc_name_list; availableFactors.value = d.fc_name_list } } catch { /* */ }; sessionStorage.removeItem('backtest_prefill') } })
</script>
