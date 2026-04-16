<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Connection /></el-icon> 因子融合</h2>
      <p>基于数据库因子做逐步融合，支持样本外优先策略，并将结果落库到 `factors.factor_fusion`</p>
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
                <el-divider content-position="left">基础参数</el-divider>
                <el-form-item label="融合方法">
                  <el-select v-model="p.fusion_method" style="width:100%">
                    <el-option label="avg_weight" value="avg_weight" />
                  </el-select>
                </el-form-item>
                <el-form-item label="来源集合">
                  <el-select v-model="p.collection" multiple collapse-tags collapse-tags-tooltip style="width:100%">
                    <el-option v-for="c in collections" :key="c" :label="c" :value="c" />
                  </el-select>
                </el-form-item>
                <el-form-item label="限定版本(可选)">
                  <el-select v-model="p.candidate_versions" multiple filterable collapse-tags collapse-tags-tooltip style="width:100%">
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
                  <el-col :span="12"><el-form-item label="开始"><el-input v-model="p.start_time" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="结束"><el-input v-model="p.end_time" /></el-form-item></el-col>
                </el-row>
              </div>

              <div class="param-section">
                <el-divider content-position="left">融合控制</el-divider>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="最大融合数"><el-input-number v-model="p.max_fusion_count" :min="1" :max="50" style="width:100%" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="并行数"><el-input-number v-model="p.n_jobs" :min="1" :max="32" style="width:100%" /></el-form-item></el-col>
                </el-row>
                <el-form-item label="优化指标">
                  <el-select v-model="p.fusion_metrics" multiple style="width:100%">
                    <el-option label="ic" value="ic" />
                    <el-option label="sharpe" value="sharpe" />
                  </el-select>
                </el-form-item>
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
                <el-divider content-position="left">样本外优先</el-divider>
                <el-form-item label="启用样本外优先"><el-switch v-model="p.consider_outsample" /></el-form-item>
                <el-row :gutter="12">
                  <el-col :span="12"><el-form-item label="样本外开始"><el-input v-model="p.outsample_start_day" placeholder="20250101" /></el-form-item></el-col>
                  <el-col :span="12"><el-form-item label="样本外结束"><el-input v-model="p.outsample_end_day" placeholder="20251231" /></el-form-item></el-col>
                </el-row>
              </div>

              <el-form-item>
                <el-button type="primary" size="large" :loading="loading" @click="run" style="width:100%;">
                  {{ loading ? '融合中...' : '开始融合' }}
                </el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
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
            <NavChart :title="fcName + ' 净值曲线'" :curve-data="curve" height="350px" />
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
import { computed, reactive, ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getVersions, runFusion } from '../api'
import NavChart from '../components/NavChart.vue'

const today = () => new Date().toISOString().slice(0, 10).replace(/-/g, '')
const defaults = () => ({
  fusion_method: 'avg_weight',
  collection: ['genetic_programming'],
  candidate_versions: [],
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
  fusion_metrics: ['ic'],
  version: `${today()}_factor_fusion_test`,
  n_jobs: 5,
  base_col_list: ['open', 'high', 'low', 'close', 'volume', 'position'],
  consider_outsample: false,
  outsample_start_day: '20250101',
  outsample_end_day: '20251231',
})

const p = reactive(defaults())
const loading = ref(false)
const result = ref(null)
const navCurves = ref({})
const collections = ref([])
const allVersions = ref([])

const selectedFactors = computed(() => result.value?.selected_factors_detail || [])

const resetParams = () => {
  Object.assign(p, defaults())
}

const formatMetrics = (m) => {
  if (!m) return ''
  return Object.keys(m).map(k => `${k}=${Number(m[k]).toFixed(6)}`).join(', ')
}

const loadMeta = async () => {
  try {
    const { data } = await getVersions()
    collections.value = data.collections || []
    allVersions.value = data.all_versions || []
  } catch {
    collections.value = ['genetic_programming', 'llm_prompt', 'factor_fusion']
    allVersions.value = []
  }
}

const run = async () => {
  loading.value = true
  try {
    const payload = {
      ...p,
      candidate_versions: (p.candidate_versions || []).length ? p.candidate_versions : null,
      relative_check_version_list: (p.relative_check_version_list || []).length ? p.relative_check_version_list : null,
      outsample_start_day: p.outsample_start_day || null,
      outsample_end_day: p.outsample_end_day || null,
    }
    const { data } = await runFusion(payload)
    result.value = data
    navCurves.value = data.nav_data?.nav_curves || {}
    ElMessage.success('融合完成')
  } catch (e) {
    ElMessage.error(`融合失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    loading.value = false
  }
}

onMounted(loadMeta)
</script>

