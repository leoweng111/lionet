<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Cpu /></el-icon> 遗传算法因子挖掘</h2>
      <p>配置 GP 遗传算法超参数，一键启动因子挖掘任务，挖掘完成后自动展示回测结果和净值曲线</p>
    </div>

    <el-row :gutter="20" class="responsive-row">
      <!-- Left: Params (scrollable) -->
      <el-col :xs="24" :sm="24" :md="12" :lg="10" :xl="10">
        <el-card shadow="hover">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">超参数配置</span>
              <el-button size="small" @click="resetParams">恢复默认</el-button>
            </div>
          </template>
          <div class="param-scroll-panel">
          <el-form :model="params" label-width="auto" label-position="right" size="small">

            <div class="param-section"><el-divider content-position="left">基础参数</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="合约"><el-input v-model="params.instrument_id_list" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="版本号"><el-input v-model="params.version" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="开始日期"><el-input v-model="params.start_time" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="结束日期"><el-input v-model="params.end_time" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="适应度"><el-select v-model="params.fitness_metric" style="width:100%"><el-option label="IC" value="ic" /><el-option label="Sharpe" value="sharpe" /></el-select></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="最大因子数"><el-input-number v-model="params.max_factor_count" :min="1" :max="500" style="width:100%" /></el-form-item></el-col>
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
              <el-form-item label="最小窗口"><el-input-number v-model="params.min_window_size" :min="1" :max="200" style="width:100%" /></el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">滚动标准化</el-divider>
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

            <div class="param-section"><el-divider content-position="left">去重 / 泄露检查</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="泄露抽样"><el-input-number v-model="params.check_leakage_count" :min="0" :max="200" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="相关阈值"><el-input-number v-model="params.relative_threshold" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-form-item label="相关去重"><el-switch v-model="params.check_relative" /></el-form-item>
            </div>

            <div class="param-section"><el-divider content-position="left">GP 核心参数</el-divider>
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

            <div class="param-section"><el-divider content-position="left">GP 高级参数</el-divider>
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
            </div>

            <div class="param-section"><el-divider content-position="left">筛选阈值</el-divider>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="净收益均值"><el-input-number v-model="params.filter_net_return_mean" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="净收益年度"><el-input-number v-model="params.filter_net_return_yearly" :step="0.01" :precision="3" style="width:100%" /></el-form-item></el-col>
              </el-row>
              <el-row :gutter="12">
                <el-col :span="12"><el-form-item label="Sharpe均值"><el-input-number v-model="params.filter_net_sharpe_mean" :step="0.1" :precision="2" style="width:100%" /></el-form-item></el-col>
                <el-col :span="12"><el-form-item label="Sharpe年度"><el-input-number v-model="params.filter_net_sharpe_yearly" :step="0.1" :precision="2" style="width:100%" /></el-form-item></el-col>
              </el-row>
            </div>

            <el-form-item>
              <el-button type="primary" size="large" :loading="mining" @click="handleStartMining" style="width:100%;">
                <el-icon v-if="!mining"><VideoPlay /></el-icon>
                {{ mining ? '挖掘中...' : '🚀 启动因子挖掘' }}
              </el-button>
            </el-form-item>
          </el-form>
          </div>
        </el-card>
      </el-col>

      <!-- Right: Result -->
      <el-col :xs="24" :sm="24" :md="12" :lg="14" :xl="14">
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
        <el-card v-if="!taskId" shadow="hover"><el-empty description="配置参数后点击「启动因子挖掘」开始" /></el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { startMining, getMiningStatus } from '../api'
import NavChart from '../components/NavChart.vue'

const defaultParams = () => ({
  instrument_type: 'futures_continuous_contract', instrument_id_list: 'C0', fc_freq: '1d',
  start_time: '20200101', end_time: '20241231',
  version: new Date().toISOString().slice(0,10).replace(/-/g,'') + '_gp_test',
  portfolio_adjust_method: '1D', interest_method: 'simple', risk_free_rate: false,
  calculate_baseline: true, apply_weighted_price: true, n_jobs: 5, max_factor_count: 50,
  min_window_size: 30, fitness_metric: 'ic',
  apply_rolling_norm: true, rolling_norm_window: 30, rolling_norm_min_periods: 20,
  rolling_norm_eps: 1e-8, rolling_norm_clip: 5.0,
  check_leakage_count: 20, check_relative: true, relative_threshold: 0.7,
  gp_generations: 20, gp_population_size: 500, gp_max_depth: 6, gp_elite_size: 50,
  gp_elite_relative_threshold: 0.65, gp_crossover_prob: 0.3, gp_mutation_prob: 0.7,
  gp_leaf_prob: 0.2, gp_const_prob: 0.02, gp_tournament_size: 3,
  gp_window_choices: [3,5,10,20,30], random_seed: null,
  gp_early_stopping_generation_count: 20, gp_depth_penalty_coef: 0.0,
  gp_depth_penalty_start_depth: 6, gp_depth_penalty_linear_coef: 0.03,
  gp_depth_penalty_quadratic_coef: 0.0, gp_log_interval: 5,
  gp_small_factor_penalty_coef: 0.0, gp_assumed_initial_capital: 100000,
  gp_elite_stagnation_generation_count: 4, gp_max_shock_generation: 3,
  filter_net_return_mean: 0.05, filter_net_return_yearly: 0.03,
  filter_net_sharpe_mean: 0.5, filter_net_sharpe_yearly: 0.3,
})
const params = reactive(defaultParams())
const resetParams = () => Object.assign(params, defaultParams())

const mining = ref(false), taskId = ref(''), taskStatus = ref(''), taskProgress = ref(''), taskError = ref('')
const miningResult = ref(null), navCurves = ref({}), perfSummary = ref([]), perfColumns = ref([])
const statusTag = computed(() => taskStatus.value==='completed'?'success':taskStatus.value==='failed'?'danger':'warning')
let pollTimer = null

const handleStartMining = async () => {
  mining.value = true; taskError.value = ''; miningResult.value = null; navCurves.value = {}; perfSummary.value = []
  try {
    const { data } = await startMining({ ...params, random_seed: params.random_seed || null })
    taskId.value = data.task_id; taskStatus.value = 'running'; taskProgress.value = '任务已提交...'
    ElMessage.success('挖掘任务已启动: ' + data.task_id); startPolling()
  } catch (err) { mining.value = false; ElMessage.error('启动失败: ' + (err.response?.data?.detail || err.message)) }
}
const startPolling = () => {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      const { data } = await getMiningStatus(taskId.value)
      taskStatus.value = data.status; taskProgress.value = data.progress; taskError.value = data.error || ''
      if (data.status === 'completed') {
        clearInterval(pollTimer); mining.value = false; miningResult.value = data.result
        if (data.result?.nav_data) {
          navCurves.value = data.result.nav_data.nav_curves || {}
          const s = data.result.nav_data.performance_summary || []; perfSummary.value = s
          if (s.length) perfColumns.value = Object.keys(s[0])
        }
        ElMessage.success('因子挖掘完成!')
      } else if (data.status === 'failed') { clearInterval(pollTimer); mining.value = false; ElMessage.error('因子挖掘失败') }
    } catch { /* keep polling */ }
  }, 3000)
}
</script>

