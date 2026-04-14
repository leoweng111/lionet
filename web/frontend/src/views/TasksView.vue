<template>
  <div>
    <div class="page-header">
      <h2><el-icon><List /></el-icon> 任务管理</h2>
      <p>查看所有因子挖掘任务的运行状态、演化进度和历史记录</p>
    </div>
    <el-card shadow="hover">
      <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><span style="font-weight:600;">任务列表</span><el-button type="primary" size="small" @click="fetchTasks" :loading="loading"><el-icon><Refresh /></el-icon> 刷新</el-button></div></template>
      <el-table :data="taskList" stripe border size="small" style="width:100%" @row-click="viewDetail">
        <el-table-column prop="task_id" label="任务ID" width="100" />
        <el-table-column prop="version" label="版本" width="220" show-overflow-tooltip />
        <el-table-column prop="status" label="状态" width="100"><template #default="{row}"><el-tag :type="stType(row.status)" size="small" effect="dark">{{ row.status }}</el-tag></template></el-table-column>
        <el-table-column prop="progress" label="进度" min-width="300" show-overflow-tooltip />
        <el-table-column label="GP演化" width="200"><template #default="{row}"><template v-if="row.gp_progress"><el-progress :percentage="Math.round(row.gp_progress.generation/row.gp_progress.total_generations*100)" :stroke-width="14" :text-inside="true" style="width:100%" /><div style="font-size:11px;color:#606266;margin-top:2px;">{{ row.gp_progress.generation }}/{{ row.gp_progress.total_generations }}代 best={{ row.gp_progress.global_best_penalized?.toFixed(4) }}</div></template><span v-else style="color:#c0c4cc;">-</span></template></el-table-column>
        <el-table-column prop="started_at" label="开始时间" width="180" show-overflow-tooltip />
        <el-table-column label="操作" width="160"><template #default="{row}"><el-button size="small" type="primary" link @click.stop="viewDetail(row)">详情</el-button><el-button size="small" type="danger" link @click.stop="handleTerminate(row)" :disabled="row.status !== 'running'">终止</el-button></template></el-table-column>
      </el-table>
      <div v-if="!taskList.length" style="text-align:center;padding:40px;color:#909399;">暂无任务记录</div>
    </el-card>

    <el-dialog v-model="dlgVisible" title="任务详情" width="75%" top="5vh" destroy-on-close>
      <template v-if="dd">
        <el-descriptions :column="2" size="small" border style="margin-bottom:16px;">
          <el-descriptions-item label="任务ID">{{ dd.task_id }}</el-descriptions-item>
          <el-descriptions-item label="状态"><el-tag :type="stType(dd.status)" size="small">{{ dd.status }}</el-tag></el-descriptions-item>
          <el-descriptions-item label="进度" :span="2">{{ dd.progress }}</el-descriptions-item>
          <el-descriptions-item label="开始时间">{{ dd.started_at }}</el-descriptions-item>
          <el-descriptions-item label="结束时间" v-if="dd.finished_at">{{ dd.finished_at }}</el-descriptions-item>
        </el-descriptions>
        <div v-if="dd.error"><el-alert type="error" :closable="false" style="margin-bottom:12px;"><pre style="white-space:pre-wrap;font-size:12px;">{{ dd.error }}</pre></el-alert></div>

        <!-- GP progress bar -->
        <div v-if="dd.gp_progress" style="margin-bottom:16px;">
          <el-descriptions :column="2" size="small" border>
            <el-descriptions-item label="当前代数">{{ dd.gp_progress.generation }} / {{ dd.gp_progress.total_generations }}</el-descriptions-item>
            <el-descriptions-item label="全局最优(penalized)">{{ dd.gp_progress.global_best_penalized?.toFixed(6) }}</el-descriptions-item>
            <el-descriptions-item label="全局最优(original)">{{ dd.gp_progress.global_best_original?.toFixed(6) }}</el-descriptions-item>
          </el-descriptions>
        </div>

        <!-- Params -->
        <el-card v-if="dd.params" shadow="never" style="margin-bottom:16px;">
          <template #header><span style="font-weight:600;">超参数配置</span></template>
          <el-descriptions :column="3" size="small" border>
            <el-descriptions-item v-for="(val, key) in dd.params" :key="key" :label="key">{{ Array.isArray(val) ? val.join(', ') : val }}</el-descriptions-item>
          </el-descriptions>
        </el-card>

        <!-- Result summary from DB or memory -->
        <template v-if="dd.result_summary">
          <el-descriptions :column="1" size="small" border style="margin-bottom:16px;">
            <el-descriptions-item label="入选因子"><el-tag v-for="f in (dd.result_summary.selected_fc_name_list||[])" :key="f" size="small" style="margin:2px;">{{ f }}</el-tag><span v-if="!(dd.result_summary.selected_fc_name_list||[]).length" style="color:#909399;">无</span></el-descriptions-item>
            <el-descriptions-item label="版本">{{ dd.result_summary.version }}</el-descriptions-item>
            <el-descriptions-item label="消息" v-if="dd.result_summary.message">{{ dd.result_summary.message }}</el-descriptions-item>
          </el-descriptions>
        </template>
        <template v-if="dd.result">
          <el-descriptions :column="1" size="small" border style="margin-bottom:16px;">
            <el-descriptions-item label="入选因子"><el-tag v-for="f in (dd.result.selected_fc_name_list||[])" :key="f" size="small" style="margin:2px;">{{ f }}</el-tag><span v-if="!(dd.result.selected_fc_name_list||[]).length" style="color:#909399;">无</span></el-descriptions-item>
            <el-descriptions-item label="版本">{{ dd.result.version }}</el-descriptions-item>
          </el-descriptions>
          <div v-if="dd.result.nav_data?.nav_curves"><el-card v-for="(curve, name) in dd.result.nav_data.nav_curves" :key="name" class="chart-card" shadow="never" style="margin-bottom:12px;"><NavChart :title="name" :curve-data="curve" height="300px" /></el-card></div>
        </template>
      </template>
    </el-dialog>
  </div>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessageBox, ElMessage } from 'element-plus'
import { getTasks, getTaskDetail, terminateMining } from '../api'
import NavChart from '../components/NavChart.vue'

const loading = ref(false), taskList = ref([]), dlgVisible = ref(false), dd = ref(null)
const stType = (s) => s === 'completed' ? 'success' : s === 'failed' ? 'danger' : s === 'terminated' ? 'info' : 'warning'
const fetchTasks = async () => { loading.value = true; try { const { data } = await getTasks(); taskList.value = data.tasks || [] } catch { /* */ } finally { loading.value = false } }
const viewDetail = async (row) => { try { const { data } = await getTaskDetail(row.task_id); dd.value = data; dlgVisible.value = true } catch { /* */ } }
const handleTerminate = async (row) => {
  try {
    await ElMessageBox.confirm(`确定要终止任务 ${row.task_id} 吗？`, '确认终止', { confirmButtonText: '终止', cancelButtonText: '取消', type: 'warning' })
    await terminateMining(row.task_id)
    ElMessage.success('任务已终止')
    fetchTasks()
  } catch (e) {
    if (e !== 'cancel') ElMessage.error('终止失败: ' + (e?.response?.data?.detail || e.message || '未知错误'))
  }
}
let timer = null
onMounted(() => { fetchTasks(); timer = setInterval(fetchTasks, 5000) })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>
