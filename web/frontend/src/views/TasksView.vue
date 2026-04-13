<template>
  <div>
    <div class="page-header">
      <h2><el-icon><List /></el-icon> 任务管理</h2>
      <p>查看所有因子挖掘任务的运行状态和历史记录</p>
    </div>

    <el-card shadow="hover">
      <template #header>
        <div style="display:flex; align-items:center; justify-content:space-between;">
          <span style="font-weight:600;">任务列表</span>
          <el-button type="primary" size="small" @click="fetchTasks" :loading="loading">
            <el-icon><Refresh /></el-icon> 刷新
          </el-button>
        </div>
      </template>

      <el-table :data="taskList" stripe border size="small" style="width:100%"
        :row-class-name="rowClassName" @row-click="handleRowClick">
        <el-table-column prop="task_id" label="任务ID" width="100" />
        <el-table-column prop="version" label="版本" width="220" show-overflow-tooltip />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small" effect="dark">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" min-width="250" show-overflow-tooltip />
        <el-table-column prop="started_at" label="开始时间" width="200" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click.stop="viewDetail(row)">查看详情</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div v-if="!taskList.length" style="text-align:center; padding:40px; color:#909399;">
        暂无任务记录，请前往「因子挖掘」页面启动任务
      </div>
    </el-card>

    <!-- Detail Dialog -->
    <el-dialog v-model="detailVisible" title="任务详情" width="70%" top="5vh">
      <template v-if="detailData">
        <el-descriptions :column="2" size="small" border style="margin-bottom:16px;">
          <el-descriptions-item label="任务ID">{{ detailData.task_id }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(detailData.status)" size="small">{{ detailData.status }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="进度" :span="2">{{ detailData.progress }}</el-descriptions-item>
          <el-descriptions-item label="开始时间" :span="2">{{ detailData.started_at }}</el-descriptions-item>
        </el-descriptions>

        <div v-if="detailData.error">
          <el-alert type="error" :closable="false" style="margin-bottom:12px;">
            <pre style="white-space:pre-wrap; font-size:12px;">{{ detailData.error }}</pre>
          </el-alert>
        </div>

        <template v-if="detailData.result">
          <el-descriptions :column="1" size="small" border style="margin-bottom:16px;">
            <el-descriptions-item label="入选因子">
              <el-tag v-for="f in detailData.result.selected_fc_name_list" :key="f" size="small" style="margin:2px;">{{ f }}</el-tag>
              <span v-if="!detailData.result.selected_fc_name_list?.length" style="color:#909399;">无</span>
            </el-descriptions-item>
            <el-descriptions-item label="版本">{{ detailData.result.version }}</el-descriptions-item>
          </el-descriptions>

          <!-- NAV Charts in dialog -->
          <div v-if="detailData.result.nav_data?.nav_curves">
            <el-card v-for="(curve, name) in detailData.result.nav_data.nav_curves" :key="name" class="chart-card" shadow="never" style="margin-bottom:12px;">
              <NavChart :title="name" :curve-data="curve" height="300px" />
            </el-card>
          </div>
        </template>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { getTasks, getMiningStatus } from '../api'
import NavChart from '../components/NavChart.vue'

const loading = ref(false)
const taskList = ref([])
const detailVisible = ref(false)
const detailData = ref(null)

const getStatusType = (s) => {
  if (s === 'completed') return 'success'
  if (s === 'failed') return 'danger'
  return 'warning'
}

const rowClassName = ({ row }) => {
  if (row.status === 'running') return 'el-table__row--warning'
  return ''
}

const fetchTasks = async () => {
  loading.value = true
  try {
    const { data } = await getTasks()
    taskList.value = data.tasks || []
  } catch { /* ignore */ }
  finally { loading.value = false }
}

const handleRowClick = (row) => viewDetail(row)

const viewDetail = async (row) => {
  try {
    const { data } = await getMiningStatus(row.task_id)
    detailData.value = data
    detailVisible.value = true
  } catch { /* ignore */ }
}

let refreshTimer = null
onMounted(() => {
  fetchTasks()
  refreshTimer = setInterval(fetchTasks, 5000)
})
onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})
</script>

