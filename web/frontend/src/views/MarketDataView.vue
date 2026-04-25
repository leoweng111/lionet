<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Histogram /></el-icon> 行情数据管理</h2>
      <p>管理数据库中的期货连续合约行情数据：更新、查看、删除</p>
    </div>

    <el-tabs v-model="activeTab" type="border-card">
      <!-- ═══ Tab 1: 合约信息更新 ═══ -->
      <el-tab-pane label="合约信息更新" name="info">
        <el-card shadow="never">
          <template #header><span style="font-weight:600;">更新连续合约详情</span></template>
          <p style="color:#606266;margin-bottom:16px;">从 AkShare 拉取最新的期货连续合约列表信息，写入 MongoDB。</p>
          <el-form :model="infoParams" label-width="120px" size="small" style="max-width:420px;">
            <el-form-item label="更新方式(method)">
              <el-select v-model="infoParams.method" style="width:100%;">
                <el-option v-for="m in updateMethods" :key="m" :label="m" :value="m" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="infoLoading" @click="handleUpdateInfo">
                <el-icon v-if="!infoLoading"><Refresh /></el-icon> {{ infoLoading ? '更新中...' : '更新合约信息' }}
              </el-button>
            </el-form-item>
          </el-form>
          <div v-if="infoLogs.length" class="log-panel" style="margin-top:16px;">
            <div class="log-panel-header">操作日志 ({{ infoLogs.length }} 条)</div>
            <div class="log-panel-body">
              <div
                v-for="(line, i) in infoLogs"
                :key="`info_${i}`"
                class="log-line"
                :class="{ 'log-warn': line.includes('WARNING'), 'log-err': line.includes('ERROR') }"
              >{{ line }}</div>
            </div>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- ═══ Tab 2: 价格数据更新 ═══ -->
      <el-tab-pane label="价格数据更新" name="price-update">
        <el-card shadow="never">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">更新连续合约价格</span>
              <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:12px;color:#909399;">每日定时更新(18:00)</span>
                <el-switch v-model="scheduleEnabled" @change="handleToggleSchedule" />
              </div>
            </div>
          </template>
          <el-form :model="priceParams" label-width="160px" size="small" style="max-width:600px;">
            <el-form-item label="合约（可多选）">
              <el-select v-model="priceParams.instrument_id" multiple filterable allow-create clearable
                placeholder="留空=更新全部" style="width:100%;">
                <el-option v-for="id in instrumentIds" :key="id" :label="id" :value="id" />
              </el-select>
            </el-form-item>
            <el-form-item label="开始日期">
              <el-input v-model="priceParams.start_date" placeholder="留空=20200101" clearable @change="onStartDateChange" />
            </el-form-item>
            <el-form-item label="结束日期">
              <el-input v-model="priceParams.end_date" :placeholder="todayStr" clearable />
            </el-form-item>
            <el-form-item label="继续后复权因子">
              <el-switch v-model="priceParams.load_prev_weighted_factor" />
            </el-form-item>
            <el-form-item label="请求间隔(秒)">
              <el-input-number v-model="priceParams.wait_time" :min="0" :max="30" :step="0.5" :precision="1" style="width:100%;" />
            </el-form-item>
            <el-form-item label="更新方式(method)">
              <el-select v-model="priceParams.method" style="width:100%;">
                <el-option v-for="m in updateMethods" :key="m" :label="m" :value="m" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="priceLoading" @click="handleUpdatePrice">
                <el-icon v-if="!priceLoading"><Upload /></el-icon> {{ priceLoading ? '更新中...' : '启动价格更新' }}
              </el-button>
            </el-form-item>
          </el-form>
          <div v-if="priceLogs.length" class="log-panel" style="margin-top:8px;">
            <div class="log-panel-header">操作日志 ({{ priceLogs.length }} 条)</div>
            <div class="log-panel-body">
              <div
                v-for="(line, i) in priceLogs"
                :key="`price_${i}`"
                class="log-line"
                :class="{ 'log-warn': line.includes('WARNING'), 'log-err': line.includes('ERROR') }"
              >{{ line }}</div>
            </div>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- ═══ Tab 3: 数据概览 ═══ -->
      <el-tab-pane label="数据概览" name="overview">
        <el-card shadow="never">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">数据库数据概览</span>
              <el-button size="small" :loading="overviewLoading" @click="loadOverview"><el-icon><Refresh /></el-icon> 刷新</el-button>
            </div>
          </template>
          <el-table :data="overviewData" stripe size="small" max-height="500" style="width:100%;" v-loading="overviewLoading">
            <el-table-column prop="instrument_id" label="合约ID" width="100" />
            <el-table-column prop="start_date" label="起始日期" width="120" />
            <el-table-column prop="end_date" label="结束日期" width="120" />
            <el-table-column prop="total_rows" label="总行数" width="90" />
            <el-table-column prop="expected_bdays" label="预期交易日" width="110" />
            <el-table-column label="缺失日期数" width="120">
              <template #default="{ row }">
                <span v-if="!row.missing_dates_count" style="color:#67c23a;">0</span>
                <el-popover v-else placement="right" :width="260" trigger="click">
                  <template #reference>
                    <el-link type="warning" :underline="true" style="font-weight:600;">{{ row.missing_dates_count }}</el-link>
                  </template>
                  <div style="font-size:12px;font-weight:600;margin-bottom:6px;">{{ row.instrument_id }} 缺失日期明细</div>
                  <div style="max-height:300px;overflow-y:auto;">
                    <div v-for="d in (row.missing_dates || [])" :key="d" style="font-size:12px;line-height:1.8;font-family:monospace;">{{ d }}</div>
                  </div>
                  <div v-if="!row.missing_dates?.length" style="color:#909399;font-size:12px;">无明细数据</div>
                </el-popover>
              </template>
            </el-table-column>
            <el-table-column label="缺失字段" min-width="150">
              <template #default="{ row }">
                <span v-if="!row.missing_fields || Object.keys(row.missing_fields).length === 0" style="color:#67c23a;">无</span>
                <span v-else style="color:#e6a23c;">{{ Object.entries(row.missing_fields).map(([k,v]) => `${k}:${v}`).join(', ') }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="80">
              <template #default="{ row }">
                <el-tag :type="row.status === '完整' ? 'success' : 'warning'" size="small">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <!-- ═══ Tab 4: 查看详细数据 ═══ -->
      <el-tab-pane label="数据查看" name="detail">
        <el-card shadow="never">
          <template #header><span style="font-weight:600;">查看详细行情数据</span></template>
          <el-form :inline="true" size="small" style="margin-bottom:12px;">
            <el-form-item label="合约">
              <el-select v-model="detailParams.instrument_id" filterable placeholder="选择合约" style="width:140px;">
                <el-option v-for="id in instrumentIds" :key="id" :label="id" :value="id" />
              </el-select>
            </el-form-item>
            <el-form-item label="开始日期">
              <el-input v-model="detailParams.start_date" placeholder="留空=全部" clearable style="width:130px;" />
            </el-form-item>
            <el-form-item label="结束日期">
              <el-input v-model="detailParams.end_date" placeholder="留空=全部" clearable style="width:130px;" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="detailLoading" @click="loadDetail">查询</el-button>
              <el-button :disabled="!detailRows.length" @click="showKline = !showKline">
                <el-icon><TrendCharts /></el-icon> {{ showKline ? '隐藏K线图' : '显示K线图' }}
              </el-button>
            </el-form-item>
          </el-form>

          <!-- K-line chart -->
          <div v-if="showKline && detailRows.length" ref="klineContainer" style="width:100%;height:480px;margin-bottom:16px;"></div>

          <el-table :data="detailRows" stripe size="small" max-height="400" style="width:100%;" v-loading="detailLoading">
            <el-table-column v-for="col in detailColumns" :key="col" :prop="col" :label="col" min-width="100" show-overflow-tooltip />
          </el-table>
        </el-card>
      </el-tab-pane>

      <!-- ═══ Tab 5: 操作日志 ═══ -->
      <el-tab-pane label="操作日志" name="op-logs">
        <el-card shadow="never">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-weight:600;">数据库更新操作日志</span>
              <div style="display:flex;align-items:center;gap:8px;">
                <el-select v-model="opLogTaskType" placeholder="任务类型" clearable size="small" style="width:140px;">
                  <el-option label="update-info" value="update-info" />
                  <el-option label="update-price" value="update-price" />
                </el-select>
                <el-date-picker
                  v-model="opLogDateRange"
                  type="daterange"
                  range-separator="至"
                  start-placeholder="开始日期"
                  end-placeholder="结束日期"
                  value-format="YYYY-MM-DD"
                  size="small"
                />
                <el-button size="small" :loading="opLogLoading" @click="loadOperationLogs">
                  <el-icon><Refresh /></el-icon> 刷新日志
                </el-button>
              </div>
            </div>
          </template>

          <el-table :data="operationLogs" stripe size="small" max-height="520" style="width:100%;" v-loading="opLogLoading">
            <el-table-column prop="started_at" label="开始时间" width="180" show-overflow-tooltip />
            <el-table-column prop="task_type" label="任务类型" width="110" />
            <el-table-column prop="task_id" label="任务ID" width="130" show-overflow-tooltip />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="stType(row.status)" size="small">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="log_count" label="日志行数" width="90" />
            <el-table-column prop="last_log" label="最后一条日志" min-width="420" show-overflow-tooltip />
            <el-table-column label="操作" width="90">
              <template #default="{ row }">
                <el-button type="info" size="small" link @click="viewOperationTaskLogs(row)">日志</el-button>
              </template>
            </el-table-column>
          </el-table>
          <div style="margin-top:8px;color:#909399;font-size:12px;">共 {{ operationLogs.length }} 个任务</div>
        </el-card>
      </el-tab-pane>

      <!-- ═══ Tab 6: 删除数据 ═══ -->
      <el-tab-pane label="数据删除" name="delete">
        <el-card shadow="never">
          <template #header><span style="font-weight:600;color:#f56c6c;">删除行情数据</span></template>
          <el-alert title="危险操作：删除的数据无法恢复，请谨慎操作！" type="error" :closable="false" show-icon style="margin-bottom:16px;" />
          <el-form :model="deleteParams" label-width="140px" size="small" style="max-width:600px;">
            <el-form-item label="合约（可多选）">
              <el-select v-model="deleteParams.instrument_id_list" multiple filterable placeholder="选择要删除的合约" style="width:100%;">
                <el-option v-for="id in instrumentIds" :key="id" :label="id" :value="id" />
              </el-select>
            </el-form-item>
            <el-form-item label="开始日期">
              <el-input v-model="deleteParams.start_date" placeholder="留空=不限" clearable />
            </el-form-item>
            <el-form-item label="结束日期">
              <el-input v-model="deleteParams.end_date" placeholder="留空=不限" clearable />
            </el-form-item>
            <el-form-item>
              <el-button type="danger" :loading="deleteLoading" @click="handleDelete" :disabled="!deleteParams.instrument_id_list.length">
                <el-icon><Delete /></el-icon> 删除数据
              </el-button>
            </el-form-item>
          </el-form>
          <div v-if="deleteResult" style="margin-top:12px;">
            <el-alert :title="deleteResult" type="success" show-icon />
          </div>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="opLogDialogVisible" title="任务日志" width="75%" top="8vh" destroy-on-close>
      <div style="margin-bottom:8px; color:#909399; font-size:12px;">任务ID: {{ opLogDialogTaskId || '-' }}</div>
      <div v-if="opLogDialogLines.length" class="op-log-detail-box">
        <div v-for="(line, idx) in opLogDialogLines" :key="`op_log_${idx}`">{{ line }}</div>
      </div>
      <el-empty v-else description="当前任务暂无日志" :image-size="60" />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, watch, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'
import {
  getInstrumentIds,
  updateContractInfo,
  updateContractPrice,
  getMarketDataTaskStatus,
  getMarketDataLogs,
  getMarketDataOverview,
  getMarketDataPrice,
  deleteMarketData,
  getScheduledStatus,
  toggleSchedule,
} from '../api'

defineOptions({ name: 'MarketDataView' })

const updateMethods = ['bulk_write_update', 'update_one', 'insert_many']
const stType = (s) => s === 'completed' ? 'success' : s === 'failed' ? 'danger' : 'warning'

// ── Shared ──
const activeTab = ref('info')
const instrumentIds = ref([])
const todayStr = new Date().toISOString().slice(0, 10).replace(/-/g, '')
const todayDashStr = new Date().toISOString().slice(0, 10)

const loadInstrumentIds = async () => {
  try {
    const { data } = await getInstrumentIds()
    instrumentIds.value = data.instrument_ids || []
  } catch { /* ignore */ }
}

onMounted(() => {
  loadInstrumentIds()
  loadScheduleStatus()
  loadOperationLogs()
})

// ── Log Panel (sub-component inline) ──
// Using a simple functional approach via template

// ── Tab 1: Update Info ──
const infoLoading = ref(false)
const infoLogs = ref([])
const infoParams = reactive({
  method: 'bulk_write_update',
})
const handleUpdateInfo = async () => {
  infoLoading.value = true
  infoLogs.value = []
  try {
    const { data } = await updateContractInfo({ method: infoParams.method })
    const taskId = data.task_id
    ElMessage.success(data.message)
    pollMarketTask(taskId, infoLogs, () => {
      infoLoading.value = false
      loadInstrumentIds()
    })
  } catch (err) {
    ElMessage.error('更新失败: ' + (err.response?.data?.detail || err.message))
    infoLoading.value = false
  }
}

// ── Tab 2: Update Price ──
const priceLoading = ref(false)
const priceLogs = ref([])
const scheduleEnabled = ref(true)
const priceParams = reactive({
  instrument_id: [],
  start_date: '',
  end_date: '',
  load_prev_weighted_factor: true,
  wait_time: 2.0,
  method: 'bulk_write_update',
})

const loadScheduleStatus = async () => {
  try {
    const { data } = await getScheduledStatus()
    scheduleEnabled.value = data.enabled
  } catch { /* ignore */ }
}
const handleToggleSchedule = async (val) => {
  try {
    await toggleSchedule(val)
    ElMessage.success(val ? '已开启定时更新' : '已关闭定时更新')
  } catch { ElMessage.error('操作失败') }
}
const onStartDateChange = (val) => {
  if (val && val !== '20200101' && val.trim()) {
    ElMessageBox.confirm(
      '修改 start_date 可能导致后复权因子不统一。目前后复权的起始日期默认为 20200101，确认继续吗？',
      '警告', { type: 'warning', confirmButtonText: '确认', cancelButtonText: '恢复默认' }
    ).catch(() => { priceParams.start_date = '' })
  }
}
const handleUpdatePrice = async () => {
  priceLoading.value = true
  priceLogs.value = []
  try {
    const payload = {
      instrument_id: priceParams.instrument_id.length ? priceParams.instrument_id : null,
      start_date: priceParams.start_date || null,
      end_date: priceParams.end_date || null,
      load_prev_weighted_factor: priceParams.load_prev_weighted_factor,
      wait_time: priceParams.wait_time,
      method: priceParams.method,
    }
    const { data } = await updateContractPrice(payload)
    ElMessage.success(data.message)
    pollMarketTask(data.task_id, priceLogs, () => { priceLoading.value = false })
  } catch (err) {
    ElMessage.error('启动失败: ' + (err.response?.data?.detail || err.message))
    priceLoading.value = false
  }
}

// ── Polling helper ──
const pollMarketTask = (taskId, logsRef, onDone) => {
  const timer = setInterval(async () => {
    try {
      const { data } = await getMarketDataTaskStatus(taskId)
      logsRef.value = data.logs || []
      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(timer)
        if (data.status === 'completed') ElMessage.success('任务完成')
        else ElMessage.error('任务失败: ' + (data.error || ''))
        if (onDone) onDone()
      }
    } catch { /* keep polling */ }
  }, 2000)
}

// ── Tab 3: Overview ──
const overviewLoading = ref(false)
const overviewData = ref([])
const loadOverview = async () => {
  overviewLoading.value = true
  try {
    const { data } = await getMarketDataOverview()
    overviewData.value = data.overview || []
  } catch (err) {
    ElMessage.error('加载失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    overviewLoading.value = false
  }
}
watch(activeTab, (val) => { if (val === 'overview') loadOverview() })

// ── Tab 4: Detail + K-line ──
const detailLoading = ref(false)
const detailRows = ref([])
const detailColumns = ref([])
const showKline = ref(false)
const klineContainer = ref(null)
let klineChart = null
const detailParams = reactive({ instrument_id: '', start_date: '', end_date: '' })

const loadDetail = async () => {
  if (!detailParams.instrument_id) { ElMessage.warning('请选择合约'); return }
  detailLoading.value = true
  showKline.value = false
  try {
    const { data } = await getMarketDataPrice({
      instrument_id: detailParams.instrument_id,
      start_date: detailParams.start_date || undefined,
      end_date: detailParams.end_date || undefined,
    })
    detailRows.value = data.rows || []
    detailColumns.value = data.columns || []
  } catch (err) {
    ElMessage.error('查询失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    detailLoading.value = false
  }
}

watch(showKline, async (val) => {
  if (val && detailRows.value.length) {
    await nextTick()
    renderKline()
  }
})

const renderKline = () => {
  if (!klineContainer.value) return
  if (klineChart) klineChart.dispose()
  klineChart = echarts.init(klineContainer.value)
  const rows = detailRows.value
  const dates = rows.map(r => r.time)
  // candlestick: [open, close, low, high]
  const candleData = rows.map(r => [r.open, r.close, r.low, r.high])
  const volumes = rows.map(r => r.volume || 0)

  klineChart.setOption({
    title: { text: `${detailParams.instrument_id} K线图`, left: 'center', textStyle: { fontSize: 14 } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
    },
    legend: { data: ['K线', '成交量'], bottom: 0 },
    grid: [
      { left: '8%', right: '8%', top: '10%', height: '55%' },
      { left: '8%', right: '8%', top: '72%', height: '18%' },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLabel: { fontSize: 10 }, boundaryGap: true },
      { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: false }, boundaryGap: true },
    ],
    yAxis: [
      { scale: true, gridIndex: 0, splitLine: { show: true, lineStyle: { type: 'dashed' } } },
      { scale: true, gridIndex: 1, splitLine: { show: false } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100 },
      { type: 'slider', xAxisIndex: [0, 1], start: 0, end: 100, top: '93%', height: 20 },
    ],
    series: [
      {
        name: 'K线', type: 'candlestick', data: candleData, xAxisIndex: 0, yAxisIndex: 0,
        itemStyle: {
          color: '#ef5350', color0: '#26a69a',
          borderColor: '#ef5350', borderColor0: '#26a69a',
        },
      },
      {
        name: '成交量', type: 'bar', data: volumes, xAxisIndex: 1, yAxisIndex: 1,
        itemStyle: { color: '#7fbbe9' },
      },
    ],
  })
}

// ── Tab 5: Operation Logs ──
const opLogLoading = ref(false)
const operationLogs = ref([])
const opLogTaskType = ref('')
const opLogDateRange = ref([todayDashStr, todayDashStr])
const opLogDialogVisible = ref(false)
const opLogDialogTaskId = ref('')
const opLogDialogLines = ref([])

const loadOperationLogs = async () => {
  opLogLoading.value = true
  try {
    const params = {
      start_date: opLogDateRange.value?.[0] || todayDashStr,
      end_date: opLogDateRange.value?.[1] || todayDashStr,
    }
    if (opLogTaskType.value) params.task_type = opLogTaskType.value
    const { data } = await getMarketDataLogs(params)
    operationLogs.value = data.logs || []
  } catch (err) {
    ElMessage.error('日志加载失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    opLogLoading.value = false
  }
}

const viewOperationTaskLogs = async (row) => {
  if (!row?.task_id) return
  try {
    const { data } = await getMarketDataTaskStatus(row.task_id)
    opLogDialogTaskId.value = row.task_id
    opLogDialogLines.value = data.logs || []
    opLogDialogVisible.value = true
  } catch (err) {
    ElMessage.error('日志加载失败: ' + (err.response?.data?.detail || err.message))
  }
}

watch(activeTab, (val) => {
  if (val === 'op-logs') loadOperationLogs()
})

// ── Tab 6: Delete ──
const deleteLoading = ref(false)
const deleteResult = ref('')
const deleteParams = reactive({ instrument_id_list: [], start_date: '', end_date: '' })

const handleDelete = async () => {
  if (!deleteParams.instrument_id_list.length) { ElMessage.warning('请选择要删除的合约'); return }
  try {
    await ElMessageBox.confirm(
      `确认删除以下合约的数据？\n合约: ${deleteParams.instrument_id_list.join(', ')}\n` +
      `日期范围: ${deleteParams.start_date || '不限'} ~ ${deleteParams.end_date || '不限'}\n\n此操作不可恢复！`,
      '⚠️ 二次确认', { type: 'error', confirmButtonText: '确认删除', cancelButtonText: '取消', confirmButtonClass: 'el-button--danger' }
    )
  } catch { return }
  deleteLoading.value = true
  deleteResult.value = ''
  try {
    const { data } = await deleteMarketData({
      instrument_id_list: deleteParams.instrument_id_list,
      start_date: deleteParams.start_date || null,
      end_date: deleteParams.end_date || null,
    })
    deleteResult.value = data.message
    ElMessage.success(data.message)
  } catch (err) {
    ElMessage.error('删除失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    deleteLoading.value = false
  }
}
</script>

<style scoped>
.page-header { margin-bottom: 16px; }
.page-header h2 { display: flex; align-items: center; gap: 8px; margin: 0 0 4px; }
.page-header p { color: #909399; margin: 0; font-size: 13px; }

.log-panel { border: 1px solid var(--el-border-color-lighter); border-radius: 6px; overflow: hidden; }
.log-panel-header { background: #f5f7fa; padding: 6px 12px; font-size: 12px; font-weight: 600; color: #606266; }
.log-panel-body { max-height: 300px; overflow-y: auto; padding: 8px 12px; background: #1e1e1e; }
.log-line { font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 11px; line-height: 1.6; color: #d4d4d4; white-space: pre-wrap; word-break: break-all; }
.log-warn { color: #e6a23c; }
.log-err { color: #f56c6c; }

.op-log-detail-box {
  max-height: 65vh;
  overflow: auto;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
  background: #0f111a;
  color: #d6deeb;
  border-radius: 6px;
  padding: 10px;
}
</style>

