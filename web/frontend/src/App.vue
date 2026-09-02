<template>
  <el-container class="layout-container">
    <!-- Sidebar -->
    <el-aside width="220px" style="background: #1d1e1f; overflow: hidden;">
      <div class="sidebar-header">
        <el-icon style="margin-right:8px; font-size:22px;"><DataLine /></el-icon>
        Lionet
      </div>
      <el-menu
        :default-active="activeMenu"
        background-color="#1d1e1f"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        router
        style="border-right: none;"
      >
        <el-menu-item v-for="r in menuRoutes" :key="r.path" :index="r.path">
          <el-icon><component :is="r.meta.icon" /></el-icon>
          <span>{{ r.meta.title }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- Main -->
    <el-container direction="vertical">
      <!-- Header -->
      <el-header style="height:60px; display:flex; align-items:center; justify-content:space-between; background:#fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding:0 24px;">
        <div style="display:flex; align-items:center;">
          <span style="font-size:16px; font-weight:600; color:#303133;">Lionet 因子挖掘平台</span>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
          <el-tag :type="backendTagType" size="small" effect="dark">
            {{ backendStatusText }}
          </el-tag>
        </div>
      </el-header>

      <!-- Content -->
      <el-main class="main-content">
        <router-view v-slot="{ Component, route: currentRoute }">
          <keep-alive include="MarketDataView">
            <component :is="Component" :key="currentRoute.fullPath" />
          </keep-alive>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { getHealth } from './api'

const route = useRoute()
const activeMenu = computed(() => route.path)

const menuRoutes = [
  { path: '/mining', meta: { title: 'GP因子挖掘', icon: 'Cpu' } },
  { path: '/llm-mining', meta: { title: 'LLM因子挖掘', icon: 'MagicStick' } },
  { path: '/fusion', meta: { title: '因子融合', icon: 'Connection' } },
  { path: '/factors', meta: { title: '因子库', icon: 'DataAnalysis' } },
  { path: '/backtest', meta: { title: '回测分析', icon: 'TrendCharts' } },
  { path: '/strategy', meta: { title: '策略分析', icon: 'Coin' } },
  { path: '/strategy-monitor', meta: { title: '策略监控', icon: 'Monitor' } },
  { path: '/market-data', meta: { title: '行情数据', icon: 'Histogram' } },
  { path: '/tasks', meta: { title: '任务管理', icon: 'List' } },
]

const BACKEND_STATUS = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  CHECKING: 'checking',
}

const backendStatus = ref(BACKEND_STATUS.CHECKING)
let healthTimer = null
let healthCheckInFlight = false
let consecutiveHealthFailures = 0

const HEALTH_CHECK_INTERVAL_MS = 10000
const HEALTH_RETRY_DELAY_MS = 800
const HEALTH_FAIL_THRESHOLD = 3

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const backendTagType = computed(() => {
  if (backendStatus.value === BACKEND_STATUS.ONLINE) return 'success'
  if (backendStatus.value === BACKEND_STATUS.OFFLINE) return 'danger'
  return 'warning'
})

const backendStatusText = computed(() => {
  if (backendStatus.value === BACKEND_STATUS.ONLINE) return '后端已连接'
  if (backendStatus.value === BACKEND_STATUS.OFFLINE) return '后端未连接'
  return '检测中/抖动中'
})


const checkHealth = async () => {
  if (healthCheckInFlight) return
  healthCheckInFlight = true
  try {
    await getHealth()
    consecutiveHealthFailures = 0
    backendStatus.value = BACKEND_STATUS.ONLINE
  } catch {
    // Retry once to absorb short transient blips.
    try {
      await sleep(HEALTH_RETRY_DELAY_MS)
      await getHealth()
      consecutiveHealthFailures = 0
      backendStatus.value = BACKEND_STATUS.ONLINE
    } catch {
      consecutiveHealthFailures += 1
      if (consecutiveHealthFailures >= HEALTH_FAIL_THRESHOLD) {
        backendStatus.value = BACKEND_STATUS.OFFLINE
      } else {
        backendStatus.value = BACKEND_STATUS.CHECKING
      }
    }
  } finally {
    healthCheckInFlight = false
  }
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, HEALTH_CHECK_INTERVAL_MS)
})
onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
})
</script>
