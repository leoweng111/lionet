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
          <el-tag :type="backendOk ? 'success' : 'danger'" size="small" effect="dark">
            {{ backendOk ? '后端已连接' : '后端未连接' }}
          </el-tag>
        </div>
      </el-header>

      <!-- Content -->
      <el-main class="main-content">
        <router-view v-slot="{ Component }">
          <keep-alive include="MarketDataView">
            <component :is="Component" />
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
  { path: '/mining', meta: { title: '因子挖掘', icon: 'Cpu' } },
  { path: '/fusion', meta: { title: '因子融合', icon: 'Connection' } },
  { path: '/factors', meta: { title: '因子库', icon: 'DataAnalysis' } },
  { path: '/backtest', meta: { title: '回测分析', icon: 'TrendCharts' } },
  { path: '/strategy', meta: { title: '策略分析', icon: 'Coin' } },
  { path: '/market-data', meta: { title: '行情数据', icon: 'Histogram' } },
  { path: '/tasks', meta: { title: '任务管理', icon: 'List' } },
]

const backendOk = ref(false)
let healthTimer = null

const checkHealth = async () => {
  try {
    await getHealth()
    backendOk.value = true
  } catch {
    backendOk.value = false
  }
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, 10000)
})
onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
})
</script>
