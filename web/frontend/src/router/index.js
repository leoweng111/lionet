import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/mining',
  },
  {
    path: '/mining',
    name: 'Mining',
    component: () => import('../views/MiningView.vue'),
    meta: { title: '因子挖掘', icon: 'Cpu' },
  },
  {
    path: '/fusion',
    name: 'Fusion',
    component: () => import('../views/FusionView.vue'),
    meta: { title: '因子融合', icon: 'Connection' },
  },
  {
    path: '/factors',
    name: 'Factors',
    component: () => import('../views/FactorsView.vue'),
    meta: { title: '因子库', icon: 'DataAnalysis' },
  },
  {
    path: '/backtest',
    name: 'Backtest',
    component: () => import('../views/BacktestView.vue'),
    meta: { title: '回测分析', icon: 'TrendCharts' },
  },
  {
    path: '/strategy',
    name: 'Strategy',
    component: () => import('../views/StrategyView.vue'),
    meta: { title: '策略分析', icon: 'Coin' },
  },
  {
    path: '/market-data',
    name: 'MarketData',
    component: () => import('../views/MarketDataView.vue'),
    meta: { title: '行情数据', icon: 'Histogram' },
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: () => import('../views/TasksView.vue'),
    meta: { title: '任务管理', icon: 'List' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
