import axios from 'axios'

const api = axios.create({
  baseURL: '',  // use Vite proxy in dev, same origin in prod
  timeout: 600000, // 10 min for long-running tasks
})

// ── Versions & Factors ────────────────────────────────
export const getVersions = () => api.get('/api/versions')
export const getFactors = (params) => api.get('/api/factors', { params })

// ── Mining ────────────────────────────────────────────
export const startMining = (data) => api.post('/api/mining/start', data)
export const getMiningStatus = (taskId) => api.get(`/api/mining/status/${taskId}`)
export const terminateMining = (taskId) => api.post(`/api/mining/terminate/${taskId}`)
export const getMiningIndicatorOptions = () => api.get('/api/mining/indicator-options')

// ── Backtest ──────────────────────────────────────────
export const runBacktest = (data) => api.post('/api/backtest', data)

// ── Fusion ────────────────────────────────────────────
export const runFusion = (data) => api.post('/api/fusion/run', data)
export const startFusion = (data) => api.post('/api/fusion/start', data)
export const getFusionStatus = (taskId) => api.get(`/api/fusion/status/${taskId}`)

// ── Strategy ──────────────────────────────────────────
export const runStrategy = (data) => api.post('/api/strategy', data)

// ── Tasks ─────────────────────────────────────────────
export const getTasks = (params) => api.get('/api/tasks', { params })
export const getTaskDetail = (taskId) => api.get(`/api/tasks/detail/${taskId}`)

// ── Health ────────────────────────────────────────────
export const getHealth = () => api.get('/api/health')

export default api
