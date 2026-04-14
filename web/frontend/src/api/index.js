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

// ── Backtest ──────────────────────────────────────────
export const runBacktest = (data) => api.post('/api/backtest', data)

// ── Strategy ──────────────────────────────────────────
export const runStrategy = (data) => api.post('/api/strategy', data)

// ── Tasks ─────────────────────────────────────────────
export const getTasks = () => api.get('/api/tasks')
export const getTaskDetail = (taskId) => api.get(`/api/tasks/detail/${taskId}`)

// ── Health ────────────────────────────────────────────
export const getHealth = () => api.get('/api/health')

export default api
