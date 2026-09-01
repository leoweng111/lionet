import axios from 'axios'

const apiBaseURL = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/$/, '')
const isNgrokPublicDomain = /ngrok-free\.(app|dev)$/i.test(new URL(apiBaseURL || 'http://localhost').host)

const api = axios.create({
  // Local dev can keep empty baseURL and rely on Vite proxy.
  // GitHub Pages should set VITE_API_BASE_URL to backend public URL.
  baseURL: apiBaseURL,
  timeout: 600000, // 10 min for long-running tasks
})

// Health check uses a short timeout so the "后端已连接/未连接" indicator
// reacts quickly instead of hanging for 10 minutes when the backend event
// loop is temporarily blocked by sync DB calls.
const healthApi = axios.create({
  baseURL: apiBaseURL,
  timeout: 5000,
})

// ngrok free public endpoint may return browser warning page (ERR_NGROK_6024)
// for browser-like requests. This header bypasses that page for API calls.
if (isNgrokPublicDomain) {
  api.defaults.headers.common['ngrok-skip-browser-warning'] = 'true'
}

// ── Versions & Factors ────────────────────────────────
export const getVersions = () => api.get('/api/versions')
export const getFactors = (params) => api.get('/api/factors', { params })

// ── Mining ────────────────────────────────────────────
export const startMining = (data) => api.post('/api/mining/start', data)
export const getMiningStatus = (taskId) => api.get(`/api/mining/status/${taskId}`)
export const terminateMining = (taskId) => api.post(`/api/mining/terminate/${taskId}`)
export const getMiningIndicatorOptions = () => api.get('/api/mining/indicator-options')
export const getMiningIntradayFeatures = () => api.get('/api/mining/intraday-features')
export const getMiningAutoConfig = () => api.get('/api/mining/auto-config')
export const updateMiningAutoConfig = (data) => api.post('/api/mining/auto-config', data)
export const getMiningAutoSchedulerStatus = () => api.get('/api/mining/auto-scheduler-status')
export const runMiningAutoSchedulerTick = () => api.post('/api/mining/auto-scheduler/tick')

// ── LLM Mining ────────────────────────────────────────
export const getLLMProfiles = () => api.get('/api/llm-mining/profiles')
export const startLLMMining = (data) => api.post('/api/llm-mining/start', data)
export const getLLMMiningStatus = (taskId) => api.get(`/api/llm-mining/status/${taskId}`)
export const terminateLLMMining = (taskId) => api.post(`/api/llm-mining/terminate/${taskId}`)

// ── Backtest ──────────────────────────────────────────
export const runBacktest = (data) => api.post('/api/backtest', data)

// ── Fusion ────────────────────────────────────────────
export const runFusion = (data) => api.post('/api/fusion/run', data)
export const startFusion = (data) => api.post('/api/fusion/start', data)
export const getFusionStatus = (taskId) => api.get(`/api/fusion/status/${taskId}`)
export const getFusionIndicatorOptions = () => api.get('/api/fusion/indicator-options')
export const updateFusionIndicatorOptions = (data) => api.post('/api/fusion/indicator-options', data)

// ── Strategy ──────────────────────────────────────────
export const runStrategy = (data) => api.post('/api/strategy', data)
export const updateStrategyMonitorPrice = (data) => api.post('/api/strategy/monitor/update-price', data)
export const generateStrategyMonitor = (data) => api.post('/api/strategy/monitor/generate', data)
export const getStrategyMonitorAutoConfig = () => api.get('/api/strategy/monitor/auto-config')
export const updateStrategyMonitorAutoConfig = (data) => api.post('/api/strategy/monitor/auto-config', data)

// ── Page Config ───────────────────────────────────────
export const getPageConfig = (pageName) => api.get(`/api/page-config/${pageName}`)
export const savePageConfig = (pageName, data) => api.post(`/api/page-config/${pageName}`, data)
export const resetPageConfig = (pageName) => api.post(`/api/page-config/${pageName}/reset`)

// ── Tasks ─────────────────────────────────────────────
export const getTasks = (params) => api.get('/api/tasks', { params })
export const getTaskDetail = (taskId) => api.get(`/api/tasks/detail/${taskId}`)

// ── Health ────────────────────────────────────────────
export const getHealth = () => healthApi.get('/api/health')

// ── Market Data Management ────────────────────────────
export const getInstrumentIds = () => api.get('/api/market-data/instrument-ids')
export const getMarketDataConfig = () => api.get('/api/market-data/config')
export const updateContractInfo = (data) => api.post('/api/market-data/update-info', data)
export const updateContractPrice = (data) => api.post('/api/market-data/update-price', data)
export const updateContractPrice1min = (data) => api.post('/api/market-data/update-price-1min', data)
export const updateContractPrice1minCsv = (data) => api.post('/api/market-data/update-price-1min-csv', data)
export const generateJoinquantScripts = (data) => api.post('/api/market-data/joinquant-scripts', data)
export const checkMarketDataPrevData = (data) => api.post('/api/market-data/check-prev-data', data)
export const getMarketDataTaskStatus = (taskId) => api.get(`/api/market-data/task-status/${taskId}`)
export const terminateMarketDataTask = (taskId) => api.post(`/api/market-data/terminate/${taskId}`)
export const getMarketDataLogs = (params) => api.get('/api/market-data/logs', { params })
export const getMarketDataOverview = () => api.get('/api/market-data/overview')
export const getMarketDataOverview1min = () => api.get('/api/market-data/overview-1min')
export const getMarketDataPrice = (params) => api.get('/api/market-data/price', { params })
export const getMarketDataPrice1min = (params) => api.get('/api/market-data/price-1min', { params })
export const deleteMarketData = (data) => api.post('/api/market-data/delete', data)
export const getScheduledStatus = () => api.get('/api/market-data/scheduled-status')
export const updateScheduledConfig = (data) => api.post('/api/market-data/schedule-config', data)
export const toggleSchedule = (enabled) => api.post(`/api/market-data/toggle-schedule?enabled=${enabled}`)
export const getScheduledStatus1min = () => api.get('/api/market-data/scheduled-status-1min')
export const updateScheduledConfig1min = (data) => api.post('/api/market-data/schedule-config-1min', data)
export const toggleSchedule1min = (enabled) => api.post(`/api/market-data/toggle-schedule-1min?enabled=${enabled}`)

export default api
