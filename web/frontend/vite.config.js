import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const isCI = process.env.GITHUB_ACTIONS === 'true'
const repoName = (process.env.GITHUB_REPOSITORY || '').split('/')[1] || 'lionet'
const githubPagesBase = `/${repoName}/`

// https://vite.dev/config/
export default defineConfig({
  // For GitHub Pages project site, app is served under /<repo>/.
  // Local dev/build keeps root path unless explicitly overridden.
  base: process.env.VITE_PUBLIC_BASE || (isCI ? githubPagesBase : '/'),
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
