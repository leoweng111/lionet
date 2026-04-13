# Lionet Web - 因子挖掘平台

> Vue 3 + FastAPI 全栈因子挖掘与回测可视化平台

## 功能

| 页面 | 功能 |
|------|------|
| **因子挖掘** | 配置 GP 遗传算法超参数，一键启动因子挖掘，实时查看进度和净值曲线 |
| **因子库** | 浏览数据库中所有已保存因子，按版本/集合筛选，支持批量选择进行回测 |
| **回测分析** | 选择已有因子运行样本内/样本外回测，展示净值曲线和绩效指标 |
| **任务管理** | 查看所有挖掘任务的运行状态和历史记录 |

## 技术栈

- **前端**: Vue 3 + Vite + Element Plus + ECharts + Vue Router
- **后端**: FastAPI + Uvicorn
- **数据**: MongoDB (通过项目已有 `mongo` 模块)

## 快速启动

### 1. 安装依赖

```bash
# Python 后端依赖
conda activate future
pip install fastapi "uvicorn[standard]"

# Vue 前端依赖
cd web/frontend
npm install
```

### 2. 启动服务

**方式一**: 使用启动脚本
```bash
cd web
bash start.sh
```

**方式二**: 分别启动
```bash
# 终端1: 启动后端
cd /path/to/lionet
conda activate future
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload

# 终端2: 启动前端
cd web/frontend
npm run dev
```

### 3. 访问

- **前端**: http://localhost:5173
- **后端 API 文档**: http://localhost:8000/docs

## 生产部署

```bash
# 构建前端
cd web/frontend
npm run build

# 启动后端 (自动 serve 前端 dist)
cd /path/to/lionet
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

构建后直接访问 http://localhost:8000 即可。

## GitHub Pages 部署

前端构建产物 (`web/frontend/dist/`) 可直接部署到 GitHub Pages:

```bash
cd web/frontend
npm run build
# 将 dist/ 内容推送到 gh-pages 分支
```

> ⚠️ GitHub Pages 仅能托管前端静态文件，需要另外部署后端 API 并修改 `src/api/index.js` 中的 `baseURL`。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/versions` | 获取所有版本列表 |
| GET | `/api/factors` | 获取因子列表 (支持 version/collection 过滤) |
| POST | `/api/mining/start` | 启动 GP 因子挖掘任务 |
| GET | `/api/mining/status/{task_id}` | 查询挖掘任务状态 |
| POST | `/api/backtest` | 运行因子回测 |
| GET | `/api/tasks` | 获取所有任务列表 |

## 目录结构

```
web/
├── start.sh              # 一键启动脚本
├── backend/
│   └── main.py           # FastAPI 后端
└── frontend/
    ├── src/
    │   ├── main.js        # Vue 入口
    │   ├── App.vue        # 主布局 (侧边栏 + 路由)
    │   ├── style.css      # 全局样式
    │   ├── api/
    │   │   └── index.js   # Axios API 封装
    │   ├── router/
    │   │   └── index.js   # Vue Router 路由
    │   ├── components/
    │   │   └── NavChart.vue  # 净值曲线 ECharts 组件
    │   └── views/
    │       ├── MiningView.vue    # 因子挖掘页
    │       ├── FactorsView.vue   # 因子库页
    │       ├── BacktestView.vue  # 回测分析页
    │       └── TasksView.vue     # 任务管理页
    └── vite.config.js     # Vite 配置 (含 API 代理)
```

