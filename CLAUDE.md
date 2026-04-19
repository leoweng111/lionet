# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**lionet** is a quantitative futures factor-generating and backtesting framework with a Vue 3 + FastAPI web interface for interactive factor mining and visualization.

## Development Commands

### Backend (Python/FastAPI)
```bash
# Start backend with auto-reload
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the startup script
cd web && bash start.sh
```

### Frontend (Vue 3/Vite)
```bash
cd web/frontend
npm install        # Install dependencies
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build
```

### Integration Test (Required After Code Changes)
```bash
# Run full integration test: single-factor backtest, fusion-factor backtest, frontend build
python -u test/integration_test.py
```
This must pass before submitting any code changes. 

It verifies:
1. Single-factor backtest consistency across 4 methods (fc_name, formula, frontend-DB, frontend-formula)
2. Fusion-factor backtest consistency across 4 methods
3. Frontend build success
Pay attention to the last words that it prints.
If it prints RESULT: false (some tests failed), means one or more tests failed, you need to check the logs above to find out which test(s) failed and fix the issue before submitting code changes.
If it prints RESULT: true  (all tests passed), means all three tests passed successfully.

## Architecture

```
lionet/
├── factors/                    # Factor generation & backtesting core
│   ├── gp_factor_engine.py   # GP evolution engine (crossover, mutation, selection)
│   ├── factor_auto_search.py # GeneticFactorGenerator, FactorFusioner
│   ├── factor_ops.py         # Factor AST nodes (binary/unary operators, rolling norm)
│   ├── factor_indicators.py  # Performance metrics (Sharpe, IC, returns)
│   ├── backtest.py           # BackTester class
│   ├── fc_from_genetic_programming/  # GP-derived factors
│   ├── fc_from_llm/          # LLM-generated factors
│   └── fc_from_tsfresh/      # tsfresh-based factors
├── strategy/
│   └── strategy.py           # Strategy: day-session open-to-open futures simulation
├── data/                      # Futures data API
│   ├── futures.py            # Continuous contract price queries
│   └── factor_data.py        # Factor formula storage/retrieval
├── mongo/                     # MongoDB integration
├── models/                    # ML models for signal generation
├── web/
│   ├── backend/main.py       # FastAPI app (GP mining, backtest, fusion endpoints)
│   └── frontend/             # Vue 3 SPA (Vite + Element Plus + ECharts)
├── test/                      # Smoke tests and analysis scripts
└── utils/
    ├── params.py              # Contract multipliers, config
    └── logging.py             # Logging utilities
```

## Key Classes

- **`GeneticFactorGenerator`** (`factors/factor_auto_search.py`) — Orchestrates GP evolution: population initialization, fitness evaluation via `BackTester`, selection, crossover, mutation, early stopping with shock mode for exploration bursts. Call `auto_mine_select_and_save_fc()` to run a full mining pipeline.

- **`BackTester`** (`factors/backtest.py`) — Computes factor performance using weighted prices, portfolio adjustment, baseline comparison. Outputs `performance_detail` (daily NAV) and `performance_summary` (yearly metrics).

- **`Strategy`** (`strategy/strategy.py`) — Simulates day-session open-to-open futures trading with margin, fees, slippage. T-signal → T+1 open execution convention. Uses `OpRollNorm` wrapper by default.

- **`FactorFusioner`** (`factors/factor_auto_search.py`) — Combines multiple factors via weighted average or other fusion methods, with leakage and similarity checks.

- **`gp_factor_engine.py`** — Low-level GP primitives: tree generation, crossover, mutation, depth penalties, tournament selection.

## Factor Formula Language

Factors are stored as formula strings parsed into AST trees (`FactorNode`):
- **Leaf nodes**: `DataNode(field)` — raw fields like `close`, `volume`; `ConstNode(value)`
- **Unary ops**: `OpNeg`, `OpRollNorm(child, window, min_periods, eps, clip)`
- **Binary ops**: `OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpPow`
- **Time-series ops**: `OpMean`, `OpStd`, `OpMax`, `OpMin`, `OpCorr`, `OpCov` — all take `(child, window)`

Formulas are evaluated via `calc_formula_series(df, formula)` in `factor_ops.py`.

## Web API

Backend serves at `:8000` with these endpoints:
- `POST /api/mining/start` — Start GP factor mining (async, returns task_id)
- `GET /api/mining/status/{task_id}` — Poll mining progress with GP generation info
- `POST /api/backtest` — Run factor backtest synchronously
- `POST /api/fusion/start` — Start factor fusion task
- `GET /api/tasks` — List all tasks (in-memory + MongoDB)
- `POST /api/strategy` — Run Strategy simulation

## Data Model

Factor formulas stored in MongoDB (`database=factors`):
- `genetic_programming` collection: GP-mined factors
- `llm_prompt` collection: LLM-generated factors
- `factor_fusion` collection: Fused factors

Task state stored in `database=task` (`gp_task`, `fusion_task` collections).

## Python Path Setup

Many scripts add `PROJECT_ROOT` to `sys.path`:
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

This ensures imports like `from factors.factor_auto_search import ...` work when running scripts from subdirectories.

请使用中文输出最终的回答和概述。