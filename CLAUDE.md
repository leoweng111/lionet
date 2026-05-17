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

### GP Mining Smoke Test
```bash
# Run GP factor mining smoke test: random fitness weights, 1-generation evolution, extreme filter thresholds
python -u test/gp_mining_smoke.py
```
This test verifies the full GP mining pipeline (`GeneticFactorGenerator.auto_mine_select_and_save_fc`) runs without errors:
- Randomly selects fitness indicator weights (sum to 1).
- Sets filter thresholds impossibly high so no factor gets saved to DB.
- Runs only 1 generation with a small population for speed.
- Asserts: factors are generated, backtest completes, no factor passes the extreme filter, formula map is populated.

It includes 3 sub-tests:
1. **GP Mining with Outsample** — Direct `GeneticFactorGenerator` call with `outsample_ratio=0.3`, verifies outsample data generation and outsample backtest.
2. **Simulate Backend Mining** — Mirrors `main.py` `_execute_mining` parameter construction (all params, normalize logic), catches param-passing mismatches between backend and core.
3. **Simulate Backend Fusion** — Mirrors `main.py` `_execute_fusion` parameter construction for `FactorFusioner`, verifies `fusion_indicator_dict` and outsample params (`outsample_ratio`, `outsample_start_time`, `outsample_end_time`) are correctly passed through.

### GP + Gradient Descent Smoke Test
```bash
# Run GP+GD mining smoke test: alternated full mining flow, consecutive elite refinement, non-differentiable fitness rejection
python -u test/gp_gradient_descent_smoke.py
```
This test verifies the optional GP+gradient-descent pipeline runs without changing the legacy GP path when disabled:
- Runs `GeneticFactorGenerator.auto_mine_select_and_save_fc()` on synthetic C0 data with `enable_gradient_descent=True`, `gradient_descent_method='alternated'`, small population/generation count, and extreme filter thresholds so no factor is saved to DB.
- Runs `run_gp_evolution()` with `gradient_descent_method='consecutive'` to verify final elite refinement.
- Asserts non-differentiable fitness metrics such as `TS RankIC` are rejected before mining starts.

### Fusion Smoke Test
```bash
# Run factor fusion smoke test: full fusion pipeline with real DB factors
python -u test/fusion_smoke.py
```
This test verifies the full `FactorFusioner.fuse()` pipeline runs without errors using real factors from MongoDB:
- Uses `genetic_programming` collection with versions `20260507_gp_test` and `20260505_gp_test`.
- Sets `max_fusion_count=2` for speed.
- Asserts: fusion completes, formula is generated, backtest runs successfully.

It includes 3 sub-tests:
1. **Direct Fusion** — Direct `FactorFusioner` call with default indicator weights, verifies end-to-end fusion flow.
2. **Simulate Backend Fusion** — Mirrors `main.py` `_execute_fusion` parameter construction (normalize `fusion_indicator_dict`, pass `use_version_dict`), catches param-passing mismatches.
3. **Simulate Frontend Payload** — Verifies the frontend's `_selectedCollections` + `_selectedVersions` → `use_version_dict` construction logic matches what `FactorFusioner` expects.

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
    ├── llm_utils.py           # LLM calling utilities (anthropic SDK, multi-profile from .env)
    └── logging.py             # Logging utilities
```

## Key Classes

- **`GeneticFactorGenerator`** (`factors/factor_auto_search.py`) — Orchestrates GP evolution: population initialization, fitness evaluation via `BackTester`, selection, crossover, mutation, early stopping with shock mode for exploration bursts. Call `auto_mine_select_and_save_fc()` to run a full mining pipeline.

- **`LLMPromptFactorGenerator`** (`factors/factor_auto_search.py`) — Uses LLM to generate factor formulas from natural language prompts. Supports multiple LLM backends via `utils/llm_utils.py` (configured in `.env` LLM_PROFILES). When `apply_rolling_norm=True`, wraps generated formulas with `OpRollNorm` (same behavior as GP). Supports cancel via `cancel_event`.

- **`BackTester`** (`factors/backtest.py`) — Computes factor performance using weighted prices, portfolio adjustment, baseline comparison. Outputs `performance_detail` (daily NAV) and `performance_summary` (yearly metrics).

- **`Strategy`** (`strategy/strategy.py`) — Simulates day-session open-to-open futures trading with margin, fees, slippage. T-signal → T+1 open execution convention. Uses `OpRollNorm` wrapper by default.

- **`FactorFusioner`** (`factors/factor_auto_search.py`) — Combines multiple factors via weighted average or other fusion methods, with leakage and similarity checks. Uses `use_version_dict` (`{collection: [version, ...]}`) to specify which factors to load from MongoDB. Uses `fusion_indicator_dict` for weighted multi-indicator scoring (same indicator system as GP fitness). Supports outsample blending via `outsample_ratio`: `blended_score = (1 - outsample_ratio) * insample_score + outsample_ratio * outsample_score`.

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
- `POST /api/llm-mining/start` — Start LLM factor mining (async, returns task_id)
- `GET /api/llm-mining/status/{task_id}` — Poll LLM mining progress
- `POST /api/llm-mining/terminate/{task_id}` — Stop LLM mining task
- `GET /api/llm-mining/profiles` — List available LLM profiles from .env
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