"""
Lionet Factor Mining Web Backend - FastAPI
==========================================
Provides REST APIs for:
  1. Running GP factor mining (async background task)
  2. Querying existing factors from MongoDB
  3. Running backtest on selected factors and returning NAV curves
  4. Listing available versions / factor names from DB
  5. Running Strategy simulation
"""

import asyncio
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make lionet project root importable ────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import lionet modules
from data.factor_data import get_factor_formula_records, get_factor_formula_map_by_version
from mongo.mongify import get_data, list_collection_names
from factors.factor_auto_search import GeneticFactorGenerator, FactorGenerator
from factors.backtest import BackTester
from strategy.strategy import Strategy

# ── App Init ───────────────────────────────────────────────────────────
app = FastAPI(title="Lionet Factor Mining Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory task store ───────────────────────────────────────────────
tasks: Dict[str, Dict[str, Any]] = {}


# ══════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ══════════════════════════════════════════════════════════════════════

class GPMiningParams(BaseModel):
    # ── 基础参数 ──
    instrument_type: str = "futures_continuous_contract"
    instrument_id_list: str = "C0"
    fc_freq: str = "1d"
    start_time: str = "20200101"
    end_time: str = "20241231"
    version: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d") + "_gp_web")
    portfolio_adjust_method: str = "1D"
    interest_method: str = "simple"
    risk_free_rate: bool = False
    calculate_baseline: bool = True
    apply_weighted_price: bool = True
    n_jobs: int = 5
    max_factor_count: int = 50
    min_window_size: int = 30
    fitness_metric: str = "ic"
    # ── 滚动标准化 ──
    apply_rolling_norm: bool = True
    rolling_norm_window: int = 30
    rolling_norm_min_periods: int = 20
    rolling_norm_eps: float = 1e-8
    rolling_norm_clip: float = 5.0
    # ── 去重 / 泄露检查 ──
    check_leakage_count: int = 20
    check_relative: bool = True
    relative_threshold: float = 0.7
    # ── GP 核心参数 ──
    gp_generations: int = 60
    gp_population_size: int = 500
    gp_max_depth: int = 6
    gp_elite_size: int = 50
    gp_elite_relative_threshold: float = 0.65
    gp_crossover_prob: float = 0.3
    gp_mutation_prob: float = 0.7
    gp_leaf_prob: float = 0.2
    gp_const_prob: float = 0.02
    gp_tournament_size: int = 3
    gp_window_choices: List[int] = [3, 5, 10, 20, 30]
    random_seed: Optional[int] = None
    # ── GP 高级参数 ──
    gp_early_stopping_generation_count: int = 20
    gp_depth_penalty_coef: float = 0.0
    gp_depth_penalty_start_depth: int = 6
    gp_depth_penalty_linear_coef: float = 0.03
    gp_depth_penalty_quadratic_coef: float = 0.0
    gp_log_interval: int = 5
    gp_small_factor_penalty_coef: float = 0.0
    gp_assumed_initial_capital: float = 100000
    gp_elite_stagnation_generation_count: int = 4
    gp_max_shock_generation: int = 3
    # ── 筛选阈值 ──
    filter_net_return_mean: float = 0.05
    filter_net_return_yearly: float = 0.03
    filter_net_sharpe_mean: float = 0.5
    filter_net_sharpe_yearly: float = 0.3


class BacktestParams(BaseModel):
    version: str
    fc_name_list: List[str]
    collection: str = "genetic_programming"
    instrument_type: str = "futures_continuous_contract"
    instrument_id_list: str = "C0"
    fc_freq: str = "1d"
    start_time: str = "20200101"
    end_time: str = "20241231"
    portfolio_adjust_method: str = "1D"
    interest_method: str = "simple"
    risk_free_rate: bool = False
    calculate_baseline: bool = True
    apply_weighted_price: bool = True
    n_jobs: int = 5


class StrategyParams(BaseModel):
    version: str
    factor_name: str
    instrument_id: str = "C0"
    start_time: str = "20200101"
    end_time: str = "20241231"
    database: str = "factors"
    collection: str = "genetic_programming"
    initial_capital: float = 1000000.0
    margin_rate: float = 0.1
    fee_per_lot: float = 2.0
    slippage: float = 1.0
    apply_rolling_norm: bool = True
    rolling_norm_window: int = 30
    rolling_norm_min_periods: int = 20
    rolling_norm_eps: float = 1e-8
    rolling_norm_clip: float = 5.0
    signal_delay_days: int = 1
    min_open_ratio: float = 1.0


class FactorQueryParams(BaseModel):
    version: Optional[str] = None
    collection: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════

def _safe_float(v):
    """Convert to JSON-safe float."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _df_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dicts with JSON-safe values."""
    records = df.to_dict(orient="records")
    safe_records = []
    for rec in records:
        safe_rec = {}
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                safe_rec[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                safe_rec[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                safe_rec[k] = _safe_float(v)
            elif isinstance(v, np.bool_):
                safe_rec[k] = bool(v)
            else:
                safe_rec[k] = v
        safe_records.append(safe_rec)
    return safe_records


def _extract_nav_data(bt: BackTester) -> Dict[str, Any]:
    """Extract NAV curves and performance summary from BackTester."""
    result = {"nav_curves": {}, "performance_summary": []}

    if bt.performance_summary is not None:
        summary_df = bt.performance_summary.copy()
        if "year" not in summary_df.columns:
            summary_df = summary_df.reset_index()
        result["performance_summary"] = _df_to_records(summary_df)

    if bt.performance_detail is not None:
        detail = bt.performance_detail.copy()
        for fc_name in detail["factor_name"].unique():
            fc_detail = detail[detail["factor_name"] == fc_name].copy()
            fc_detail = fc_detail.sort_values("time")

            curve_data = {
                "time": [t.isoformat() if isinstance(t, (pd.Timestamp, datetime)) else str(t) for t in fc_detail["time"]],
                "gross_nav": [_safe_float(v) for v in fc_detail.get("daily_gross_nav", [])],
                "net_nav": [_safe_float(v) for v in fc_detail.get("daily_net_nav", [])],
            }

            if "daily_gross_nav_baseline_long" in fc_detail.columns:
                curve_data["gross_nav_baseline_long"] = [_safe_float(v) for v in fc_detail["daily_gross_nav_baseline_long"]]
                curve_data["gross_nav_baseline_short"] = [_safe_float(v) for v in fc_detail["daily_gross_nav_baseline_short"]]
                curve_data["net_nav_baseline_long"] = [_safe_float(v) for v in fc_detail["daily_net_nav_baseline_long"]]
                curve_data["net_nav_baseline_short"] = [_safe_float(v) for v in fc_detail["daily_net_nav_baseline_short"]]

            result["nav_curves"][fc_name] = curve_data

    return result


def _extract_strategy_nav_data(strat: Strategy) -> Dict[str, Any]:
    """Extract NAV curves and performance summary from Strategy."""
    result = {"nav_curves": {}, "performance_summary": [], "trade_detail": []}

    if strat.performance_summary is not None:
        summary_df = strat.performance_summary.copy()
        if "year" not in summary_df.columns:
            summary_df = summary_df.reset_index()
        result["performance_summary"] = _df_to_records(summary_df)

    if strat.performance_detail is not None:
        detail = strat.performance_detail.copy().sort_values("time")

        curve_data = {
            "time": [t.isoformat() if isinstance(t, (pd.Timestamp, datetime)) else str(t) for t in detail["time"]],
            "gross_nav": [_safe_float(v) for v in detail.get("daily_gross_nav", [])],
            "net_nav": [_safe_float(v) for v in detail.get("daily_net_nav", [])],
            "equity_nav": [_safe_float(v) for v in detail.get("nav", [])],
        }
        if "baseline_nav_long" in detail.columns:
            curve_data["gross_nav_baseline_long"] = [_safe_float(v) for v in detail["baseline_nav_long"]]
            curve_data["gross_nav_baseline_short"] = [_safe_float(v) for v in detail["baseline_nav_short"]]

        result["nav_curves"][strat.factor_name] = curve_data

        # Trade detail (subset of columns)
        keep_cols = [c for c in [
            "time", "factor_value", "position_lots", "target_lots", "delta_lots",
            "open", "close", "next_open", "future_ret",
            "daily_gross_pnl", "daily_net_pnl", "fee", "slippage_cost",
            "equity", "required_margin", "available_cash", "is_rebalanced", "warning",
        ] if c in detail.columns]
        result["trade_detail"] = _df_to_records(detail[keep_cols])

    return result


# ══════════════════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


# ── 1. List versions from DB ──────────────────────────────────────────

@app.get("/api/versions")
async def list_versions():
    """Return all available versions grouped by collection."""
    try:
        collections = list_collection_names(database="factors")
    except Exception:
        collections = ["genetic_programming", "llm_prompt"]

    version_map: Dict[str, List[str]] = {}
    all_versions = set()
    for col in collections:
        try:
            df = get_data(database="factors", collection=col, mongo_operator={})
            if isinstance(df, pd.DataFrame) and not df.empty and "version" in df.columns:
                versions = sorted(df["version"].dropna().unique().tolist(), reverse=True)
                version_map[col] = versions
                all_versions.update(versions)
        except Exception:
            continue
    return {
        "collections": collections,
        "version_map": version_map,
        "all_versions": sorted(all_versions, reverse=True),
    }


# ── 2. List factors by version ────────────────────────────────────────

@app.get("/api/factors")
async def list_factors(version: Optional[str] = None, collection: Optional[str] = None):
    """List factors (factor_name, formula, etc.) with optional version/collection filter."""
    try:
        collections = [collection] if collection else None
        versions = [version] if version else None
        df = get_factor_formula_records(collections=collections, versions=versions)
        if df.empty:
            return {"factors": [], "count": 0}

        records = _df_to_records(df)
        return {"factors": records, "count": len(records)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Start GP mining (background task) ──────────────────────────────

@app.post("/api/mining/start")
async def start_mining(params: GPMiningParams):
    """Launch a GP factor mining task in background. Returns task_id."""
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "params": params.dict(),
        "progress": "初始化中...",
        "result": None,
        "error": None,
    }

    async def _run():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _execute_mining, params, task_id)
            tasks[task_id]["result"] = result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = "完成"
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = traceback.format_exc()
            tasks[task_id]["progress"] = f"失败: {str(e)}"

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


def _execute_mining(params: GPMiningParams, task_id: str) -> Dict[str, Any]:
    """Execute GP mining synchronously (runs in thread pool)."""
    tasks[task_id]["progress"] = "创建 GeneticFactorGenerator..."

    fg = GeneticFactorGenerator(
        instrument_type=params.instrument_type,
        instrument_id_list=params.instrument_id_list,
        fc_freq=params.fc_freq,
        start_time=params.start_time,
        end_time=params.end_time,
        version=params.version,
        portfolio_adjust_method=params.portfolio_adjust_method,
        interest_method=params.interest_method,
        risk_free_rate=params.risk_free_rate,
        calculate_baseline=params.calculate_baseline,
        apply_weighted_price=params.apply_weighted_price,
        n_jobs=params.n_jobs,
        max_factor_count=params.max_factor_count,
        min_window_size=params.min_window_size,
        apply_rolling_norm=params.apply_rolling_norm,
        rolling_norm_window=params.rolling_norm_window,
        rolling_norm_min_periods=params.rolling_norm_min_periods,
        rolling_norm_eps=params.rolling_norm_eps,
        rolling_norm_clip=params.rolling_norm_clip,
        check_leakage_count=params.check_leakage_count,
        check_relative=params.check_relative,
        relative_threshold=params.relative_threshold,
        gp_generations=params.gp_generations,
        fitness_metric=params.fitness_metric,
        gp_max_depth=params.gp_max_depth,
        gp_population_size=params.gp_population_size,
        gp_elite_size=params.gp_elite_size,
        gp_elite_relative_threshold=params.gp_elite_relative_threshold,
        gp_crossover_prob=params.gp_crossover_prob,
        gp_mutation_prob=params.gp_mutation_prob,
        gp_leaf_prob=params.gp_leaf_prob,
        gp_const_prob=params.gp_const_prob,
        gp_tournament_size=params.gp_tournament_size,
        gp_window_choices=params.gp_window_choices,
        gp_depth_penalty_coef=params.gp_depth_penalty_coef,
        gp_depth_penalty_start_depth=params.gp_depth_penalty_start_depth,
        gp_depth_penalty_linear_coef=params.gp_depth_penalty_linear_coef,
        gp_depth_penalty_quadratic_coef=params.gp_depth_penalty_quadratic_coef,
        gp_early_stopping_generation_count=params.gp_early_stopping_generation_count,
        gp_log_interval=params.gp_log_interval,
        random_seed=params.random_seed,
        gp_assumed_initial_capital=params.gp_assumed_initial_capital,
        gp_small_factor_penalty_coef=params.gp_small_factor_penalty_coef,
        gp_elite_stagnation_generation_count=params.gp_elite_stagnation_generation_count,
        gp_max_shock_generation=params.gp_max_shock_generation,
    )

    filter_indicator_dict = {
        "Net Return": (params.filter_net_return_mean, params.filter_net_return_yearly, 1),
        "Net Sharpe": (params.filter_net_sharpe_mean, params.filter_net_sharpe_yearly, 1),
    }

    tasks[task_id]["progress"] = "正在运行遗传算法挖掘因子..."
    result = fg.auto_mine_select_and_save_fc(
        filter_indicator_dict=filter_indicator_dict,
        n_jobs=params.n_jobs,
        require_all_row=True,
        require_all_instruments=True,
    )

    tasks[task_id]["progress"] = "提取回测结果..."

    bt = result.get("bt")
    nav_data = _extract_nav_data(bt) if bt else {"nav_curves": {}, "performance_summary": []}

    formulas = {}
    if hasattr(fg, "factor_formula_map") and fg.factor_formula_map:
        formulas = dict(fg.factor_formula_map)

    return {
        "config_path": result.get("config_path"),
        "selected_fc_name_list": result.get("selected_fc_name_list", []),
        "message": result.get("message", ""),
        "nav_data": nav_data,
        "factor_formulas": formulas,
        "version": params.version,
    }


# ── 4. Check mining task status ───────────────────────────────────────

@app.get("/api/mining/status/{task_id}")
async def mining_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    resp = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "started_at": task["started_at"],
        "error": task.get("error"),
    }
    if task["status"] == "completed" and task["result"]:
        resp["result"] = task["result"]
    return resp


# ── 5. Run backtest on existing DB factors ────────────────────────────

@app.post("/api/backtest")
async def run_backtest(params: BacktestParams):
    """Run backtest on existing factors (from DB) and return NAV data."""
    try:
        bt = BackTester(
            fc_name_list=params.fc_name_list,
            version=params.version,
            collection=params.collection,
            instrument_type=params.instrument_type,
            instrument_id_list=params.instrument_id_list,
            fc_freq=params.fc_freq,
            start_time=params.start_time,
            end_time=params.end_time,
            portfolio_adjust_method=params.portfolio_adjust_method,
            interest_method=params.interest_method,
            risk_free_rate=params.risk_free_rate,
            calculate_baseline=params.calculate_baseline,
            apply_weighted_price=params.apply_weighted_price,
            n_jobs=params.n_jobs,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, bt.backtest)

        nav_data = _extract_nav_data(bt)
        return {
            "status": "ok",
            "nav_data": nav_data,
            "fc_name_list": params.fc_name_list,
            "version": params.version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── 6. Run Strategy simulation ────────────────────────────────────────

@app.post("/api/strategy")
async def run_strategy(params: StrategyParams):
    """Run Strategy simulation and return NAV data + trade detail."""
    try:
        strat = Strategy(
            version=params.version,
            factor_name=params.factor_name,
            instrument_id=params.instrument_id,
            start_time=params.start_time,
            end_time=params.end_time,
            database=params.database,
            collection=params.collection,
            initial_capital=params.initial_capital,
            margin_rate=params.margin_rate,
            fee_per_lot=params.fee_per_lot,
            slippage=params.slippage,
            apply_rolling_norm=params.apply_rolling_norm,
            rolling_norm_window=params.rolling_norm_window,
            rolling_norm_min_periods=params.rolling_norm_min_periods,
            rolling_norm_eps=params.rolling_norm_eps,
            rolling_norm_clip=params.rolling_norm_clip,
            signal_delay_days=params.signal_delay_days,
            min_open_ratio=params.min_open_ratio,
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, strat.backtest)

        nav_data = _extract_strategy_nav_data(strat)
        return {
            "status": "ok",
            "nav_data": nav_data,
            "factor_name": params.factor_name,
            "version": params.version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── 7. List all running/completed tasks ───────────────────────────────

@app.get("/api/tasks")
async def list_tasks():
    task_list = []
    for tid, info in tasks.items():
        task_list.append({
            "task_id": tid,
            "status": info["status"],
            "progress": info["progress"],
            "started_at": info["started_at"],
            "version": info.get("params", {}).get("version", ""),
        })
    return {"tasks": sorted(task_list, key=lambda x: x["started_at"], reverse=True)}


# ── Entry point ───────────────────────────────────────────────────────

# Serve Vue frontend dist in production mode
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

