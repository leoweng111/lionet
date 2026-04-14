"""
Lionet Factor Mining Web Backend - FastAPI
"""

import asyncio
import copy
import json
import logging
import os
import re
import sys
import threading
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
from mongo.mongify import get_data, list_collection_names, update_one_data
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
#  GP progress log handler — intercepts log messages to extract
#  generation progress and global best fitness in real-time
# ══════════════════════════════════════════════════════════════════════

_GP_GEN_RE = re.compile(
    r'GP generation (\d+)/(\d+).*'
    r'global_best_penalized=([\d.eE+-]+).*'
    r'global_best_original=([\d.eE+-]+)'
)


class _GPProgressHandler(logging.Handler):
    """Attach to the lionet logger to capture GP generation progress."""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def emit(self, record):
        if self.task_id not in tasks:
            return
        msg = record.getMessage()
        m = _GP_GEN_RE.search(msg)
        if m:
            gen_cur, gen_total, best_pen, best_orig = m.group(1), m.group(2), m.group(3), m.group(4)
            tasks[self.task_id]["progress"] = (
                f"演化进度: {gen_cur}/{gen_total} 代 | "
                f"全局最优 penalized={best_pen}, original={best_orig}"
            )
            tasks[self.task_id]["gp_progress"] = {
                "generation": int(gen_cur),
                "total_generations": int(gen_total),
                "global_best_penalized": float(best_pen),
                "global_best_original": float(best_orig),
            }


# ══════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ══════════════════════════════════════════════════════════════════════

class GPMiningParams(BaseModel):
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
    apply_rolling_norm: bool = True
    rolling_norm_window: int = 30
    rolling_norm_min_periods: int = 20
    rolling_norm_eps: float = 1e-8
    rolling_norm_clip: float = 5.0
    check_leakage_count: int = 20
    check_relative: bool = True
    relative_threshold: float = 0.7
    gp_generations: int = 20
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
    filter_net_return_mean: float = 0.05
    filter_net_return_yearly: float = 0.03
    filter_net_sharpe_mean: float = 0.5
    filter_net_sharpe_yearly: float = 0.3


class BacktestParams(BaseModel):
    version: Optional[str] = None
    fc_name_list: Optional[List[str]] = None
    formula: Optional[str] = None
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


class StrategyBatchParams(BaseModel):
    version: str
    factor_name_list: List[str]
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
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _df_to_records(df: pd.DataFrame) -> List[Dict]:
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
    result = {"nav_curves": {}, "performance_summary": []}
    if bt.performance_summary is not None:
        summary_df = bt.performance_summary.copy()
        if "year" not in summary_df.columns:
            summary_df = summary_df.reset_index()
        result["performance_summary"] = _df_to_records(summary_df)
    if bt.performance_detail is not None:
        detail = bt.performance_detail.copy()
        for fc_name in detail["factor_name"].unique():
            fc_detail = detail[detail["factor_name"] == fc_name].copy().sort_values("time")
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
        keep_cols = [c for c in [
            "time", "factor_value", "position_lots", "target_lots", "delta_lots",
            "open", "close", "next_open", "future_ret",
            "daily_gross_pnl", "daily_net_pnl", "fee", "slippage_cost",
            "equity", "required_margin", "available_cash", "is_rebalanced", "warning",
        ] if c in detail.columns]
        result["trade_detail"] = _df_to_records(detail[keep_cols])
    return result


def _save_task_to_db(task_id: str, params_dict: dict, status: str,
                     result_summary: Optional[dict] = None):
    """Persist task detail into MongoDB (database=task, collection=task_detail)."""
    try:
        record = {
            "task_id": task_id,
            "status": status,
            "params": params_dict,
            "started_at": tasks.get(task_id, {}).get("started_at", ""),
            "finished_at": datetime.now().isoformat(),
        }
        if result_summary:
            record["result_summary"] = result_summary
        update_one_data(
            database="task",
            collection="task_detail",
            mongo_operator={"task_id": task_id},
            data=record,
            upsert=True,
        )
    except Exception as e:
        print(f"[WARNING] Failed to save task {task_id} to DB: {e}")


# ══════════════════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/api/versions")
async def list_versions():
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
    return {"collections": collections, "version_map": version_map, "all_versions": sorted(all_versions, reverse=True)}


@app.get("/api/factors")
async def list_factors(version: Optional[str] = None, collection: Optional[str] = None):
    try:
        cols = [collection] if collection else None
        vers = [version] if version else None
        df = get_factor_formula_records(collections=cols, versions=vers)
        if df.empty:
            return {"factors": [], "count": 0}
        return {"factors": _df_to_records(df), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GP Mining ─────────────────────────────────────────────────────────

@app.post("/api/mining/start")
async def start_mining(params: GPMiningParams):
    task_id = str(uuid.uuid4())[:8]
    cancel_event = threading.Event()
    tasks[task_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "params": params.dict(),
        "progress": "初始化中...",
        "gp_progress": None,
        "result": None,
        "error": None,
        "cancel_event": cancel_event,
    }

    async def _run():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _execute_mining, params, task_id, cancel_event)
            # If terminated by user, still save partial results but keep terminated status
            if tasks[task_id]["status"] == "terminated":
                tasks[task_id]["result"] = result
                _save_task_to_db(task_id, params.dict(), "terminated", {
                    "selected_fc_name_list": result.get("selected_fc_name_list", []),
                    "version": result.get("version", ""),
                    "message": result.get("message", "Task terminated by user."),
                })
                return
            tasks[task_id]["result"] = result
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = "完成"
            _save_task_to_db(task_id, params.dict(), "completed", {
                "selected_fc_name_list": result.get("selected_fc_name_list", []),
                "version": result.get("version", ""),
                "message": result.get("message", ""),
            })
        except Exception as e:
            # If terminated by user, keep status but note the error
            if tasks[task_id]["status"] == "terminated":
                tasks[task_id]["progress"] = "已终止（用户手动终止）"
                return
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = traceback.format_exc()
            tasks[task_id]["progress"] = f"失败: {str(e)}"
            _save_task_to_db(task_id, params.dict(), "failed", {"error": str(e)})

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


def _execute_mining(params: GPMiningParams, task_id: str, cancel_event: threading.Event) -> Dict[str, Any]:
    tasks[task_id]["progress"] = "创建 GeneticFactorGenerator..."

    # Attach log handler to capture GP progress
    from utils.logging import log as lionet_logger
    handler = _GPProgressHandler(task_id)
    lionet_logger.addHandler(handler)

    try:
        fg = GeneticFactorGenerator(
            instrument_type=params.instrument_type,
            instrument_id_list=params.instrument_id_list,
            fc_freq=params.fc_freq,
            start_time=params.start_time, end_time=params.end_time,
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

        # Attach cancel_event so GP evolution can be interrupted
        fg.cancel_event = cancel_event

        filter_indicator_dict = {
            "Net Return": (params.filter_net_return_mean, params.filter_net_return_yearly, 1),
            "Net Sharpe": (params.filter_net_sharpe_mean, params.filter_net_sharpe_yearly, 1),
        }

        tasks[task_id]["progress"] = "正在运行遗传算法挖掘因子 (0/" + str(params.gp_generations) + " 代)..."
        result = fg.auto_mine_select_and_save_fc(
            filter_indicator_dict=filter_indicator_dict,
            n_jobs=params.n_jobs, require_all_row=True, require_all_instruments=True,
        )

        tasks[task_id]["progress"] = "提取回测结果..."
        bt = result.get("bt")
        nav_data = _extract_nav_data(bt) if bt else {"nav_curves": {}, "performance_summary": []}
        formulas = dict(fg.factor_formula_map) if hasattr(fg, "factor_formula_map") and fg.factor_formula_map else {}

        return {
            "config_path": result.get("config_path"),
            "selected_fc_name_list": result.get("selected_fc_name_list", []),
            "message": result.get("message", ""),
            "nav_data": nav_data,
            "factor_formulas": formulas,
            "version": params.version,
        }
    finally:
        lionet_logger.removeHandler(handler)


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
        "gp_progress": task.get("gp_progress"),
        "params": task.get("params"),
    }
    if task["status"] == "completed" and task["result"]:
        resp["result"] = task["result"]
    return resp


@app.post("/api/mining/terminate/{task_id}")
async def terminate_mining(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Task is not running (status={task['status']})")
    cancel_event = task.get("cancel_event")
    if cancel_event is None:
        raise HTTPException(status_code=400, detail="Task does not support cancellation")
    cancel_event.set()
    task["status"] = "terminated"
    task["progress"] = "已终止（用户手动终止）"
    _save_task_to_db(task_id, task.get("params", {}), "terminated", {"message": "用户手动终止"})
    return {"task_id": task_id, "status": "terminated", "message": "任务已终止"}


# ── Backtest ──────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def run_backtest(params: BacktestParams):
    try:
        bt_kwargs = dict(
            collection=params.collection, instrument_type=params.instrument_type,
            instrument_id_list=params.instrument_id_list, fc_freq=params.fc_freq,
            start_time=params.start_time, end_time=params.end_time,
            portfolio_adjust_method=params.portfolio_adjust_method,
            interest_method=params.interest_method, risk_free_rate=params.risk_free_rate,
            calculate_baseline=params.calculate_baseline,
            apply_weighted_price=params.apply_weighted_price, n_jobs=params.n_jobs,
        )
        if params.formula:
            bt_kwargs["formula"] = params.formula
            bt_kwargs["version"] = params.version or "__formula__"
            bt_kwargs["fc_name_list"] = ["formula_factor"]
        else:
            if not params.version or not params.fc_name_list:
                raise ValueError("version and fc_name_list are required when formula is not provided.")
            bt_kwargs["version"] = params.version
            bt_kwargs["fc_name_list"] = params.fc_name_list

        bt = BackTester(**bt_kwargs)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, bt.backtest)
        nav_data = _extract_nav_data(bt)

        # In formula mode, rename 'formula_factor' to the actual formula in the response
        if params.formula and "formula_factor" in nav_data.get("nav_curves", {}):
            nav_data["nav_curves"][params.formula] = nav_data["nav_curves"].pop("formula_factor")
            for rec in nav_data.get("performance_summary", []):
                if rec.get("Factor Name") == "formula_factor":
                    rec["Factor Name"] = params.formula

        return {
            "status": "ok", "nav_data": nav_data,
            "fc_name_list": [params.formula] if params.formula else params.fc_name_list,
            "version": params.version or "",
            "formula": params.formula,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Strategy (batch) ──────────────────────────────────────────────────

@app.post("/api/strategy")
async def run_strategy(params: StrategyBatchParams):
    """Run Strategy simulation for one or multiple factors."""
    try:
        all_results = []
        for factor_name in params.factor_name_list:
            strat = Strategy(
                version=params.version, factor_name=factor_name,
                instrument_id=params.instrument_id,
                start_time=params.start_time, end_time=params.end_time,
                database=params.database, collection=params.collection,
                initial_capital=params.initial_capital, margin_rate=params.margin_rate,
                fee_per_lot=params.fee_per_lot, slippage=params.slippage,
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
            all_results.append({"factor_name": factor_name, "nav_data": nav_data})

        return {"status": "ok", "results": all_results, "version": params.version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── Task list + DB-persisted detail ───────────────────────────────────

@app.get("/api/tasks")
async def list_tasks():
    task_list = []
    for tid, info in tasks.items():
        task_list.append({
            "task_id": tid, "status": info["status"], "progress": info["progress"],
            "started_at": info["started_at"],
            "version": info.get("params", {}).get("version", ""),
            "gp_progress": info.get("gp_progress"),
        })
    # Also fetch from DB for historical tasks
    try:
        db_tasks = get_data(database="task", collection="task_detail", mongo_operator={})
        if isinstance(db_tasks, pd.DataFrame) and not db_tasks.empty:
            for _, row in db_tasks.iterrows():
                tid = row.get("task_id", "")
                if tid and tid not in tasks:
                    task_list.append({
                        "task_id": tid, "status": row.get("status", "unknown"),
                        "progress": "已完成 (历史记录)",
                        "started_at": row.get("started_at", ""),
                        "version": row.get("params", {}).get("version", "") if isinstance(row.get("params"), dict) else "",
                        "gp_progress": None,
                    })
    except Exception:
        pass
    return {"tasks": sorted(task_list, key=lambda x: x.get("started_at", ""), reverse=True)}


@app.get("/api/tasks/detail/{task_id}")
async def get_task_detail(task_id: str):
    """Get full task detail including params, from memory or DB."""
    if task_id in tasks:
        task = tasks[task_id]
        return {
            "task_id": task_id, "status": task["status"], "progress": task["progress"],
            "started_at": task["started_at"], "error": task.get("error"),
            "gp_progress": task.get("gp_progress"), "params": task.get("params"),
            "result": task.get("result") if task["status"] in ("completed", "terminated") else None,
        }
    # Fallback to DB
    try:
        df = get_data(database="task", collection="task_detail", mongo_operator={"task_id": task_id})
        if isinstance(df, pd.DataFrame) and not df.empty:
            row = df.iloc[0].to_dict()
            return {
                "task_id": task_id, "status": row.get("status", "unknown"),
                "progress": "已完成 (历史记录)", "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "params": row.get("params"), "result_summary": row.get("result_summary"),
                "error": None, "gp_progress": None,
            }
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Task not found")


# ── Entry point ───────────────────────────────────────────────────────

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

