"""
Lionet Factor Mining Web Backend - FastAPI
"""

import asyncio
import logging
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
from data.factor_data import get_factor_formula_records
from mongo.mongify import get_data, delete_data, list_collection_names, update_one_data, update_many_data
from data.futures import (
    get_futures_continuous_contract_info,
    get_futures_continuous_contract_price,
    update_futures_continuous_contract_info,
    update_futures_continuous_contract_price,
)
from factors.factor_auto_search import GeneticFactorGenerator, FactorFusioner
from factors.backtest import BackTester
from strategy.strategy import Strategy
from utils.params import (
    GP_DEFAULT_FILTER_INDICATOR_DICT,
    GP_DEFAULT_FITNESS_INDICATOR_WEIGHT,
    GP_INDICATOR_DIRECTION,
    GP_SUPPORTED_INDICATOR,
)

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

GP_TASK_COLLECTION = "gp_task"
FUSION_TASK_COLLECTION = "fusion_task"
TASK_TYPE_GP = "gp_mining"
TASK_TYPE_FUSION = "factor_fusion"


def _task_type_label(task_type: Optional[str]) -> str:
    return "因子融合" if task_type == TASK_TYPE_FUSION else "因子挖掘"


def _task_collection(task_type: Optional[str]) -> str:
    return FUSION_TASK_COLLECTION if task_type == TASK_TYPE_FUSION else GP_TASK_COLLECTION


# ── Startup: clean up stale "running" tasks in DB ─────────────────────
@app.on_event("startup")
async def _cleanup_stale_tasks():
    """Mark any DB tasks stuck in 'running' as 'interrupted'.

    When the server process restarts, those tasks are definitely dead.
    """
    try:
        total_modified = 0
        for collection in (GP_TASK_COLLECTION, FUSION_TASK_COLLECTION):
            result = update_many_data(
                database="task",
                collection=collection,
                mongo_operator={"status": "running"},
                update_data={
                    "status": "interrupted",
                    "progress": "服务重启，任务已中断",
                    "finished_at": datetime.now().isoformat(),
                },
            )
            total_modified += int(result.get("modified_count", 0) or 0)
        if total_modified > 0:
            print(f"[STARTUP] Marked {total_modified} stale running task(s) as interrupted.")
    except Exception as e:
        print(f"[STARTUP] Failed to clean up stale tasks: {e}")


# ══════════════════════════════════════════════════════════════════════
#  GP progress log handler — intercepts log messages to extract
#  generation progress and global best fitness in real-time
# ══════════════════════════════════════════════════════════════════════

_GP_GEN_RE = re.compile(
    r'GP generation (\d+)/(\d+).*'
    r'global_best_penalized=([\d.eE+-]+).*'
    r'global_best_original=([\d.eE+-]+)'
)


def _append_task_log(task_id: str, record: logging.LogRecord, owner_thread_id: Optional[int] = None) -> Optional[str]:
    if task_id not in tasks:
        return None
    if owner_thread_id is not None and int(record.thread) != int(owner_thread_id):
        return None
    msg = record.getMessage()
    log_line = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {record.levelname} - {msg}'
    logs = tasks[task_id].setdefault("logs", [])
    logs.append(log_line)
    if len(logs) > 500:
        del logs[:-500]
    return msg


class _GPProgressHandler(logging.Handler):
    """Attach to the lionet logger to capture GP generation progress."""

    def __init__(self, task_id: str, owner_thread_id: Optional[int] = None):
        super().__init__()
        self.task_id = task_id
        self.owner_thread_id = owner_thread_id

    def emit(self, record):
        msg = _append_task_log(self.task_id, record, self.owner_thread_id)
        if msg is None:
            return
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
            # Persist live progress so /api/tasks works even across multi-worker processes.
            try:
                update_one_data(
                    database="task",
                    collection=GP_TASK_COLLECTION,
                    mongo_operator={"task_id": self.task_id},
                    data={
                        "task_id": self.task_id,
                        "task_type": TASK_TYPE_GP,
                        "status": tasks[self.task_id].get("status", "running"),
                        "progress": tasks[self.task_id].get("progress", ""),
                        "gp_progress": tasks[self.task_id].get("gp_progress"),
                        "started_at": tasks[self.task_id].get("started_at", ""),
                        "params": tasks[self.task_id].get("params", {}),
                    },
                    upsert=True,
                )
            except Exception:
                pass


class _FusionLogHandler(logging.Handler):
    """Capture fusion task logs into the shared in-memory task store."""

    def __init__(self, task_id: str, owner_thread_id: Optional[int] = None):
        super().__init__()
        self.task_id = task_id
        self.owner_thread_id = owner_thread_id

    def emit(self, record):
        _append_task_log(self.task_id, record, self.owner_thread_id)


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
    fitness_indicator_dict: Dict[str, Optional[float]] = Field(default_factory=lambda: dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT))
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
    gp_log_interval: int = 1
    gp_small_factor_penalty_coef: float = 0.0
    gp_assumed_initial_capital: float = 100000
    gp_elite_stagnation_generation_count: int = 4
    gp_max_shock_generation: int = 3
    filter_net_return_mean: float = 0.05
    filter_net_return_yearly: float = 0.03
    filter_net_sharpe_mean: float = 0.5
    filter_net_sharpe_yearly: float = 0.3
    # indicator -> {mean_threshold, yearly_threshold, direction}
    filter_indicator_dict: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)
    consistency_penalty_enabled: bool = False
    consistency_penalty_coef: float = 1.0


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


class FusionParams(BaseModel):
    fusion_method: str = 'avg_weight'
    collection: List[str] = ['genetic_programming']
    candidate_versions: Optional[List[str]] = None
    instrument_type: str = 'futures_continuous_contract'
    instrument_id_list: str = 'C0'
    fc_freq: str = '1d'
    start_time: str = '20200101'
    end_time: str = '20241231'
    portfolio_adjust_method: str = '1D'
    interest_method: str = 'simple'
    risk_free_rate: bool = False
    apply_weighted_price: bool = True
    check_leakage_count: int = 20
    check_relative: bool = True
    relative_threshold: float = 0.7
    relative_check_version_list: Optional[List[str]] = None
    max_fusion_count: int = 5
    fusion_metrics: List[str] = ['ic']
    version: str = Field(default_factory=lambda: datetime.now().strftime('%Y%m%d') + '_factor_fusion_test')
    n_jobs: int = 5
    base_col_list: Optional[List[str]] = None
    consider_outsample: bool = False
    outsample_start_day: Optional[str] = None
    outsample_end_day: Optional[str] = None


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


def _sanitize_nan(v, default=None):
    """Convert pandas NaN / numpy NaN to a JSON-safe *default* value."""
    if v is None:
        return default
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return default
    return v


def _normalize_fitness_indicator_dict(params: GPMiningParams) -> Dict[str, float]:
    indicator_weight_raw = dict(params.fitness_indicator_dict or {})
    indicator_weight: Dict[str, float] = {}

    for indicator in GP_SUPPORTED_INDICATOR:
        raw_weight = indicator_weight_raw.get(indicator, None)
        if raw_weight is None:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if abs(weight) <= 1e-12:
            continue
        indicator_weight[indicator] = weight

    # 兼容旧参数：未提供新字典时，沿用 fitness_metric。
    if not indicator_weight:
        metric = str(params.fitness_metric or '').strip().lower()
        if metric == 'sharpe':
            indicator_weight = {'Gross Sharpe': 1.0}
        else:
            indicator_weight = {'TS IC': 1.0}
    return indicator_weight


def _normalize_filter_indicator_dict(
    params: GPMiningParams,
) -> Dict[str, tuple[Optional[float], Optional[float], int]]:
    raw = dict(params.filter_indicator_dict or {})
    out: Dict[str, tuple[Optional[float], Optional[float], int]] = {}

    for indicator, conf in raw.items():
        if indicator not in GP_SUPPORTED_INDICATOR:
            continue
        conf = conf or {}

        mean_threshold_raw = conf.get('mean_threshold')
        yearly_threshold_raw = conf.get('yearly_threshold')
        direction_raw = conf.get('direction')

        mean_threshold = None if mean_threshold_raw is None else float(mean_threshold_raw)
        yearly_threshold = None if yearly_threshold_raw is None else float(yearly_threshold_raw)

        direction_default = int(GP_INDICATOR_DIRECTION.get(indicator, 1))
        direction = direction_default if direction_raw is None else int(direction_raw)
        if direction not in (1, -1):
            direction = direction_default

        out[indicator] = (mean_threshold, yearly_threshold, direction)

    # 兼容旧参数：若新结构为空，使用原有 Net Return / Net Sharpe 阈值。
    if not out:
        for indicator, (default_mean, default_yearly, default_direction) in GP_DEFAULT_FILTER_INDICATOR_DICT.items():
            if indicator == 'Net Return':
                out[indicator] = (float(params.filter_net_return_mean), float(params.filter_net_return_yearly), default_direction)
            elif indicator == 'Net Sharpe':
                out[indicator] = (float(params.filter_net_sharpe_mean), float(params.filter_net_sharpe_yearly), default_direction)
            else:
                out[indicator] = (default_mean, default_yearly, default_direction)

    return out


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


def _build_mining_result_overview(result: Dict[str, Any]) -> str:
    selected = result.get('selected_fc_name_list', []) or []
    config_path = result.get('config_path')
    msg = str(result.get('message', '') or '').strip()
    formulas = result.get('factor_formulas') or {}
    if selected:
        lines = [
            f'挖到 {len(selected)} 个有效因子。',
            f'已储存到: {config_path or "(未返回存储路径)"}',
            '入选因子与公式:',
        ]
        for i, fc in enumerate(selected, start=1):
            lines.append(f'{i}. {fc}')
            formula = formulas.get(fc)
            if formula:
                lines.append(f'   Formula: {formula}')
        if msg:
            lines.append(f'补充信息: {msg}')
        return '\n'.join(lines)

    best_failed = result.get('best_failed_indicator_metrics') or {}
    lines = ['未挖到有效因子。']
    if msg:
        lines.append(f'筛选/失败信息: {msg}')
    if best_failed:
        lines.append('各指标最佳候选（含公式）:')
        for indicator, info in best_failed.items():
            info = info or {}
            sample_fc = str(info.get('factor_name', 'N/A'))
            lines.append(f'- 指标 {indicator}: {sample_fc}')
            formula = formulas.get(sample_fc)
            if formula:
                lines.append(f'  Formula: {formula}')
            extra_metrics = {k: v for k, v in info.items() if k != 'factor_name'}
            if extra_metrics:
                lines.extend(_format_indicator_metrics_lines(extra_metrics, indent='  '))
    return '\n'.join(lines)


def _format_indicator_metrics_lines(metrics: Dict[str, Any], indent: str = '') -> List[str]:
    lines: List[str] = []
    for k, v in metrics.items():
        if k == "yearly_detail" and isinstance(v, list):
            lines.append(f"{indent}{k}:")
            for yearly_item in v:
                if not isinstance(yearly_item, dict):
                    lines.append(f"{indent}  - {yearly_item}")
                    continue
                year = yearly_item.get("year", "N/A")
                val = yearly_item.get("value")
                threshold = yearly_item.get("threshold")
                passed = yearly_item.get("pass")
                lines.append(
                    f"{indent}  - {year}: value={val}, threshold={threshold}, pass={passed}"
                )
        elif isinstance(v, dict):
            lines.append(f"{indent}{k}:")
            lines.extend(_format_indicator_metrics_lines(v, indent=indent + '  '))
        elif isinstance(v, list):
            lines.append(f"{indent}{k}:")
            for item in v:
                lines.append(f"{indent}  - {item}")
        else:
            lines.append(f"{indent}{k}: {v}")
    return lines


def _build_fusion_result_overview(result: Dict[str, Any]) -> str:
    lines = ["因子融合任务完成。"]
    lines.append(f"落库状态: {'已落库' if result.get('persisted') else '未落库'}")
    if result.get("collection") and result.get("version"):
        lines.append(f"落库位置: factors.{result.get('collection')}@{result.get('version')}")
    if result.get("fusion_factor_name"):
        lines.append(f"融合因子名: {result.get('fusion_factor_name')}")
    if result.get("fusion_formula"):
        lines.append(f"融合公式: {result.get('fusion_formula')}")
    if result.get("final_metrics"):
        lines.append("最终指标:")
        lines.extend(_format_indicator_metrics_lines(result.get("final_metrics") or {}, indent='  '))
    if result.get("final_metrics_outsample"):
        lines.append("样本外指标:")
        lines.extend(_format_indicator_metrics_lines(result.get("final_metrics_outsample") or {}, indent='  '))
    return "\n".join(lines)


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
                     result_summary: Optional[dict] = None,
                     task_type: str = TASK_TYPE_GP,
                     result: Optional[dict] = None):
    """Persist task detail into MongoDB (database=task, task-type collection)."""
    try:
        record = {
            "task_id": task_id,
            "task_type": task_type,
            "status": status,
            "params": params_dict,
            "progress": tasks.get(task_id, {}).get("progress", ""),
            "gp_progress": tasks.get(task_id, {}).get("gp_progress"),
            "started_at": tasks.get(task_id, {}).get("started_at", ""),
            "finished_at": datetime.now().isoformat(),
            "logs": tasks.get(task_id, {}).get("logs", [])[-500:],
        }
        if result_summary:
            record["result_summary"] = result_summary
        if result is not None:
            record["result"] = result
        update_one_data(
            database="task",
            collection=_task_collection(task_type),
            mongo_operator={"task_id": task_id},
            data=record,
            upsert=True,
        )
    except Exception as e:
        print(f"[WARNING] Failed to save task {task_id} to DB: {e}")


def _load_task_from_db(task_id: str) -> Optional[Dict[str, Any]]:
    for task_type, collection in ((TASK_TYPE_GP, GP_TASK_COLLECTION), (TASK_TYPE_FUSION, FUSION_TASK_COLLECTION)):
        try:
            df = get_data(database="task", collection=collection, mongo_operator={"task_id": task_id})
            if isinstance(df, pd.DataFrame) and not df.empty:
                row = df.iloc[0].to_dict()
                row = {k: _sanitize_nan(v) for k, v in row.items()}
                if not row.get("task_type"):
                    row["task_type"] = task_type
                return row
        except Exception:
            continue
    return None


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

@app.get('/api/mining/indicator-options')
async def get_mining_indicator_options():
    return {
        'supported_indicator': list(GP_SUPPORTED_INDICATOR),
        'indicator_direction': dict(GP_INDICATOR_DIRECTION),
        'default_fitness_indicator_weight': dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT),
        'default_filter_indicator_dict': {
            k: {
                'mean_threshold': v[0],
                'yearly_threshold': v[1],
                'direction': v[2],
            }
            for k, v in GP_DEFAULT_FILTER_INDICATOR_DICT.items()
        },
    }

@app.post("/api/mining/start")
async def start_mining(params: GPMiningParams):
    req_version = str(params.version).strip()
    if req_version:
        duplicated_running = [
            tid for tid, info in tasks.items()
            if info.get("status") == "running"
            and str((info.get("params") or {}).get("version", "")).strip() == req_version
        ]
        if duplicated_running:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"已有相同版本正在运行: version={req_version}, "
                    f"task_id={duplicated_running[0]}"
                ),
            )

    task_id = str(uuid.uuid4())[:8]
    cancel_event = threading.Event()
    tasks[task_id] = {
        "task_type": TASK_TYPE_GP,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "params": params.dict(),
        "progress": "初始化中...",
        "gp_progress": None,
        "result": None,
        "result_overview": None,
        "error": None,
        "cancel_event": cancel_event,
        "logs": [],
    }
    # Persist initial running record immediately for cross-worker visibility.
    _save_task_to_db(task_id, params.dict(), "running", {"message": "任务已提交"}, task_type=TASK_TYPE_GP)

    async def _run():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _execute_mining, params, task_id, cancel_event)
            # If terminated by user, still save partial results but keep terminated status
            if tasks[task_id]["status"] == "terminated":
                tasks[task_id]["result"] = result
                tasks[task_id]["result_overview"] = _build_mining_result_overview(result)
                _save_task_to_db(task_id, params.dict(), "terminated", {
                    "selected_fc_name_list": result.get("selected_fc_name_list", []),
                    "version": result.get("version", ""),
                    "message": result.get("message", "Task terminated by user."),
                    "config_path": result.get("config_path"),
                    "factor_formulas": result.get("factor_formulas", {}),
                    "best_failed_indicator_metrics": result.get("best_failed_indicator_metrics"),
                    "result_overview": tasks[task_id]["result_overview"],
                }, task_type=TASK_TYPE_GP, result=result)
                return
            tasks[task_id]["result"] = result
            tasks[task_id]["result_overview"] = _build_mining_result_overview(result)
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = "完成"
            tasks[task_id]["finished_at"] = datetime.now().isoformat()
            _save_task_to_db(task_id, params.dict(), "completed", {
                "selected_fc_name_list": result.get("selected_fc_name_list", []),
                "version": result.get("version", ""),
                "message": result.get("message", ""),
                "config_path": result.get("config_path"),
                "factor_formulas": result.get("factor_formulas", {}),
                "best_failed_indicator_metrics": result.get("best_failed_indicator_metrics"),
                "result_overview": tasks[task_id]["result_overview"],
            }, task_type=TASK_TYPE_GP, result=result)
        except Exception as e:
            # If terminated by user, keep status but note the error
            if tasks[task_id]["status"] == "terminated":
                tasks[task_id]["progress"] = "已终止（用户手动终止）"
                tasks[task_id]["finished_at"] = datetime.now().isoformat()
                return
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = traceback.format_exc()
            tasks[task_id]["progress"] = f"失败: {str(e)}"
            tasks[task_id]["finished_at"] = datetime.now().isoformat()
            _save_task_to_db(task_id, params.dict(), "failed", {"error": str(e)}, task_type=TASK_TYPE_GP)

    asyncio.create_task(_run())
    return {"task_id": task_id, "status": "running"}


def _execute_mining(params: GPMiningParams, task_id: str, cancel_event: threading.Event) -> Dict[str, Any]:
    tasks[task_id]["progress"] = "创建 GeneticFactorGenerator..."

    fitness_indicator_dict = _normalize_fitness_indicator_dict(params)
    filter_indicator_dict = _normalize_filter_indicator_dict(params)

    # Attach log handler to capture GP progress
    from utils.logging import log as lionet_logger
    handler = _GPProgressHandler(task_id=task_id, owner_thread_id=threading.get_ident())
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
            fitness_indicator_dict=fitness_indicator_dict,
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
            consistency_penalty_enabled=params.consistency_penalty_enabled,
            consistency_penalty_coef=params.consistency_penalty_coef,
        )

        # Attach cancel_event so GP evolution can be interrupted
        fg.cancel_event = cancel_event


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
            "best_failed_indicator_metrics": result.get("best_failed_indicator_metrics"),
            "nav_data": nav_data,
            "factor_formulas": formulas,
            "version": params.version,
            "collection": "genetic_programming",
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
        "task_type": task.get("task_type", TASK_TYPE_GP),
        "status": task["status"],
        "progress": task["progress"],
        "started_at": task["started_at"],
        "finished_at": task.get("finished_at", ""),
        "error": task.get("error"),
        "gp_progress": task.get("gp_progress"),
        "params": task.get("params"),
        "logs": task.get("logs", [])[-200:],
    }
    if task["status"] in ("completed", "terminated") and task["result"]:
        resp["result"] = task["result"]
        resp["result_overview"] = task.get("result_overview")
    return resp


@app.post("/api/mining/terminate/{task_id}")
async def terminate_mining(task_id: str):
    # ── Case 1: task lives in memory (current process) ────────────────
    if task_id in tasks:
        task = tasks[task_id]
        if task["status"] != "running":
            raise HTTPException(status_code=400, detail=f"Task is not running (status={task['status']})")
        cancel_event = task.get("cancel_event")
        if cancel_event is None:
            raise HTTPException(status_code=400, detail="Task does not support cancellation")
        cancel_event.set()
        task["status"] = "terminated"
        task["progress"] = "已终止（用户手动终止）"
        task["finished_at"] = datetime.now().isoformat()
        _save_task_to_db(
            task_id,
            task.get("params", {}),
            "terminated",
            {"message": "用户手动终止"},
            task_type=task.get("task_type", TASK_TYPE_GP),
            result=task.get("result"),
        )
        return {"task_id": task_id, "status": "terminated", "message": "任务已终止"}

    # ── Case 2: task only exists in DB (e.g. stale after server restart)
    try:
        df = get_data(database="task", collection=GP_TASK_COLLECTION, mongo_operator={"task_id": task_id})
        if isinstance(df, pd.DataFrame) and not df.empty:
            current_status = str(df.iloc[0].get("status", "") or "")
            if current_status == "running":
                update_one_data(
                    database="task",
                    collection=GP_TASK_COLLECTION,
                    mongo_operator={"task_id": task_id},
                    data={
                        "status": "terminated",
                        "progress": "已终止（用户手动终止）",
                        "finished_at": datetime.now().isoformat(),
                    },
                    upsert=False,
                )
                return {"task_id": task_id, "status": "terminated", "message": "任务已终止（该任务实际已不在运行）"}
            else:
                raise HTTPException(status_code=400, detail=f"Task is not running (status={current_status})")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[WARNING] terminate_mining DB fallback failed for task {task_id}: {e}")
        traceback.print_exc()

    raise HTTPException(status_code=404, detail="Task not found")


# ── Backtest ──────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def run_backtest(params: BacktestParams):
    try:
        bt_kwargs: Dict[str, Any] = dict(
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


def _execute_fusion(params: FusionParams, task_id: Optional[str] = None) -> Dict[str, Any]:
    handler = None
    lionet_logger = None
    if task_id and task_id in tasks:
        from utils.logging import log as lionet_logger
        handler = _FusionLogHandler(task_id=task_id, owner_thread_id=threading.get_ident())
        lionet_logger.addHandler(handler)

    try:
        raw_factor_dict = None
        if params.candidate_versions:
            records = get_factor_formula_records(
                collections=params.collection,
                versions=params.candidate_versions,
                database='factors',
            )
            if records.empty:
                raise ValueError(
                    f'No factor formulas found for collections={params.collection}, '
                    f'versions={params.candidate_versions}.'
                )
            raw_factor_dict = {}
            for version, df_v in records.groupby('version', sort=False):
                names = [str(x) for x in df_v['factor_name'].dropna().astype(str).tolist()]
                if names:
                    raw_factor_dict[str(version)] = list(dict.fromkeys(names))

        fusioner = FactorFusioner(
            fusion_method=params.fusion_method,
            raw_factor_dict=raw_factor_dict,
            collection=params.collection,
            instrument_type=params.instrument_type,
            instrument_id_list=params.instrument_id_list,
            fc_freq=params.fc_freq,
            start_time=params.start_time,
            end_time=params.end_time,
            portfolio_adjust_method=params.portfolio_adjust_method,
            interest_method=params.interest_method,
            risk_free_rate=params.risk_free_rate,
            apply_weighted_price=params.apply_weighted_price,
            check_leakage_count=params.check_leakage_count,
            check_relative=params.check_relative,
            relative_threshold=params.relative_threshold,
            relative_check_version_list=params.relative_check_version_list,
            max_fusion_count=params.max_fusion_count,
            fusion_metrics=params.fusion_metrics,
            version=params.version,
            n_jobs=params.n_jobs,
            base_col_list=params.base_col_list,
            consider_outsample=params.consider_outsample,
            outsample_start_day=params.outsample_start_day,
            outsample_end_day=params.outsample_end_day,
        )
        result = fusioner.fuse()
        bt = result.get('bt')
        nav_data = _extract_nav_data(bt) if bt else {'nav_curves': {}, 'performance_summary': []}

        fusion_formula = result.get('fusion_formula')
        fusion_curve_name = result.get('fusion_factor_name') or (fusion_formula if fusion_formula else 'fusion_factor')
        has_oos = bool(params.outsample_start_day and params.outsample_end_day)
        nav_split_date = params.outsample_start_day if has_oos else None
        curve_end_time = params.outsample_end_day if has_oos else params.end_time

        if fusion_formula and has_oos:
            # Build a continuous in-sample + out-of-sample curve in one run.
            bt_full = BackTester(
                collection='genetic_programming',
                instrument_type=params.instrument_type,
                instrument_id_list=params.instrument_id_list,
                fc_freq=params.fc_freq,
                start_time=params.start_time,
                end_time=curve_end_time,
                portfolio_adjust_method=params.portfolio_adjust_method,
                interest_method=params.interest_method,
                risk_free_rate=params.risk_free_rate,
                calculate_baseline=True,
                apply_weighted_price=params.apply_weighted_price,
                n_jobs=params.n_jobs,
                formula=fusion_formula,
                version=params.version or '__formula__',
                fc_name_list=['formula_factor'],
            )
            bt_full.backtest()
            nav_data = _extract_nav_data(bt_full)
            if 'formula_factor' in nav_data.get('nav_curves', {}):
                nav_data['nav_curves'][fusion_curve_name] = nav_data['nav_curves'].pop('formula_factor')
                for rec in nav_data.get('performance_summary', []):
                    if rec.get('Factor Name') == 'formula_factor':
                        rec['Factor Name'] = fusion_curve_name

        # Add NAV curves of raw factors selected for fusion so users can compare before/after fusion.
        selected_details = result.get('selected_factors_detail') or []
        grouped_raw: Dict[tuple, List[str]] = {}
        for item in selected_details:
            if not isinstance(item, dict):
                continue
            raw_collection = str(item.get('collection') or '')
            raw_version = str(item.get('version') or '')
            raw_factor_name = str(item.get('factor_name') or '')
            if not (raw_collection and raw_version and raw_factor_name):
                continue
            grouped_raw.setdefault((raw_collection, raw_version), [])
            grouped_raw[(raw_collection, raw_version)].append(raw_factor_name)

        for (raw_collection, raw_version), raw_names in grouped_raw.items():
            raw_names = list(dict.fromkeys(raw_names))
            if not raw_names:
                continue
            try:
                bt_raw = BackTester(
                    collection=raw_collection,
                    instrument_type=params.instrument_type,
                    instrument_id_list=params.instrument_id_list,
                    fc_freq=params.fc_freq,
                    start_time=params.start_time,
                    end_time=curve_end_time,
                    portfolio_adjust_method=params.portfolio_adjust_method,
                    interest_method=params.interest_method,
                    risk_free_rate=params.risk_free_rate,
                    calculate_baseline=True,
                    apply_weighted_price=params.apply_weighted_price,
                    n_jobs=params.n_jobs,
                    version=raw_version,
                    fc_name_list=raw_names,
                )
                bt_raw.backtest()
                raw_nav = _extract_nav_data(bt_raw)
                for raw_name, raw_curve in raw_nav.get('nav_curves', {}).items():
                    base_key = f"{raw_version} | {raw_name}"
                    final_key = base_key
                    dup_idx = 2
                    while final_key in nav_data['nav_curves']:
                        final_key = f"{base_key} #{dup_idx}"
                        dup_idx += 1
                    nav_data['nav_curves'][final_key] = raw_curve
            except Exception as e:
                logging.warning(
                    'Failed to append raw factor NAV for fusion task '
                    f'(collection={raw_collection}, version={raw_version}): {e}'
                )

        return {
            'status': 'ok',
            'version': params.version,
            'collection': result.get('fusion_collection', 'factor_fusion'),
            'persisted': bool(result.get('persisted', False)),
            'fusion_factor_name': result.get('fusion_factor_name'),
            'fusion_curve_name': fusion_curve_name,
            'fusion_formula': fusion_formula,
            'fusion_info': result.get('fusion_info'),
            'selected_factor_keys': result.get('selected_factor_keys', []),
            'selected_factors_detail': result.get('selected_factors_detail', []),
            'final_metrics': result.get('final_metrics', {}),
            'final_metrics_outsample': result.get('final_metrics_outsample'),
            'leakage_check': result.get('leakage_check'),
            'relative_check': result.get('relative_check'),
            'consider_outsample': result.get('consider_outsample', False),
            'outsample_start_day': result.get('outsample_start_day'),
            'outsample_end_day': result.get('outsample_end_day'),
            'nav_split_date': nav_split_date,
            'nav_data': nav_data,
        }
    finally:
        if handler is not None and lionet_logger is not None:
            lionet_logger.removeHandler(handler)


@app.post('/api/fusion/start')
async def start_fusion(params: FusionParams):
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "task_type": TASK_TYPE_FUSION,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "params": params.dict(),
        "progress": "初始化融合任务...",
        "gp_progress": None,
        "result": None,
        "result_overview": None,
        "error": None,
        "logs": [],
    }
    _save_task_to_db(task_id, params.dict(), "running", {"message": "融合任务已提交"}, task_type=TASK_TYPE_FUSION)

    async def _run():
        try:
            tasks[task_id]["progress"] = "正在执行融合计算..."
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _execute_fusion, params, task_id)
            tasks[task_id]["result"] = result
            tasks[task_id]["result_overview"] = _build_fusion_result_overview(result)
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = "完成"
            tasks[task_id]["finished_at"] = datetime.now().isoformat()
            _save_task_to_db(
                task_id,
                params.dict(),
                "completed",
                {
                    "version": result.get("version", ""),
                    "collection": result.get("collection"),
                    "persisted": result.get("persisted"),
                    "fusion_factor_name": result.get("fusion_factor_name"),
                    "fusion_formula": result.get("fusion_formula"),
                    "final_metrics": result.get("final_metrics"),
                    "selected_factors_detail": result.get("selected_factors_detail", []),
                    "result_overview": tasks[task_id]["result_overview"],
                },
                task_type=TASK_TYPE_FUSION,
                result=result,
            )
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = traceback.format_exc()
            tasks[task_id]["progress"] = f"失败: {str(e)}"
            tasks[task_id]["finished_at"] = datetime.now().isoformat()
            _save_task_to_db(
                task_id,
                params.dict(),
                "failed",
                {"error": str(e)},
                task_type=TASK_TYPE_FUSION,
            )
        finally:
            pass

    asyncio.create_task(_run())
    return {"task_id": task_id, "task_type": TASK_TYPE_FUSION, "status": "running"}


@app.get('/api/fusion/status/{task_id}')
async def fusion_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("task_type") != TASK_TYPE_FUSION:
        raise HTTPException(status_code=400, detail="Not a fusion task")
    resp = {
        "task_id": task_id,
        "task_type": task.get("task_type"),
        "status": task["status"],
        "progress": task["progress"],
        "started_at": task["started_at"],
        "finished_at": task.get("finished_at", ""),
        "error": task.get("error"),
        "params": task.get("params"),
        "logs": task.get("logs", [])[-200:],
    }
    if task["status"] in ("completed", "terminated") and task.get("result"):
        resp["result"] = task.get("result")
        resp["result_overview"] = task.get("result_overview")
    return resp


@app.post('/api/fusion/run')
async def run_fusion(params: FusionParams):
    """Backward-compatible sync fusion endpoint."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _execute_fusion, params, None)
        return result
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
async def list_tasks(start_date: Optional[str] = None, end_date: Optional[str] = None):
    def _in_date_range(ts: str) -> bool:
        if not ts:
            return False
        d = str(ts)[:10]
        if start_date and d < start_date:
            return False
        if end_date and d > end_date:
            return False
        return True

    task_list = []
    for tid, info in tasks.items():
        finished_at = info.get("finished_at", "")
        candidate_time = finished_at or info.get("started_at", "")
        if (start_date or end_date) and not _in_date_range(candidate_time):
            continue
        task_list.append({
            "task_id": tid,
            "task_type": _task_type_label(info.get("task_type")),
            "status": info["status"],
            "progress": info["progress"],
            "started_at": info["started_at"],
            "finished_at": finished_at,
            "version": info.get("params", {}).get("version", ""),
            "gp_progress": info.get("gp_progress"),
        })
    # Also fetch from DB for historical tasks
    for task_type, collection in ((TASK_TYPE_GP, GP_TASK_COLLECTION), (TASK_TYPE_FUSION, FUSION_TASK_COLLECTION)):
        try:
            db_tasks = get_data(database="task", collection=collection, mongo_operator={})
            if isinstance(db_tasks, pd.DataFrame) and not db_tasks.empty:
                for _, row in db_tasks.iterrows():
                    tid = _sanitize_nan(row.get("task_id"), "")
                    if tid and tid not in tasks:
                        db_status = _sanitize_nan(row.get("status"), "unknown")
                        db_progress = _sanitize_nan(row.get("progress"), "")
                        if db_status == "running":
                            db_status = "interrupted"
                            db_progress = "服务重启，任务已中断"
                            try:
                                update_one_data(
                                    database="task",
                                    collection=collection,
                                    mongo_operator={"task_id": tid},
                                    data={
                                        "status": "interrupted",
                                        "progress": db_progress,
                                        "finished_at": datetime.now().isoformat(),
                                    },
                                    upsert=False,
                                )
                            except Exception:
                                pass
                        params_val = _sanitize_nan(row.get("params"))
                        started_at = _sanitize_nan(row.get("started_at"), "")
                        finished_at = _sanitize_nan(row.get("finished_at"), "")
                        candidate_time = finished_at or started_at
                        if (start_date or end_date) and not _in_date_range(candidate_time):
                            continue
                        task_list.append({
                            "task_id": tid,
                            "task_type": _task_type_label(_sanitize_nan(row.get("task_type"), task_type)),
                            "status": db_status,
                            "progress": db_progress,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "version": params_val.get("version", "") if isinstance(params_val, dict) else "",
                            "gp_progress": _sanitize_nan(row.get("gp_progress")),
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
            "task_id": task_id,
            "task_type": _task_type_label(task.get("task_type", TASK_TYPE_GP)),
            "status": task["status"],
            "progress": task["progress"],
            "started_at": task["started_at"],
            "finished_at": task.get("finished_at", ""),
            "error": task.get("error"),
            "gp_progress": task.get("gp_progress"), "params": task.get("params"),
            "result": task.get("result") if task["status"] in ("completed", "terminated") else None,
            "result_overview": task.get("result_overview"),
            "logs": task.get("logs", [])[-200:],
        }
    # Fallback to DB
    try:
        row = _load_task_from_db(task_id)
        if row:
            # Sanitize NaN values from pandas to prevent JSON serialization errors
            # Self-heal: if DB says "running" but task is not in memory, it's dead.
            db_status = row.get("status") or "unknown"
            db_task_type = row.get("task_type") or TASK_TYPE_GP
            db_collection = _task_collection(db_task_type)
            if db_status == "running":
                db_status = "interrupted"
                row["progress"] = "服务重启，任务已中断"
                try:
                    update_one_data(
                        database="task",
                        collection=db_collection,
                        mongo_operator={"task_id": task_id},
                        data={
                            "status": "interrupted",
                            "progress": row["progress"],
                            "finished_at": datetime.now().isoformat(),
                        },
                        upsert=False,
                    )
                except Exception:
                    pass
            result_summary = row.get("result_summary")
            logs_val = row.get("logs")
            return {
                "task_id": task_id,
                "task_type": _task_type_label(db_task_type),
                "status": db_status,
                "progress": row.get("progress") or "",
                "started_at": row.get("started_at") or "",
                "finished_at": row.get("finished_at") or "",
                "params": row.get("params"),
                "result": row.get("result"),
                "result_summary": result_summary,
                "result_overview": result_summary.get("result_overview") if isinstance(result_summary, dict) else None,
                "logs": logs_val if isinstance(logs_val, list) else [],
                "error": None,
                "gp_progress": row.get("gp_progress"),
            }
    except Exception as e:
        print(f"[WARNING] get_task_detail DB fallback failed for task {task_id}: {e}")
        traceback.print_exc()
    raise HTTPException(status_code=404, detail="Task not found")


# ══════════════════════════════════════════════════════════════════════
#  Market Data Management APIs
# ══════════════════════════════════════════════════════════════════════

# In-memory store for market-data tasks (update-info / update-price)
market_data_tasks: Dict[str, Dict[str, Any]] = {}

# ── Scheduled daily update ────────────────────────────────────────────
_daily_update_enabled = True
_daily_update_lock = threading.Lock()
_daily_update_thread: Optional[threading.Thread] = None


def _run_daily_price_update():
    """Background: update all instrument prices with default params."""
    task_id = f"scheduled_{uuid.uuid4().hex[:8]}"
    market_data_tasks[task_id] = {
        "type": "update-price", "status": "running",
        "started_at": datetime.now().isoformat(), "logs": [],
        "params": {"instrument_id": None, "scheduled": True},
    }
    from utils.logging import log as lionet_logger
    handler = _MarketDataLogHandler(task_id)
    lionet_logger.addHandler(handler)
    try:
        update_futures_continuous_contract_price(
            instrument_id=None,
            start_date=None,
            end_date=None,
            load_prev_weighted_factor=True,
            wait_time=2.0,
        )
        market_data_tasks[task_id]["status"] = "completed"
    except Exception as e:
        market_data_tasks[task_id]["status"] = "failed"
        market_data_tasks[task_id]["error"] = str(e)
    finally:
        market_data_tasks[task_id]["finished_at"] = datetime.now().isoformat()
        lionet_logger.removeHandler(handler)


def _daily_scheduler_loop():
    """Simple scheduler: sleep until next day's target time then run update."""
    import time as _time
    target_hour, target_minute = 18, 0  # 18:00 daily
    while _daily_update_enabled:
        now = datetime.now()
        target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now >= target:
            from datetime import timedelta
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        # Sleep in small increments so we can respond to shutdown
        while wait_seconds > 0 and _daily_update_enabled:
            _time.sleep(min(wait_seconds, 30))
            wait_seconds -= 30
        if _daily_update_enabled:
            try:
                _run_daily_price_update()
            except Exception:
                pass


class _MarketDataLogHandler(logging.Handler):
    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def emit(self, record):
        if self.task_id not in market_data_tasks:
            return
        msg = record.getMessage()
        log_line = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {record.levelname} - {msg}'
        logs = market_data_tasks[self.task_id].setdefault("logs", [])
        logs.append(log_line)
        if len(logs) > 2000:
            del logs[:-2000]


@app.on_event("startup")
async def _start_daily_scheduler():
    global _daily_update_thread
    _daily_update_thread = threading.Thread(target=_daily_scheduler_loop, daemon=True)
    _daily_update_thread.start()


class UpdatePriceParams(BaseModel):
    instrument_id: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    load_prev_weighted_factor: bool = True
    wait_time: float = 2.0


class DeleteDataParams(BaseModel):
    instrument_id_list: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.get("/api/market-data/instrument-ids")
async def get_instrument_ids():
    """Get all available instrument_ids from DB."""
    try:
        df = get_futures_continuous_contract_info(instrument_id=None, from_database=True)
        if df is None or df.empty:
            return {"instrument_ids": []}
        ids = sorted(df["instrument_id"].dropna().unique().tolist())
        return {"instrument_ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market-data/update-info")
async def api_update_info():
    """Update continuous contract info."""
    task_id = f"info_{uuid.uuid4().hex[:8]}"
    market_data_tasks[task_id] = {
        "type": "update-info", "status": "running",
        "started_at": datetime.now().isoformat(), "logs": [],
    }

    def _run():
        from utils.logging import log as lionet_logger
        handler = _MarketDataLogHandler(task_id)
        lionet_logger.addHandler(handler)
        try:
            lionet_logger.info('合约信息更新任务启动...')
            update_futures_continuous_contract_info()
            lionet_logger.info('合约信息更新任务完成')
            market_data_tasks[task_id]["status"] = "completed"
        except Exception as e:
            market_data_tasks[task_id]["status"] = "failed"
            market_data_tasks[task_id]["error"] = str(e)
            lionet_logger.error(f'合约信息更新失败: {e}')
        finally:
            market_data_tasks[task_id]["finished_at"] = datetime.now().isoformat()
            lionet_logger.removeHandler(handler)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "message": "合约信息更新任务已启动"}


@app.post("/api/market-data/update-price")
async def api_update_price(params: UpdatePriceParams):
    """Update continuous contract price data (async task)."""
    task_id = f"price_{uuid.uuid4().hex[:8]}"
    market_data_tasks[task_id] = {
        "type": "update-price", "status": "running",
        "started_at": datetime.now().isoformat(), "logs": [],
        "params": params.dict(),
    }

    def _run():
        from utils.logging import log as lionet_logger
        handler = _MarketDataLogHandler(task_id)
        lionet_logger.addHandler(handler)
        try:
            lionet_logger.info(
                f'价格数据更新任务启动: instrument_id={params.instrument_id}, '
                f'start_date={params.start_date}, end_date={params.end_date}, '
                f'load_prev_weighted_factor={params.load_prev_weighted_factor}, '
                f'wait_time={params.wait_time}'
            )
            update_futures_continuous_contract_price(
                instrument_id=params.instrument_id,
                start_date=params.start_date,
                end_date=params.end_date,
                load_prev_weighted_factor=params.load_prev_weighted_factor,
                wait_time=params.wait_time,
            )
            lionet_logger.info('价格数据更新任务完成')
            market_data_tasks[task_id]["status"] = "completed"
        except Exception as e:
            market_data_tasks[task_id]["status"] = "failed"
            market_data_tasks[task_id]["error"] = str(e)
            lionet_logger.error(f'价格数据更新任务失败: {e}')
            traceback.print_exc()
        finally:
            market_data_tasks[task_id]["finished_at"] = datetime.now().isoformat()
            lionet_logger.removeHandler(handler)

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "message": "价格数据更新任务已启动"}


@app.get("/api/market-data/task-status/{task_id}")
async def api_market_data_task_status(task_id: str):
    """Poll market-data task status and logs."""
    if task_id not in market_data_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = market_data_tasks[task_id]
    return {
        "task_id": task_id,
        "type": t.get("type"),
        "status": t.get("status"),
        "started_at": t.get("started_at"),
        "finished_at": t.get("finished_at"),
        "error": t.get("error"),
        "logs": t.get("logs", [])[-500:],
    }


@app.get("/api/market-data/overview")
async def api_market_data_overview():
    """Return data overview: per instrument_id stats (queried per-instrument to avoid OOM)."""
    import asyncio

    def _compute_overview():
        try:
            # Get all known instrument_ids from contract info
            from data.futures import get_futures_continuous_contract_info as _get_info
            df_info = _get_info(instrument_id=None, from_database=True)
            if df_info is None or df_info.empty:
                return {"overview": []}
            all_ids = sorted(df_info["instrument_id"].dropna().unique().tolist())

            result = []
            for ins_id in all_ids:
                try:
                    df = get_futures_continuous_contract_price(
                        instrument_id=ins_id,
                        start_date="20000101",
                        end_date="20991231",
                        from_database=True,
                    )
                    if df is None or df.empty:
                        result.append({
                            "instrument_id": str(ins_id),
                            "start_date": "-", "end_date": "-",
                            "total_rows": 0, "expected_bdays": 0,
                            "missing_dates_count": 0, "missing_fields": {},
                            "status": "无数据",
                        })
                        continue

                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                    df = df.dropna(subset=["time"]).sort_values("time")
                    if df.empty:
                        continue

                    min_date = df["time"].min()
                    max_date = df["time"].max()
                    total_rows = len(df)
                    bdays = pd.bdate_range(min_date, max_date)
                    expected_count = len(bdays)

                    price_cols = ["open", "high", "low", "close", "volume", "position"]
                    missing_fields = {}
                    for c in price_cols:
                        if c in df.columns:
                            null_count = int(df[c].isna().sum())
                            if null_count > 0:
                                missing_fields[c] = null_count

                    actual_dates = set(df["time"].dt.date)
                    bday_dates = set(d.date() for d in bdays)
                    missing_dates_count = len(bday_dates - actual_dates)

                    result.append({
                        "instrument_id": str(ins_id),
                        "start_date": min_date.strftime("%Y-%m-%d"),
                        "end_date": max_date.strftime("%Y-%m-%d"),
                        "total_rows": total_rows,
                        "expected_bdays": expected_count,
                        "missing_dates_count": missing_dates_count,
                        "missing_fields": missing_fields,
                        "status": "完整" if missing_dates_count <= 5 and not missing_fields else "有缺失",
                    })
                except Exception:
                    result.append({
                        "instrument_id": str(ins_id),
                        "start_date": "-", "end_date": "-",
                        "total_rows": 0, "expected_bdays": 0,
                        "missing_dates_count": 0, "missing_fields": {},
                        "status": "查询异常",
                    })
            return {"overview": result}
        except Exception as e:
            return {"error": str(e)}

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, _compute_overview)
    if "error" in resp:
        raise HTTPException(status_code=500, detail=resp["error"])
    return resp


@app.get("/api/market-data/price")
async def api_get_price(
    instrument_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get price data for one instrument, for table display and K-line chart."""
    import asyncio

    def _query():
        df = get_futures_continuous_contract_price(
            instrument_id=instrument_id,
            start_date=start_date or "20000101",
            end_date=end_date or None,
            from_database=True,
        )
        if df is None or df.empty:
            return {"rows": [], "columns": []}
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time").reset_index(drop=True)
        df["time"] = df["time"].dt.strftime("%Y-%m-%d")
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        for c in ["open", "high", "low", "close", "settle", "volume", "position",
                   "weighted_factor", "cur_weighted_factor"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        columns = df.columns.tolist()
        rows = df.where(df.notna(), None).to_dict(orient="records")
        return {"rows": rows, "columns": columns}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market-data/delete")
async def api_delete_price(params: DeleteDataParams):
    """Delete price data for given instrument_ids and optional date range."""
    if not params.instrument_id_list:
        raise HTTPException(status_code=400, detail="instrument_id_list is required")
    try:
        mongo_op: Dict[str, Any] = {"instrument_id": {"$in": params.instrument_id_list}}
        date_conds = []
        if params.start_date:
            date_conds.append({"time": {"$gte": pd.Timestamp(params.start_date)}})
        if params.end_date:
            date_conds.append({"time": {"$lte": pd.Timestamp(params.end_date)}})
        if date_conds:
            mongo_op = {"$and": [mongo_op] + date_conds}
        delete_data(
            database="futures",
            collection="continuous_contract_price_daily",
            mongo_operator=mongo_op,
        )
        return {"message": f"已删除 {params.instrument_id_list} 的数据", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market-data/scheduled-status")
async def api_scheduled_status():
    """Return whether daily scheduled update is enabled."""
    return {"enabled": _daily_update_enabled}


@app.post("/api/market-data/toggle-schedule")
async def api_toggle_schedule(enabled: bool = True):
    """Toggle daily scheduled update."""
    global _daily_update_enabled, _daily_update_thread
    _daily_update_enabled = enabled
    if enabled and (_daily_update_thread is None or not _daily_update_thread.is_alive()):
        _daily_update_thread = threading.Thread(target=_daily_scheduler_loop, daemon=True)
        _daily_update_thread.start()
    return {"enabled": _daily_update_enabled, "message": "已开启定时更新" if enabled else "已关闭定时更新"}


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

