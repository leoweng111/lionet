"""Standalone task runner subprocess.

Runs GP/LLM/fusion/market-data tasks in a **separate OS process** so they
survive backend restarts.  Status/progress/logs are written to MongoDB
(database=task) and the backend reads them back purely for monitoring.

Why separate process?
  Previously tasks ran as daemon threads inside the FastAPI process.  When the
  backend restarted, those threads died and tasks were marked 'interrupted'.
  A subprocess has an independent lifetime: even if the backend restarts,
  the task keeps running to completion.

Usage (invoked by main.py via ``subprocess.Popen``):
    python -m web.backend.task_runner <task_type> <task_id> <params_json>

Exit codes:
    0  -> task completed
    1  -> task failed
    2  -> task terminated / cancelled
"""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing web.backend.main does NOT start uvicorn (guarded by __main__).
# It lets us reuse _execute_* functions and the global tasks dicts.
from web.backend import main as main_mod  # noqa: E402

_HEARTBEAT_INTERVAL = 3.0  # seconds


# ── Task dispatch ──────────────────────────────────────────────────

def _dispatch_gp(params: Dict[str, Any], task_id: str,
                 cancel_event: threading.Event) -> Dict[str, Any]:
    p = main_mod.GPMiningParams(**params)
    return main_mod._execute_mining(p, task_id, cancel_event)


def _dispatch_llm(params: Dict[str, Any], task_id: str,
                  cancel_event: threading.Event) -> Dict[str, Any]:
    p = main_mod.LLMMiningParams(**params)
    return main_mod._execute_llm_mining(p, task_id, cancel_event)


def _dispatch_fusion(params: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    p = main_mod.FusionParams(**params)
    return main_mod._execute_fusion(p, task_id)


def _dispatch_market_data(task_type: str, params: Dict[str, Any],
                          task_id: str,
                          cancel_event: threading.Event) -> Dict[str, Any]:
    """Execute one manual market-data task type.

    Mirrors the inline ``_run`` bodies of the /api/market-data endpoints but
    runs in the subprocess.  Status lives in ``main_mod.market_data_tasks``
    and is periodically flushed to MongoDB by the heartbeat thread.
    """
    from utils.logging import log as lionet_logger
    from data.futures import (
        update_futures_continuous_contract_info,
        update_futures_continuous_contract_price,
        update_futures_continuous_contract_price_from_minute,
        update_futures_continuous_contract_price_1min,
    )
    handler = main_mod._MarketDataLogHandler(task_id)
    lionet_logger.addHandler(handler)
    try:
        if task_type == 'update-info':
            p = main_mod.UpdateInfoParams(**params)
            lionet_logger.info(f'合约信息更新任务启动: method={p.method}')
            update_futures_continuous_contract_info(method=p.method)
            lionet_logger.info('合约信息更新任务完成')

        elif task_type == 'update-price':
            p = main_mod.UpdatePriceParams(**params)
            effective_start = p.start_date or main_mod.RESEARCH_START_DATE
            lionet_logger.info(
                f'价格数据更新任务启动: source={p.source}, instrument_id={p.instrument_id}, '
                f'start_date={effective_start}, end_date={p.end_date}'
            )
            if p.source in ('joinquant', 'tqsdk_edb'):
                update_futures_continuous_contract_price_from_minute(
                    instrument_id=p.instrument_id,
                    start_date=p.start_date,
                    end_date=p.end_date,
                    method=p.method,
                    cancel_event=cancel_event,
                    source=p.source,
                )
            else:
                update_futures_continuous_contract_price(
                    instrument_id=p.instrument_id,
                    start_date=effective_start,
                    end_date=p.end_date,
                    load_prev_weighted_factor=p.load_prev_weighted_factor,
                    wait_time=p.wait_time,
                    method=p.method,
                    only_update_new=p.only_update_new,
                    cancel_event=cancel_event,
                )
            lionet_logger.info('价格数据更新任务完成')

        elif task_type == 'update-price-1min':
            p = main_mod.UpdatePrice1minParams(**params)
            lionet_logger.info(
                f'分钟价格更新任务启动: source={p.source}, instrument_id={p.instrument_id}, '
                f'start_date={p.start_date}, end_date={p.end_date}'
            )
            update_futures_continuous_contract_price_1min(
                instrument_id=p.instrument_id,
                start_date=p.start_date,
                end_date=p.end_date,
                wait_time=p.wait_time,
                method=p.method,
                cancel_event=cancel_event,
                source=p.source,
                load_prev_weighted_factor=p.load_prev_weighted_factor,
            )
            lionet_logger.info('分钟价格更新任务完成')

        elif task_type == 'update-price-1min-csv':
            from data.futures import import_1min_csv_to_db
            p = main_mod.UpdatePrice1minCsvParams(**params)
            lionet_logger.info(
                f'分钟 CSV 导入任务启动: instrument_id={p.instrument_id}, '
                f'source={p.source}, main_csv={p.main_csv}, fix_csv={p.fix_csv}'
            )
            result = import_1min_csv_to_db(
                main_csv=p.main_csv,
                fix_csv=p.fix_csv,
                instrument_id=p.instrument_id,
                source=p.source,
                method=p.method,
                load_prev_weighted_factor=p.load_prev_weighted_factor,
                cancel_event=cancel_event,
            )
            lionet_logger.info(f'分钟 CSV 导入完成: {result.get("message", "")}')
            main_mod.market_data_tasks[task_id]["result"] = result

        else:
            raise ValueError(f'Unsupported market-data task type: {task_type}')

        main_mod.market_data_tasks[task_id]["status"] = "completed"
        main_mod.market_data_tasks[task_id]["progress"] = "完成"
        return {"task_id": task_id, "status": "completed"}

    except main_mod.UpdateCancelledError as e:
        main_mod.market_data_tasks[task_id]["status"] = "terminated"
        main_mod.market_data_tasks[task_id]["error"] = str(e)
        lionet_logger.warning(f'任务已终止: {e}')
        return {"task_id": task_id, "status": "terminated"}

    finally:
        lionet_logger.removeHandler(handler)


# ── Heartbeat ─────────────────────────────────────────────────────

def _heartbeat_gp(task_id: str, params: Dict[str, Any], task_type: str,
                  stop_flag: threading.Event,
                  cancel_event: threading.Event) -> None:
    """Periodically flush in-memory task snapshot to MongoDB.

    **Read-then-write ordering matters**: the backend's terminate endpoint marks
    the task 'terminated' in DB.  If we write the in-memory 'running' status
    first, we'd overwrite that flag and never observe the termination.  So we
    always read the DB status first, and if it is 'terminated' we set the
    cancel_event and stop WITHOUT writing anything.
    """
    while not stop_flag.is_set():
        try:
            # 1) Read DB status FIRST (authoritative for termination).
            row = main_mod._load_task_from_db(task_id)
            db_status = str(row.get("status")) if row else "running"
            if db_status == "terminated":
                cancel_event.set()
                if task_id in main_mod.tasks:
                    main_mod.tasks[task_id]["status"] = "terminated"
                stop_flag.set()
                break

            # 2) Write the snapshot using the DB's own status (never downgrade
            #    a terminated status back to running).
            current_status = main_mod.tasks.get(task_id, {}).get("status", "running")
            if current_status == "terminated":
                stop_flag.set()
                break
            main_mod._save_task_to_db(task_id, params, db_status, task_type=task_type)
        except Exception:
            pass
        stop_flag.wait(_HEARTBEAT_INTERVAL)


def _heartbeat_market(task_id: str, stop_flag: threading.Event,
                      cancel_event: threading.Event) -> None:
    while not stop_flag.is_set():
        try:
            # Read-then-write: check for termination BEFORE writing snapshot.
            row = main_mod._load_market_data_task_from_db(task_id)
            if row and str(row.get("status")) == "terminated":
                cancel_event.set()
                if task_id in main_mod.market_data_tasks:
                    main_mod.market_data_tasks[task_id]["status"] = "terminated"
                stop_flag.set()
                break
            current_status = main_mod.market_data_tasks.get(task_id, {}).get("status", "running")
            if current_status == "terminated":
                stop_flag.set()
                break
            main_mod._save_market_data_task_to_db(task_id)
        except Exception:
            pass
        stop_flag.wait(_HEARTBEAT_INTERVAL)


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 4:
        print('Usage: python -m web.backend.task_runner <task_type> <task_id> <params_json>')
        return 1

    task_type = sys.argv[1]
    task_id = sys.argv[2]
    params: Dict[str, Any] = json.loads(sys.argv[3])

    cancel_event = threading.Event()
    pid = os.getpid()

    # Register task in the module globals so _execute_* / _save_* work.
    is_market = (task_type == main_mod.TASK_TYPE_MARKET_DATA)
    if is_market:
        md_type = str(params.get('type') or task_type)
        main_mod.market_data_tasks[task_id] = {
            "type": md_type,
            "status": "running",
            "started_at": main_mod.datetime.now().isoformat(),
            "logs": [],
            "params": params,
            "cancel_event": cancel_event,
            "pid": pid,
        }
        main_mod._save_market_data_task_to_db(task_id)
    else:
        main_mod.tasks[task_id] = {
            "task_type": task_type,
            "status": "running",
            "started_at": main_mod.datetime.now().isoformat(),
            "params": params,
            "progress": "初始化中...",
            "gp_progress": None,
            "result": None,
            "result_overview": None,
            "error": None,
            "logs": [],
            "cancel_event": cancel_event,
            "pid": pid,
        }
        main_mod._save_task_to_db(task_id, params, "running",
                                  {"message": "任务已提交"},
                                  task_type=task_type)

    # Heartbeat thread keeps MongoDB in sync with in-memory progress/logs.
    stop_heartbeat = threading.Event()
    if is_market:
        hb = threading.Thread(target=_heartbeat_market,
                              args=(task_id, stop_heartbeat, cancel_event),
                              daemon=True)
    else:
        hb = threading.Thread(target=_heartbeat_gp,
                              args=(task_id, params, task_type, stop_heartbeat, cancel_event),
                              daemon=True)
    hb.start()

    result: Optional[Dict[str, Any]] = None
    status: str = 'completed'
    error_tb: Optional[str] = None

    try:
        if task_type == main_mod.TASK_TYPE_GP:
            result = _dispatch_gp(params, task_id, cancel_event)
        elif task_type == main_mod.TASK_TYPE_LLM:
            result = _dispatch_llm(params, task_id, cancel_event)
        elif task_type == main_mod.TASK_TYPE_FUSION:
            result = _dispatch_fusion(params, task_id)
        elif is_market:
            result = _dispatch_market_data(md_type, params, task_id, cancel_event)
        else:
            raise ValueError(f'Unknown task type: {task_type}')

        # Check for cooperative cancellation.
        if is_market:
            if main_mod.market_data_tasks.get(task_id, {}).get("status") == "terminated":
                status = 'terminated'
        else:
            if main_mod.tasks.get(task_id, {}).get("status") == "terminated":
                status = 'terminated'
    except Exception as exc:
        status = 'failed'
        error_tb = traceback.format_exc()
        print(f'[task_runner] {task_type} {task_id} FAILED: {exc}')
        traceback.print_exc()

    stop_heartbeat.set()
    finished_at = main_mod.datetime.now().isoformat()

    if is_market:
        md = main_mod.market_data_tasks.get(task_id, {})
        if status == 'terminated':
            md["status"] = 'terminated'
            md["finished_at"] = finished_at
        elif status == 'failed':
            md["status"] = 'failed'
            md["error"] = error_tb
            md["progress"] = f'失败: {error_tb}'
            md["finished_at"] = finished_at
        else:
            md["status"] = 'completed'
            md["finished_at"] = finished_at
        main_mod._save_market_data_task_to_db(task_id)
    else:
        t = main_mod.tasks.get(task_id, {})
        if status == 'terminated':
            t["status"] = 'terminated'
            t["progress"] = '已终止（用户手动终止）'
            t["finished_at"] = finished_at
        elif status == 'failed':
            t["status"] = 'failed'
            t["error"] = error_tb
            t["progress"] = f'失败: {error_tb}'
            t["finished_at"] = finished_at
        else:
            t["status"] = 'completed'
            t["result"] = result
            t["result_overview"] = main_mod._build_mining_result_overview(result) if result else None
            t["progress"] = '完成'
            t["finished_at"] = finished_at

        summary = {}
        if result:
            summary = {
                'selected_fc_name_list': result.get('selected_fc_name_list', []),
                'version': result.get('version', ''),
                'message': result.get('message', ''),
                'config_path': result.get('config_path'),
                'factor_formulas': result.get('factor_formulas', {}),
                'best_failed_indicator_metrics': result.get('best_failed_indicator_metrics'),
                'result_overview': t.get('result_overview'),
            }
        main_mod._save_task_to_db(task_id, params, status, summary,
                                  task_type=task_type, result=result)

    print(f'[task_runner] {task_type} {task_id} -> {status}')
    return 0 if status == 'completed' else (2 if status == 'terminated' else 1)


if __name__ == '__main__':
    sys.exit(main())
