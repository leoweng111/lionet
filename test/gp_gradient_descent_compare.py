"""
Compare classic GP vs GP+Gradient-Descent mining using config/gp_auto_search_params.json.

Default mode uses the full config. For fast validation, pass --quick to cap generations,
population, factors and GD steps while preserving the same parameter source.

Usage:
    python -u test/gp_gradient_descent_compare.py
    python -u test/gp_gradient_descent_compare.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator
from factors.gp_gradient_descent_config import DIFFERENTIABLE_GP_FITNESS_INDICATORS
from utils.params import GP_DEFAULT_FILTER_INDICATOR_DICT, GP_SUPPORTED_INDICATOR


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _normalize_indicator_weight(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for indicator in GP_SUPPORTED_INDICATOR:
        try:
            weight = float(raw.get(indicator, 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        if abs(weight) > 1e-12:
            out[indicator] = weight
    return out or {'TS IC': 1.0}


def _differentiable_fitness_for_pair(raw: Dict[str, Any]) -> Dict[str, float]:
    normalized = _normalize_indicator_weight(raw)
    filtered = {k: v for k, v in normalized.items() if k in DIFFERENTIABLE_GP_FITNESS_INDICATORS}
    if not filtered:
        return {'TS IC': 1.0}
    total_abs = sum(abs(v) for v in filtered.values())
    if total_abs <= 1e-12:
        return {'TS IC': 1.0}
    return {k: float(v) / total_abs for k, v in filtered.items()}


def _build_filter_dict(raw: Dict[str, Any]) -> Dict[str, tuple[Optional[float], Optional[float], int]]:
    if not isinstance(raw, dict) or not raw:
        return dict(GP_DEFAULT_FILTER_INDICATOR_DICT)
    out: Dict[str, tuple[Optional[float], Optional[float], int]] = {}
    for indicator, conf in raw.items():
        if indicator not in GP_SUPPORTED_INDICATOR or not isinstance(conf, dict):
            continue
        mean_raw = conf.get('mean_threshold')
        yearly_raw = conf.get('yearly_threshold')
        direction_raw = conf.get('direction', 1)
        mean = None if mean_raw in (None, '') else float(mean_raw)
        yearly = None if yearly_raw in (None, '') else float(yearly_raw)
        try:
            direction = int(direction_raw)
        except (TypeError, ValueError):
            direction = 1
        if direction not in (1, -1):
            direction = 1
        out[indicator] = (mean, yearly, direction)
    return out or dict(GP_DEFAULT_FILTER_INDICATOR_DICT)


def _apply_quick_caps(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    p['gp_generations'] = min(int(p.get('gp_generations', 1)), 1)
    p['gp_population_size'] = min(int(p.get('gp_population_size', 8)), 8)
    p['gp_elite_size'] = min(int(p.get('gp_elite_size', 3)), 3)
    p['max_factor_count'] = min(int(p.get('max_factor_count', 3)), 3)
    p['gradient_descent_steps'] = min(int(p.get('gradient_descent_steps', 1)), 1)
    p['gp_early_stopping_generation_count'] = 0
    p['check_relative'] = False
    p['check_leakage_count'] = min(int(p.get('check_leakage_count', 3)), 3)
    p['n_jobs'] = 1
    p['calculate_baseline'] = False
    return p


def _ctor_kwargs(params: Dict[str, Any], enable_gd: bool, fitness_indicator_dict: Dict[str, float]) -> Dict[str, Any]:
    keys = [
        'instrument_type', 'instrument_id_list', 'fc_freq', 'start_time', 'end_time',
        'portfolio_adjust_method', 'interest_method', 'risk_free_rate', 'calculate_baseline',
        'apply_weighted_price', 'n_jobs', 'min_window_size', 'max_factor_count',
        'apply_rolling_norm', 'rolling_norm_window', 'rolling_norm_min_periods',
        'rolling_norm_eps', 'rolling_norm_clip', 'check_leakage_count', 'check_relative',
        'relative_threshold', 'gp_generations', 'gp_population_size', 'gp_max_depth',
        'gp_elite_size', 'gp_elite_relative_threshold', 'gp_tournament_size',
        'gp_crossover_prob', 'gp_mutation_prob', 'gp_leaf_prob', 'gp_const_prob',
        'gp_window_choices', 'random_seed', 'gp_early_stopping_generation_count',
        'gp_depth_penalty_coef', 'gp_depth_penalty_start_depth',
        'gp_depth_penalty_linear_coef', 'gp_depth_penalty_quadratic_coef', 'gp_log_interval',
        'gp_small_factor_penalty_coef', 'gp_assumed_initial_capital',
        'gp_elite_stagnation_generation_count', 'gp_max_shock_generation',
        'consistency_penalty_enabled', 'consistency_penalty_coef', 'outsample_ratio',
        'outsample_start_time', 'outsample_end_time', 'gradient_descent_method',
        'generation_per_gradient_descent', 'gradient_descent_steps', 'parametric_method',
        'gradient_descent_optimizer', 'learning_rate', 'gradient_descent_early_stopping_steps',
        'gradient_clip_norm', 'gradient_soft_temperature',
    ]
    kwargs = {k: params[k] for k in keys if k in params}
    kwargs['version'] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{'gpgd' if enable_gd else 'gp'}_compare"
    kwargs['fitness_indicator_dict'] = fitness_indicator_dict
    kwargs['enable_gradient_descent'] = bool(enable_gd)
    return kwargs


def _summary_records(summary: Optional[pd.DataFrame], label: str) -> List[Dict[str, Any]]:
    if summary is None or summary.empty:
        return []
    df = summary.copy()
    index_name = df.index.name or 'year'
    df = df.reset_index().rename(columns={'index': index_name})
    if index_name != 'year' and 'year' not in df.columns:
        df = df.rename(columns={index_name: 'year'})
    df['sample'] = label
    return df.to_dict(orient='records')


def _compact_all_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    out = []
    for r in records:
        year = str(r.get('year', r.get('Year', ''))).lower()
        if year == 'all':
            out.append({
                'sample': r.get('sample'),
                'factor': r.get('Factor Name') or r.get('factor_name'),
                'Gross Sharpe': r.get('Gross Sharpe'),
                'Net Sharpe': r.get('Net Sharpe'),
                'TS IC': r.get('TS IC'),
                'TS RankIC': r.get('TS RankIC'),
                'Net Return': r.get('Net Return'),
            })
    return out


def _run_one(params: Dict[str, Any], enable_gd: bool, fitness: Dict[str, float]) -> Dict[str, Any]:
    t0 = time.time()
    fg = GeneticFactorGenerator(**_ctor_kwargs(params, enable_gd=enable_gd, fitness_indicator_dict=fitness))
    fg.generate()
    fc_names = list(fg.generated_fc_name_list or [])
    bt = fg.backtest(
        data=fg.generated_data,
        fc_name_list=fc_names,
        n_jobs=int(params.get('n_jobs', 1)),
        start_time=params.get('start_time'),
        end_time=params.get('end_time'),
    )
    is_records = _summary_records(bt.performance_summary, 'insample')

    oos_records: List[Dict[str, Any]] = []
    if getattr(fg, 'generated_outsample_data', None) is not None:
        bt_oos = fg.backtest(
            data=fg.generated_outsample_data,
            fc_name_list=fc_names,
            n_jobs=int(params.get('n_jobs', 1)),
            start_time=params.get('outsample_start_time'),
            end_time=params.get('outsample_end_time'),
            calculate_baseline=False,
        )
        oos_records = _summary_records(bt_oos.performance_summary, 'outsample')

    return {
        'mode': 'GP+GD' if enable_gd else 'GP',
        'enable_gradient_descent': bool(enable_gd),
        'elapsed_seconds': round(time.time() - t0, 3),
        'factor_count': len(fc_names),
        'factor_names': fc_names,
        'factor_formulas': getattr(fg, 'factor_formula_map', {}),
        'insample': is_records,
        'outsample': oos_records,
        'compact': _compact_all_rows(is_records + oos_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config' / 'gp_auto_search_params.json'))
    parser.add_argument('--quick', action='store_true', help='Cap generations/population/GD steps for fast validation.')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'test' / 'artifacts'))
    args = parser.parse_args()

    config_path = Path(args.config)
    params = _load_json(config_path)
    if args.quick:
        params = _apply_quick_caps(params)

    raw_fitness = params.get('fitness_indicator_dict') or {}
    original_fitness = _normalize_indicator_weight(raw_fitness)
    paired_fitness = _differentiable_fitness_for_pair(raw_fitness)
    dropped = sorted(set(original_fitness) - set(paired_fitness))
    if dropped:
        print(f"[WARN] Current config contains non-differentiable fitness for GP+GD: {dropped}")
        print(f"[WARN] For fair paired comparison, both GP and GP+GD use differentiable fitness: {paired_fitness}")

    print('=' * 100)
    print(f"Config: {config_path}")
    print(f"Quick mode: {args.quick}")
    print(f"Fitness used by both modes: {paired_fitness}")
    print(f"Generations={params.get('gp_generations')}, population={params.get('gp_population_size')}, max_factor_count={params.get('max_factor_count')}, GD steps={params.get('gradient_descent_steps')}")
    print('=' * 100)

    results = []
    for enable_gd in [False, True]:
        print(f"\nRunning {'GP+GD' if enable_gd else 'GP'} ...")
        result = _run_one(params, enable_gd=enable_gd, fitness=paired_fitness)
        results.append(result)
        print(f"Done {result['mode']}: elapsed={result['elapsed_seconds']}s, factors={result['factor_count']}")
        print(pd.DataFrame(result['compact']).to_string(index=False) if result['compact'] else 'No compact all-row metrics.')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gp_gradient_descent_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}{'_quick' if args.quick else ''}.json"
    payload = {
        'config': str(config_path),
        'quick': bool(args.quick),
        'fitness_original': original_fitness,
        'fitness_used_by_both_modes': paired_fitness,
        'dropped_non_differentiable_fitness': dropped,
        'params_effective': params,
        'results': results,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('\n' + '=' * 100)
    print(f"Saved comparison JSON: {output_file}")
    print('RESULT: true')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


