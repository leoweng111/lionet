"""
GP因子挖掘冒烟测试：验证 GeneticFactorGenerator + auto_mine_select_and_save_fc 流程是否正常。

测试策略：
- 随机生成 fitness_indicator_dict（权重总和为 1）。
- 将 filter_indicator_dict 的门槛设置得极高，确保没有因子能通过筛选（避免因子入库）。
- 仅运行 1 代演化（gp_generations=1），快速验证流程完整性。
- 默认开启样本外比例 = 0.3，样本外区间 20250101~20251231。
- 模拟后端 _execute_mining / _execute_fusion 的参数构造逻辑，确保前后端调用链路一致。

Usage:
    python -u test/gp_mining_smoke.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator
from utils.params import (
    GP_DEFAULT_FITNESS_INDICATOR_WEIGHT,
    GP_DEFAULT_FILTER_INDICATOR_DICT,
    GP_INDICATOR_DIRECTION,
    GP_SUPPORTED_INDICATOR,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _random_fitness_indicator_dict(seed: int = 42) -> dict:
    """随机生成 fitness_indicator_dict，权重总和为 1。"""
    config_path = PROJECT_ROOT / 'config' / 'gp_supported_indicator.json'
    with open(config_path, 'r') as f:
        all_indicators = json.load(f)

    rng = np.random.default_rng(seed)

    # 随机选取 3~6 个指标
    n_selected = rng.integers(3, min(7, len(all_indicators) + 1))
    selected = rng.choice(all_indicators, size=n_selected, replace=False).tolist()

    # 随机权重，归一化到总和为 1
    raw_weights = rng.random(n_selected)
    weights = raw_weights / raw_weights.sum()

    return {indicator: round(float(w), 4) for indicator, w in zip(selected, weights)}


def _build_extreme_filter_indicator_dict() -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
    """构建极端门槛的 filter_indicator_dict，确保没有因子能通过筛选。

    filter_indicator_dict 格式: {indicator: (mean_threshold, yearly_threshold, direction)}
    direction=1 表示 >=，direction=-1 表示 <=。
    """
    filter_dict = {}
    # 设置 Gross Sharpe 极高门槛：均值 >= 999，每年 >= 999
    filter_dict['Gross Sharpe'] = (999.0, 999.0, GP_INDICATOR_DIRECTION.get('Gross Sharpe', 1))
    # 设置 TS IC 极高门槛
    filter_dict['TS IC'] = (999.0, 999.0, GP_INDICATOR_DIRECTION.get('TS IC', 1))
    return filter_dict


def _simulate_backend_normalize_fitness(raw_dict: Dict[str, float]) -> Dict[str, float]:
    """模拟 main.py 中 _normalize_fitness_indicator_dict 的逻辑。"""
    indicator_weight: Dict[str, float] = {}
    for indicator in GP_SUPPORTED_INDICATOR:
        raw_weight = raw_dict.get(indicator, None)
        if raw_weight is None:
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if abs(weight) <= 1e-12:
            continue
        indicator_weight[indicator] = weight

    if not indicator_weight:
        indicator_weight = {'TS IC': 1.0}
    return indicator_weight


def _simulate_backend_normalize_filter(
    raw_filter_dict: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
    """模拟 main.py 中 _normalize_filter_indicator_dict 的逻辑。"""
    out: Dict[str, Tuple[Optional[float], Optional[float], int]] = {}
    for indicator, conf in raw_filter_dict.items():
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

    if not out:
        for indicator, (default_mean, default_yearly, default_direction) in GP_DEFAULT_FILTER_INDICATOR_DICT.items():
            out[indicator] = (default_mean, default_yearly, default_direction)
    return out


# ── Test 1: Direct GeneticFactorGenerator (with outsample) ──────────

def test_gp_mining_with_outsample():
    """测试 GP 因子挖掘全流程（含样本外回测）。"""
    print('=' * 60)
    print('TEST 1: GP Mining with Outsample')
    print('=' * 60)

    seed = int(datetime.now().strftime('%S'))
    fitness_indicator_dict = _random_fitness_indicator_dict(seed=seed)
    filter_indicator_dict = _build_extreme_filter_indicator_dict()

    print(f'  Random seed: {seed}')
    print(f'  fitness_indicator_dict: {fitness_indicator_dict}')
    print(f'  filter_indicator_dict: {filter_indicator_dict}')

    version = f'__smoke_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}__'

    fg = GeneticFactorGenerator(
        instrument_id_list='C0',
        start_time='20230101',
        end_time='20241231',
        calculate_baseline=False,
        n_jobs=1,
        max_factor_count=5,
        check_leakage_count=3,
        check_relative=False,
        version=version,
        gp_generations=1,
        gp_population_size=20,
        gp_max_depth=4,
        gp_elite_size=5,
        fitness_indicator_dict=fitness_indicator_dict,
        random_seed=seed,
        outsample_ratio=0.3,
        outsample_start_time='20250101',
        outsample_end_time='20251231',
    )

    print(f'\n  Running auto_mine_select_and_save_fc ...')
    result = fg.auto_mine_select_and_save_fc(
        filter_indicator_dict=filter_indicator_dict,
        n_jobs=1,
        require_all_row=True,
        require_all_instruments=True,
    )

    selected = result.get('selected_fc_name_list', [])
    bt = result.get('bt')

    print(f'\n  ── Results ──')
    print(f'  selected_fc_name_list: {selected}')
    print(f'  generated_fc_name_list: {fg.generated_fc_name_list}')

    # 1) 因子应该被生成
    assert fg.generated_data is not None, 'generated_data should not be None'
    assert len(fg.generated_fc_name_list) > 0, 'Should generate at least 1 factor'
    print(f'  ✓ Factor generation OK: {len(fg.generated_fc_name_list)} factors generated')

    # 2) 回测应该完成
    assert bt is not None, 'BackTester should not be None'
    assert bt.performance_summary is not None, 'performance_summary should not be None'
    print(f'  ✓ Backtest OK: performance_summary shape={bt.performance_summary.shape}')

    # 3) 因为门槛极高，不应该有因子通过筛选（即不会入库）
    assert len(selected) == 0, f'No factors should pass extreme thresholds, but got: {selected}'
    print(f'  ✓ Filter OK: no factors passed extreme thresholds (as expected)')

    # 4) 因子公式应该存在
    assert len(fg.factor_formula_map) > 0, 'factor_formula_map should not be empty'
    print(f'  ✓ Formula map OK: {len(fg.factor_formula_map)} formulas')

    # 5) 样本外数据应该被生成
    assert fg.outsample_data is not None, 'outsample_data should not be None (outsample_ratio=0.3)'
    assert fg.generated_outsample_data is not None, 'generated_outsample_data should not be None'
    print(f'  ✓ Outsample data OK: generated_outsample_data shape={fg.generated_outsample_data.shape}')

    print(f'\n  ✓ TEST 1 PASSED')
    return True


# ── Test 2: Simulate backend _execute_mining ────────────────────────

def test_simulate_backend_mining():
    """模拟后端 _execute_mining 的完整参数构造和调用链路。

    复现 main.py 中 GPMiningParams → _normalize_*_dict → GeneticFactorGenerator
    的参数转换逻辑，确保所有参数能正确传递给 GeneticFactorGenerator。
    """
    print()
    print('=' * 60)
    print('TEST 2: Simulate Backend _execute_mining')
    print('=' * 60)

    seed = 7
    version = f'__smoke_backend_{datetime.now().strftime("%Y%m%d_%H%M%S")}__'

    # 模拟前端传入的 fitness_indicator_dict（与 GPMiningParams 格式一致）
    raw_fitness = dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)
    raw_fitness['TS IC'] = 0.5
    raw_fitness['TS ICIR'] = 0.5

    # 模拟前端传入的 filter_indicator_dict（与 GPMiningParams 格式一致）
    raw_filter: Dict[str, Dict[str, Optional[float]]] = {
        'Gross Sharpe': {'mean_threshold': 999.0, 'yearly_threshold': 999.0, 'direction': 1},
        'TS IC': {'mean_threshold': 999.0, 'yearly_threshold': 999.0, 'direction': 1},
    }

    # 模拟后端的归一化逻辑
    fitness_indicator_dict = _simulate_backend_normalize_fitness(raw_fitness)
    filter_indicator_dict = _simulate_backend_normalize_filter(raw_filter)

    print(f'  fitness_indicator_dict (after normalize): {fitness_indicator_dict}')
    print(f'  filter_indicator_dict (after normalize): {filter_indicator_dict}')

    # 模拟后端构造 GeneticFactorGenerator 的完整参数列表（与 main.py _execute_mining 一致）
    fg = GeneticFactorGenerator(
        instrument_type='futures_continuous_contract',
        instrument_id_list='C0',
        fc_freq='1d',
        start_time='20230101',
        end_time='20241231',
        version=version,
        portfolio_adjust_method='1D',
        interest_method='simple',
        risk_free_rate=False,
        calculate_baseline=False,
        apply_weighted_price=True,
        n_jobs=1,
        max_factor_count=5,
        min_window_size=30,
        apply_rolling_norm=True,
        rolling_norm_window=30,
        rolling_norm_min_periods=20,
        rolling_norm_eps=1e-8,
        rolling_norm_clip=5.0,
        check_leakage_count=3,
        check_relative=False,
        relative_threshold=0.7,
        gp_generations=1,
        fitness_indicator_dict=fitness_indicator_dict,
        gp_max_depth=4,
        gp_population_size=20,
        gp_elite_size=5,
        gp_elite_relative_threshold=0.65,
        gp_crossover_prob=0.3,
        gp_mutation_prob=0.7,
        gp_leaf_prob=0.2,
        gp_const_prob=0.02,
        gp_tournament_size=3,
        gp_window_choices=[3, 5, 10, 20, 30],
        gp_depth_penalty_coef=0.0,
        gp_depth_penalty_start_depth=6,
        gp_depth_penalty_linear_coef=0.03,
        gp_depth_penalty_quadratic_coef=0.0,
        gp_early_stopping_generation_count=20,
        gp_log_interval=1,
        random_seed=seed,
        gp_assumed_initial_capital=100000,
        gp_small_factor_penalty_coef=0.0,
        gp_elite_stagnation_generation_count=4,
        gp_max_shock_generation=3,
        consistency_penalty_enabled=False,
        consistency_penalty_coef=1.0,
        outsample_ratio=0.3,
        outsample_start_time='20250101',
        outsample_end_time='20251231',
    )

    print(f'\n  Running auto_mine_select_and_save_fc (simulating backend) ...')
    result = fg.auto_mine_select_and_save_fc(
        filter_indicator_dict=filter_indicator_dict,
        n_jobs=1,
        require_all_row=True,
        require_all_instruments=True,
    )

    bt = result.get('bt')
    selected = result.get('selected_fc_name_list', [])

    assert fg.generated_data is not None, 'generated_data should not be None'
    assert len(fg.generated_fc_name_list) > 0, 'Should generate at least 1 factor'
    print(f'  ✓ Factor generation OK: {len(fg.generated_fc_name_list)} factors generated')

    assert bt is not None, 'BackTester should not be None'
    assert bt.performance_summary is not None, 'performance_summary should not be None'
    print(f'  ✓ Backtest OK')

    assert len(selected) == 0, f'No factors should pass extreme thresholds, but got: {selected}'
    print(f'  ✓ Filter OK: no factors passed extreme thresholds (as expected)')

    assert fg.outsample_data is not None, 'outsample_data should not be None'
    assert fg.generated_outsample_data is not None, 'generated_outsample_data should not be None'
    print(f'  ✓ Outsample OK')

    assert len(fg.factor_formula_map) > 0, 'factor_formula_map should not be empty'
    print(f'  ✓ Formula map OK: {len(fg.factor_formula_map)} formulas')

    print(f'\n  ✓ TEST 2 PASSED')
    return True


# ── Test 3: Simulate backend _execute_fusion (param construction) ───

def test_simulate_backend_fusion():
    """模拟后端 _execute_fusion 的参数构造逻辑。

    验证 FusionParams 中的 outsample_ratio/outsample_start_time/outsample_end_time
    和 fusion_indicator_dict 能正确传递给 FactorFusioner。
    """
    print()
    print('=' * 60)
    print('TEST 3: Simulate Backend _execute_fusion (param construction)')
    print('=' * 60)

    from factors.factor_auto_search import FactorFusioner

    # 模拟 FusionParams 中的参数
    outsample_ratio = 0.3
    outsample_start_time = '20250101'
    outsample_end_time = '20251231'

    # 模拟后端归一化 fusion_indicator_dict
    fusion_indicator_dict = {'TS IC': 0.3, 'TS ICIR': 0.7}

    # 模拟 main.py _execute_fusion 中的参数映射
    try:
        fusioner = FactorFusioner(
            fusion_method='avg_weight',
            use_version_dict={'genetic_programming': ['__test_version__']},
            instrument_type='futures_continuous_contract',
            instrument_id_list='C0',
            fc_freq='1d',
            start_time='20230101',
            end_time='20241231',
            portfolio_adjust_method='1D',
            interest_method='simple',
            risk_free_rate=False,
            apply_weighted_price=True,
            check_leakage_count=20,
            check_relative=False,
            relative_threshold=0.7,
            relative_check_version_list=None,
            max_fusion_count=5,
            fusion_indicator_dict=fusion_indicator_dict,
            version='__smoke_fusion_test__',
            n_jobs=1,
            base_col_list=None,
            outsample_ratio=outsample_ratio,
            outsample_start_time=outsample_start_time,
            outsample_end_time=outsample_end_time,
        )
        print(f'  ✓ FactorFusioner construction OK')
        print(f'    outsample_ratio={fusioner.outsample_ratio}')
        print(f'    outsample_start_time={fusioner.outsample_start_time}')
        print(f'    outsample_end_time={fusioner.outsample_end_time}')
        print(f'    fusion_indicator_dict={fusioner.fusion_indicator_dict}')

        assert fusioner.outsample_ratio == outsample_ratio, f'outsample_ratio mismatch: {fusioner.outsample_ratio}'
        assert fusioner.outsample_start_time == outsample_start_time
        assert fusioner.outsample_end_time == outsample_end_time
        assert fusioner.fusion_indicator_dict == fusion_indicator_dict, \
            f'fusion_indicator_dict mismatch: {fusioner.fusion_indicator_dict}'
        print(f'  ✓ Param mapping OK')

    except TypeError as e:
        print(f'  ✗ FactorFusioner construction FAILED: {e}')
        raise

    print(f'\n  ✓ TEST 3 PASSED')
    return True


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = {}
    test_funcs = [
        ('GP Mining with Outsample', test_gp_mining_with_outsample),
        ('Simulate Backend Mining', test_simulate_backend_mining),
        ('Simulate Backend Fusion', test_simulate_backend_fusion),
    ]

    for name, func in test_funcs:
        try:
            results[name] = func()
        except Exception as e:
            print(f'\n  ✗ {name} FAILED: {e}')
            import traceback
            traceback.print_exc()
            results[name] = False

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    all_passed = True
    for name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {status}: {name}')
        if not passed:
            all_passed = False

    print(f'RESULT: {"true" if all_passed else "false"}  ({"all tests passed" if all_passed else "some tests failed"})')
    sys.exit(0 if all_passed else 1)
