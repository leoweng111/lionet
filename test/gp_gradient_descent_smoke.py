"""
GP+梯度下降因子挖掘冒烟测试。

Usage:
    python -u test/gp_gradient_descent_smoke.py

测试内容：
1. alternated 模式完整跑通 GeneticFactorGenerator.auto_mine_select_and_save_fc。
   使用合成 C0 日频数据、极端筛选阈值，避免写入 MongoDB。
2. consecutive 模式直接跑通 run_gp_evolution，验证最终精英库优化路径。
3. EMA/TsDecayExp 在梯度阶段使用指数加权均值，而不是误用 SMA。
4. 启用梯度下降时，如果 fitness 包含不可微指标（如 TS RankIC），应立即报错。
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator
from factors.gp_factor_engine import run_gp_evolution
from factors.gp_gradient_descent_config import validate_gradient_descent_fitness_indicators
from utils.params import GP_INDICATOR_DIRECTION


def _require_torch() -> None:
    if importlib.util.find_spec('torch') is None:
        raise RuntimeError('PyTorch is required for GP+gradient descent smoke test. Please install requirements.txt first.')


def _build_synthetic_daily_data(n: int = 120, seed: int = 20260516) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range('2020-01-01', periods=n, freq='D')
    ret = rng.normal(0.0005, 0.012, size=n)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=n))
    volume = rng.uniform(1000, 5000, size=n)
    position = rng.uniform(10000, 30000, size=n)
    return pd.DataFrame({
        'time': time,
        'instrument_id': 'C0',
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'position': position,
    })


def _extreme_filter() -> Dict[str, Tuple[Optional[float], Optional[float], int]]:
    return {
        'TS IC': (999.0, 999.0, GP_INDICATOR_DIRECTION.get('TS IC', 1)),
        'Gross Sharpe': (999.0, 999.0, GP_INDICATOR_DIRECTION.get('Gross Sharpe', 1)),
    }


def test_alternated_full_generator_flow() -> None:
    data = _build_synthetic_daily_data()
    fg = GeneticFactorGenerator(
        instrument_id_list='C0',
        data=data,
        start_time='20200101',
        end_time='20200430',
        apply_weighted_price=False,
        check_relative=False,
        max_factor_count=2,
        gp_generations=1,
        gp_population_size=6,
        gp_max_depth=3,
        gp_elite_size=2,
        gp_window_choices=[3, 5],
        gp_early_stopping_generation_count=0,
        gp_log_interval=1,
        fitness_indicator_dict={'TS IC': 1.0},
        random_seed=7,
        apply_rolling_norm=True,
        rolling_norm_window=5,
        rolling_norm_min_periods=3,
        enable_gradient_descent=True,
        gradient_descent_method='alternated',
        generation_per_gradient_descent=1,
        gradient_descent_steps=2,
        parametric_method='opgd',
        gradient_descent_optimizer='adam',
        learning_rate=0.01,
        gradient_descent_early_stopping_steps=1,
    )
    result = fg.auto_mine_select_and_save_fc(filter_indicator_dict=_extreme_filter(), n_jobs=1)
    assert fg.generated_data is not None and not fg.generated_data.empty
    assert fg.factor_formula_map, 'factor_formula_map should be populated after GP+GD mining.'
    assert result['selected_fc_name_list'] == [], 'Extreme filter should avoid DB saving in smoke test.'


def test_consecutive_engine_flow() -> None:
    data = _build_synthetic_daily_data(n=90, seed=20260517)
    # run_gp_evolution expects future_ret in df; GeneticFactorGenerator normally adds it.
    close = data['close'].astype(float)
    data = data.copy()
    data['future_ret'] = close.pct_change().shift(-1).fillna(0.0)
    candidates = run_gp_evolution(
        df=data,
        data_fields=['open', 'high', 'low', 'close', 'volume', 'position'],
        fitness_indicator_dict={'TS IC': 1.0},
        max_factor_count=2,
        generations=1,
        population_size=6,
        max_depth=3,
        elite_size=2,
        tournament_size=2,
        crossover_prob=0.2,
        mutation_prob=0.6,
        window_choices=[3, 5],
        const_prob=0.1,
        leaf_prob=0.35,
        random_seed=11,
        early_stopping_generation_count=0,
        apply_rolling_norm=True,
        rolling_norm_window=5,
        rolling_norm_min_periods=3,
        enable_gradient_descent=True,
        gradient_descent_method='consecutive',
        generation_per_gradient_descent=1,
        gradient_descent_steps=2,
        parametric_method='gpgd',
        gradient_descent_optimizer='adamw',
        learning_rate=0.01,
        gradient_descent_early_stopping_steps=1,
    )
    assert candidates, 'consecutive GP+GD should return candidates.'
    assert all(c.formula for c in candidates)


def test_non_differentiable_fitness_rejected() -> None:
    try:
        validate_gradient_descent_fitness_indicators({'TS RankIC': 1.0})
    except ValueError as exc:
        assert '不可微' in str(exc) or 'non' in str(exc).lower()
        return
    raise AssertionError('TS RankIC should be rejected when GP+GD is enabled.')


def test_ema_uses_ewm_not_sma_in_torch_evaluator() -> None:
    from factors.factor_ops import DataNode, OpEma
    from factors.gp_gradient_descent import GradientDescentConfig, _ParametricTorchEvaluator

    data = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=6, freq='D'),
        'instrument_id': 'C0',
        'open': [1, 2, 3, 4, 5, 6],
        'high': [1, 2, 3, 4, 5, 6],
        'low': [1, 2, 3, 4, 5, 6],
        'close': [1, 2, 3, 4, 5, 6],
        'volume': [10, 11, 12, 13, 14, 15],
        'position': [100, 101, 102, 103, 104, 105],
        'future_ret': [0.0, 0.01, -0.01, 0.02, -0.02, 0.0],
    })
    cfg = GradientDescentConfig.from_kwargs(
        enable_gradient_descent=True,
        gradient_descent_steps=1,
        min_window=3,
        max_window=3,
        window_choices=[3],
    )
    model = _ParametricTorchEvaluator(
        root=OpEma(DataNode('close'), 3),
        df=data,
        cfg=cfg,
        apply_rolling_norm=False,
        rolling_norm_window=5,
        rolling_norm_min_periods=3,
        rolling_norm_eps=1e-8,
        rolling_norm_clip=5.0,
    )
    actual = np.asarray(model.forward().detach().cpu().tolist(), dtype=float)
    expected_ema = data['close'].ewm(span=3, adjust=False).mean().to_numpy(dtype=float)
    wrong_sma = data['close'].rolling(3).mean().fillna(0.0).to_numpy(dtype=float)
    assert np.allclose(actual, expected_ema, atol=1e-6), f'EMA evaluator mismatch: {actual} vs {expected_ema}'
    assert not np.allclose(actual, wrong_sma, atol=1e-6), 'EMA evaluator should not collapse to rolling SMA.'


def test_rolling_std_matches_pandas_ddof_one() -> None:
    from factors.factor_ops import DataNode, OpTsStd
    from factors.gp_gradient_descent import GradientDescentConfig, _ParametricTorchEvaluator

    data = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=6, freq='D'),
        'instrument_id': 'C0',
        'open': [1, 2, 4, 7, 11, 16],
        'high': [1, 2, 4, 7, 11, 16],
        'low': [1, 2, 4, 7, 11, 16],
        'close': [1, 2, 4, 7, 11, 16],
        'volume': [10, 11, 12, 13, 14, 15],
        'position': [100, 101, 102, 103, 104, 105],
        'future_ret': [0.0, 0.01, -0.01, 0.02, -0.02, 0.0],
    })
    cfg = GradientDescentConfig.from_kwargs(
        enable_gradient_descent=True,
        gradient_descent_steps=1,
        min_window=3,
        max_window=3,
        window_choices=[3],
    )
    model = _ParametricTorchEvaluator(
        root=OpTsStd(DataNode('close'), 3),
        df=data,
        cfg=cfg,
        apply_rolling_norm=False,
        rolling_norm_window=5,
        rolling_norm_min_periods=3,
        rolling_norm_eps=1e-8,
        rolling_norm_clip=5.0,
    )
    actual = np.asarray(model.forward().detach().cpu().tolist(), dtype=float)
    expected = data['close'].rolling(3).std().fillna(0.0).to_numpy(dtype=float)
    wrong_population = data['close'].rolling(3).std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    assert np.allclose(actual, expected, atol=1e-6), f'TsStd evaluator mismatch: {actual} vs {expected}'
    assert not np.allclose(actual, wrong_population, atol=1e-6), 'TsStd evaluator should use pandas ddof=1, not population std.'


def main() -> None:
    _require_torch()
    tests = [
        test_alternated_full_generator_flow,
        test_consecutive_engine_flow,
        test_ema_uses_ewm_not_sma_in_torch_evaluator,
        test_rolling_std_matches_pandas_ddof_one,
        test_non_differentiable_fitness_rejected,
    ]
    ok = True
    for fn in tests:
        name = fn.__name__
        try:
            print(f'Running {name} ...')
            fn()
            print(f'PASS {name}')
        except Exception as exc:
            ok = False
            print(f'FAIL {name}: {exc}')
            import traceback
            traceback.print_exc()
    print(f'RESULT: {str(ok).lower()}')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()


