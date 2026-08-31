"""
日内分钟特征冒烟测试：验证日内特征从计算到GP挖掘到回测的全链路。

测试策略：
- 构造合成分钟数据（含夜盘），验证 compute_intraday_daily_features 产出正确。
- 验证日内算子（OpRV 等）能被 parse_formula_to_node 解析、calc 读取预计算列。
- 验证 get_unary_ops 按选择过滤算子池。
- 验证 extract_intraday_features_from_formula 能从公式提取所需特征。
- 验证 ensure_intraday_features_in_df 能透明预计算（mock 分钟数据不可用时优雅降级）。
- 验证 GP+GD 可微评估器能处理 OpIntradayFeature（不报 NotImplementedError）。

Usage:
    python -u test/intraday_features_smoke.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_synthetic_minute_df(n_days: int = 5, seed: int = 42) -> pd.DataFrame:
    """构造合成分钟数据（含夜盘），用于日内特征计算测试。

    每个交易日含：前一日 21:00-22:59 夜盘（120根）+ 当日 09:00-11:29+13:30-14:59 日盘（300根）。
    """
    rng = np.random.default_rng(seed)
    bars = []
    base = datetime(2024, 1, 15)  # Monday

    for day_offset in range(n_days):
        # Night session: 21:00-22:59
        night_date = base + timedelta(days=day_offset)
        for h in range(21, 23):
            for m in range(60):
                t = night_date.replace(hour=h, minute=m)
                px = 100 + rng.normal(0, 0.3) + day_offset * 0.5
                bars.append({
                    'time': t, 'instrument_id': 'C0', 'open': px, 'high': px + 0.2,
                    'low': px - 0.2, 'close': px + rng.normal(0, 0.1),
                    'volume': float(rng.integers(50, 200)),
                    'position': float(rng.integers(4000, 6000)),
                    'weighted_factor': 1.0,
                })
        # Day session: 09:00-11:29 + 13:30-14:59
        day_date = base + timedelta(days=day_offset + 1)
        for h in [9, 10, 11, 13, 14]:
            for m in range(60):
                if h == 11 and m > 29:
                    continue
                if h == 13 and m < 30:
                    continue
                t = day_date.replace(hour=h, minute=m)
                px = 100 + rng.normal(0, 0.3) + day_offset * 0.5
                bars.append({
                    'time': t, 'instrument_id': 'C0', 'open': px, 'high': px + 0.2,
                    'low': px - 0.2, 'close': px + rng.normal(0, 0.1),
                    'volume': float(rng.integers(50, 200)),
                    'position': float(rng.integers(4000, 6000)),
                    'weighted_factor': 1.0,
                })

    return pd.DataFrame(bars)


# ── Test 1: Feature computation correctness ───────────────────────

def test_feature_computation():
    """验证 compute_intraday_daily_features 产出全部 22 个特征，且关键不变量成立。"""
    print('=' * 60)
    print('TEST 1: Feature Computation Correctness')
    print('=' * 60)

    from factors.factor_intraday_features import (
        compute_intraday_daily_features, ALL_FEATURE_NAMES,
    )

    minute_df = _build_synthetic_minute_df(n_days=5, seed=42)
    print(f'  Synthetic minute bars: {len(minute_df)} rows')

    result = compute_intraday_daily_features(
        minute_df, feature_names=ALL_FEATURE_NAMES,
    )
    assert not result.empty, 'Result should not be empty'
    print(f'  Daily features: {len(result)} rows, {len(result.columns)} columns')

    # All 22 features should be present (+ time + instrument_id = 24 columns)
    expected_cols = set(ALL_FEATURE_NAMES)
    actual_cols = set(result.columns) - {'time', 'instrument_id'}
    missing = expected_cols - actual_cols
    assert not missing, f'Missing features: {missing}'
    print(f'  ✓ All {len(ALL_FEATURE_NAMES)} features computed')

    # Invariants
    for _, row in result.iterrows():
        if pd.notna(row['rv']):
            assert abs(row['rv'] - (row['rv_neg'] + row['rv_pos'])) < 1e-6, 'rv != rv_neg + rv_pos'
            assert abs(row['jump'] - max(row['rv'] - row['bpv'], 0)) < 1e-10, 'jump != max(rv-bpv, 0)'
    print(f'  ✓ Invariants OK: rv=rv_neg+rv_pos, jump=max(rv-bpv,0)')

    # Volatility features should be non-negative
    for col in ['rv', 'bpv', 'rv_neg', 'rv_pos', 'pk', 'gk', 'rs_vol', 'medrv']:
        vals = result[col].dropna()
        assert (vals >= 0).all(), f'{col} should be non-negative, got min={vals.min()}'
    print(f'  ✓ Volatility features non-negative')

    # YZ: first day NaN (no prev close), rest non-NaN
    assert pd.isna(result['yz'].iloc[0]), 'YZ first day should be NaN'
    assert result['yz'].iloc[1:].notna().all(), 'YZ day 2+ should be non-NaN'
    print(f'  ✓ YZ: first day NaN, rest valid')

    print(f'\n  ✓ TEST 1 PASSED')
    return True


# ── Test 2: Operator parsing and calc ──────────────────────────────

def test_operator_parsing_and_calc():
    """验证日内算子能被解析、calc 读取预计算列。"""
    print()
    print('=' * 60)
    print('TEST 2: Operator Parsing and Calc')
    print('=' * 60)

    from factors.factor_ops import (
        parse_formula_to_node, calc_formula_series,
        INTRADAY_FEATURE_OPS, INTRADAY_FEATURE_OP_MAP,
        get_unary_ops, UNARY_OPS, UNARY_CHILD_OPS, FactorDataType,
    )

    # 2a: All operators registered in OP_CLASS_BY_NAME
    for cls in INTRADAY_FEATURE_OPS:
        name = cls.__name__.removeprefix('Op')
        node = parse_formula_to_node(f'{name}(close)')
        assert isinstance(node, cls), f'{name} -> {type(node)}'
        assert node.data_type == cls.OUTPUT_TYPE
    print(f'  ✓ All 22 operators parse correctly')

    # 2b: Complex formulas
    formulas = [
        'RollNorm(RV(close), 30, 20, 1e-08, 5)',
        'Mul(RetOvernight(close), RollNorm(OIFlow(close), 30, 20, 1e-08, 5))',
        'TsMean(RV(close), 5)',
        'Add(RollNorm(RV(close), 30, 20, 1e-08, 5), RollNorm(BPV(close), 30, 20, 1e-08, 5))',
    ]
    for f in formulas:
        node = parse_formula_to_node(f)
        assert node is not None
    print(f'  ✓ Complex formulas parse OK')

    # 2c: calc reads pre-computed column
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'time': dates, 'instrument_id': 'C0',
        'open': 100 + np.arange(10), 'high': 101 + np.arange(10),
        'low': 99 + np.arange(10), 'close': 100.5 + np.arange(10),
        'volume': np.arange(100, 110), 'position': np.arange(1000, 1010),
        'rv': np.random.rand(10) * 0.1,
    })
    result = calc_formula_series(df, formula='RV(close)')
    assert np.allclose(result, df['rv'], equal_nan=True)
    print(f'  ✓ RV(close) reads pre-computed rv column')

    # 2d: Missing column raises KeyError
    try:
        calc_formula_series(df, formula='CVD(close)')
        raise AssertionError('Should have raised KeyError for missing cvd column')
    except KeyError:
        pass
    print(f'  ✓ CVD(close) raises KeyError when cvd missing')

    # 2e: get_unary_ops filtering
    ops_none = get_unary_ops(None)
    assert all(cls not in INTRADAY_FEATURE_OPS for cls in ops_none)
    ops_partial = get_unary_ops(['rv', 'oi_flow'])
    assert any(cls.__name__ == 'OpRV' for cls in ops_partial)
    assert any(cls.__name__ == 'OpOIFlow' for cls in ops_partial)
    assert not any(cls.__name__ == 'OpBPV' for cls in ops_partial)
    print(f'  ✓ get_unary_ops filters correctly')

    # 2f: UNARY_CHILD_OPS includes intraday (for isinstance checks)
    from factors.factor_ops import OpRV
    assert OpRV in UNARY_CHILD_OPS
    assert OpRV not in UNARY_OPS
    print(f'  ✓ UNARY_CHILD_OPS includes OpRV, UNARY_OPS excludes it')

    print(f'\n  ✓ TEST 2 PASSED')
    return True


# ── Test 3: Formula feature extraction ─────────────────────────────

def test_formula_feature_extraction():
    """验证 extract_intraday_features_from_formula 能正确提取公式中的日内特征。"""
    print()
    print('=' * 60)
    print('TEST 3: Formula Feature Extraction')
    print('=' * 60)

    from factors.factor_intraday_features import extract_intraday_features_from_formula

    tests = [
        ('RollNorm(RV(close), 30, 20, 1e-08, 5)', {'rv'}),
        ('Mul(OIFlow(close), RetOpen30(close))', {'oi_flow', 'ret_open30'}),
        ('TsMean(close, 5)', set()),
        ('Add(RV(close), BPV(close))', {'rv', 'bpv'}),
        ('RollNorm(RetOvernight(close), 30, 20, 1e-08, 5)', {'ret_overnight'}),
        ('Mul(RetOpen30(close), RollNorm(KyleLambda(close), 30, 20, 1e-08, 5))',
         {'ret_open30', 'kyle_lambda'}),
        ('', set()),
        ('not a formula', set()),
    ]
    for formula, expected in tests:
        needed = set(extract_intraday_features_from_formula(formula))
        assert needed == expected, f'{formula}: got {needed}, expected {expected}'
    print(f'  ✓ All {len(tests)} extraction tests passed')

    print(f'\n  ✓ TEST 3 PASSED')
    return True


# ── Test 4: ensure_intraday_features_in_df graceful degradation ────

def test_ensure_graceful_degradation():
    """验证 ensure_intraday_features_in_df 在无分钟数据时优雅降级。"""
    print()
    print('=' * 60)
    print('TEST 4: ensure_intraday_features_in_df Graceful Degradation')
    print('=' * 60)

    from factors.factor_intraday_features import ensure_intraday_features_in_df

    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'time': dates, 'instrument_id': 'C0',
        'open': 100 + np.arange(5), 'high': 101 + np.arange(5),
        'low': 99 + np.arange(5), 'close': 100.5 + np.arange(5),
        'volume': np.arange(100, 105), 'position': np.arange(1000, 1005),
    })

    # No intraday ops → df unchanged
    result = ensure_intraday_features_in_df(
        df=df, formulas=['TsMean(close, 5)'],
        instrument_id_list=['C0'], start_date='20240101', end_date='20240105',
    )
    assert list(result.columns) == list(df.columns), 'Should not add columns for non-intraday formula'
    print(f'  ✓ Non-intraday formula → df unchanged')

    # Intraday ops but no minute data → df unchanged (graceful)
    result2 = ensure_intraday_features_in_df(
        df=df, formulas=['RollNorm(RV(close), 30, 20, 1e-08, 5)'],
        instrument_id_list=['C0'], start_date='20240101', end_date='20240105',
        source='__nonexistent_source__',
    )
    assert list(result2.columns) == list(df.columns), 'Should not add columns when minute data unavailable'
    print(f'  ✓ Intraday formula but no minute data → df unchanged (graceful)')

    # If rv already in df → no re-computation needed
    df_with_rv = df.copy()
    df_with_rv['rv'] = 0.05
    result3 = ensure_intraday_features_in_df(
        df=df_with_rv, formulas=['RollNorm(RV(close), 30, 20, 1e-08, 5)'],
        instrument_id_list=['C0'], start_date='20240101', end_date='20240105',
        source='__nonexistent_source__',
    )
    assert 'rv' in result3.columns
    print(f'  ✓ Pre-existing rv column → not recomputed')

    print(f'\n  ✓ TEST 4 PASSED')
    return True


# ── Test 5: GP tree generation respects feature filter ─────────────

def test_gp_tree_respects_filter():
    """验证 GP 树生成遵守日内特征选择过滤。"""
    print()
    print('=' * 60)
    print('TEST 5: GP Tree Generation Respects Feature Filter')
    print('=' * 60)

    import random
    from factors.factor_ops import get_unary_ops, INTRADAY_FEATURE_OPS
    from factors.gp_factor_engine import _generate_valid_random_tree

    rng = random.Random(42)
    data_fields = ['open', 'high', 'low', 'close', 'volume', 'position', 'rv', 'oi_flow']

    # With partial filter — forbidden ops should never appear
    ops_partial = get_unary_ops(['rv', 'oi_flow'])
    forbidden_names = {'BPV', 'CVD', 'Jump', 'KyleLambda', 'RetOpen30', 'VRK', 'YZ'}
    for _ in range(300):
        tree = _generate_valid_random_tree(
            data_fields, 5, 0, [3, 5, 10], 0.02, 0.3, rng, unary_ops=ops_partial)
        formula = tree.to_formula()
        for name in forbidden_names:
            assert name not in formula, f'Forbidden {name} in: {formula}'
    print(f'  ✓ 300 trees with [rv,oi_flow]: no forbidden operators')

    # With all features — RV should appear at least once
    all_features = [cls.FEATURE_COLUMN for cls in INTRADAY_FEATURE_OPS]
    ops_all = get_unary_ops(all_features)
    df_all = data_fields + [c for c in all_features if c not in data_fields]
    found_rv = sum(
        1 for _ in range(300)
        if 'RV(' in _generate_valid_random_tree(
            df_all, 5, 0, [3, 5, 10], 0.02, 0.3, rng, unary_ops=ops_all).to_formula()
    )
    assert found_rv > 0, 'RV should appear in trees when all features enabled'
    print(f'  ✓ 300 trees with all features: {found_rv} contain RV')

    # With no features — no intraday ops at all
    ops_none = get_unary_ops(None)
    for _ in range(300):
        tree = _generate_valid_random_tree(
            data_fields, 5, 0, [3, 5, 10], 0.02, 0.3, rng, unary_ops=ops_none)
        formula = tree.to_formula()
        for cls in INTRADAY_FEATURE_OPS:
            name = cls.__name__.removeprefix('Op')
            assert name not in formula, f'{name} appeared when intraday disabled'
    print(f'  ✓ 300 trees with no intraday: zero intraday operators')

    print(f'\n  ✓ TEST 5 PASSED')
    return True


# ── Test 6: GP+GD differentiable evaluator handles intraday ops ─────

def test_gp_gd_handles_intraday_ops():
    """验证 GP+梯度下降可微评估器能处理 OpIntradayFeature。"""
    print()
    print('=' * 60)
    print('TEST 6: GP+GD Differentiable Evaluator Handles Intraday Ops')
    print('=' * 60)

    try:
        import torch
    except ImportError:
        print('  ⚠ PyTorch not available, skipping GP+GD intraday test')
        return True

    from factors.factor_ops import DataNode, OpAdd, OpRV
    from factors.gp_gradient_descent import GradientDescentConfig, _ParametricTorchEvaluator

    n = 60
    close = np.linspace(100, 110, n)
    rv_vals = np.random.rand(n) * 0.1
    data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='D'),
        'instrument_id': 'C0',
        'open': close, 'high': close + 0.5, 'low': close - 0.5,
        'close': close, 'volume': np.arange(1000, 1000 + n),
        'position': np.arange(10000, 10000 + n),
        'rv': rv_vals,
        'future_ret': np.random.randn(n) * 0.01,
    })

    cfg = GradientDescentConfig.from_kwargs(
        enable_gradient_descent=True,
        gradient_descent_steps=1,
        min_window=3,
        max_window=10,
        window_choices=[3, 5, 10],
    )

    # Tree with intraday operator: Add(RV(close), DataNode('open'))
    model = _ParametricTorchEvaluator(
        root=OpAdd(OpRV(DataNode('close')), DataNode('open')),
        df=data,
        cfg=cfg,
        apply_rolling_norm=False,
        rolling_norm_window=5,
        rolling_norm_min_periods=3,
        rolling_norm_eps=1e-8,
        rolling_norm_clip=5.0,
    )

    factor = model.forward()
    factor_np = factor.detach().cpu().numpy()
    expected = rv_vals + close
    assert np.allclose(factor_np, expected, atol=1e-5), \
        f'OpRV+OpAdd forward mismatch: {factor_np[:3]} vs {expected[:3]}'
    print(f'  ✓ OpIntradayFeature forward OK (reads rv field + edge weight)')

    # Gradient should flow (edge weight on RV's child)
    loss = -model.score(factor, {'TS IC': 1.0})
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, 'Should have gradients from edge weights'
    assert all(np.isfinite(float(g.sum().item())) for g in grads)
    print(f'  ✓ Gradients finite through OpIntradayFeature')

    # Materialize should preserve the operator
    node = model.materialize()
    formula = node.to_formula()
    assert 'RV(' in formula, f'Materialized formula should contain RV: {formula}'
    print(f'  ✓ Materialize preserves OpRV: {formula}')

    print(f'\n  ✓ TEST 6 PASSED')
    return True


# ── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = {}
    test_funcs = [
        ('Feature Computation', test_feature_computation),
        ('Operator Parsing and Calc', test_operator_parsing_and_calc),
        ('Formula Feature Extraction', test_formula_feature_extraction),
        ('Ensure Graceful Degradation', test_ensure_graceful_degradation),
        ('GP Tree Respects Filter', test_gp_tree_respects_filter),
        ('GP+GD Handles Intraday Ops', test_gp_gd_handles_intraday_ops),
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
