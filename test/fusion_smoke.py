"""
因子融合冒烟测试：验证 FactorFusioner 融合流程是否正常。

测试策略：
- 使用 genetic_programming 集合下的 20260507_gp_test、20260505_gp_test 两个 version。
- max_fusion_count=2，加速测试。
- 其余参数使用默认值。
- 同时模拟后端 _execute_fusion 的参数构造逻辑，确保前后端调用链路一致。

Usage:
    python -u test/fusion_smoke.py
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import FactorFusioner
from utils.params import (
    GP_DEFAULT_FITNESS_INDICATOR_WEIGHT,
    GP_SUPPORTED_INDICATOR,
)


# ── Test 1: Direct FactorFusioner call ───────────────────────────────

def test_fusion_direct():
    """直接调用 FactorFusioner，验证融合流程完整性。"""
    print()
    print('=' * 60)
    print('TEST 1: Direct FactorFusioner Fusion')
    print('=' * 60)

    version = f'__smoke_fusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}__'

    fusioner = FactorFusioner(
        fusion_method='avg_weight',
        use_version_dict={
            'genetic_programming': ['20260507_gp_test', '20260505_gp_test'],
        },
        instrument_type='futures_continuous_contract',
        instrument_id_list='C0',
        fc_freq='1d',
        start_time='20200101',
        end_time='20241231',
        portfolio_adjust_method='1D',
        interest_method='simple',
        risk_free_rate=False,
        apply_weighted_price=True,
        check_leakage_count=20,
        check_relative=False,
        relative_threshold=0.7,
        max_fusion_count=2,
        version=version,
        n_jobs=1,
    )

    print(f'  use_version_dict={fusioner.use_version_dict}')
    print(f'  fusion_indicator_dict={fusioner.fusion_indicator_dict}')
    print(f'  max_fusion_count={fusioner.max_fusion_count}')

    result = fusioner.fuse()

    assert result is not None, 'fuse() should return a dict'
    print(f'  ✓ fuse() returned result')

    fusion_formula = result.get('fusion_formula')
    print(f'  fusion_formula={fusion_formula}')

    selected = result.get('selected_factors_detail') or []
    print(f'  selected_factors_detail count={len(selected)}')

    bt = result.get('bt')
    if bt is not None:
        assert bt.performance_summary is not None, 'performance_summary should not be None'
        print(f'  ✓ Backtest OK')
    else:
        print(f'  ⚠ No BackTester returned (fusion may not have found valid factors)')

    print(f'\n  ✓ TEST 1 PASSED')
    return True


# ── Test 2: Simulate backend _execute_fusion ─────────────────────────

def test_simulate_backend_fusion():
    """模拟后端 _execute_fusion 的完整参数构造和调用链路。

    复现 main.py 中 FusionParams → FactorFusioner 的参数转换逻辑，
    确保 use_version_dict 和其他参数能正确传递。
    """
    print()
    print('=' * 60)
    print('TEST 2: Simulate Backend _execute_fusion')
    print('=' * 60)

    version = f'__smoke_backend_fusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}__'

    # 模拟前端传入的参数（与 FusionParams 格式一致）
    raw_fusion_indicator_dict = dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)
    raw_fusion_indicator_dict['TS IC'] = 0.5
    raw_fusion_indicator_dict['TS ICIR'] = 0.5

    # 模拟后端归一化 fusion_indicator_dict（与 main.py _execute_fusion 一致）
    fusion_indicator_dict = {}
    for k, v in raw_fusion_indicator_dict.items():
        if k in GP_SUPPORTED_INDICATOR:
            fv = float(v) if v is not None else 0.0
            if abs(fv) > 1e-12:
                fusion_indicator_dict[k] = fv
    if not fusion_indicator_dict:
        fusion_indicator_dict = dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)

    print(f'  fusion_indicator_dict (after normalize): {fusion_indicator_dict}')

    # 模拟前端构造 use_version_dict
    use_version_dict = {
        'genetic_programming': ['20260507_gp_test', '20260505_gp_test'],
    }

    # 模拟后端构造 FactorFusioner 的完整参数列表（与 main.py _execute_fusion 一致）
    fusioner = FactorFusioner(
        fusion_method='avg_weight',
        use_version_dict=use_version_dict,
        instrument_type='futures_continuous_contract',
        instrument_id_list='C0',
        fc_freq='1d',
        start_time='20200101',
        end_time='20241231',
        portfolio_adjust_method='1D',
        interest_method='simple',
        risk_free_rate=False,
        apply_weighted_price=True,
        check_leakage_count=20,
        check_relative=False,
        relative_threshold=0.7,
        relative_check_version_list=None,
        max_fusion_count=2,
        fusion_indicator_dict=fusion_indicator_dict,
        version=version,
        n_jobs=1,
        base_col_list=None,
        outsample_ratio=0.0,
        outsample_start_time=None,
        outsample_end_time=None,
    )

    print(f'  ✓ FactorFusioner construction OK')
    print(f'    use_version_dict={fusioner.use_version_dict}')
    print(f'    fusion_indicator_dict={fusioner.fusion_indicator_dict}')

    assert fusioner.use_version_dict == use_version_dict, \
        f'use_version_dict mismatch: {fusioner.use_version_dict}'
    print(f'  ✓ Param mapping OK')

    print(f'\n  Running fuse() (simulating backend) ...')
    result = fusioner.fuse()

    assert result is not None, 'fuse() should return a dict'
    print(f'  ✓ fuse() returned result')

    fusion_formula = result.get('fusion_formula')
    print(f'  fusion_formula={fusion_formula}')

    selected = result.get('selected_factors_detail') or []
    print(f'  selected_factors_detail count={len(selected)}')

    bt = result.get('bt')
    if bt is not None:
        assert bt.performance_summary is not None, 'performance_summary should not be None'
        print(f'  ✓ Backtest OK')

    print(f'\n  ✓ TEST 2 PASSED')
    return True


# ── Test 3: Simulate frontend payload construction ───────────────────

def test_simulate_frontend_payload():
    """模拟前端构造 use_version_dict 的逻辑。

    验证前端从 _selectedCollections + _selectedVersions 构造 use_version_dict
    的逻辑与后端 FactorFusioner 的 use_version_dict 参数格式一致。
    """
    print()
    print('=' * 60)
    print('TEST 3: Simulate Frontend Payload Construction')
    print('=' * 60)

    # 模拟前端的 _selectedCollections 和 _selectedVersions
    selected_collections = ['genetic_programming']
    selected_versions = ['20260507_gp_test', '20260505_gp_test']

    # 模拟前端构造 use_version_dict 的逻辑
    use_version_dict = {}
    cols = selected_collections if selected_collections else ['genetic_programming']
    vers = selected_versions if selected_versions else []
    for c in cols:
        use_version_dict[c] = list(vers) if vers else []

    print(f'  selected_collections={selected_collections}')
    print(f'  selected_versions={selected_versions}')
    print(f'  use_version_dict={use_version_dict}')

    assert use_version_dict == {'genetic_programming': ['20260507_gp_test', '20260505_gp_test']}
    print(f'  ✓ use_version_dict construction OK')

    # 验证能被 FactorFusioner 接受
    version = f'__smoke_frontend_{datetime.now().strftime("%Y%m%d_%H%M%S")}__'
    fusioner = FactorFusioner(
        fusion_method='avg_weight',
        use_version_dict=use_version_dict,
        version=version,
        max_fusion_count=2,
        check_relative=False,
        n_jobs=1,
    )
    assert fusioner.use_version_dict == use_version_dict
    print(f'  ✓ FactorFusioner accepts frontend-constructed use_version_dict')

    print(f'\n  ✓ TEST 3 PASSED')
    return True


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = {}
    test_funcs = [
        ('Direct Fusion', test_fusion_direct),
        ('Simulate Backend Fusion', test_simulate_backend_fusion),
        ('Simulate Frontend Payload', test_simulate_frontend_payload),
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

