"""
Integration test for lionet: single-factor backtest, fusion-factor backtest, and frontend build.

This script tests:
1. Single-factor backtest consistency across 4 methods (fc_name, formula, frontend-DB, frontend-formula)
2. Fusion-factor backtest consistency across 4 methods
3. Frontend build success

Usage:
    python -u test/integration_test.py
"""

import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ──────────────────────────────────────────────────────────
EPSILON = 1e-6

IS_START = "20200101"
IS_END = "20241231"
OOS_START = "20250101"
OOS_END = "20251231"
REAL_OOS_START = "20260101"
REAL_OOS_END = "20260330"
# Frontend calculates finalEndTime = max(IS_END, OOS_END, REAL_OOS_END)
FULL_END = REAL_OOS_END

# Test factor for single-factor test
TEST_SINGLE_VERSION = "20260417_gp_test"
TEST_SINGLE_FC_NAME = "fac_gp_0041"
TEST_SINGLE_COLLECTION = "genetic_programming"

# Test factor for fusion test
TEST_FUSION_VERSION = "20260417_factor_fusion_test"
TEST_FUSION_FC_NAME = "fusion_avg_weight_4"
TEST_FUSION_COLLECTION = "factor_fusion"

# Key metrics to compare
COMPARE_METRICS = [
    "Gross Sharpe",
    "Net Sharpe",
    "TS IC",
    "TS RankIC",
]


# ── Helpers ───────────────────────────────────────────────────────────

def safe_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_formula_from_db(fc_name: str, version: str, collection: str) -> str:
    """Retrieve factor formula from MongoDB."""
    from data import get_factor_formula_map_by_version

    formula_map = get_factor_formula_map_by_version(
        fc_name_list=[fc_name],
        version=version,
        collections=[collection],
    )
    formula = formula_map.get(fc_name, "").strip()
    if not formula:
        raise ValueError(
            f"Formula not found for {fc_name} in {collection}@{version}"
        )
    return formula


def run_backtester_by_fc_name(
    fc_name: str,
    version: str,
    collection: str,
    instrument_id: str = "C0",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Method 1: BackTester with fc_name_list (database factor)."""
    from factors.backtest import BackTester

    bt = BackTester(
        fc_name_list=[fc_name],
        version=version,
        collection=collection,
        instrument_id_list=instrument_id,
        fc_freq="1d",
        start_time=IS_START,
        end_time=FULL_END,
        portfolio_adjust_method="1D",
        interest_method="simple",
        risk_free_rate=False,
        calculate_baseline=True,
        apply_weighted_price=True,
        n_jobs=5,
    )
    bt.backtest()
    return bt.performance_detail, bt.performance_summary


def run_backtester_by_formula(
    formula: str,
    instrument_id: str = "C0",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Method 2: BackTester with formula string."""
    from factors.backtest import BackTester

    bt = BackTester(
        formula=formula,
        version="__formula__",
        collection="genetic_programming",
        instrument_id_list=instrument_id,
        fc_freq="1d",
        start_time=IS_START,
        end_time=FULL_END,
        portfolio_adjust_method="1D",
        interest_method="simple",
        risk_free_rate=False,
        calculate_baseline=True,
        apply_weighted_price=True,
        n_jobs=5,
    )
    bt.backtest()
    return bt.performance_detail, bt.performance_summary


def _build_summary_by_range(
    summary: pd.DataFrame,
    fc_name: str,
    start_day: str,
    end_day: str,
) -> pd.DataFrame:
    """Extract performance summary for a factor within a date range (by year)."""
    start_year = int(str(start_day)[:4])
    end_year = int(str(end_day)[:4])

    # year is in the index, reset it to a column
    df = summary.reset_index()
    if "year" not in df.columns:
        # fallback: index name is 'year'
        df = df.rename(columns={df.columns[0]: "year"})

    scoped = df[df["Factor Name"] == fc_name].copy()
    if scoped.empty:
        return scoped

    scoped["year_num"] = pd.to_numeric(scoped["year"], errors="coerce")
    scoped = scoped[
        (scoped["year_num"] >= start_year) & (scoped["year_num"] <= end_year)
    ]
    scoped = scoped.drop(columns=["year_num"])

    # If only one year row, add an 'all' row
    if len(scoped) == 1:
        all_row = scoped.copy()
        all_row["year"] = "all"
        scoped = pd.concat([scoped, all_row], ignore_index=True)

    return scoped


def extract_metrics(summary: pd.DataFrame, fc_name: str) -> Dict[str, float]:
    """Extract key metrics (Gross Sharpe, Net Sharpe, TS IC, TS RankIC) for 'all' year."""
    # year is in the index
    df = summary.reset_index()
    if "year" not in df.columns:
        df = df.rename(columns={df.columns[0]: "year"})

    all_rows = df[df["year"] == "all"]
    if all_rows.empty:
        all_rows = df[df["Factor Name"] == fc_name]
    if all_rows.empty:
        return {}

    row = all_rows.iloc[0]
    return {m: safe_float(row[m]) for m in COMPARE_METRICS}


def compare_metrics(
    m1: Dict[str, float],
    m2: Dict[str, float],
    name1: str = "Method1",
    name2: str = "Method2",
) -> List[str]:
    """Compare two metric dicts. Return list of error messages (empty = pass)."""
    errors = []
    for metric in COMPARE_METRICS:
        v1 = m1.get(metric)
        v2 = m2.get(metric)
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            errors.append(
                f"  {metric}: {name1}={v1}, {name2}={v2} (one is None)"
            )
            continue
        diff = abs(v1 - v2)
        rel_diff = diff / (abs(v1) + EPSILON)
        if diff > EPSILON and rel_diff > EPSILON:
            errors.append(
                f"  {metric}: {name1}={v1:.6f}, {name2}={v2:.6f}, "
                f"diff={diff:.6f}, rel_diff={rel_diff:.6f}"
            )
    return errors


# ── Backtest Test ─────────────────────────────────────────────────────

def test_single_factor_backtest() -> Tuple[bool, str]:
    """Test single-factor backtest consistency across 4 methods."""
    print("\n" + "=" * 60)
    print("TEST 1: Single-Factor Backtest (fac_gp_0041)")
    print("=" * 60)

    # Get formula from DB
    try:
        formula = get_formula_from_db(
            TEST_SINGLE_FC_NAME, TEST_SINGLE_VERSION, TEST_SINGLE_COLLECTION
        )
        print(f"  Formula from DB: {formula[:80]}...")
    except Exception as e:
        return False, f"Failed to get formula from DB: {e}"

    # ── Method 1: BackTester with fc_name ──────────────────────────────
    print("\n  [Method 1] BackTester with fc_name_list...")
    try:
        _, summary_1 = run_backtester_by_fc_name(
            TEST_SINGLE_FC_NAME,
            TEST_SINGLE_VERSION,
            TEST_SINGLE_COLLECTION,
        )
    except Exception as e:
        return False, f"Method 1 (fc_name) failed: {e}"

    # ── Method 2: BackTester with formula ──────────────────────────────
    print("  [Method 2] BackTester with formula...")
    try:
        _, summary_2 = run_backtester_by_formula(formula)
    except Exception as e:
        return False, f"Method 2 (formula) failed: {e}"

    # ── Method 3: Frontend DB mode ────────────────────────────────────
    # Frontend DB mode calls /api/backtest with fc_name_list (no formula)
    print("  [Method 3] Simulate frontend DB mode (no formula)...")
    from factors.backtest import BackTester

    try:
        bt3 = BackTester(
            fc_name_list=[TEST_SINGLE_FC_NAME],
            version=TEST_SINGLE_VERSION,
            collection=TEST_SINGLE_COLLECTION,
            instrument_id_list="C0",
            fc_freq="1d",
            start_time=IS_START,
            end_time=FULL_END,
            portfolio_adjust_method="1D",
            interest_method="simple",
            risk_free_rate=False,
            calculate_baseline=True,
            apply_weighted_price=True,
            n_jobs=5,
        )
        bt3.backtest()
        _, summary_3 = bt3.performance_detail, bt3.performance_summary
    except Exception as e:
        return False, f"Method 3 (frontend DB) failed: {e}"

    # ── Method 4: Frontend formula mode ────────────────────────────────
    # Frontend formula mode calls /api/backtest with formula (version=None, fc_name_list=[])
    # The backend maps this to: formula=formula, version="__formula__", fc_name_list=["formula_factor"]
    print("  [Method 4] Simulate frontend formula mode (with formula)...")
    try:
        bt4 = BackTester(
            formula=formula,
            version="__formula__",
            collection=TEST_SINGLE_COLLECTION,
            instrument_id_list="C0",
            fc_freq="1d",
            start_time=IS_START,
            end_time=FULL_END,
            portfolio_adjust_method="1D",
            interest_method="simple",
            risk_free_rate=False,
            calculate_baseline=True,
            apply_weighted_price=True,
            n_jobs=5,
        )
        bt4.backtest()
        _, summary_4 = bt4.performance_detail, bt4.performance_summary
    except Exception as e:
        return False, f"Method 4 (frontend formula) failed: {e}"

    # ── Compare IS / OOS / Real-OOS metrics ───────────────────────────
    summaries = {
        "Method1 (fc_name)": summary_1,
        "Method2 (formula)": summary_2,
        "Method3 (frontend_DB)": summary_3,
        "Method4 (frontend_formula)": summary_4,
    }

    ranges = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("Real-OOS", REAL_OOS_START, REAL_OOS_END),
    ]

    for range_name, start_day, end_day in ranges:
        print(f"\n  ── {range_name} ({start_day} ~ {end_day}) ──")
        all_errors = []

        # Extract metrics for each method
        method_metrics = {}
        for method_name, summary in summaries.items():
            # Formula mode renames factor to "formula_factor"
            fc_name_to_use = "formula_factor" if "formula" in method_name else TEST_SINGLE_FC_NAME
            range_summary = _build_summary_by_range(
                summary, fc_name_to_use, start_day, end_day
            )
            metrics = extract_metrics(range_summary, fc_name_to_use)
            method_metrics[method_name] = metrics
            print(f"    {method_name}: {metrics}")

        # Compare all pairs
        method_names = list(method_metrics.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                errs = compare_metrics(
                    method_metrics[m1], method_metrics[m2], name1=m1, name2=m2
                )
                if errs:
                    all_errors.extend(
                        [f"  [{range_name}] {m1} vs {m2}:"]
                        + errs
                    )

        if all_errors:
            return False, (
                f"Single-factor {range_name} metrics mismatch:\n"
                + "\n".join(all_errors)
            )

    print(f"\n  ✓ Single-factor backtest: all ranges, all methods PASS")
    return True, ""


def test_fusion_factor_backtest() -> Tuple[bool, str]:
    """Test fusion-factor backtest consistency across 4 methods."""
    print("\n" + "=" * 60)
    print("TEST 2: Fusion-Factor Backtest (fusion_avg_weight_4)")
    print("=" * 60)

    # Get formula from DB
    try:
        formula = get_formula_from_db(
            TEST_FUSION_FC_NAME, TEST_FUSION_VERSION, TEST_FUSION_COLLECTION
        )
        print(f"  Formula from DB: {formula[:80]}...")
    except Exception as e:
        return False, f"Failed to get fusion formula from DB: {e}"

    # ── Method 1: BackTester with fc_name ──────────────────────────────
    print("\n  [Method 1] BackTester with fc_name_list...")
    try:
        _, summary_1 = run_backtester_by_fc_name(
            TEST_FUSION_FC_NAME,
            TEST_FUSION_VERSION,
            TEST_FUSION_COLLECTION,
        )
    except Exception as e:
        return False, f"Fusion Method 1 (fc_name) failed: {e}"

    # ── Method 2: BackTester with formula ──────────────────────────────
    print("  [Method 2] BackTester with formula...")
    try:
        _, summary_2 = run_backtester_by_formula(formula)
    except Exception as e:
        return False, f"Fusion Method 2 (formula) failed: {e}"

    # ── Method 3: Frontend DB mode ────────────────────────────────────
    print("  [Method 3] Simulate frontend DB mode (no formula)...")
    from factors.backtest import BackTester

    try:
        bt3 = BackTester(
            fc_name_list=[TEST_FUSION_FC_NAME],
            version=TEST_FUSION_VERSION,
            collection=TEST_FUSION_COLLECTION,
            instrument_id_list="C0",
            fc_freq="1d",
            start_time=IS_START,
            end_time=FULL_END,
            portfolio_adjust_method="1D",
            interest_method="simple",
            risk_free_rate=False,
            calculate_baseline=True,
            apply_weighted_price=True,
            n_jobs=5,
        )
        bt3.backtest()
        _, summary_3 = bt3.performance_detail, bt3.performance_summary
    except Exception as e:
        return False, f"Fusion Method 3 (frontend DB) failed: {e}"

    # ── Method 4: Frontend formula mode ────────────────────────────────
    print("  [Method 4] Simulate frontend formula mode (with formula)...")
    try:
        bt4 = BackTester(
            formula=formula,
            version="__formula__",
            collection=TEST_FUSION_COLLECTION,
            instrument_id_list="C0",
            fc_freq="1d",
            start_time=IS_START,
            end_time=FULL_END,
            portfolio_adjust_method="1D",
            interest_method="simple",
            risk_free_rate=False,
            calculate_baseline=True,
            apply_weighted_price=True,
            n_jobs=5,
        )
        bt4.backtest()
        _, summary_4 = bt4.performance_detail, bt4.performance_summary
    except Exception as e:
        return False, f"Fusion Method 4 (frontend formula) failed: {e}"

    # ── Compare IS / OOS / Real-OOS metrics ───────────────────────────
    summaries = {
        "Method1 (fc_name)": summary_1,
        "Method2 (formula)": summary_2,
        "Method3 (frontend_DB)": summary_3,
        "Method4 (frontend_formula)": summary_4,
    }

    ranges = [
        ("IS", IS_START, IS_END),
        ("OOS", OOS_START, OOS_END),
        ("Real-OOS", REAL_OOS_START, REAL_OOS_END),
    ]

    for range_name, start_day, end_day in ranges:
        print(f"\n  ── {range_name} ({start_day} ~ {end_day}) ──")
        all_errors = []

        method_metrics = {}
        for method_name, summary in summaries.items():
            fc_name_to_use = (
                "formula_factor"
                if "formula" in method_name
                else TEST_FUSION_FC_NAME
            )
            range_summary = _build_summary_by_range(
                summary, fc_name_to_use, start_day, end_day
            )
            metrics = extract_metrics(range_summary, fc_name_to_use)
            method_metrics[method_name] = metrics
            print(f"    {method_name}: {metrics}")

        method_names = list(method_metrics.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                errs = compare_metrics(
                    method_metrics[m1], method_metrics[m2], name1=m1, name2=m2
                )
                if errs:
                    all_errors.extend(
                        [f"  [{range_name}] {m1} vs {m2}:"]
                        + errs
                    )

        if all_errors:
            return False, (
                f"Fusion-factor {range_name} metrics mismatch:\n"
                + "\n".join(all_errors)
            )

    print(f"\n  ✓ Fusion-factor backtest: all ranges, all methods PASS")
    return True, ""


# ── Build Test ─────────────────────────────────────────────────────────

def test_frontend_build() -> Tuple[bool, str]:
    """Test frontend build succeeds."""
    print("\n" + "=" * 60)
    print("TEST 3: Frontend Build")
    print("=" * 60)

    frontend_dir = PROJECT_ROOT / "web" / "frontend"
    if not frontend_dir.exists():
        return False, f"Frontend directory not found: {frontend_dir}"

    print(f"  Running npm run build in {frontend_dir}...")
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "Frontend build timed out (> 5 min)"
    except Exception as e:
        return False, f"Frontend build failed to start: {e}"

    if result.returncode != 0:
        return False, (
            f"Frontend build failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists():
        return False, f"Frontend build succeeded but dist/ not found"

    index_html = dist_dir / "index.html"
    if not index_html.exists():
        return False, f"Frontend build succeeded but dist/index.html not found"

    print(f"  ✓ Frontend build PASS (dist/ generated)")
    return True, ""


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  Lionet Integration Test Suite                       ║")
    print("╚════════════════════════════════════════════════════════╝")

    results = []
    failures = []

    # Test 1
    ok, err = test_single_factor_backtest()
    results.append(("Single-Factor Backtest", ok, err))
    if not ok:
        failures.append(("Single-Factor Backtest", err))
        print(f"\n  ✗ FAIL: {err[:500]}")
    else:
        print(f"\n  ✓ PASS")

    # Test 2
    ok, err = test_fusion_factor_backtest()
    results.append(("Fusion-Factor Backtest", ok, err))
    if not ok:
        failures.append(("Fusion-Factor Backtest", err))
        print(f"\n  ✗ FAIL: {err[:500]}")
    else:
        print(f"\n  ✓ PASS")

    # Test 3
    ok, err = test_frontend_build()
    results.append(("Frontend Build", ok, err))
    if not ok:
        failures.append(("Frontend Build", err))
        print(f"\n  ✗ FAIL: {err[:500]}")
    else:
        print(f"\n  ✓ PASS")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok, err in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False

    if failures:
        print("\n" + "-" * 60)
        print("FAILURES")
        print("-" * 60)
        for name, err in failures:
            print(f"\n[{name}]")
            print(err[:1000])

    if all_pass:
        print("RESULT: true  (all tests passed)")
    else:
        print("RESULT: false (some tests failed)")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
