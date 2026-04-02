"""Quick check for relative-check similarity on same formula/data.

This script prints Spearman correlation between a freshly generated factor
and the same formula recomputed via calc_formula_df on the same base data.
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator
from factors.factor_ops import calc_formula_df


def main():
    fg = GeneticFactorGenerator(
        instrument_id_list='C0',
        start_time='20200101',
        end_time='20241231',
        version='relative_check_verify',
        calculate_baseline=True,
        max_factor_count=10,
        rolling_norm_clip=5.0,
        gp_generations=2,
        fitness_metric='ic',
        gp_max_depth=4,
        gp_population_size=20,
        gp_elite_size=5,
        gp_elite_relative_threshold=0.75,
        gp_crossover_prob=0.7,
        gp_leaf_prob=0.3,
        gp_const_prob=0.02,
        gp_window_choices=[5, 10, 20],
        gp_depth_penalty_coef=0.0,
        gp_depth_penalty_start_depth=4,
        gp_depth_penalty_linear_coef=0.05,
        gp_depth_penalty_quadratic_coef=0.0,
        gp_early_stopping_generation_count=2,
        random_seed=42,
        gp_assumed_initial_capital=100000,
        gp_small_factor_penalty_coef=0.0,
        gp_elite_stagnation_generation_count=5,
        gp_max_shock_generation=3,
    )

    generated = fg.generate()
    if not fg.generated_fc_name_list:
        print('No factors generated.')
        return

    fc_name = fg.generated_fc_name_list[0]
    formula = fg.factor_formula_map[fc_name]
    print('Selected factor:', fc_name)
    print('Formula:', formula)

    base_df = fg.load_base_data()
    df_eval = fg._prepare_df_for_gp(base_df)

    # Recompute the same formula on the same data for comparison.
    recomputed = calc_formula_df(
        df=df_eval,
        formula_map={'recomputed': formula},
        data_fields=fg.base_col_list,
    )

    left = generated[['time', 'instrument_id', fc_name]].copy()
    right = recomputed[['time', 'instrument_id', 'recomputed']].copy()
    merged = left.merge(right, on=['time', 'instrument_id'], how='inner')

    series_a = pd.to_numeric(merged[fc_name], errors='coerce')
    series_b = pd.to_numeric(merged['recomputed'], errors='coerce')

    corr = pd.concat([series_a, series_b], axis=1).corr(method='spearman').iloc[0, 1]
    print('Spearman corr (same formula, same data):', corr)
    print('Merged rows:', len(merged))


if __name__ == '__main__':
    main()
