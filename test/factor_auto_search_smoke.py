import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.factor_auto_search import GeneticFactorGenerator


def build_mock_data(rows: int = 80) -> pd.DataFrame:
    time_idx = pd.date_range('2024-01-01', periods=rows, freq='D')
    close = pd.Series(range(100, 100 + rows), dtype=float)
    df = pd.DataFrame({
        'time': time_idx,
        'instrument_id': ['C0'] * rows,
        'open': close,
        'high': close * 1.01,
        'low': close * 0.99,
        'close': close,
        'volume': 1000 + close,
        'position': 500 + close,
    })
    return df


if __name__ == '__main__':
    data = build_mock_data()
    fg = GeneticFactorGenerator(
        instrument_id_list='C0',
        data=data,
        fc_freq='1d',
        portfolio_adjust_method='1D',
        min_window_size=20,
        max_factor_count=6,
        gp_generations=3,
        gp_population_size=16,
        gp_elite_size=4,
        gp_tournament_size=3,
        version='smoke_20260317_000000',
        n_jobs=1,
    )
    generated = fg.generate()
    print('generated shape:', generated.shape)
    print('factor count:', len(fg.generated_fc_name_list))

    selected = fg.generated_fc_name_list[:2]
    saved_path = fg.save_fc_value(selected, file_name='smoke_gp_factor', file_format='csv')
    print('saved path:', saved_path)


    leakage_check = fg.check_if_leakage(fc_name_list=selected, raise_error=True)
    print('leakage check:', leakage_check)

    bt = fg.backtest(fc_name_list=selected, n_jobs=1)
    print('backtest summary head:')
    print(bt.performance_summary.head())

