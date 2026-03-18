import pandas as pd

from factors.factor_auto_search import FactorGenerator, TsfreshFactorGenerator


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
    fg = TsfreshFactorGenerator(
        instrument_id_list='C0',
        data=data,
        fc_freq='1d',
        portfolio_adjust_method='1D',
        min_window_size=20,
        max_factor_count=8,
        tsfresh_profile='minimal',
        version='smoke_20260317_000000',
        n_jobs=1,
    )
    generated = fg.generate()
    print('generated shape:', generated.shape)
    print('factor count:', len(fg.generated_fc_name_list))

    selected = fg.generated_fc_name_list[:2]
    saved_path = fg.save_fc_value(selected, file_name='smoke_tsfresh_factor', file_format='csv')
    print('saved path:', saved_path)

    fc_config_path = fg.save_fc(selected)
    loaded_fc = FactorGenerator.load_fc(fc_config_path)
    generated_subset = fg.generate_with_fc(loaded_fc)
    print('saved fc config path:', fc_config_path)
    print('generated subset shape:', generated_subset.shape)

    bt2 = fg.backtest_from_fc_config(fc_config_path, n_jobs=1)
    print('one-step backtest summary rows:', len(bt2.performance_summary))

    auto_result = fg.auto_mine_select_and_save_fc(
        net_ret_threshold=-1.0,
        sharpe_threshold=-1.0,
        n_jobs=1,
        require_all_instruments=False,
    )
    print('auto selected factor count:', len(auto_result['selected_fc_name_list']))
    print('auto config path:', auto_result['config_path'])

    leakage_check = fg.check_if_leakage(fc_name_list=selected, raise_error=True)
    print('leakage check:', leakage_check)

    bt = fg.backtest(fc_name_list=selected, n_jobs=1)
    print('backtest summary head:')
    print(bt.performance_summary.head())

