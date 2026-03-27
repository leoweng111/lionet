"""
This script contains methods for calculating factor indicators.
"""
import pandas as pd
import numpy as np
from typing import Union
from joblib import delayed, Parallel
from stats import merge_dataframe
from utils.params import FEE

EPSILON = 1e-6


def _safe_corr(x: pd.Series,
               y: pd.Series,
               method: str = 'pearson') -> float:
    """Return correlation without triggering warnings on constant/empty inputs."""
    s = pd.concat([pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')], axis=1).dropna()
    if s.empty:
        return np.nan
    if s.iloc[:, 0].nunique(dropna=True) <= 1:
        return np.nan
    if s.iloc[:, 1].nunique(dropna=True) <= 1:
        return np.nan
    return float(s.iloc[:, 0].corr(s.iloc[:, 1], method=method))


def _get_fee_for_instrument(Data: pd.DataFrame) -> float:
    instrument_ids = Data['instrument_id'].dropna().unique().tolist()
    if len(instrument_ids) != 1:
        raise ValueError('Fee lookup expects data of one instrument only.')
    instrument_id = instrument_ids[0]
    if instrument_id not in FEE:
        raise ValueError(f'Missing fee config for instrument_id={instrument_id}. Please update utils/params.py FEE.')
    return float(FEE[instrument_id])


def get_annualized_ts_ic_and_t_corr(Data: pd.DataFrame,
                                    fc_col: str,
                                    fc_freq: str,
                                    portfolio_adjust_method: str):
    """
    计算时序IC和t_corr。

    :param Data:
    :param fc_col:
    :param ret_freq:
    :return:
    """
    for col in ['time', 'instrument_id', 'future_ret', fc_col]:
        assert col in Data.columns, f'df does not contain columns {col}.'

    df = Data.copy()
    if df['instrument_id'].nunique() != 1:
        raise ValueError('Time-series IC must be calculated on one instrument at a time.')
    df = df.sort_values(by='time')
    df['year'] = pd.to_datetime(df['time']).dt.year

    ic_ts_year = df.groupby('year')[[fc_col, 'future_ret']].apply(
        lambda g: _safe_corr(g[fc_col], g['future_ret'], method='pearson')
    )
    ic_ts_all = pd.Series(_safe_corr(df[fc_col], df['future_ret'], method='pearson'), index=['all'])
    ic_ts = pd.concat([ic_ts_year, ic_ts_all]).to_frame('TS IC')

    rank_ic_ts_year = df.groupby('year')[[fc_col, 'future_ret']].apply(
        lambda g: _safe_corr(g[fc_col], g['future_ret'], method='spearman')
    )
    rank_ic_ts_all = pd.Series(_safe_corr(df[fc_col], df['future_ret'], method='spearman'), index=['all'])
    rank_ic_ts = pd.concat([rank_ic_ts_year, rank_ic_ts_all]).to_frame('TS RankIC')

    ic_ts = pd.concat([ic_ts, rank_ic_ts], axis=1)
    ic_ts.index.name = 'year'
    if portfolio_adjust_method == '1D':
        ret_freq = 1
    elif portfolio_adjust_method == '1M':
        ret_freq = 30
    else:
        ret_freq = 1

    t_corr_year = df.groupby('year')['future_ret'].apply(lambda x: np.sqrt(x.size / ret_freq))
    t_corr_all = pd.Series(len(df) / ret_freq, index=['all'])
    t_corr = pd.concat([t_corr_year, t_corr_all]).to_frame('T-corr')
    t_corr.index.name = 'year'

    return ic_ts, t_corr


def get_ts_ret_and_turnover(Data: pd.DataFrame,
                            fc_col: Union[str, list]):
    # time series gross ret and net ret
    # time series way of calculating ret and turnover is simpler
    if isinstance(fc_col, str):
        fc_col = [fc_col]
    for col in ['time', 'instrument_id', 'future_ret'] + fc_col:
        assert col in Data.columns, f'Data does not contain column {col}.'

    df = Data.copy()
    if df['instrument_id'].nunique() != 1:
        raise ValueError('Time-series return/turnover must be calculated on one instrument at a time.')
    fee = _get_fee_for_instrument(df)

    # if the factor value or label is missing, keep the position of previous day
    df[fc_col] = df[fc_col].ffill().fillna(0)
    df['future_ret'] = df['future_ret'].fillna(0)
    df = df.set_index('time')
    # df = df.dropna(subset=fc_col + ['future_ret']).sort_values(by='time').set_index('time')

    # For a single instrument, TS gross return is signal_t * future_ret_t.
    df_gross_ret_ts = df[fc_col].mul(df['future_ret'], axis=0)
    # Physical turnover:
    # - Normal day: |position_t - position_{t-1}|
    # - Rollover day: close old + open new => |position_{t-1}| + |position_t|
    normal_turnover = df[fc_col].diff().abs()
    rollover_turnover = df[fc_col].shift(1).abs() + df[fc_col].abs()
    if 'is_rollover' in df.columns:
        rollover_mask = df['is_rollover'].fillna(False).astype(bool)
        turnover_values = np.where(
            rollover_mask.to_numpy().reshape(-1, 1),
            rollover_turnover.to_numpy(),
            normal_turnover.to_numpy(),
        )
        df_turnover_ts = pd.DataFrame(turnover_values, index=normal_turnover.index, columns=normal_turnover.columns)
    else:
        df_turnover_ts = normal_turnover

    first_row_idx = df_turnover_ts.index[0]
    df_turnover_ts.loc[first_row_idx] = df[fc_col].loc[first_row_idx].abs()

    # df['prev_fc_col'] = df[fc_col].shift(1).fillna(0)
    df_net_ret_ts = df_gross_ret_ts - df_turnover_ts * fee
    # These three dataframe should not contain any nan values.
    return df_gross_ret_ts, df_net_ret_ts, df_turnover_ts


def get_annualized_ret(Data: pd.DataFrame,
                       ret_col: Union[str, list],
                       interest_method: str = 'simple'):
    """
    Get annualized return for every year in data and total annualized return.

    :param Data: Data should be a dataframe with columns: 'time' and ret_col.
        It can represent one strategy return series or multiple factor return columns.
    :param ret_col: columns of Data which stand for ret
    :param interest_method: can be 'simple' or 'compound'
    :return:
    """
    if isinstance(ret_col, str):
        ret_col = [ret_col]
    for col in ['time'] + ret_col:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)
    df['year'] = pd.to_datetime(df['time']).dt.year

    if interest_method == 'simple':
        ret_year = df.groupby('year')[ret_col].mean() * 252
        ret_all = (df[ret_col].mean() * 252).to_frame('all').T
    # 复利的做法更加常见，代表着每日将收益或亏损计入总资金量，之后的调仓根据总资金量和因子值变动。
    elif interest_method == 'compound':
        df = df.set_index('time')
        df_cumret = (1 + df[ret_col]).cumprod().reset_index()
        df_cumret['year'] = pd.to_datetime(df_cumret['time']).dt.year

        ret_year = df_cumret.groupby('year')[ret_col].apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) ** (252 / len(x)) - 1)
        ret_all = (df_cumret[ret_col].iloc[-1] ** (252 / len(df_cumret[ret_col])) - 1).to_frame('all').T
    else:
        raise NotImplementedError(f'Does not support method {interest_method}.')

    ret = pd.concat([ret_year, ret_all])
    ret.index.name = 'year'

    return ret


def get_annualized_volatility(Data: pd.DataFrame,
                              ret_col: Union[str, list]):
    """

    :param Data:
    :param ret_col:
    :return:
    """
    if isinstance(ret_col, str):
        ret_col = [ret_col]
    for col in ['time'] + ret_col:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)
    df['year'] = pd.to_datetime(df['time']).dt.year

    volatility_year = df.groupby('year')[ret_col].std() * np.sqrt(252)
    volatility_all = (df[ret_col].std() * np.sqrt(252)).to_frame('all').T
    volatility = pd.concat([volatility_year, volatility_all])
    volatility.index.name = 'year'

    return volatility


def get_annualized_turnover(Data: pd.DataFrame,
                            ret_col: Union[str, list]):
    """

    :param Data:
    :param ret_col:
    :return:
    """
    if isinstance(ret_col, str):
        ret_col = [ret_col]
    for col in ['time'] + ret_col:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)
    df['year'] = pd.to_datetime(df['time']).dt.year

    turnover_year = df.groupby('year')[ret_col].mean()
    turnover_all = (df[ret_col].mean()).to_frame('all').T
    turnover = pd.concat([turnover_year, turnover_all])
    turnover.index.name = 'year'

    return turnover


def get_annualized_sharpe(ret: pd.DataFrame,
                          volatility: pd.DataFrame):
    """
    Calculate annualized sharpe, does not consider risk-free rate.
    :param ret:
    :param volatility:
    :return:
    """
    assert ret.shape == volatility.shape
    assert all(ret.index == volatility.index)
    assert all(ret.columns == volatility.columns)

    sharpe = ret / volatility
    sharpe.index.name = 'year'

    return sharpe


def get_annualized_sortino_ratio(Data: pd.DataFrame,
                                 ret_col: Union[str, list],
                                 interest_method: str = 'simple'):
    """

    :param Data:
    :param ret_col:
    :param interest_method:
    :return:
    """
    if isinstance(ret_col, str):
        ret_col = [ret_col]
    # 采用下行波动率计算
    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)

    volatility_list = []
    ret = get_annualized_ret(df, ret_col, interest_method)
    for col in ret_col:
        df_down = df.loc[df[col] < 0]
        volatility = get_annualized_volatility(df_down, [col])
        volatility_list.append(volatility)

    volatility = pd.concat(volatility_list, axis=1)
    sortino_ratio = ret / volatility
    sortino_ratio.index.name = 'year'

    return sortino_ratio


def get_annualized_drawdown(Data: pd.DataFrame,
                            ret_col: Union[str, list]):
    """
    Get drawdown by compound rate.
    """
    # maximum_drawdown = max((nav - max(nav)) / max(nav))
    # if lost all money, just stop. This algorithm does not consider adding extra capital.
    # So the maximum drawdown is 1.
    if isinstance(ret_col, str):
        ret_col = [ret_col]

    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)

    nav = (1 + df[ret_col]).cumprod()
    dd = 1 - nav / nav.cummax()

    df['year'] = pd.to_datetime(df['time']).dt.year
    max_dd_year = df.groupby('year')[ret_col].apply(_get_max_dd)
    max_dd_all = (dd.max()).to_frame('all').T
    max_dd = pd.concat([max_dd_year, max_dd_all])
    max_dd.index.name = 'year'

    return dd, max_dd


def _get_max_dd(ret: pd.DataFrame):
    ret_nav = (1 + ret).cumprod()
    dd = 1 - ret_nav / ret_nav.cummax()
    return dd.max()


def get_annualized_calmar_ratio(ret: pd.DataFrame,
                                max_dd: pd.DataFrame):
    assert ret.shape == max_dd.shape
    assert all(ret.index == max_dd.index)
    assert all(ret.columns == max_dd.columns)

    calmar_ratio = ret / max_dd
    calmar_ratio.index.name = 'year'

    return calmar_ratio


def get_annualized_win_rate(Data: pd.DataFrame,
                            ret_col: Union[str, list]):
    """
    Get win rate, which is the ratio of profitable trades.
    """
    if isinstance(ret_col, str):
        ret_col = [ret_col]
    df = Data.copy()
    df = df.sort_values(by='time', ascending=True)
    df['year'] = pd.to_datetime(df['time']).dt.year

    win_rate_list = []
    for col in ret_col:
        win_rate_list.append(df.groupby('year')[col].apply(lambda x: len(x[x > 0]) / len(x)))
    win_rate_year = pd.concat(win_rate_list, axis=1)

    win_rate_all = (df[ret_col].apply(lambda x: len(x[x > 0]) / len(x))).to_frame('all').T
    win_rate = pd.concat([win_rate_year, win_rate_all])
    win_rate.index.name = 'year'

    return win_rate


def get_annualized_ts_instrument_count(Data: pd.DataFrame):
    """
    Get time-series annualized instrument count
    :param
    :return:
    """
    for col in ['time', 'instrument_id']:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    df['year'] = pd.to_datetime(df['time']).dt.year

    count_year = df.groupby('year')['instrument_id'].nunique().to_frame('count')
    count_all = pd.Series(df['instrument_id'].nunique(), index=['all']).to_frame('count').T

    instrument_count = pd.concat([count_year, count_all])
    instrument_count.index.name = 'year'

    return instrument_count


def get_performance(Data: pd.DataFrame,
                    fc_name_list: Union[str, list],
                    fc_freq: str = '1d',
                    portfolio_adjust_method: str = '1D',
                    interest_method: str = 'simple',
                    calculate_baseline: bool = False,
                    n_jobs=5):

    if isinstance(fc_name_list, str):
        fc_name_list = [fc_name_list]
    for col in ['time', 'instrument_id', 'future_ret'] + fc_name_list:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    if df['instrument_id'].nunique() != 1:
        raise ValueError('get_performance expects Data of one instrument only.')
    f = lambda x: get_performance_for_one_factor(df,
                                                 x,
                                                 fc_freq,
                                                 portfolio_adjust_method,
                                                 interest_method,
                                                 calculate_baseline=calculate_baseline
                                                 )

    with Parallel(n_jobs=n_jobs) as parallel:
        performance_list = parallel(delayed(f)(fc_name) for fc_name in fc_name_list)

    performance_dc_list = [x[0] for x in performance_list]
    performance_summary_list = [x[1] for x in performance_list]

    performance_dc = {k: v for k, v in zip(fc_name_list, performance_dc_list)}
    performance_summary = pd.concat(performance_summary_list)

    return performance_dc, performance_summary


def get_performance_for_one_factor(Data: pd.DataFrame,
                                   fc_name: str,
                                   fc_freq: str = '1d',
                                   portfolio_adjust_method: str = '1D',
                                   interest_method: str = 'simple',
                                   calculate_baseline: bool = False):
    """
    Get time-series performance for one factor in single-instrument strategy research.
    """
    df = Data.copy()
    fee = _get_fee_for_instrument(df)

    # time-series ret and turnover
    df_gross_ret_ts, df_net_ret_ts, df_turnover_ts = get_ts_ret_and_turnover(df, fc_name)
    df_gross_ret_ts = df_gross_ret_ts.reset_index()
    df_net_ret_ts = df_net_ret_ts.reset_index()
    df_turnover_ts = df_turnover_ts.reset_index()

    # Gross Performance
    gross_annualized_ret_ts = get_annualized_ret(df_gross_ret_ts, fc_name, interest_method)
    gross_annualized_volatility_ts = get_annualized_volatility(df_gross_ret_ts, fc_name)
    gross_annualized_sharpe_ts = get_annualized_sharpe(gross_annualized_ret_ts, gross_annualized_volatility_ts)
    gross_annualized_sortino_ratio_ts = get_annualized_sortino_ratio(df_gross_ret_ts, fc_name, interest_method)
    gross_drawdown_ts, gross_annualized_max_drawdown_ts = get_annualized_drawdown(df_gross_ret_ts, fc_name)
    gross_annualized_calmar_ratio_ts = \
        get_annualized_calmar_ratio(gross_annualized_ret_ts, gross_annualized_volatility_ts)
    gross_annualized_win_rate_ts = get_annualized_win_rate(df_gross_ret_ts, fc_name)

    # Net Performance
    net_annualized_ret_ts = get_annualized_ret(df_net_ret_ts, fc_name, interest_method)
    net_annualized_volatility_ts = get_annualized_volatility(df_net_ret_ts, fc_name)
    net_annualized_sharpe_ts = get_annualized_sharpe(net_annualized_ret_ts, net_annualized_volatility_ts)
    net_annualized_sortino_ratio_ts = get_annualized_sortino_ratio(df_net_ret_ts, fc_name, interest_method)
    net_drawdown_ts, net_annualized_max_drawdown_ts = get_annualized_drawdown(df_net_ret_ts, fc_name)
    net_annualized_calmar_ratio_ts = get_annualized_calmar_ratio(net_annualized_ret_ts, net_annualized_volatility_ts)
    net_annualized_win_rate_ts = get_annualized_win_rate(df_net_ret_ts, fc_name)

    # IC, ICIR , turnover and instrument count does not depend on fee
    annualized_ic_ts, annualized_tcorr_ts = \
        get_annualized_ts_ic_and_t_corr(df, fc_name, fc_freq, portfolio_adjust_method)
    annualized_turnover_ts = get_annualized_turnover(df_turnover_ts, fc_name)
    annualized_instrument_count_ts = get_annualized_ts_instrument_count(df)

    ts_performance_dc = {'gross_annualized_ret': gross_annualized_ret_ts,
                         'gross_annualized_volatility': gross_annualized_volatility_ts,
                         'gross_annualized_sharpe': gross_annualized_sharpe_ts,
                         'gross_annualized_sortino': gross_annualized_sortino_ratio_ts,
                         'gross_annualized_max_drawdown': gross_annualized_max_drawdown_ts,
                         'gross_drawdown': gross_drawdown_ts,
                         'gross_annualized_calmar': gross_annualized_calmar_ratio_ts,
                         'gross_annualized_win_rate': gross_annualized_win_rate_ts,

                         'net_annualized_ret': net_annualized_ret_ts,
                         'net_annualized_volatility': net_annualized_volatility_ts,
                         'net_annualized_sharpe': net_annualized_sharpe_ts,
                         'net_annualized_sortino': net_annualized_sortino_ratio_ts,
                         'net_annualized_max_drawdown': net_annualized_max_drawdown_ts,
                         'net_drawdown': net_drawdown_ts,
                         'net_annualized_calmar': net_annualized_calmar_ratio_ts,
                         'net_annualized_win_rate': net_annualized_win_rate_ts,

                         # data which does not depend on fee
                         'annualized_turnover': annualized_turnover_ts,
                         'annualized_ic': annualized_ic_ts,
                         'annualized_tcorr': annualized_tcorr_ts,
                         'annualized_instrument_count': annualized_instrument_count_ts,

                         # daily data
                         'daily_gross_ret': df_gross_ret_ts,
                         'daily_net_ret': df_net_ret_ts,
                         'daily_turnover': df_turnover_ts}

    baseline_col_long = '__baseline_long__'
    baseline_col_short = '__baseline_short__'
    if calculate_baseline:
        baseline_df = df[['time', 'instrument_id', 'future_ret']].copy()
        baseline_df[baseline_col_long] = 1.0
        baseline_df[baseline_col_short] = -1.0

        baseline_cols = [baseline_col_long, baseline_col_short]
        baseline_gross_ret_ts, baseline_net_ret_ts, baseline_turnover_ts = get_ts_ret_and_turnover(baseline_df, baseline_cols)
        baseline_gross_ret_ts = baseline_gross_ret_ts.reset_index()
        baseline_net_ret_ts = baseline_net_ret_ts.reset_index()
        baseline_turnover_ts = baseline_turnover_ts.reset_index()

        # Gross baseline metrics
        baseline_gross_ret_annual = get_annualized_ret(baseline_gross_ret_ts, baseline_cols, interest_method)
        baseline_gross_vol_annual = get_annualized_volatility(baseline_gross_ret_ts, baseline_cols)
        baseline_gross_sharpe_annual = get_annualized_sharpe(baseline_gross_ret_annual, baseline_gross_vol_annual)
        baseline_gross_sortino_annual = get_annualized_sortino_ratio(baseline_gross_ret_ts, baseline_cols, interest_method)
        _, baseline_gross_maxdd_annual = get_annualized_drawdown(baseline_gross_ret_ts, baseline_cols)
        baseline_gross_calmar_annual = get_annualized_calmar_ratio(baseline_gross_ret_annual, baseline_gross_vol_annual)
        baseline_gross_win_rate_annual = get_annualized_win_rate(baseline_gross_ret_ts, baseline_cols)

        # Net baseline metrics
        baseline_net_ret_annual = get_annualized_ret(baseline_net_ret_ts, baseline_cols, interest_method)
        baseline_net_vol_annual = get_annualized_volatility(baseline_net_ret_ts, baseline_cols)
        baseline_net_sharpe_annual = get_annualized_sharpe(baseline_net_ret_annual, baseline_net_vol_annual)
        baseline_net_sortino_annual = get_annualized_sortino_ratio(baseline_net_ret_ts, baseline_cols, interest_method)
        _, baseline_net_maxdd_annual = get_annualized_drawdown(baseline_net_ret_ts, baseline_cols)
        baseline_net_calmar_annual = get_annualized_calmar_ratio(baseline_net_ret_annual, baseline_net_vol_annual)
        baseline_net_win_rate_annual = get_annualized_win_rate(baseline_net_ret_ts, baseline_cols)
        baseline_turnover_annual = get_annualized_turnover(baseline_turnover_ts, baseline_cols)

        ts_performance_dc['daily_gross_ret_baseline'] = baseline_gross_ret_ts
        ts_performance_dc['daily_net_ret_baseline'] = baseline_net_ret_ts
        ts_performance_dc['daily_turnover_baseline'] = baseline_turnover_ts

    ts_performance_summary_list = \
        [gross_annualized_ret_ts[[fc_name]].rename(columns={fc_name: 'Gross Return'}),
         net_annualized_ret_ts[[fc_name]].rename(columns={fc_name: 'Net Return'}),

         gross_annualized_volatility_ts[[fc_name]].rename(columns={fc_name: 'Gross Volatility'}),
         net_annualized_volatility_ts[[fc_name]].rename(columns={fc_name: 'Net Volatility'}),

         gross_annualized_sharpe_ts[[fc_name]].rename(columns={fc_name: 'Gross Sharpe'}),
         net_annualized_sharpe_ts[[fc_name]].rename(columns={fc_name: 'Net Sharpe'}),

         gross_annualized_sortino_ratio_ts[[fc_name]].rename(columns={fc_name: 'Gross Sortino'}),
         net_annualized_sortino_ratio_ts[[fc_name]].rename(columns={fc_name: 'Net Sortino'}),

         gross_annualized_max_drawdown_ts[[fc_name]].rename(columns={fc_name: 'Gross MaxDD'}),
         net_annualized_max_drawdown_ts[[fc_name]].rename(columns={fc_name: 'Net MaxDD'}),

         gross_annualized_calmar_ratio_ts[[fc_name]].rename(columns={fc_name: 'Gross Calmar'}),
         net_annualized_calmar_ratio_ts[[fc_name]].rename(columns={fc_name: 'Net Calmar'}),

         gross_annualized_win_rate_ts[[fc_name]].rename(columns={fc_name: 'Gross Win Rate'}),
         net_annualized_win_rate_ts[[fc_name]].rename(columns={fc_name: 'Net Win Rate'}),

         annualized_turnover_ts[[fc_name]].rename(columns={fc_name: 'Turnover'}),
         annualized_ic_ts,
         annualized_tcorr_ts,
         annualized_instrument_count_ts
         ]

    ts_performance_summary = merge_dataframe(ts_performance_summary_list, on='year')

    if calculate_baseline:
        baseline_metric_map = {
            'Gross Return': baseline_gross_ret_annual,
            'Net Return': baseline_net_ret_annual,
            'Gross Volatility': baseline_gross_vol_annual,
            'Net Volatility': baseline_net_vol_annual,
            'Gross Sharpe': baseline_gross_sharpe_annual,
            'Net Sharpe': baseline_net_sharpe_annual,
            'Gross Sortino': baseline_gross_sortino_annual,
            'Net Sortino': baseline_net_sortino_annual,
            'Gross MaxDD': baseline_gross_maxdd_annual,
            'Net MaxDD': baseline_net_maxdd_annual,
            'Gross Calmar': baseline_gross_calmar_annual,
            'Net Calmar': baseline_net_calmar_annual,
            'Gross Win Rate': baseline_gross_win_rate_annual,
            'Net Win Rate': baseline_net_win_rate_annual,
            'Turnover': baseline_turnover_annual,
        }

        def _format_triplet(val, long_val, short_val) -> str:
            def _fmt(x):
                if pd.isna(x):
                    return 'nan'
                return f'{float(x):.6g}'
            return f'{_fmt(val)}({_fmt(long_val)},{_fmt(short_val)})'

        for metric_col, baseline_df_metric in baseline_metric_map.items():
            if metric_col not in ts_performance_summary.columns:
                continue
            long_series = baseline_df_metric[baseline_col_long].reindex(ts_performance_summary.index)
            short_series = baseline_df_metric[baseline_col_short].reindex(ts_performance_summary.index)
            ts_performance_summary[f'{metric_col} (With Baseline)'] = [
                _format_triplet(v, lv, sv)
                for v, lv, sv in zip(ts_performance_summary[metric_col], long_series, short_series)
            ]

    ts_performance_summary['Factor Name'] = fc_name
    ts_performance_summary['Factor Freq'] = fc_freq
    ts_performance_summary['Fee'] = fee

    return ts_performance_dc, ts_performance_summary
