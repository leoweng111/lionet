"""
This script contains methods for calculating factor indicators.
"""
import pandas as pd
import numpy as np
from typing import Union
from joblib import delayed, Parallel
from .factor import *
from .factor_utils import join_fc_name_and_parameter
from lwpackage.lwstats import merge_dataframe, iterdict

EPSILON = 1e-6


def get_grouped_ret_and_turnover_for_one_factor(Data: pd.DataFrame,
                                                fc_col: str,
                                                portfolio_number=10,
                                                portfolio_method: str = 'longshort',
                                                fc_freq: str = '1d',
                                                fee: float = 0.00025):
    """
    Get the ret, net ret, instrument count, turnover data for one factor.

    :param Data: a dataframe with columns 'time', 'instrument_id', fc_name, 'ret'
    :param fc_col:
    :param portfolio_number:
    :param portfolio_method:
    :param fc_freq:
    :param fee:
    :return: daily-freq data of ret, net ret, instrument count, turnover
    """
    assert fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, but got {fc_freq} instead.'
    assert portfolio_number > 1, f'Portfolio number should be larger than 1, but got {portfolio_number} instead.'
    for col in ['time', 'instrument_id', fc_col, 'future_ret']:
        assert col in Data.columns, f'Data does not contain column {col}.'

    df = Data.copy()
    # 此处需要特别注意nan value的处理，即future ret列中可能包含nan值。
    # 1. 如果某个时间截面上部分股票的future ret的值为nan，就剔除这些股票，用剩下的股票进行分组。
    # 2. 如果某个时间截面上部分股票的fc_col的值为nan（不存在因子值），也剔除这些股票，用剩下的股票进行分组。
    # 3. 如果某个时间截面不存在股票数据，或者剔除后的有效标的数量小于portfolio number，则不进行分组。
    # 综上考虑，在一开始求解的时候先对fc_col和future_ret dropna，再将有效标的数量小于portfolio number的时间戳删除后进行计算即可。
    # 注意最后需要将计算的结果和完整的时间戳merge，如果最终结果存在nan，则说明这个时间戳我们无法进行计算。

    df = df.dropna(subset=[fc_col, 'future_ret'])
    count = df.groupby('time').size()
    invaild_time_idx = count[count < portfolio_number].index.to_list()
    df = df[~df['time'].isin(invaild_time_idx)]

    df = df.set_index(['time', 'instrument_id'])
    df['quantile_0'] = float('-inf')
    for i in range(1, portfolio_number):
        df[f'quantile_{i}'] = df.groupby('time')[fc_col].transform(lambda x: x.quantile(i / portfolio_number))
    df[f'quantile_{portfolio_number}'] = float('inf')

    df_gross_ret, df_n, df_turnover = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(portfolio_number):

        df_gross_ret[f'{fc_col} {i + 1}'] = df.groupby('time').apply(
            lambda x: (((x[fc_col] > x[f'quantile_{i}']) &
                        (x[fc_col] < x[f'quantile_{i + 1}'])) * x['future_ret']).sum() / (
                              (x[fc_col] > x[f'quantile_{i}']) &
                              (x[fc_col] < x[f'quantile_{i + 1}'])).sum(), include_groups=False)

        df_n[f'{fc_col} {i + 1}'] = df.groupby('time').apply(
            lambda x: ((x[fc_col] > x[f'quantile_{i}']) &
                       (x[fc_col] < x[f'quantile_{i + 1}'])).sum(), include_groups=False)

        turnover_arr = df.groupby('time', as_index=False).apply(lambda x: (x[fc_col] > x[f'quantile_{i}']) &
                                                                          (x[fc_col] < x[f'quantile_{i + 1}']),
                                                                include_groups=False).droplevel(0)
        turnover_arr = (turnover_arr.unstack() * 1).astype(float).fillna(0)
        turnover_arr = turnover_arr.apply(lambda x: x / df_n[f'{fc_col} {i + 1}'])
        turnover_arr = turnover_arr.diff().abs().sum(axis=1)

        df_turnover[f'{fc_col} {i + 1}'] = turnover_arr

    df_gross_ret[f'{fc_col} {portfolio_number}-1'] = (df_gross_ret[f'{fc_col} {portfolio_number}'] - df_gross_ret[
        f'{fc_col} {1}']) / 2
    # Assume that at any time, the long portfolio and short portfolio do not contain replicated instruments.
    df_turnover[f'{fc_col} {portfolio_number}-1'] = \
        (df_turnover[f'{fc_col} {portfolio_number}'] + df_turnover[f'{fc_col} {1}'])

    # When data is not day-freq, we calculate the minute-freq ret and then aggregate it.
    if fc_freq != '1d':
        df_gross_ret = df_gross_ret.reset_index()
        df_gross_ret['time'] = pd.to_datetime(df_gross_ret['time']).dt.date
        df_gross_ret = df_gross_ret.groupby('time').sum()

        df_turnover = df_turnover.reset_index()
        df_turnover['time'] = pd.to_datetime(df_turnover['time']).dt.date
        df_turnover = df_turnover.groupby('time').sum()

        df_n = df_n.reset_index()
        df_n['time'] = pd.to_datetime(df_n['time']).dt.date
        df_n = df_n.groupby('time').mean()

    # net ret = gross ret - turnover * fee
    df_net_ret = df_gross_ret - df_turnover * fee

    if portfolio_method == 'longshort':
        df_gross_ret['ret'] = df_gross_ret[f'{fc_col} {portfolio_number}-1']
        df_net_ret['ret'] = df_net_ret[f'{fc_col} {portfolio_number}-1']
        df_turnover['ret'] = df_turnover[f'{fc_col} {portfolio_number}-1']

    elif portfolio_method == 'long_only':
        df_gross_ret['ret'] = df_gross_ret[f'{fc_col} {portfolio_number}']
        df_net_ret['ret'] = df_net_ret[f'{fc_col} {portfolio_number}']
        df_turnover['ret'] = df_turnover[f'{fc_col} {portfolio_number}']

    else:
        raise NotImplementedError(f'Do not support method {portfolio_method}.')

    df_gross_ret = df_gross_ret.reset_index()
    df_gross_ret['time'] = pd.to_datetime(df_gross_ret['time']).dt.date
    df_gross_ret = df_gross_ret.groupby('time').sum()

    # These four dataframe should not contain any nan values according to the above procedure.
    # 如果某一个时间截面的数据无法计算，df中就直接不包含这一个时间截面，而不是将这个时间截面的计算结果设置为nan
    return df_gross_ret, df_net_ret, df_n, df_turnover


def get_annualized_ic_and_icir(Data: pd.DataFrame,
                               fc_col: Union[str, list]):
    """
    计算因子值和future ret之间的相关系数，则计算IC的周期和未来收益率future ret的计算周期相同，也和调仓周期相同。
    e.g 因子频率是5min，而调仓周期为一天，则我们在每天对因子值求平均值，再计算这个平均值和未来一天收益率的相关系数。
    :param Data:
    :param fc_col:
    :return:
    """

    if isinstance(fc_col, str):
        fc_col = [fc_col]
    for col in ['time', 'instrument_id', 'future_ret'] + fc_col:
        assert col in Data.columns, f'df does not contain columns {col}.'
    num_ic = Data.groupby('time').corr('pearson').loc[(slice(None), fc_col), ['future_ret']].unstack(level=1)
    num_ic.columns = num_ic.columns.droplevel(0)

    rank_ic = Data.groupby('time').corr('spearman').loc[(slice(None), fc_col), ['future_ret']].unstack(level=1)
    rank_ic.columns = rank_ic.columns.droplevel(0)

    if len(fc_col) == 1:
        ic_cols = ['NumIC', 'RankIC']
        icir_cols = ['NumICIR', 'RankICIR']
    else:
        ic_cols = [x + ' NumIC' for x in fc_col] + [x + ' RankIC' for x in fc_col]
        icir_cols = [x + ' NumICIR' for x in fc_col] + [x + ' RankICIR' for x in fc_col]

    ic_all = pd.concat([num_ic, rank_ic], axis=1)
    ic_all.columns = ic_cols
    ic_year = ic_all.copy()
    ic_year = ic_year.reset_index()
    ic_year['year'] = pd.to_datetime(ic_year['time']).dt.year
    ic_year = ic_year.groupby('year')[ic_cols].mean()

    ic_all = ic_all.mean().to_frame('all').T
    ic = pd.concat([ic_year, ic_all])
    ic.index.name = 'year'

    icir = pd.concat([num_ic, rank_ic], axis=1)
    icir.columns = icir_cols

    icir_year = icir.copy()
    icir_year = icir_year.reset_index()
    icir_year['year'] = pd.to_datetime(icir_year['time']).dt.year
    icir_year = icir_year.groupby('year')[icir_cols].apply(lambda x: x.mean() / x.std())

    icir_all = (icir.mean() / icir.std()).to_frame('all').T
    icir = pd.concat([icir_year, icir_all])
    icir.index.name = 'year'

    return ic, icir


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
    df = df.sort_values(by='instrument_id')
    df['year'] = pd.to_datetime(df['time']).dt.year

    ic_ts_year = df.groupby('year')[fc_col].corr(df['future_ret'], method='pearson')
    ic_ts_all = pd.Series(df[fc_col].corr(df['future_ret'], method='pearson'), index=['all'])
    ic_ts = pd.concat([ic_ts_year, ic_ts_all]).to_frame('TS IC')

    rank_ic_ts_year = df.groupby('year')[fc_col].corr(df['future_ret'], method='spearman')
    rank_ic_ts_all = pd.Series(df[fc_col].corr(df['future_ret'], method='spearman'), index=['all'])
    rank_ic_ts = pd.concat([rank_ic_ts_year, rank_ic_ts_all]).to_frame('TS RankIC')

    ic_ts = pd.concat([ic_ts, rank_ic_ts], axis=1)
    ic_ts.index.name = 'year'
    if portfolio_adjust_method == '1D':
        ret_fre1 = 1

    elif portfolio_adjust_method == '1M':
        ret_freq = 30
    else:
        pass

    t_corr_year = df.groupby('year')['future_ret'].apply(lambda x: np.sqrt(x.size / ret_freq))
    t_corr_all = pd.Series(len(df) / ret_freq, index=['all'])
    t_corr = pd.concat([t_corr_year, t_corr_all]).to_frame('T-corr')
    t_corr.index.name = 'year'

    return ic_ts, t_corr


def get_ts_ret_and_turnover(Data: pd.DataFrame,
                            fc_col: Union[str, list],
                            fee: float = 0.00025):
    # time series gross ret and net ret
    # time series way of calculating ret and turnover is simpler
    # todo: 最后的performance中加入这个时序指标
    if isinstance(fc_col, str):
        fc_col = [fc_col]
    for col in ['time', 'instrument_id', 'future_ret'] + fc_col:
        assert col in Data.columns, f'Data does not contain column {col}.'

    df = Data.copy()
    # if one of value in fc_col + ['future_ret'] is nan, we will not consider this line of data
    df = df.dropna(subset=fc_col + ['future_ret'])
    df = df.set_index(['time', 'instrument_id'])
    df_gross_ret_ts = pd.DataFrame(df[fc_col].values * df[['future_ret']].values,
                                   index=df.index,
                                   columns=fc_col)
    df_gross_ret_ts = df_gross_ret_ts.groupby('time').mean()
    # the main difference between ts-turnover and cr-turnover is that ts-turnover takes time-cross-section mean here
    df_turnover_ts = df.unstack(level=1).diff().abs().fillna(0).mean(axis=1).to_frame()
    df_turnover_ts.columns = fc_col
    df_net_ret_ts = pd.DataFrame(df_gross_ret_ts.values - (df_turnover_ts * fee).values,
                                 index=df_gross_ret_ts.index,
                                 columns=fc_col)
    # These three dataframe should not contain any nan values.
    return df_gross_ret_ts, df_net_ret_ts, df_turnover_ts


def get_annualized_ret(Data: pd.DataFrame,
                       ret_col: Union[str, list],
                       interest_method: str = 'simple'):
    """
    Get annualized return for every year in data and total annualized return.

    :param Data: Data should be a dataframe with columns: 'time' and ret_col, it can stand for either grouped ret for
        one factor or ret of different factors. This func will calculate indicator on every ret col of Data.
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


def get_annualized_instrument_count(df_n: pd.DataFrame):
    """
    Get annualized instrument count for every portfolio group
    :param df_n: the func return df_n of get_grouped_ret_and_turnover(need to reset_index)
    :return:
    """
    assert 'time' in df_n.columns
    df = df_n.copy()

    df['year'] = pd.to_datetime(df['time']).dt.year
    n_cols = list(set(df.columns.to_list()) - {'time', 'year'})
    n_cols = sorted(n_cols, key=lambda x: int(x.split(' ')[1]))

    count_year = df.groupby('year')[n_cols].mean().round().astype(int)
    count_all = df[n_cols].mean().round().astype(int).to_frame('all').T

    instrument_count = pd.concat([count_year, count_all])
    instrument_count.index.name = 'year'

    return instrument_count


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

    count_year = df.groupby('year').size().to_frame('count')
    count_all = int(df.groupby('year').size().mean().round())
    count_all = pd.Series(count_all, index=['all']).to_frame('count').T

    instrument_count = pd.concat([count_year, count_all])
    instrument_count.index.name = 'year'

    return instrument_count


def get_performance(Data: pd.DataFrame,
                    fc_name_list: Union[str, list],
                    fc_freq: str = '1d',
                    portfolio_adjust_method: str = '1D',
                    portfolio_number: int = 10,
                    portfolio_method: str = 'longshort',
                    interest_method: str = 'simple',
                    fee: float = 0.00025,
                    n_jobs=5):

    if isinstance(fc_name_list, str):
        fc_name_list = [fc_name_list]
    for col in ['time', 'instrument_id', 'future_ret'] + fc_name_list:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()
    f = lambda x: get_performance_for_one_factor(df, x, fc_freq, portfolio_adjust_method, portfolio_number,
                                                 portfolio_method, interest_method, fee)

    with Parallel(n_jobs=n_jobs) as parallel:
        performance_list = parallel(delayed(f)(fc_name) for fc_name in fc_name_list)

    performance_dc_list = [x[0] for x in performance_list]
    performance_summary_list = [x[1] for x in performance_list]
    ts_performance_dc_list = [x[2] for x in performance_list]
    ts_performance_summary_list = [x[3] for x in performance_list]

    performance_dc = {k: v for k, v in zip(fc_name_list, performance_dc_list)}
    performance_summary = pd.concat(performance_summary_list)
    ts_performance_dc = {k: v for k, v in zip(fc_name_list, ts_performance_dc_list)}
    ts_performance_summary = pd.concat(ts_performance_summary_list)

    return performance_dc, performance_summary, ts_performance_dc, ts_performance_summary


def get_performance_for_one_factor(Data: pd.DataFrame,
                                   fc_name: str,
                                   fc_freq: str = '1d',
                                   portfolio_adjust_method: str = '1D',
                                   portfolio_number: int = 10,
                                   portfolio_method: str = 'longshort',
                                   interest_method: str = 'simple',
                                   fee: float = 0.00025):
    """
    Get all performance.
    """

    ## cross-sectional performance

    # cross-sectional ret and turnover
    df = Data.copy()
    df_gross_ret, df_net_ret, df_n, df_turnover = \
        get_grouped_ret_and_turnover_for_one_factor(df, fc_name, portfolio_number, portfolio_method, fc_freq, fee)

    ret_col = [fc_name + ' ' + str(i) for i in range(1, portfolio_number + 1)] + \
              [f'{fc_name} {portfolio_number}-1', 'ret']
    df_gross_ret = df_gross_ret.reset_index()
    df_net_ret = df_net_ret.reset_index()
    df_n = df_n.reset_index()
    df_turnover = df_turnover.reset_index()

    # Gross Performance
    gross_annualized_ret = get_annualized_ret(df_gross_ret, ret_col, interest_method)
    gross_annualized_volatility = get_annualized_volatility(df_gross_ret, ret_col)
    gross_annualized_sharpe = get_annualized_sharpe(gross_annualized_ret, gross_annualized_volatility)
    gross_annualized_sortino_ratio = get_annualized_sortino_ratio(df_gross_ret, ret_col, interest_method)
    gross_drawdown, gross_annualized_max_drawdown = get_annualized_drawdown(df_gross_ret, ret_col)
    gross_annualized_calmar_ratio = get_annualized_calmar_ratio(gross_annualized_ret, gross_annualized_volatility)
    gross_annualized_win_rate = get_annualized_win_rate(df_gross_ret, ret_col)

    # Net Performance
    net_annualized_ret = get_annualized_ret(df_net_ret, ret_col, interest_method)
    net_annualized_volatility = get_annualized_volatility(df_net_ret, ret_col)
    net_annualized_sharpe = get_annualized_sharpe(net_annualized_ret, net_annualized_volatility)
    net_annualized_sortino_ratio = get_annualized_sortino_ratio(df_net_ret, ret_col, interest_method)
    net_drawdown, net_annualized_max_drawdown = get_annualized_drawdown(df_net_ret, ret_col)
    net_annualized_calmar_ratio = get_annualized_calmar_ratio(net_annualized_ret, net_annualized_volatility)
    net_annualized_win_rate = get_annualized_win_rate(df_net_ret, ret_col)

    # IC, ICIR , turnover and instrument count does not depend on fee
    annualized_ic, annualized_icir = get_annualized_ic_and_icir(df, fc_name)
    annualized_turnover = get_annualized_turnover(df_turnover, ret_col)
    annualized_instrument_count = get_annualized_instrument_count(df_n)

    performance_dc = {'gross_annualized_ret': gross_annualized_ret,
                      'gross_annualized_volatility': gross_annualized_volatility,
                      'gross_annualized_sharpe': gross_annualized_sharpe,
                      'gross_annualized_sortino': gross_annualized_sortino_ratio,
                      'gross_annualized_max_drawdown': gross_annualized_max_drawdown,
                      'gross_drawdown': gross_drawdown,
                      'gross_annualized_calmar': gross_annualized_calmar_ratio,
                      'gross_annualized_win_rate': gross_annualized_win_rate,

                      'net_annualized_ret': net_annualized_ret,
                      'net_annualized_volatility': net_annualized_volatility,
                      'net_annualized_sharpe': net_annualized_sharpe,
                      'net_annualized_sortino': net_annualized_sortino_ratio,
                      'net_annualized_max_drawdown': net_annualized_max_drawdown,
                      'net_drawdown': net_drawdown,
                      'net_annualized_calmar': net_annualized_calmar_ratio,
                      'net_annualized_win_rate': net_annualized_win_rate,

                      # data which does not depend on fee
                      'annualized_turnover': annualized_turnover,
                      'annualized_ic': annualized_ic,
                      'annualized_icir': annualized_icir,
                      'annualized_instrument_count': annualized_instrument_count,

                      # daily data
                      'daily_gross_ret': df_gross_ret,
                      'daily_net_ret': df_net_ret,
                      'daily_instrument_count': df_n,
                      'daily_turnover': df_turnover}

    performance_summary_list = \
        [gross_annualized_ret[['ret']].rename(columns={'ret': 'Gross Return'}),
         net_annualized_ret[['ret']].rename(columns={'ret': 'Net Return'}),

         gross_annualized_volatility[['ret']].rename(columns={'ret': 'Gross Volatility'}),
         net_annualized_volatility[['ret']].rename(columns={'ret': 'Net Volatility'}),

         gross_annualized_sharpe[['ret']].rename(columns={'ret': 'Gross Sharpe'}),
         net_annualized_sharpe[['ret']].rename(columns={'ret': 'Net Sharpe'}),

         gross_annualized_sortino_ratio[['ret']].rename(columns={'ret': 'Gross Sortino'}),
         net_annualized_sortino_ratio[['ret']].rename(columns={'ret': 'Net Sortino'}),

         gross_annualized_max_drawdown[['ret']].rename(columns={'ret': 'Gross MaxDD'}),
         net_annualized_max_drawdown[['ret']].rename(columns={'ret': 'Net MaxDD'}),

         gross_annualized_calmar_ratio[['ret']].rename(columns={'ret': 'Gross Calmar'}),
         net_annualized_calmar_ratio[['ret']].rename(columns={'ret': 'Net Calmar'}),

         gross_annualized_win_rate[['ret']].rename(columns={'ret': 'Gross Win Rate'}),
         net_annualized_win_rate[['ret']].rename(columns={'ret': 'Net Win Rate'}),

         annualized_turnover[['ret']].rename(columns={'ret': 'Turnover'}),
         annualized_ic,
         annualized_icir,
         # special calculating method here: the sum of all groups at any cross-section
         annualized_instrument_count.sum(axis=1).round(0).astype(int).to_frame('Average Instrument Count')
         ]

    performance_summary = merge_dataframe(performance_summary_list, on='year')
    performance_summary['Factor Name'] = fc_name
    performance_summary['Factor Freq'] = fc_freq
    performance_summary['Fee'] = fee

    ## time series performance

    # time-series ret and turnover
    df_gross_ret_ts, df_net_ret_ts, df_turnover_ts = get_ts_ret_and_turnover(df, fc_name, fee)
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
    ts_performance_summary['Factor Name'] = fc_name
    ts_performance_summary['Factor Freq'] = fc_freq
    ts_performance_summary['Fee'] = fee

    return performance_dc, performance_summary, ts_performance_dc, ts_performance_summary
