from typing import Union
from joblib import Parallel, delayed
from .factor import *
from lwpackage.lwdata import get_risk_free_rate
from lwpackage.lwstats import merge_dataframe, iterdict


def get_factor_value(Data: pd.DataFrame,
                     fc_name_list: Union[str, list],
                     n_jobs=5):
    """
    Calculate factor value.

    :param Data:
    :param fc_name_list:
    :param n_jobs:
    :return: a dataframe with factor values in fc_name_list
    """
    if isinstance(fc_name_list, str):
        fc_name_list = [fc_name_list]
    for col in ['time', 'instrument_id']:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()

    df = df.sort_values(by='time', ascending=True)
    df = df.set_index(['time', 'instrument_id'])

    fc_class_list = [eval(fc_name) for fc_name in fc_name_list]
    f = lambda x: get_factor_value_for_one_factor(df, x)

    with Parallel(n_jobs=n_jobs) as parallel:
        mapper_list = parallel(delayed(f)(fc_class) for fc_class in fc_class_list)

    df = merge_dataframe([df] + mapper_list, on=['time', 'instrument_id'])
    df = df.reset_index()

    return df


def get_factor_value_for_one_factor(Data: pd.DataFrame,
                                    fc_class):
    parameters = iterdict(fc_class.param_range)
    fc_func = fc_class.operate
    mapper_list = []
    for parameter in parameters:
        fc_name_with_parameter = join_fc_name_and_parameter(fc_class.__name__, parameter)
        mapper = Data.groupby('instrument_id', as_index=False, sort=False).apply(
            lambda x: fc_func(x, **parameter).reset_index().set_index(
                ['time', 'instrument_id']).iloc[:, -1].to_frame(fc_name_with_parameter)
                             )
        mapper_list.append(mapper)

    return merge_dataframe(mapper_list, on=['time', 'instrument_id'])


def join_fc_name_and_parameter(fc_name, parameter):

    return fc_name + '_' + '_'.join([str(value) for _, value in parameter.items()])


def cross_sectional_norm(Data: pd.DataFrame,
                         fc_name_list: Union[str, list]):
    """
    Standardize the factor value on every time cross-section.

    :param Data:
    :param fc_name_list: the factor value column, every factor value column is named after its name
    :return:
    """
    df = Data.copy()

    # todo: robust standardize
    # for the nan values, Series.mean() equals Series.mean(skipna=True), this will ignore the Nan values and calculate
    # the mean of the rest, this is the same for Series.std()
    for fc_name in fc_name_list:
        df[fc_name] = \
            df.groupby('time')[fc_name].transform(lambda x: (x - x.mean()) / x.std())

    return df


def get_future_ret(Data: pd.DataFrame,
                   portfolio_adjust_method: Union[str, None] = None,
                   rfr: bool = True):
    """
    Calculate return based on average transaction price. The future ret must match the position-adjust period.

    :param Data: panel price Data with open, high, low, close price
    :param portfolio_adjust_method: transaction_period
    :param rfr: considering risk-free rate or not
    :return: a dataframe with a ret column
    """
    for col in ['time', 'instrument_id']:
        assert col in Data.columns, f'df does not contain columns {col}.'
    df = Data.copy()

    # # special logic for transaction period
    # if not transaction_period:
    #     if fc_freq == '1d':
    #         transaction_period = 1
    #     elif fc_freq == '5m':
    #         transaction_period = 1
    #     else:  # 1m freq
    #         transaction_period = 3

    # assert ret_freq > 0
    # assert transaction_period > 0
    # assert ret_freq >= 2 * transaction_period, f'Transaction period is ' \
    #                                            f'{transaction_period}, ' \
    #                                            f'while ret freq is {ret_freq}, invalid!'
    # 未来收益率计算周期和调仓周期要相同！
    # 假设使用T-1的因子值，在T0的收盘时刻以T0收盘价开仓，然后在T1的收盘时刻以T1收盘价平仓
    # 所以，我们要保证T0的因子值是利用了直到T-1的信息计算出来的，不能用到T的信息
    # 那么在每个bar
    if portfolio_adjust_method == '1D':
        df = df.sort_values(by='time')
        df['future_ret'] = df.groupby('instrument_id')['close'].transform(lambda x: x.pct_change())

    # df['transaction_price'] = df['close'].pct_change()
    # # transaction price of one bar is the average transaction price for a transaction that is completed at
    # # the close time of this bar.
    # df['transaction_price'] = \
    #     df[['open', 'high', 'low', 'close']].mean(axis=1).rolling(
    #         transaction_period).mean()
    #
    # mapper = df.groupby('instrument_id')['transaction_price'].apply(
    #     lambda x: (x.shift(-ret_freq - 1) / x.shift(-transaction_period + 1) - 1)).droplevel(0)
    #
    # df['future_ret'] = df.index.map(mapper)

    if rfr:  # risk-free rate will have effect on sharpe only if long_only == True
        # annualized risk-free rate
        # we fill nan values of risk-free rate with next day's risk-free rate if available
        df['date'] = df['time'].dt.date

        df_rfr = get_risk_free_rate()[['date', 'rate']].copy()
        df_rfr['date'] = df_rfr['date'].dt.date
        df = df.merge(df_rfr, on='date', how='left', validate='m:1')
        df['rate'] = df['rate'].bfill().ffill()
        df = df.rename(columns={'rate': 'rfr'})

        # simple interest rate
        df['rfr'] = df['rfr'] / 252

        # excess ret = ret - risk free rate
        df['future_ret'] = df['future_ret'] - df['rfr']
        df = df.drop(columns=['rfr', 'date'])

    # 计算结果中，部分时间截面的future ret可能为nan，原因是：
    # 此时间截面到最终时间截面的长度小于transaction_period，导致无法求出future ret
    return df
