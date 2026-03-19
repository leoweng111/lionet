from typing import Union
from collections import Counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .factor import *
from . import factor as factor_module
from data import get_risk_free_rate
from stats import merge_dataframe, iterdict


def resolve_factor_class(fc_name: str):
    """Resolve factor class by name from factors.factor module."""
    fc_class = getattr(factor_module, fc_name, None)
    if fc_class is None or not isinstance(fc_class, type):
        available = sorted(
            name for name, obj in vars(factor_module).items()
            if isinstance(obj, type) and hasattr(obj, 'param_range') and hasattr(obj, 'operate')
        )
        raise NameError(f'Factor class `{fc_name}` is not defined. Available factors: {available}')
    return fc_class


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
    fc_name_counter = Counter(fc_name_list)
    duplicated_fc_names = sorted([x for x, cnt in fc_name_counter.items() if cnt > 1])
    if duplicated_fc_names:
        raise ValueError(f'fc_name_list contains duplicated factor names: {duplicated_fc_names}')
    for col in ['time', 'instrument_id']:
        assert col in Data.columns, f'Data does not contain column {col}.'
    df = Data.copy()

    df = df.sort_values(by='time', ascending=True)
    df = df.set_index(['time', 'instrument_id'])

    fc_class_list = [resolve_factor_class(fc_name) for fc_name in fc_name_list]
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


def get_future_ret(Data: pd.DataFrame,
                   portfolio_adjust_method: Union[str, None] = None,
                   rfr: bool = False) -> pd.DataFrame:
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
        # open to open ret. For day T, using (T+2 open - T+1 open) / T+1 open
        df = df.sort_values(by='time')
        df['future_ret'] = df.groupby('instrument_id')['open'].transform(lambda x: x.pct_change().shift(-2))

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


def rolling_normalize_features(df: pd.DataFrame,
                               factor_cols: Union[str, list],
                               rolling_norm_window: int = 30,
                               rolling_norm_min_periods: int = 20,
                               rolling_norm_eps: float = 1e-8,
                               rolling_norm_clip: float = 10.0,
                               instrument_col: str = 'instrument_id') -> pd.DataFrame:
    """Rolling normalize factor columns without look-ahead leakage.

    At time t, normalization uses history up to t-1 via shift(1) + rolling(...).
    """
    if isinstance(factor_cols, str):
        factor_cols = [factor_cols]
    if not factor_cols:
        return df.copy()

    df_out = df.copy()
    sort_cols = [instrument_col, 'time'] if instrument_col in df_out.columns else ['time']
    if 'time' in df_out.columns:
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    for col in factor_cols:
        series = pd.to_numeric(df_out[col], errors='coerce')
        if instrument_col in df_out.columns:
            hist_mean = series.groupby(df_out[instrument_col]).transform(
                lambda x: x.shift(1).rolling(rolling_norm_window, min_periods=rolling_norm_min_periods).mean()
            )
            hist_std = series.groupby(df_out[instrument_col]).transform(
                lambda x: x.shift(1).rolling(rolling_norm_window, min_periods=rolling_norm_min_periods).std()
            )
        else:
            hist_mean = series.shift(1).rolling(rolling_norm_window, min_periods=rolling_norm_min_periods).mean()
            hist_std = series.shift(1).rolling(rolling_norm_window, min_periods=rolling_norm_min_periods).std()

        hist_std = hist_std.replace(0, np.nan)
        z = (series - hist_mean) / (hist_std + rolling_norm_eps)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df_out[col] = z.clip(-rolling_norm_clip, rolling_norm_clip)

    return df_out

