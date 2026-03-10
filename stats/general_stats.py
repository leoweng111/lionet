"""
This script contains some general mathematics formulas or stats methods.
"""
import pandas as pd
from typing import Union
from itertools import product
from functools import reduce


def annual_to_daily_ret(annualized_ret: float):
    """

    :param annualized_ret:
    :return:
    """
    # assert -1 <= annualized_ret <= 1, f'Ret must between -1 and 1, but get {annualized_ret} instead.'
    return (1 + annualized_ret) ** (1 / 252) - 1


def daily_to_annual_ret(daily_ret: float):
    """

    :param daily_ret:
    :return:
    """
    # if not
    # assert -1 <= daily_ret <= 1, f'Ret must between 0 and 1, but get {daily_ret} instead.'
    return ((1 + daily_ret) ** 252) - 1


def adjust_ret_freq(ret: float, prev_freq: int, after_freq: int):
    """
    Adjust ret frequency from prev to after, calculating at the compound interest rate.
    :param ret:
    :param prev_freq: unit is day
    :param after_freq: unit is day
    :return:
    """
    if not ret:
        return ret
    if pd.isna(ret):
        return ret
    # assert -1 <= ret <= 1, f'Ret must between 0 and 1, but get {ret} instead.'
    assert prev_freq > 0
    assert after_freq > 0

    # adjust to one-day ret
    ret_one_day = (1 + ret) ** (1 / prev_freq) - 1

    # the ret on after_freq
    new_ret = (1 + ret_one_day) ** after_freq - 1

    return new_ret


def row_count_at_percentage(df: pd.DataFrame, percentage: float = 0.2):
    assert 0 <= percentage <= 1, 'Percentage should be a number between 0 and 1.'
    rows = int(df.shape[0] * percentage)

    return rows


def merge_dataframe(dataframe_list: list,
                    on: Union[str, list],
                    how='left',
                    validate='1:1',
                    **kwargs):
    """

    :param dataframe_list:
    :param on:
    :param how:
    :param validate:
    :param kwargs:
    :return:
    """
    return reduce(lambda x, y: pd.merge(x, y, on=on, how=how, validate=validate, **kwargs), dataframe_list)


def split_dataframe(Data: pd.DataFrame,
                    units_per_chunk: int,
                    split_unit: str = 'month'):
    """
    Splits a DataFrame by month into multiple DataFrames.

    :param Data: The input DataFrame containing a date column.
    :param units_per_chunk: Number of units per chunk.
    :param split_unit: split unit, can be 'year', 'month' or 'day'.
    :return: List[pd.DataFrame]: A list of DataFrames, each containing data for a specific month.
    """
    assert Data.index.names[0] == 'time'
    assert split_unit in ['year', 'month', 'day']
    df = Data.copy()
    df = df.reset_index()
    # Convert the date column to Pandas datetime
    df['date'] = pd.to_datetime(df['time'])

    # Group by month and create a list of DataFrames
    if split_unit == 'year':
        freq_signal = 'Y'
    elif split_unit == 'month':
        freq_signal = 'M'
    else:
        freq_signal = 'D'

    g = df.groupby(pd.Grouper(key='date', freq=f'{units_per_chunk}{freq_signal}'))
    chunks = [group for _, group in g]

    # if the data does not start from the first day of a month, the first of chunk will be shorter than others
    return chunks[1:]


def iterdict(parameter):
    """

    :param parameter:
    :return: a list
    """
    para_keys = parameter.keys()
    para_list = []
    for key in para_keys:
        para_list.append(parameter[key])
    para_rows = list(product(*para_list))
    para_rows = [dict(zip(para_keys, x)) for x in para_rows]

    return para_rows


def get_nan_summary(Data: pd.DataFrame):
    """
    Get the summary of missing value of Data.
    :param Data:
    :return:
    """
    df = Data.copy()
    nan_list = []
    for col in df.columns:
        vc = df[col].isna().value_counts()
        if len(vc) == 1:
            if vc.index[0]:
                nan_list.append(1)
            else:
                nan_list.append(0)
        else:
            ratio = vc[True] / (vc[True] + vc[False])
            nan_list.append(ratio)
    nan_describe = pd.Series(nan_list, index=df.columns)

    return nan_describe
