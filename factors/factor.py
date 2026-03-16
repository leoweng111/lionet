"""
This script contains all manually-made factors.
"""
import pandas as pd


class fac_cumret:
    param_range = {'a': [1, 2, 3],
                   'b': [50, 100]}

    @staticmethod
    def operate(Data: pd.DataFrame, **kwargs):
        hash_tb = {chr(i): 0 for i in range(97, 123)}
        for key, value in kwargs.items():
            hash_tb[key] = value
        a = int(hash_tb['a'])
        b = int(hash_tb['b'])

        df = Data.copy()
        df['ret'] = df['close'].pct_change(a)
        df['mom'] = df['ret'].rolling(b, 1).mean()

        return df['mom']


class fac_upperline:
    param_range = {'a': [10, 30]}

    @staticmethod
    def operate(Data: pd.DataFrame, **kwargs):
        hash_tb = {chr(i): 0 for i in range(97, 123)}
        for key, value in kwargs.items():
            hash_tb[key] = value
        a = int(hash_tb['a'])

        df = Data.copy()
        df['upper_line'] = (df['high'] - df['open']) / (df['open'] - df['low'])
        df['upper_line'] = df['upper_line'].rolling(a, 1).mean()

        return df['upper_line']


class fac_winrate:
    param_range = {'a': [10, 20, 30]}

    @staticmethod
    def operate(Data: pd.DataFrame, **kwargs):
        hash_tb = {chr(i): 0 for i in range(97, 123)}
        for key, value in kwargs.items():
            hash_tb[key] = value
        a = int(hash_tb['a'])

        df = Data.copy()
        df['ret'] = df['close'].pct_change()
        df['win_ratio'] = (df['ret'] > 0) * 1
        df['win_ratio'] = df['win_ratio'].rolling(a).mean()

        return df['win_ratio']


def factormaker4(Data):
    df = Data.copy()
    df['close_max'] = df['close'].rolling(10).max()
    df['close_min'] = df['close'].rolling(10).min()
    df['location'] = (df['close'] - df['close_min']) / (df['close_max'] - df['close_min'])
    df['location'] = df['location'].rolling(5).mean()

    return df['location']


def factormaker5(Data):
    # trend strength
    df = Data.copy()
    df['price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['price_change'] = df['price'].pct_change(10)
    df['abs_total_price_change'] = df['price'].pct_change(1).abs().rolling(10).sum()
    df['trend_strength'] = df['price_change'] / df['abs_total_price_change']
    df['trend_strength'] = df['trend_strength'].rolling(20).mean()

    return df['trend_strength']


def factormaker6(Data):
    df = Data.copy()
    df['sign'] = (df['close'].diff() < 0).map({True: 1, False: -1})
    df['signed_volume'] = df['volume'] * df['sign']
    df['obv'] = df['signed_volume'].rolling(10).sum()

    return df['obv']

# def factorMaker7(Data):
# def factorMaker8(Data):


# def factormaker5(Data):
#     # simply for testing
#     df = Data.copy()
#     df['ret30'] = df['close'].shift(-30) / df['close'] - 1
#
#     return df['ret30']
