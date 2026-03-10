"""
This script is to get and deal with stock data based on akshare.
"""
import time
import datetime
from typing import List, Union, Literal
import pandas as pd
import numpy as np

import akshare as ak
from lwpackage.lwmongo.mongify import get_data, update_data

from lwpackage.lwutils.lwpath import START_TIME, START_DATE, END_DATE
from lwpackage.lwutils.logging import log
from lwpackage.lwerror.errors import StockDataError


def get_stock_info(instrument_id: Union[str, List, None] = None,
                   from_database: bool = True):
    """
    Get basic stock_info with optional filters.

    :param instrument_id: instrument_id
    :param from_database: get data from database or not
    :return: stock data
    """
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if not from_database:
        rename_dc = {'代码': 'instrument_id', '名称': 'name', '最新价': 'latest_price', '总市值': 'total_market_value',
                     '流通市值': 'current_market_value'}
        df_instrument = ak.stock_zh_a_spot_em()
        df_instrument = df_instrument.rename(columns=rename_dc).loc[:, [key for key in rename_dc.values()]]
        if instrument_id:
            df_instrument = df_instrument.loc[df_instrument['instrument_id'].isin(instrument_id)]
            _check_no_missing_instrument_id(df_instrument, instrument_id)
    else:
        if instrument_id:
            mongo_operator = {'instrument_id': {'$in': instrument_id}}
            df_instrument = get_data(database='stock',
                                     collection='instrument_info',
                                     mongo_operator=mongo_operator)

            # if the instrument_id is given, we need to check whether we have get all instrument_id required.
            _check_no_missing_instrument_id(df_instrument, instrument_id)
        else:
            df_instrument = get_data(database='stock',
                                     collection='instrument_info')

    return df_instrument


def _check_no_missing_instrument_id(df_instrument: pd.DataFrame,
                                    instrument_id: List):
    """
    Check if df_instrument contains all instrument id in instrument_id. Raise error if missing.

    :param df_instrument: df_instrument
    :param instrument_id: instrument_id
    :return: None
    """
    if len(df_instrument) != len(instrument_id):
        missing_instrument_id = list(set(instrument_id) - set(df_instrument['instrument_id']))
        raise StockDataError(f'The instrument id data in {missing_instrument_id} '
                             f'of stock.instrument_info database is missing, will not return any stock sata.')


def update_stock_info(instrument_id: Union[str, List, None] = None,
                      method: str = 'insert_many'):
    """
    Update stock info in stock.instrument_info collection with the latest data from akshare.

    :param instrument_id: the instrument ids need to be updated
    :param method: updating method
    :return: None
    """

    df_latest_stock_info = get_stock_info(instrument_id=instrument_id, from_database=False)
    update_data(database='stock',
                collection='instrument_info',
                df=df_latest_stock_info,
                method=method)

    log.info(f'Successfully update instrument info.')


def get_stock_price(frequency: str,
                    instrument_id: Union[str, List, None] = None,
                    start_time: Union[str, datetime.date, datetime.datetime, pd.Timestamp, None] = None,
                    end_time: Union[str, datetime.date, datetime.datetime, pd.Timestamp, None] = None,
                    from_database: bool = True,
                    adjust: Literal['', 'qfq', 'hfq'] = '',
                    sort_time: bool = True,
                    wait_time: float = 0.3):
    """
    Get stock price data from database or using ak.stock_zh_a_minute & ak.stock_zh_a_hist to get stock price data from
    akshare.

    :param frequency:
    :param instrument_id: instrument_id
    :param start_time: default is the oldest price data that can be found
    :param end_time: default is now
    :param from_database: get data from database or not
    :param adjust: adjust param of ak.stock_zh_a_minute & ak.stock_zh_a_hist
    :param sort_time: whether to sort data by time
    :param wait_time: wait time between query from akshare
    :return: stock price data
    """
    assert frequency in ['1m', '5m', '1d'], f"frequency {frequency} not support, only support '1m', '5m' or '1d'."

    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if start_time:
        start_time = pd.to_datetime(start_time)
    else:
        start_time = START_TIME
    if end_time:
        end_time = pd.to_datetime(end_time)
    if not end_time:
        end_time = datetime.datetime.now()

    #  latest instrument_id from akshare
    if not instrument_id:
        instrument_id = get_stock_info(from_database=True)['instrument_id'].to_list()
    if not from_database:
        df_list = []
        if frequency in ['1m', '5m']:
            for ins_id in instrument_id:
                ins_id = add_stock_suffix(ins_id)
                df_price = ak.stock_zh_a_minute(symbol=ins_id, period=frequency[0], adjust=adjust)
                time.sleep(wait_time)
                if len(df_price) > 0:
                    df_price['instrument_id'] = ins_id
                    df_list += [df_price]
                else:
                    # If no data, which may indicate that the stock has already exited from the market.
                    print(f'No data for instrument {ins_id}.')
                    log.info(f'No data for instrument {ins_id}.')

            if len(df_list) == 0:
                return pd.DataFrame(columns=['day', 'open', 'high', 'low', 'close', 'volume'])
            df_price = pd.concat(df_list)
            df_price = df_price.rename(columns={'day': 'time'})
            df_price['time'] = pd.to_datetime(df_price['time'])
        else:  # day frequency data
            # stock_zh_a_hist returns all data available if the format of start_date or end_date is illegal.
            for ins_id in instrument_id:
                df_price = ak.stock_zh_a_hist(symbol=ins_id, period='daily', adjust=adjust)
                time.sleep(wait_time)
                if len(df_price) > 0:
                    df_price['instrument_id'] = ins_id
                    df_list += [df_price]
                else:
                    # If no data, which may indicate that the stock has already exited from the market.
                    log.info(f'No data for instrument {ins_id}.')

            if len(df_list) == 0:
                return pd.DataFrame(columns=['day', 'open', 'high', 'low', 'close', 'volume'])
            df_price = pd.concat(df_list)
            df_price = df_price[['日期', '开盘', '最高', '最低', '收盘', '成交量', 'instrument_id']].rename(
                columns={'日期': 'time', '开盘': 'open', '最高': 'high', '最低': 'low',
                         '收盘': 'close', '成交量': 'volume'})

    else:  # price data from database
        if not start_time:
            mongo_operator = {
                '$and': [
                    {'time': {'$lte': end_time}},
                    {'instrument_id': {"$in": instrument_id}}
                ]}
        else:
            mongo_operator = {
                '$and': [
                    {'time': {'$gte': start_time}},
                    {'time': {'$lte': end_time}},
                    {'instrument_id': {"$in": instrument_id}}
                ]}

        if frequency in ['1m', '5m']:
            df_price = get_data(database='stock',
                                collection=f'price_{frequency}in',
                                mongo_operator=mongo_operator)

        else:
            df_price = get_data(database='stock',
                                collection='price_daily',
                                mongo_operator=mongo_operator)

    df_price = clean_stock_price_data(df_price)
    df_price = df_price.loc[(df_price['time'] >= start_time) & (df_price['time'] <= end_time)]

    if sort_time and len(df_price) > 0:
        df_price = df_price.sort_values(by='time')

    return df_price


def update_stock_price(frequency: Union[str, List] = None,
                       instrument_id: Union[str, List, None] = None,
                       update_existing_instrument_id: bool = True,
                       wait_time: float = 3,
                       method: str = 'insert_many'):
    """
    Update stock price collection with the latest data from akshare. In order to be robust, only update one instrument
    once.

    :param frequency: the price databases with frequency will be updated, can be '1m', '5m' or '1d', if not specify,
        then all three price databases will be updated
    :param instrument_id: instrument id
    :param update_existing_instrument_id: if update, we update existing data; otherwise we only add non-existing data.
    :param wait_time: wait time between query from akshare
    :param method: Method for updating data. View details on lwpackage.lwmongo.mongify.update_data.
    :return: None
    """
    if frequency:
        assert frequency in ['1m', '5m', '1d'], f"frequency {frequency} not support, only support '1m', '5m' or '1d'."

    if not frequency:
        frequency = ['1m', '5m', '1d']
    if isinstance(frequency, str):
        frequency = [frequency]
    for freq in frequency:

        # for robustness, we update one instrument once.
        # When we need to update all data, in order to update all data, we shuffle the instrument id list we get.
        if not instrument_id:
            instrument_id = np.random.permutation(get_stock_info(from_database=False)['instrument_id']).tolist()
        if isinstance(instrument_id, str):
            instrument_id = [instrument_id]

        # we update instrument_id one by one, this is to avoid reaching the maximum-visit limit of websites.
        for ins_id in instrument_id:
            try:
                print(f'Begin updating instrument id {ins_id} of frequency {freq}.')
                # if we only consider instrument_id that do not exist in database.
                if not update_existing_instrument_id:
                    df_data = get_stock_price(frequency=freq,
                                              instrument_id=ins_id,
                                              from_database=True,
                                              wait_time=wait_time)
                    if len(df_data) == 0:
                        _update_stock_price_for_one_instrument(frequency=freq,
                                                               instrument_id=ins_id,
                                                               wait_time=wait_time,
                                                               method=method)
                    else:
                        log.info(f'Price {freq} data for instrument {ins_id} already exists in database, skip.')

                else:  # we update database data with all the latest data from akshare.
                    _update_stock_price_for_one_instrument(frequency=freq,
                                                           instrument_id=ins_id,
                                                           wait_time=wait_time,
                                                           method=method)
            except Exception as e:
                print(f'Error occurs: {e}')
                log.info(f'Error occurs: {e}')

        log.info(f'Successfully update all the {freq} price data.')

    log.info(f'Successfully update all the required price data.')


def _update_stock_price_for_one_instrument(frequency: str,
                                           instrument_id: str,
                                           wait_time: float = 0.03,
                                           method: str = 'insert_many',
                                           filter_column: Union[str, list] = None):
    """
    Update stock price for one instrument using specified updating method.

    :param frequency: the price databases with frequency will be updated, can be '1m', '5m' or '1d', if not specify,
        then all three price databases will be updated.
    :param instrument_id: instrument_id
    :param wait_time: wait time between query from akshare
    :param method: Method for updating data. View details on lwpackage.lwmongo.mongify.update_data.
    :return:
    """
    update_start_time = time.perf_counter()
    if frequency == '1m':
        col = 'price_1min'
    elif frequency == '5m':
        col = 'price_5min'
    else:
        col = 'price_daily'

    df_latest_price = get_stock_price(frequency=frequency,
                                      instrument_id=instrument_id,
                                      from_database=False,
                                      wait_time=wait_time)
    if len(df_latest_price) > 0:
        assert not any(df_latest_price.duplicated(['instrument_id', 'time'])), \
            f' Wrong data with duplicated instrument_id and time, please check.'
        # for one instrument, we update one tick once, for robustness.

        update_data(database='stock',
                    collection=col,
                    df=df_latest_price,
                    method=method,
                    filter_column=filter_column)

        update_end_time = time.perf_counter()
        elapsed_time = update_end_time - update_start_time
        log.info(f'Successfully update {frequency} price data for instrument {instrument_id} in database {col} costing '
                 f'{elapsed_time: .2f} seconds.')
    else:
        print(f'No data for instrument {instrument_id}.')


def get_risk_free_rate(start_year: int = START_DATE.year,
                       end_year: int = END_DATE.year,
                       from_database: bool = True):
    """
    We use 10-year China National Bond rate as the risk-free rate.

    :param start_year: start_year
    :param end_year: end_year
    :param from_database: get data from database or not
    :return: None
    """

    if from_database:
        start_date = pd.to_datetime(str(start_year) + '0101')
        end_date = pd.to_datetime(str(end_year) + '1231')

        mongo_operator = {
            '$and': [
                {'date': {'$gte': start_date}},
                {'date': {'$lte': end_date}}
            ]}

        df_rfr = get_data(database='stock',
                          collection='risk_free_rate',
                          mongo_operator=mongo_operator)

    else:
        df_list = []
        for year in range(start_year, end_year + 1):
            start_date = str(year) + '0101'
            end_date = str(year) + '1231'
            df = ak.bond_china_yield(start_date, end_date)
            df = df.loc[df['曲线名称'] == '中债国债收益率曲线'][['曲线名称', '10年', '日期']].copy()
            df = df.rename(columns={'曲线名称': 'instrument_id', '10年': 'rate', '日期': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df['rate'] /= 100
            df_list.append(df)
        df_rfr = pd.concat(df_list)

    df_rfr = df_rfr.sort_values(by='date')

    return df_rfr.dropna()


def update_risk_free_rate(method: str = 'insert_many'):
    """
    Update risk free rate data.

    :param method: updating method
    :return: None
    """
    df_rfr = get_risk_free_rate(from_database=False)
    update_data(database='stock', collection='risk_free_rate', df=df_rfr, method=method)


def add_stock_suffix(instrument_id: str):
    if instrument_id.startswith('60') or instrument_id.startswith('688') or instrument_id.startswith('900'):
        return 'sh' + instrument_id
    elif instrument_id.startswith('00') or instrument_id.startswith('30') or instrument_id.startswith('200'):
        return 'sz' + instrument_id
    elif instrument_id.startswith('83') or instrument_id.startswith('87') or instrument_id.startswith('88') or \
            instrument_id.startswith('82') or instrument_id.startswith('889') or instrument_id.startswith('4'):
        return 'bj' + instrument_id
    else:
        raise StockDataError(f'Unknown instrument id: {instrument_id}.')


def remove_stock_suffix(instrument_id: Union[str, int]):
    if isinstance(instrument_id, int):
        instrument_id = str(instrument_id)
    if instrument_id[:2].isalpha():
        instrument_id = instrument_id[2:]
    assert instrument_id.isdigit() and len(instrument_id) == 6, f'Invalid value for instrument_id {instrument_id}.'
    return instrument_id


def clean_stock_price_data(df: pd.DataFrame):
    if len(df) == 0:
        return df
    for col in ['time', 'open', 'high', 'low', 'close', 'volume', 'instrument_id']:
        assert col in df.columns, f'DataFrame does not contain column {col}.'
    df['instrument_id'] = df['instrument_id'].apply(remove_stock_suffix)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['volume'] = df['volume'].astype(int)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume']].copy()

    df = df.drop_duplicates(subset=['time', 'instrument_id'])
    assert not any(df.duplicated(['time', 'instrument_id']))

    return df
