"""
This script is to get and deal with futures data based on akshare.
"""
import time
from typing import List, Union
import pandas as pd

import akshare as ak
from mongo.mongify import get_data, update_data
from utils.path import START_DATE_STR, END_DATE_STR, START_DATE, END_DATE
from utils.logging import log


def get_futures_continuous_contract_info(instrument_id: Union[str, List, None] = None,
                                         from_database: bool = True):
    """
    Get futures continuous contract info with optional filters.

    :param instrument_id: instrument_id
    :param from_database: get data from database or not
    :return: futures continuous contract info data
    """
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if not from_database:
        df_futures_info = ak.futures_display_main_sina()
        df_futures_info = df_futures_info.rename(columns={'symbol': 'instrument_id'})

        if instrument_id:
            df_futures_info = df_futures_info.loc[df_futures_info['instrument_id'].isin(instrument_id)]
    else:
        if instrument_id:
            mongo_operator = {'instrument_id': {'$in': instrument_id}}
            df_futures_info = get_data(database='futures',
                                       collection='continuous_contract_info',
                                       mongo_operator=mongo_operator)

        else:
            df_futures_info = get_data(database='futures',
                                       collection='continuous_contract_info')

    return df_futures_info


def update_futures_continuous_contract_info(instrument_id: Union[str, List, None] = None,
                                            method: str = 'insert_many'):
    """
    Update futures continuous contract info in database.

    :param instrument_id: the instrument ids need to be updated
    :param method: updating method
    :return: None
    """

    df_futures_info = get_futures_continuous_contract_info(instrument_id=instrument_id,
                                                           from_database=False)
    update_data(database='futures',
                collection='continuous_contract_info',
                df=df_futures_info,
                method=method,
                filter_column=['instrument_id'])

    log.info(f'Successfully update futures continuous contract info.')


def get_futures_continuous_contract_price(instrument_id: Union[str, List, None] = None,
                                          start_date: str = None,
                                          end_date: str = None,
                                          from_database: bool = True,
                                          wait_time: float = 0.3):
    """
    Get futures continuous contract daily price with optional filters.

    :param instrument_id: instrument_id
    :param start_date: start_date
    :param end_date: end_date
    :param from_database: get data from database or not
    :param wait_time: wait time between query from akshare
    :return: futures continuous contract daily price data
    """
    if not instrument_id:
        instrument_id = get_futures_continuous_contract_info(from_database=True)['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if not start_date:
        start_date = START_DATE_STR
    if not end_date:
        end_date = END_DATE_STR

    if not from_database:
        df_list = []
        for ins_id in instrument_id:
            df_futures = ak.futures_main_sina(symbol=ins_id, start_date=start_date, end_date=end_date)
            time.sleep(wait_time)
            df_futures['instrument_id'] = ins_id
            df_list.append(df_futures)
        df_futures_price = pd.concat(df_list)
        rename_dc = {'日期': 'time', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
                     '收盘价': 'close', '成交量': 'volume', '持仓量': 'position'}

        df_futures_price = df_futures_price.rename(columns=rename_dc)
        df_futures_price = df_futures_price[['time', 'instrument_id',
                                             'open', 'high', 'low', 'close', 'volume', 'position']].copy()
        df_futures_price = df_futures_price.loc[df_futures_price['instrument_id'].isin(instrument_id)]
        df_futures_price['time'] = pd.to_datetime(df_futures_price['time'])

    else:
        mongo_operator = {
            '$and': [
                {'time': {'$gte': pd.Timestamp(start_date)}},
                {'time': {'$lte': pd.Timestamp(end_date)}},
                {'instrument_id': {"$in": instrument_id}}
            ]}
        df_futures_price = get_data(database='futures',
                                    collection='continuous_contract_price_daily',
                                    mongo_operator=mongo_operator)

    return df_futures_price


def update_futures_continuous_contract_price(instrument_id: Union[str, List, None] = None,
                                             start_date: str = None,
                                             end_date: str = None,
                                             wait_time: float = 0.3,
                                             method: str = 'insert_many'):
    """
    Update futures continuous contract daily price in database.

    :param instrument_id: the instrument ids need to be updated
    :param start_date: start_date
    :param end_date: end_date
    :param wait_time: wait time between query from akshare
    :param method: updating method
    :return: None
    """

    if not instrument_id:
        instrument_id = get_futures_continuous_contract_info()['instrument_id'].tolist()

    df_futures_price = get_futures_continuous_contract_price(instrument_id=instrument_id,
                                                             start_date=start_date,
                                                             end_date=end_date,
                                                             from_database=False,
                                                             wait_time=wait_time)
    update_data(database='futures',
                collection='continuous_contract_price_daily',
                df=df_futures_price,
                method=method)

    log.info(f'Successfully update futures continuous contract daily price.')


def get_risk_free_rate(start_year: int = START_DATE.year,
                       end_year: int = END_DATE.year,
                       from_database: bool = True):
    """
    Use 10-year China National Bond yield as risk-free rate.
    """
    if from_database:
        start_date = pd.to_datetime(str(start_year) + '0101')
        end_date = pd.to_datetime(str(end_year) + '1231')
        mongo_operator = {
            '$and': [
                {'date': {'$gte': start_date}},
                {'date': {'$lte': end_date}},
            ]
        }
        df_rfr = get_data(database='futures',
                          collection='risk_free_rate',
                          mongo_operator=mongo_operator)
    else:
        df_list = []
        for year in range(start_year, end_year + 1):
            start_date = f'{year}0101'
            end_date = f'{year}1231'
            df = ak.bond_china_yield(start_date, end_date)
            df = df.loc[df['曲线名称'] == '中债国债收益率曲线'][['曲线名称', '10年', '日期']].copy()
            df = df.rename(columns={'曲线名称': 'instrument_id', '10年': 'rate', '日期': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df['rate'] /= 100
            df_list.append(df)
        df_rfr = pd.concat(df_list) if df_list else pd.DataFrame(columns=['instrument_id', 'rate', 'date'])

    return df_rfr.sort_values(by='date').dropna()


def update_risk_free_rate(method: str = 'insert_many'):
    """
    Update risk free rate data in futures database.
    """
    df_rfr = get_risk_free_rate(from_database=False)
    update_data(database='futures', collection='risk_free_rate', df=df_rfr, method=method)
