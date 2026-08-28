"""
This script is to get and deal with futures data based on akshare.
"""
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
import numpy as np

import akshare as ak
from mongo.mongify import get_data, update_data
from utils.params import (
    RESEARCH_START_DATE,
    FUTURES_FIXED_LISTING_MONTHS,
)
from utils.logging import log

# 数据来源标识: 每条价格记录都带 source 字段
SOURCE_AKSHARE = 'akshare'        # 日频数据(akshare 接口)
SOURCE_JOINQUANT = 'joinquant'    # 分钟频数据(聚宽), 及由其聚合的日频
SOURCE_EDB = 'tqsdk_edb'          # 分钟频数据(天勤 EDB 免费接口)


class UpdateCancelledError(RuntimeError):
    """Raised when an update task is cancelled by user."""


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        raise UpdateCancelledError('Update cancelled by user.')


def _today_date_str() -> str:
    return date.today().strftime('%Y%m%d')


def get_trading_days(start_date: str, end_date: str) -> List[pd.Timestamp]:
    """Return a list of Chinese futures trading days between start_date and end_date (inclusive).

    Uses the ``chinese_calendar`` package which covers Chinese public holidays
    and weekend make-up workdays. Futures exchanges (SHFE/DCE/CZCE/CFFEX)
    follow the same holiday schedule as the national statutory holidays.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYYMMDD' or 'YYYY-MM-DD' format.

    Returns
    -------
    List[pd.Timestamp]
        Sorted list of trading day timestamps.
    """
    import chinese_calendar as cc

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_days = pd.date_range(start, end, freq='D')
    trading_days = [d for d in all_days if cc.is_workday(d.date())]
    return trading_days


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
                                            method: str = 'bulk_write_update'):
    """
    Update futures continuous contract info in database.
    If a record with the same instrument_id already exists, it will be skipped.

    :param instrument_id: the instrument ids need to be updated
    :param method: updating method
    :return: None
    """

    df_futures_info = get_futures_continuous_contract_info(instrument_id=instrument_id,
                                                           from_database=False)

    # Filter out records that already exist in DB (by instrument_id)
    try:
        df_existing = get_futures_continuous_contract_info(instrument_id=None, from_database=True)
        if df_existing is not None and not df_existing.empty:
            existing_ids = set(df_existing['instrument_id'].dropna().unique())
            before_count = len(df_futures_info)
            df_futures_info = df_futures_info[~df_futures_info['instrument_id'].isin(existing_ids)]
            skipped = before_count - len(df_futures_info)
            if skipped > 0:
                log.info(f'Skipped {skipped} existing instrument(s), {len(df_futures_info)} new to insert.')
    except Exception:
        pass  # If DB query fails, proceed with full insert

    if df_futures_info.empty:
        log.info('No new continuous contract info to insert (all already exist).')
        return

    update_data(database='futures',
                collection='continuous_contract_info',
                df=df_futures_info,
                method=method,
                filter_column=['instrument_id'])

    log.info(f'Successfully update futures continuous contract info ({len(df_futures_info)} records).')


def get_futures_continuous_contract_price(instrument_id: Union[str, List, None] = None,
                                          start_date: str = None,
                                          end_date: str = None,
                                          from_database: bool = True,
                                          load_prev_weighted_factor: bool = True,
                                          wait_time: float = 2.0,
                                          cancel_event=None,
                                          source: Optional[Union[str, List[str]]] = 'joinquant'):
    """Get futures continuous contract daily price with optional filters.

    source: 数据来源过滤, 支持单个字符串或列表。默认 'joinquant' 只读 joinquant 日频
            (由聚宽分钟聚合而来, 与分钟/回测口径一致; 兼容无 source 旧记录);
            传 'akshare' 读取 akshare 日频; 传 None/'' 表示不过滤(返回全部来源, 注意可能多行);
            传列表如 ['akshare','joinquant'] 表示同时读取多个来源。
    """
    """
    Get futures continuous contract daily price with optional filters.

    :param instrument_id: instrument_id
    :param start_date: start_date
    :param end_date: end_date
    :param from_database: get data from database or not
    :param load_prev_weighted_factor: when building continuous data, whether to continue
        weighted_factor from the latest DB record before start_date.
    :param wait_time: wait time between query from akshare
    :return: futures continuous contract daily price data
    """
    if not instrument_id:
        instrument_id = get_futures_continuous_contract_info(from_database=True)['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if not start_date:
        # For continuous-contract research/update, default anchor starts from RESEARCH_START_DATE.
        start_date = RESEARCH_START_DATE
    if not end_date:
        end_date = _today_date_str()

    if not from_database:
        df_list = []
        for idx, ins_id in enumerate(instrument_id, 1):
            _raise_if_cancelled(cancel_event)
            root_instrument = _to_root_instrument(ins_id)
            log.info(f'[{idx}/{len(instrument_id)}] 正在获取 {ins_id} (root={root_instrument}) 的连续合约数据...')
            df_futures = build_roll_adjusted_continuous_contract_price(
                instrument_id=root_instrument,
                start_date=start_date,
                end_date=end_date,
                from_database=False,
                continuous_instrument_id=ins_id,
                load_prev_weighted_factor=load_prev_weighted_factor,
                wait_time=wait_time,
                research_start_date=RESEARCH_START_DATE,
                cancel_event=cancel_event,
            )
            if not isinstance(df_futures, pd.DataFrame):
                continue
            df_futures['instrument_id'] = ins_id
            log.info(f'[{idx}/{len(instrument_id)}] {ins_id} 获取完成, {len(df_futures)} 行')
            if not df_futures.empty:
                df_list.append(df_futures)
        if not df_list:
            return pd.DataFrame(columns=[
                'time', 'instrument_id', 'symbol',
                'open', 'high', 'low', 'close', 'settle',
                'volume', 'position',
                'weighted_factor', 'cur_weighted_factor', 'is_rollover',
            ])
        df_futures_price = pd.concat(df_list, ignore_index=True)
        df_futures_price = df_futures_price.loc[df_futures_price['instrument_id'].isin(instrument_id)]
        df_futures_price['time'] = pd.to_datetime(df_futures_price['time'])

    else:
        mongo_operator = {
            '$and': [
                {'time': {'$gte': pd.Timestamp(start_date)}},
                {'time': {'$lte': pd.Timestamp(end_date)}},
                {'instrument_id': {"$in": instrument_id}}
            ]}
        # 按 source 过滤(默认 akshare), 避免 joinquant/tqsdk_edb 日频混入重复
        if isinstance(source, (list, tuple)):
            src_list = [s for s in source if s]
        elif source:
            src_list = [source]
        else:
            src_list = None
        if src_list:
            mongo_operator['$and'].append({'source': {'$in': src_list + [None]}})
        df_futures_price = get_data(database='futures',
                                    collection='continuous_contract_price_daily',
                                    mongo_operator=mongo_operator)

    return df_futures_price


def update_futures_continuous_contract_price(instrument_id: Union[str, List, None] = None,
                                             start_date: str = None,
                                             end_date: str = None,
                                             load_prev_weighted_factor: bool = True,
                                             wait_time: float = 2.0,
                                             method: str = 'bulk_write_update',
                                             only_update_new: bool = False,
                                             cancel_event=None):
    """
    Update futures continuous contract daily price in database.

    :param instrument_id: the instrument ids need to be updated
    :param start_date: start_date
    :param end_date: end_date
    :param load_prev_weighted_factor: if True, continue weighted_factor from DB record
        before start_date; otherwise start from 1.0 behavior.
    :param wait_time: wait time between query from akshare
    :param method: updating method
    :param only_update_new:
        - False: 按传入区间直接拉取并写入（原行为）。
        - True: 仅拉取数据库中不存在的【日期, 合约】组合。
          实现方式：先读取 DB 已有记录，按交易日历计算缺失日期段，再仅对缺失段调用行情接口。
    :return: None
    """

    if not instrument_id:
        instrument_id = get_futures_continuous_contract_info()['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    log.info(f'开始更新 {len(instrument_id)} 个合约的价格数据: {instrument_id}')

    # Keep default aligned with the research back-adjustment anchor date.
    start_date = start_date or RESEARCH_START_DATE
    end_date = end_date or _today_date_str()

    if not only_update_new:
        _raise_if_cancelled(cancel_event)
        df_futures_price = get_futures_continuous_contract_price(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            from_database=False,
            load_prev_weighted_factor=load_prev_weighted_factor,
            wait_time=wait_time,
            cancel_event=cancel_event,
        )
    else:
        # only_update_new=True: 简化策略
        # 仅基于“数据库已有最早/最晚日期”与输入区间做前补/后补，
        # 不再对区间内部缺口做细粒度分段补齐。
        _raise_if_cancelled(cancel_event)
        req_start_ts = pd.Timestamp(start_date)
        req_end_ts = pd.Timestamp(end_date)
        if req_start_ts > req_end_ts:
            raise ValueError(f'start_date > end_date: {start_date} > {end_date}')

        update_ranges: Dict[str, List[tuple[str, str]]] = {}

        def _to_trading_range(seg_start_ts: pd.Timestamp,
                              seg_end_ts: pd.Timestamp) -> Optional[tuple[str, str]]:
            """Convert a calendar segment to [first_trading_day, last_trading_day]."""
            if seg_start_ts > seg_end_ts:
                return None
            tds = get_trading_days(
                start_date=seg_start_ts.strftime('%Y%m%d'),
                end_date=seg_end_ts.strftime('%Y%m%d'),
            )
            if not tds:
                return None
            return tds[0].strftime('%Y%m%d'), tds[-1].strftime('%Y%m%d')
        for ins in instrument_id:
            ins_key = str(ins)
            mongo_operator = {
                '$and': [
                    {'instrument_id': ins_key},
                    {'time': {'$gte': req_start_ts}},
                    {'time': {'$lte': req_end_ts}},
                ]
            }
            df_existing = get_data(
                database='futures',
                collection='continuous_contract_price_daily',
                mongo_operator=mongo_operator,
            )

            # DB 在请求区间内无数据 => 整段都需要更新。
            if not isinstance(df_existing, pd.DataFrame) or df_existing.empty:
                whole_seg = _to_trading_range(req_start_ts, req_end_ts)
                if whole_seg is not None:
                    update_ranges[ins_key] = [whole_seg]
                    days_all = get_trading_days(start_date=whole_seg[0], end_date=whole_seg[1])
                else:
                    days_all = []
                log.info(
                    f'only_update_new=True instrument={ins_key}, '
                    f'existing_range=none, update_ranges={update_ranges.get(ins_key, [])}, '
                    f'update_trading_dates_count={len(days_all)}'
                )
                continue

            df_existing = df_existing.copy()
            df_existing['time'] = pd.to_datetime(df_existing['time'], errors='coerce')
            df_existing = df_existing.dropna(subset=['time'])
            if df_existing.empty:
                whole_seg = _to_trading_range(req_start_ts, req_end_ts)
                if whole_seg is not None:
                    update_ranges[ins_key] = [whole_seg]
                    days_all = get_trading_days(start_date=whole_seg[0], end_date=whole_seg[1])
                else:
                    days_all = []
                log.info(
                    f'only_update_new=True instrument={ins_key}, '
                    f'existing_range=invalid, update_ranges={update_ranges.get(ins_key, [])}, '
                    f'update_trading_dates_count={len(days_all)}'
                )
                continue

            db_min = pd.Timestamp(df_existing['time'].min())
            db_max = pd.Timestamp(df_existing['time'].max())

            ranges: List[tuple[str, str]] = []
            # 前补区间: [req_start, db_min-1]
            if req_start_ts < db_min:
                front_end = min(req_end_ts, db_min - pd.Timedelta(days=1))
                if req_start_ts <= front_end:
                    front_seg = _to_trading_range(req_start_ts, front_end)
                    if front_seg is not None:
                        ranges.append(front_seg)

            # 后补区间: [db_max+1, req_end]
            if req_end_ts > db_max:
                back_start = max(req_start_ts, db_max + pd.Timedelta(days=1))
                if back_start <= req_end_ts:
                    back_seg = _to_trading_range(back_start, req_end_ts)
                    if back_seg is not None:
                        ranges.append(back_seg)

            if ranges:
                update_ranges[ins_key] = ranges

            update_days_count = 0
            for r_start, r_end in ranges:
                update_days_count += len(get_trading_days(start_date=r_start, end_date=r_end))
            log.info(
                f'only_update_new=True instrument={ins_key}, '
                f'existing_range=[{db_min.strftime("%Y%m%d")}, {db_max.strftime("%Y%m%d")}], '
                f'update_ranges={ranges if ranges else "[]"}, '
                f'update_trading_dates_count={update_days_count}'
            )

        if not update_ranges:
            log.info(
                f'only_update_new=True: 数据库中已包含全部目标【日期, 合约】, 无需更新。'
                f'range=[{start_date}, {end_date}], instruments={instrument_id}'
            )
            return

        # 针对每个合约仅拉取缺失日期段，减少不必要的 AkShare 请求和写入。
        df_list: List[pd.DataFrame] = []
        total_segments = sum(len(v) for v in update_ranges.values())
        seg_idx = 0
        for ins_key, ranges in update_ranges.items():
            for seg_start, seg_end in ranges:
                _raise_if_cancelled(cancel_event)
                seg_idx += 1
                log.info(
                    f'only_update_new=True [{seg_idx}/{total_segments}] '
                    f'更新缺失段 instrument={ins_key}, range=[{seg_start}, {seg_end}]'
                )
                df_seg = get_futures_continuous_contract_price(
                    instrument_id=[ins_key],
                    start_date=seg_start,
                    end_date=seg_end,
                    from_database=False,
                    load_prev_weighted_factor=load_prev_weighted_factor,
                    wait_time=wait_time,
                    cancel_event=cancel_event,
                )
                if isinstance(df_seg, pd.DataFrame) and not df_seg.empty:
                    df_list.append(df_seg)

        if not df_list:
            log.warning('only_update_new=True: 目标缺失段均未拉取到可写入数据，跳过写入。')
            return
        df_futures_price = pd.concat(df_list, ignore_index=True)

    if df_futures_price is None or df_futures_price.empty:
        log.warning('所有合约均无数据，跳过写入。')
        return

    log.info(f'共获取 {len(df_futures_price)} 行数据，开始写入数据库...')

    # 日频 akshare 来源: 加 source 字段, 并与 joinquant 聚合的日频共存(唯一键含 source)
    df_futures_price = df_futures_price.copy()
    df_futures_price['source'] = SOURCE_AKSHARE
    update_data(database='futures',
                collection='continuous_contract_price_daily',
                df=df_futures_price,
                method=method,
                filter_column=['time', 'instrument_id', 'source'])

    log.info(f'Successfully update futures continuous contract daily price ({len(df_futures_price)} rows, source={SOURCE_AKSHARE}).')


def _to_root_instrument(instrument_id: str) -> str:
    ins = str(instrument_id).upper().strip()
    if not ins:
        raise ValueError('instrument_id is empty.')
    return ins[:-1] if ins.endswith('0') else ins


def get_available_symbol(instrument_id: str,
                         year: Union[str, int],
                         month_list: Optional[Sequence[int]] = None,
                         wait_time: float = 0.5,
                         cancel_event=None) -> List[str]:
    """Return available listed contract symbols for one product and year.

    Example: instrument_id='C', year='2025' -> ['C2501', 'C2505', ...]
    """
    root = _to_root_instrument(instrument_id)
    yy = str(year).strip()[-2:]
    # If this product has configured fixed listing months, directly return symbols
    # instead of probing all months via AkShare.
    fixed_months = FUTURES_FIXED_LISTING_MONTHS.get(root)
    if fixed_months:
        base_months = [int(m) for m in fixed_months]
        if month_list is not None:
            month_set = {int(x) for x in month_list}
            base_months = [m for m in base_months if m in month_set]
        return [f'{root}{yy}{m:02d}' for m in base_months]

    months = list(month_list) if month_list is not None else list(range(1, 13))

    available: List[str] = []
    for m in months:
        _raise_if_cancelled(cancel_event)
        symbol = f'{root}{yy}{int(m):02d}'
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                available.append(symbol)
        except Exception:
            pass
        if wait_time > 0:
            time.sleep(wait_time)
    return available


def _normalize_zh_daily_symbol_df(df_raw: pd.DataFrame,
                                  symbol: str) -> pd.DataFrame:
    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame()
    df = df_raw.copy()
    rename_dc: Dict[str, str] = {
        'date': 'time',
        'hold': 'position',
    }
    df = df.rename(columns=rename_dc)
    required = ['time', 'open', 'high', 'low', 'close', 'volume', 'position']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'futures_zh_daily_sina({symbol}) missing columns: {missing}')
    if 'settle' not in df.columns:
        df['settle'] = df['close']

    df['time'] = pd.to_datetime(df['time'])
    for c in ['open', 'high', 'low', 'close', 'settle', 'volume', 'position']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['settle'] = df['settle'].fillna(df['close'])
    df['volume'] = df['volume'].fillna(0.0)
    df['position'] = df['position'].fillna(0.0)
    df = df.dropna(subset=['time', 'open', 'high', 'low', 'close'])
    df['symbol'] = symbol
    return df[['time', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position']]


def get_futures_symbol_info(instrument_id: Union[str, List, None] = None,
                            start_date: str = None,
                            end_date: str = None,
                            wait_time: float = 0.5,
                            cancel_event=None) -> List[str]:
    """Get available listed symbols for one/many products in a date range."""
    if not instrument_id:
        instrument_id = get_futures_continuous_contract_info(from_database=True)['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    if not start_date:
        start_date = RESEARCH_START_DATE
    if not end_date:
        end_date = _today_date_str()

    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    years = list(range(start_year - 1, end_year + 2))
    symbols: List[str] = []
    for ins_id in instrument_id:
        _raise_if_cancelled(cancel_event)
        root = _to_root_instrument(ins_id)
        for y in years:
            symbols.extend(get_available_symbol(
                instrument_id=root,
                year=y,
                wait_time=wait_time,
                cancel_event=cancel_event,
            ))
    return sorted(list(dict.fromkeys(symbols)))


def _infer_root_from_symbol(symbol: str) -> str:
    s = str(symbol).upper().strip()
    if not s:
        return s
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    return s[:i]


def get_futures_symbol_price(instrument_id: Union[str, List, None] = None,
                             symbol_list: Union[str, List, None] = None,
                             *,
                             start_date: str,
                             end_date: str,
                             from_database: bool = True,
                             wait_time: float = 2.0,
                             cancel_event=None) -> pd.DataFrame:
    """Get symbol-level futures daily price either from DB cache or AkShare API.

    Output columns include:
    ['time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position']
    """
    if not str(start_date or '').strip() or not str(end_date or '').strip():
        raise ValueError('get_futures_symbol_price requires non-empty start_date and end_date.')

    if isinstance(symbol_list, str):
        symbol_list = [symbol_list]

    if not symbol_list:
        symbol_list = get_futures_symbol_info(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            wait_time=min(wait_time, 0.5),
            cancel_event=cancel_event,
        )

    if not symbol_list:
        log.warning(f'No symbol found for instrument_id={instrument_id}, range=[{start_date}, {end_date}].')
        return pd.DataFrame(columns=[
            'time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position'
        ])

    if from_database:
        mongo_operator = {
            '$and': [
                {'time': {'$gte': pd.Timestamp(start_date)}},
                {'time': {'$lte': pd.Timestamp(end_date)}},
                {'symbol': {'$in': list(symbol_list)}},
            ]
        }
        df = get_data(database='futures', collection='symbol_price_daily', mongo_operator=mongo_operator)
        if not isinstance(df, pd.DataFrame) or df.empty:
            log.warning(
                f'No symbol price found in DB futures.symbol_price_daily for symbols={len(symbol_list)}, '
                f'range=[{start_date}, {end_date}]'
            )
            return pd.DataFrame(columns=[
                'time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position'
            ])
        if 'instrument_id' not in df.columns:
            df['instrument_id'] = df['symbol'].map(_infer_root_from_symbol)
        return df.sort_values(['symbol', 'time']).reset_index(drop=True)

    df_list: List[pd.DataFrame] = []
    for symbol in symbol_list:
        _raise_if_cancelled(cancel_event)
        try:
            df_raw = ak.futures_zh_daily_sina(symbol=symbol)
            df_symbol = _normalize_zh_daily_symbol_df(df_raw, symbol=symbol)
            if df_symbol.empty:
                log.warning(f'{symbol} has no valid data from ak.futures_zh_daily_sina, skip.')
                continue
            root = _infer_root_from_symbol(symbol)
            df_symbol['instrument_id'] = root
            df_symbol = df_symbol[(df_symbol['time'] >= pd.Timestamp(start_date)) & (df_symbol['time'] <= pd.Timestamp(end_date))]
            if not df_symbol.empty:
                df_list.append(df_symbol)
                log.info(f'Fetched symbol={symbol} from ak.futures_zh_daily_sina, rows={len(df_symbol)}, range=[{start_date}, {end_date}]')
            else:
                log.warning(f'symbol={symbol} has no data in the specified date range from ak.futures_zh_daily_sina, skip.')
        except Exception as e:
            log.warning(f'Failed to fetch symbol={symbol} from ak.futures_zh_daily_sina: {e}')
        if wait_time > 0:
            time.sleep(wait_time)

    if not df_list:
        return pd.DataFrame(columns=[
            'time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position'
        ])
    out = pd.concat(df_list, ignore_index=True)
    return out[['time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle', 'volume', 'position']]


def update_futures_symbol_price(instrument_id: Union[str, List, None] = None,
                                symbol_list: Union[str, List, None] = None,
                                *,
                                start_date: str,
                                end_date: str,
                                wait_time: float = 2.0,
                                method: str = 'insert_many') -> None:
    """Update symbol-level futures daily price into futures.symbol_price_daily.

    For each symbol, write to DB immediately and log success/failure explicitly.
    """
    if not str(start_date or '').strip() or not str(end_date or '').strip():
        raise ValueError('update_futures_symbol_price requires non-empty start_date and end_date.')

    if isinstance(symbol_list, str):
        symbol_list = [symbol_list]
    if not symbol_list:
        symbol_list = get_futures_symbol_info(
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            wait_time=min(wait_time, 0.5),
        )

    if not symbol_list:
        log.warning('No symbols to update for futures.symbol_price_daily.')
        return

    success_symbols: List[str] = []
    failed_symbols: List[str] = []
    for symbol in symbol_list:
        try:
            df_symbol = get_futures_symbol_price(
                symbol_list=[symbol],
                start_date=start_date,
                end_date=end_date,
                from_database=False,
                wait_time=wait_time,
            )
            if df_symbol.empty:
                failed_symbols.append(symbol)
                log.warning(
                    f'[symbol_price_daily] skip empty symbol={symbol}, range=[{start_date}, {end_date}]'
                )
                continue

            update_data(
                database='futures',
                collection='symbol_price_daily',
                df=df_symbol,
                method=method,
                filter_column=['time', 'symbol'],
            )
            success_symbols.append(symbol)
            log.info(
                f'[symbol_price_daily] updated symbol={symbol}, rows={len(df_symbol)}, '
                f'range=[{start_date}, {end_date}], method={method}'
            )
        except Exception as e:
            failed_symbols.append(symbol)
            log.warning(f'[symbol_price_daily] failed symbol={symbol}: {e}')

    log.info(
        f'update_futures_symbol_price finished: success={len(success_symbols)}, failed={len(failed_symbols)}'
    )
    if failed_symbols:
        log.warning(f'Failed symbols: {failed_symbols}')


def _empty_continuous_price_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'time', 'symbol',
        'open', 'high', 'low', 'close', 'settle',
        'volume', 'position',
        'weighted_factor', 'cur_weighted_factor', 'is_rollover',
    ])


def _load_prev_weighted_factor(continuous_instrument_id: str,
                               start_date: str) -> float:
    """Load weighted_factor from the latest DB row before start_date.

    Raises
    ------
    ValueError
        If no previous row exists, required columns are missing, or weighted_factor is invalid.
    """
    mongo_operator = {
        '$and': [
            {'instrument_id': continuous_instrument_id},
            {'time': {'$lt': pd.Timestamp(start_date)}},
            {'source': {'$in': [SOURCE_AKSHARE, None]}},
        ]
    }
    df_prev = get_data(
        database='futures',
        collection='continuous_contract_price_daily',
        mongo_operator=mongo_operator,
    )
    if not isinstance(df_prev, pd.DataFrame) or df_prev.empty:
        raise ValueError(
            f'No previous continuous price data found before start_date. '
            f'instrument_id={continuous_instrument_id}, start_date={start_date}'
        )

    df_prev = df_prev.copy()
    if 'time' not in df_prev.columns or 'weighted_factor' not in df_prev.columns:
        raise ValueError(
            f'Previous continuous data missing required columns. '
            f'instrument_id={continuous_instrument_id}, start_date={start_date}, '
            f'columns={list(df_prev.columns)}'
        )
    df_prev['time'] = pd.to_datetime(df_prev['time'], errors='coerce')
    df_prev['weighted_factor'] = pd.to_numeric(df_prev['weighted_factor'], errors='coerce')
    df_prev = df_prev.dropna(subset=['time', 'weighted_factor'])
    if df_prev.empty:
        raise ValueError(
            f'Previous continuous data has no valid (time, weighted_factor). '
            f'instrument_id={continuous_instrument_id}, start_date={start_date}'
        )

    df_prev = df_prev.sort_values('time', ascending=False)
    last_row = df_prev.iloc[0]
    last_time = pd.Timestamp(last_row['time']).strftime('%Y-%m-%d')
    last_wf = float(last_row['weighted_factor'])
    if not np.isfinite(last_wf) or last_wf <= 0:
        raise ValueError(
            f'Invalid previous weighted_factor. '
            f'instrument_id={continuous_instrument_id}, start_date={start_date}, '
            f'last_time={last_time}, weighted_factor={last_wf}'
        )
    return last_wf


def _build_roll_adjusted_continuous_from_panel(df_panel: pd.DataFrame,
                                               start_date: str,
                                               end_date: str,
                                               instrument_id: str,
                                               research_start_date: str,
                                               initial_weighted_factor: float = 1.0) -> pd.DataFrame:
    if df_panel.empty:
        return _empty_continuous_price_df()

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    research_ts = pd.to_datetime(research_start_date)

    panel = df_panel.copy()
    panel = panel[(panel['time'] >= start_ts) & (panel['time'] <= end_ts)].copy()
    if panel.empty:
        return _empty_continuous_price_df()

    panel = panel.sort_values(['time', 'symbol']).reset_index(drop=True)
    # Build per-date contract ranking matrices.
    # We use volume as the primary dominant criterion, while checking consistency with
    # position-based dominant symbol for diagnostics.
    vol_df = panel.pivot_table(index='time', columns='symbol', values='volume', aggfunc='last').sort_index()
    pos_df = panel.pivot_table(index='time', columns='symbol', values='position', aggfunc='last').sort_index()

    dominant_by_volume = vol_df.idxmax(axis=1)
    dominant_by_position = pos_df.idxmax(axis=1)

    # Primary dominant symbol is volume-max.
    dominant_today = dominant_by_volume.copy()

    # If volume-max and position-max are different, we log detailed diagnostics.
    mismatch_mask = (
        dominant_by_volume.notna()
        & dominant_by_position.notna()
        & (dominant_by_volume != dominant_by_position)
    )
    for t in dominant_today.index[mismatch_mask]:
        vol_symbol = str(dominant_by_volume.loc[t])
        pos_symbol = str(dominant_by_position.loc[t])
        vol_value = pd.to_numeric(vol_df.loc[t, vol_symbol], errors='coerce') if vol_symbol in vol_df.columns else np.nan
        pos_value = pd.to_numeric(pos_df.loc[t, pos_symbol], errors='coerce') if pos_symbol in pos_df.columns else np.nan
        log.warning(
            '[DominantMismatch] '
            f'instrument={instrument_id}, date={pd.Timestamp(t).strftime("%Y-%m-%d")}, '
            f'volume_symbol={vol_symbol}, volume={float(vol_value) if pd.notna(vol_value) else np.nan}, '
            f'position_symbol={pos_symbol}, position={float(pos_value) if pd.notna(pos_value) else np.nan}, '
            'decision=use_volume_symbol'
        )
    dominant_used = dominant_today.shift(1)
    if not dominant_used.empty:
        dominant_used.iloc[0] = dominant_today.iloc[0]
    dominant_used = dominant_used.ffill().fillna(dominant_today)

    panel_indexed = panel.set_index(['time', 'symbol']).sort_index()
    time_list = dominant_used.index.tolist()

    weighted_factor = float(initial_weighted_factor)
    cur_weighted_factor = 1.0
    started = False
    prev_symbol = None
    rows: List[Dict[str, object]] = []

    def _row_by_key(key: tuple) -> Optional[pd.Series]:
        if key not in panel_indexed.index:
            return None
        row = panel_indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row

    for t in time_list:
        symbol = dominant_used.loc[t]
        if pd.isna(symbol):
            continue
        symbol = str(symbol)

        row_key = (t, symbol)
        row = _row_by_key(row_key)
        if row is None:
            fallback = dominant_today.loc[t]
            if pd.isna(fallback):
                continue
            symbol = str(fallback)
            row_key = (t, symbol)
            row = _row_by_key(row_key)
            if row is None:
                continue

        if not started and t >= research_ts:
            weighted_factor = float(initial_weighted_factor)
            cur_weighted_factor = 1.0
            started = True

        is_rollover = bool(prev_symbol is not None and symbol != prev_symbol)
        if started and is_rollover:
            cur_ratio = 1.0
            old_row = _row_by_key((t, prev_symbol))
            new_row = _row_by_key((t, symbol))
            if old_row is not None and new_row is not None:
                old_open = float(pd.to_numeric(old_row.get('open'), errors='coerce'))
                new_open = float(pd.to_numeric(new_row.get('open'), errors='coerce'))
                if np.isfinite(old_open) and np.isfinite(new_open) and abs(new_open) > 1e-12:
                    cur_ratio = old_open / new_open
            cur_weighted_factor = float(cur_ratio)
            weighted_factor = float(weighted_factor) * float(cur_ratio)

        # Keep raw unadjusted prices in output.
        # Back-adjusted prices should be calculated on demand via: raw_price * weighted_factor.
        adj = float(weighted_factor) if started else 1.0
        rows.append({
            'time': t,
            'symbol': symbol,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'settle': float(row['settle']) if pd.notna(row['settle']) else np.nan,
            'volume': float(row['volume']) if pd.notna(row['volume']) else np.nan,
            'position': float(row['position']) if pd.notna(row['position']) else np.nan,
            'weighted_factor': float(adj),
            'cur_weighted_factor': float(cur_weighted_factor if started else 1.0),
            'is_rollover': bool(started and is_rollover),
        })
        prev_symbol = symbol

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values('time').reset_index(drop=True)
    return out


def build_roll_adjusted_continuous_contract_price(instrument_id: str,
                                                  start_date: str,
                                                  end_date: str,
                                                  from_database: bool = True,
                                                  continuous_instrument_id: Optional[str] = None,
                                                  load_prev_weighted_factor: bool = True,
                                                  wait_time: float = 2.0,
                                                  research_start_date: str = RESEARCH_START_DATE,
                                                  cancel_event=None) -> pd.DataFrame:
    """Build continuous daily price from symbol-level data with anti-leakage rollover rule.

    Output prices are RAW (non-adjusted). Use `price * weighted_factor` when adjusted
    prices are needed in research/backtest.
    """
    root = _to_root_instrument(instrument_id)
    continuous_id = continuous_instrument_id or (instrument_id if str(instrument_id).endswith('0') else f'{root}0')
    log.info(f'[continuous] {continuous_id}: 获取可用合约列表 ({start_date}~{end_date})...')
    symbols = get_futures_symbol_info(
        instrument_id=root,
        start_date=start_date,
        end_date=end_date,
        wait_time=min(wait_time, 0.5),
        cancel_event=cancel_event,
    )
    if not symbols:
        log.warning(f'No available symbols found for instrument={root} in range [{start_date}, {end_date}].')
        return _empty_continuous_price_df()

    log.info(f'[continuous] {continuous_id}: 找到 {len(symbols)} 个合约, 开始获取价格数据...')
    panel_df = get_futures_symbol_price(
        instrument_id=root,
        symbol_list=symbols,
        start_date=start_date,
        end_date=end_date,
        from_database=from_database,
        wait_time=wait_time,
        cancel_event=cancel_event,
    )
    if panel_df.empty:
        if from_database:
            log.warning(
                f'No symbol data in DB for instrument={root}. '
                f'Please run update_futures_symbol_price first. range=[{start_date}, {end_date}]'
            )
        else:
            log.warning(f'No symbol data from AkShare for instrument={root}, range=[{start_date}, {end_date}]')
        return pd.DataFrame()

    initial_weighted_factor = 1.0
    if load_prev_weighted_factor:
        try:
            initial_weighted_factor = _load_prev_weighted_factor(
                continuous_instrument_id=continuous_id,
                start_date=start_date,
            )
            log.info(
                f'[continuous][weighted_factor] instrument={continuous_id}, '
                f'start_date={start_date}, initial_weighted_factor={initial_weighted_factor}'
            )
        except Exception as e:
            # Strict mode by requirement: stop immediately instead of fallback=1.0
            log.error(
                f'[continuous][weighted_factor] strict load failed, terminate update. '
                f'instrument_id={continuous_id}, start_date={start_date}, error={e}'
            )
            raise

    return _build_roll_adjusted_continuous_from_panel(
        df_panel=panel_df,
        start_date=start_date,
        end_date=end_date,
        instrument_id=root,
        research_start_date=research_start_date,
        initial_weighted_factor=initial_weighted_factor,
    )


def compare_with_ak_main_continuous(instrument_id: str,
                                    start_date: str,
                                    end_date: str,
                                    wait_time: float = 2.0,
                                    atol: float = 1e-8) -> pd.DataFrame:
    """Compare custom stitched continuous vs ak.futures_main_sina; return mismatch rows."""
    root = _to_root_instrument(instrument_id)
    custom_df = build_roll_adjusted_continuous_contract_price(
        instrument_id=root,
        start_date=start_date,
        end_date=end_date,
        from_database=False,
        wait_time=wait_time,
        research_start_date=RESEARCH_START_DATE,
    )
    if custom_df.empty:
        return pd.DataFrame()

    main_df = ak.futures_main_sina(symbol=f'{root}0', start_date=start_date, end_date=end_date)
    rename_dc = {
        '日期': 'time',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'position',
    }
    main_df = main_df.rename(columns=rename_dc)
    main_df['time'] = pd.to_datetime(main_df['time'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'position']:
        if c in main_df.columns:
            main_df[c] = pd.to_numeric(main_df[c], errors='coerce')

    custom_cmp = custom_df[['time', 'open', 'high', 'low', 'close', 'symbol', 'is_rollover']].copy()
    merged = custom_cmp[['time', 'symbol', 'is_rollover', 'open', 'high', 'low', 'close']].merge(
        main_df[['time', 'open', 'high', 'low', 'close']],
        on='time', how='inner', suffixes=('_custom', '_main')
    )
    if merged.empty:
        return pd.DataFrame()

    mismatch_mask = np.zeros(len(merged), dtype=bool)
    for c in ['open', 'high', 'low', 'close']:
        left = pd.to_numeric(merged[f'{c}_custom'], errors='coerce')
        right = pd.to_numeric(merged[f'{c}_main'], errors='coerce')
        mismatch_mask |= ~np.isclose(left, right, atol=atol, rtol=0.0, equal_nan=True)
    return merged.loc[mismatch_mask].copy().reset_index(drop=True)


def get_risk_free_rate(start_year: int = int(RESEARCH_START_DATE[:4]),
                       end_year: int = date.today().year,
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


# ================== 分钟频率价格更新（天勤 EDB 免费接口） ==================
# 数据源: 天勤 EDB 行情历史服务 https://edb.shinnytech.com
#   免费可获取「近 1 年」的主力连续合约 1 分钟线(period=60), 无需 token。
#   主连 symbol: KQ.m@{交易所}.{品种}  郑商所品种代码大写, 其余小写。
# 换月因子(symbol/weighted_factor/is_rollover)复用日频库 continuous_contract_price_daily。
# 写入 collection: futures.continuous_contract_price_1min。

_FUTURES_ROOT_TO_EXCHANGE: Dict[str, str] = {
    # 上期所 SHFE
    'RB': 'SHFE', 'CU': 'SHFE', 'AL': 'SHFE', 'ZN': 'SHFE', 'AU': 'SHFE', 'AG': 'SHFE',
    'NI': 'SHFE', 'SN': 'SHFE', 'PB': 'SHFE', 'FU': 'SHFE', 'BU': 'SHFE', 'RU': 'SHFE',
    'SP': 'SHFE', 'SS': 'SHFE', 'HC': 'SHFE', 'WR': 'SHFE', 'AO': 'SHFE', 'BR': 'SHFE', 'AD': 'SHFE',
    # 大商所 DCE
    'C': 'DCE', 'M': 'DCE', 'Y': 'DCE', 'A': 'DCE', 'B': 'DCE', 'CS': 'DCE', 'JD': 'DCE',
    'L': 'DCE', 'V': 'DCE', 'PP': 'DCE', 'J': 'DCE', 'JM': 'DCE', 'I': 'DCE', 'EG': 'DCE',
    'EB': 'DCE', 'PG': 'DCE', 'LH': 'DCE', 'P': 'DCE', 'RR': 'DCE', 'BB': 'DCE', 'FB': 'DCE', 'LG': 'DCE',
    # 郑商所 CZCE
    'TA': 'CZCE', 'SR': 'CZCE', 'CF': 'CZCE', 'MA': 'CZCE', 'FG': 'CZCE', 'SA': 'CZCE',
    'UR': 'CZCE', 'AP': 'CZCE', 'CJ': 'CZCE', 'OI': 'CZCE', 'RM': 'CZCE', 'PF': 'CZCE',
    'PK': 'CZCE', 'SF': 'CZCE', 'SM': 'CZCE', 'PX': 'CZCE', 'PR': 'CZCE', 'CY': 'CZCE',
    'WH': 'CZCE', 'SH': 'CZCE', 'ZC': 'CZCE',
    # 能源 INE
    'SC': 'INE', 'NR': 'INE', 'LU': 'INE', 'BC': 'INE', 'EC': 'INE',
    # 广期所 GFEX
    'SI': 'GFEX', 'LC': 'GFEX',
}


def _root_to_edb_symbol(root: str) -> Optional[str]:
    """把品种 root(如 C/RB/TA)转成 EDB 主连 symbol(如 KQ.m@DCE.c / KQ.m@SHFE.rb / KQ.m@CZCE.TA)。"""
    r = str(root).upper().strip()
    if not r:
        return None
    exch = _FUTURES_ROOT_TO_EXCHANGE.get(r)
    if not exch:
        return None
    code = r if exch == 'CZCE' else r.lower()
    return f'KQ.m@{exch}.{code}'


def _fetch_edb_kline(symbol: str, start_time: str, end_time: str,
                     period: int = 60, wait_time: float = 0.5) -> pd.DataFrame:
    """调用天勤 EDB 免费接口获取主连分钟线(period=60 秒=1分钟), 返回含 datetime 的 DataFrame。"""
    import io
    import urllib.parse
    import urllib.request

    params = {'period': period, 'symbol': symbol,
              'start_time': start_time, 'end_time': end_time}
    url = 'https://edb.shinnytech.com/md/kline?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode('utf-8', 'ignore')
    if wait_time > 0:
        time.sleep(wait_time)
    if not raw.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(raw))
    if 'datetime_nano' in df.columns:
        df['datetime'] = (pd.to_datetime(df['datetime_nano'], unit='ns', utc=True)
                          .dt.tz_convert('Asia/Shanghai').dt.tz_localize(None))
    return df


def _gen_minute_windows(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                        days: int = 30) -> List[tuple]:
    """按天生成分钟拉取窗口, 避免单次请求数据量过大。"""
    windows: List[tuple] = []
    cur = start_dt
    while cur < end_dt:
        win_end = min(cur + pd.Timedelta(days=days), end_dt)
        windows.append((cur, win_end))
        cur = win_end
    return windows


def assign_trading_day_1min(datetime_series: pd.Series, trading_days=None) -> pd.Series:
    """给分钟 bar 打「交易日」标签。

    中国商品期货夜盘(小时>=20)属于「下一个交易日」; 日盘(09:00-15:00)属于当天;
    周五夜盘属于下周一(不能简单+1天)。交易日列表默认从「日盘 bar(小时<20)」推断。
    """
    if trading_days is None:
        day_mask = datetime_series.dt.hour < 20
        tds = np.array(sorted(pd.unique(datetime_series[day_mask].dt.normalize())),
                       dtype='datetime64[D]')
    else:
        tds = np.array(pd.to_datetime(list(trading_days)).normalize(), dtype='datetime64[D]')
        tds = np.unique(tds)
    night = datetime_series.dt.hour >= 20
    cal = datetime_series.dt.normalize().values.astype('datetime64[D]')
    out = pd.Series(pd.NaT, index=datetime_series.index, dtype='datetime64[ns]')
    if tds.size == 0:
        return out
    day_pos = np.minimum(np.searchsorted(tds, cal[~night], side='left'), tds.size - 1)
    out.loc[~night] = pd.to_datetime(tds[day_pos])
    night_pos = np.minimum(np.searchsorted(tds, cal[night], side='right'), tds.size - 1)
    out.loc[night] = pd.to_datetime(tds[night_pos])
    return out


def detect_rollover_from_minute_df(df: pd.DataFrame,
                                   initial_weighted_factor: float = 1.0,
                                   gap_threshold: float = 0.01,
                                   oi_chg_threshold: float = 0.15) -> pd.DataFrame:
    """基于分钟数据自身检测主力切换日, 并计算后复权因子链。

    - 换月日信号: ① 隔夜价格跳空 |gap|>gap_threshold; ② 持仓量单日|变化|>oi_chg_threshold
      (价格接近但持仓量跳变, 如某天持仓 +44% 就是主力切换)。
    - 换月比例 ≈ prev_close / open: 连续序列只有主力价, 用跳空比例近似消除换月跳空。
    - 返回 schedule: td, symbol, weighted_factor, cur_weighted_factor, is_rollover。
    """
    first = df.groupby('td').first()
    last = df.groupby('td').last()
    pos_col = 'position' if 'position' in last.columns else \
        ('open_interest' if 'open_interest' in last.columns else None)
    daily = pd.DataFrame({
        'open': first['open'],
        'close': last['close'],
        'position': last[pos_col] if pos_col else 0,
    }).sort_index()
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap_ret'] = daily['open'] / daily['prev_close'] - 1.0
    daily['oi_chg'] = pd.to_numeric(daily['position'], errors='coerce').pct_change()
    daily['is_rollover'] = ((daily['gap_ret'].abs() > gap_threshold)
                            | (daily['oi_chg'].abs() > oi_chg_threshold))

    # 与日频 _build_roll_adjusted_continuous_from_panel 语义一致:
    #   weighted_factor 累乘; cur_weighted_factor 换月日=换月比例, 非换月日沿用上一次值。
    wf = float(initial_weighted_factor)
    cur_cwf = 1.0
    wfs = []
    cwfs = []
    for t, row in daily.iterrows():
        if bool(row['is_rollover']) and pd.notna(row['prev_close']) and abs(row['open']) > 1e-12:
            cur_cwf = float(row['prev_close'] / row['open'])
            wf *= cur_cwf
        wfs.append(wf)
        cwfs.append(cur_cwf)
    daily['weighted_factor'] = wfs
    daily['cur_weighted_factor'] = cwfs
    daily['symbol'] = ''
    return daily.reset_index()[['td', 'symbol', 'weighted_factor', 'cur_weighted_factor', 'is_rollover']]


def build_minute_continuous_df_from_edb(df: pd.DataFrame,
                                        schedule: pd.DataFrame,
                                        instrument_id: str,
                                        symbol: str = '',
                                        mark_first_only: bool = True) -> pd.DataFrame:
    """把 EDB 分钟 df 与换月 schedule 合并, 生成待入库 DataFrame。

    df 需含: datetime, open, high, low, close, volume, open_interest, td
    """
    out = df.merge(schedule, on='td', how='left')
    out['symbol'] = out['symbol'].fillna('').ffill()
    if symbol:
        out['symbol'] = out['symbol'].replace('', symbol).fillna(symbol)
    out['weighted_factor'] = pd.to_numeric(out['weighted_factor'], errors='coerce').ffill().fillna(1.0)
    out['cur_weighted_factor'] = pd.to_numeric(out['cur_weighted_factor'], errors='coerce').ffill().fillna(1.0)
    daily_is_rollover = out['is_rollover'].fillna(False).astype(bool)
    first_of_day = ~out.duplicated(subset='td', keep='first')
    out['is_rollover'] = daily_is_rollover & first_of_day if mark_first_only else daily_is_rollover

    out['instrument_id'] = instrument_id
    out['settle'] = out['close']
    out = out.rename(columns={'datetime': 'time', 'open_interest': 'position'})
    out['time'] = pd.to_datetime(out['time'], errors='coerce')
    price_cols = ['open', 'high', 'low', 'close', 'settle', 'volume', 'position', 'money',
                  'weighted_factor', 'cur_weighted_factor']
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    if 'money' not in out.columns:
        out['money'] = np.nan

    out = out.dropna(subset=['time', 'open', 'high', 'low', 'close'])
    for c, fill_val in [('settle', None), ('volume', 0.0), ('position', 0.0),
                        ('money', 0.0), ('weighted_factor', 1.0), ('cur_weighted_factor', 1.0)]:
        if c == 'settle':
            out['settle'] = out['settle'].fillna(out['close'])
        else:
            out[c] = out[c].fillna(fill_val)

    cols = ['time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle',
            'volume', 'position', 'money', 'weighted_factor', 'cur_weighted_factor', 'is_rollover']
    out = out[cols].sort_values('time').reset_index(drop=True)
    out = out.drop_duplicates(subset=['time', 'instrument_id'], keep='last').reset_index(drop=True)
    return out


def _load_latest_wf_1min(instrument_id: str) -> float:
    """读取分钟库该品种最新一天的 weighted_factor, 用于增量更新时锚定 wf 链连续。"""
    try:
        df = get_data('futures', 'continuous_contract_price_1min', {'instrument_id': instrument_id})
        if df is None or df.empty:
            return 1.0
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        if df.empty or 'weighted_factor' not in df.columns:
            return 1.0
        latest = df.loc[df['time'].idxmax()]
        try:
            return float(latest['weighted_factor'])
        except Exception:
            return 1.0
    except Exception:
        return 1.0


def _load_prev_wf_1min(instrument_id: str,
                       before_time,
                       source: Optional[str] = None) -> Optional[float]:
    """读取分钟库中该品种在 before_time 之前的最后一条 weighted_factor。

    用于 CSV 导入时「继续后复权因子」: 把 CSV 的 wf 链锚定到库中已有链上。
    - source: 只锚定相同 source 的记录(如 'joinquant'), 避免混用其他来源(如 tqsdk_edb)。
    - 无数据/无效返回 None(表示无法接续, 从 CSV 自身链或 1.0 起算)。
    """
    try:
        mongo_filter: Dict[str, Any] = {
            'instrument_id': str(instrument_id),
            'time': {'$lt': pd.Timestamp(before_time)},
        }
        if source:
            # 只接续同 source 的记录; 兼容历史无 source 的记录
            mongo_filter['source'] = {'$in': [source, None]}
        df = get_data('futures', 'continuous_contract_price_1min', mongo_filter)
        if df is None or df.empty or 'weighted_factor' not in df.columns:
            return None
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['weighted_factor'] = pd.to_numeric(df['weighted_factor'], errors='coerce')
        df = df.dropna(subset=['time', 'weighted_factor'])
        if df.empty:
            return None
        latest = df.loc[df['time'].idxmax(), 'weighted_factor']
        wf = float(latest)
        return wf if np.isfinite(wf) and wf > 0 else None
    except Exception:
        return None


def update_futures_continuous_contract_price_1min(
    instrument_id: Union[str, List[str], None] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    wait_time: float = 0.5,
    method: str = 'bulk_write_update',
    cancel_event=None,
    source: str = SOURCE_EDB,
    load_prev_weighted_factor: bool = True,
) -> None:
    """从天勤 EDB 免费接口获取主力连续合约的近期分钟线, 写入 continuous_contract_price_1min。

    - 数据源: 天勤 EDB 免费接口, **免费额度为近 1 年分钟线**(period=60)。默认取最近 90 天。
    - 换月日/后复权因子: **基于分钟数据自身检测**(隔夜跳空 + 持仓量跳变),
      保证分钟数据内部自洽; 与日频库换月日相互独立, 不依赖日频数据。
    - load_prev_weighted_factor=True: 以数据库已有最新 weighted_factor 锚定, 保证 wf 链连续(默认);
      =False: 从 1.0 重新开始(类似日频的"不继续后复权因子")。
    - 写入唯一键含 source(默认 tqsdk_edb), 与 joinquant 分钟数据并存。
    - 注意: 若 start_date 早于近 1 年, EDB 免费接口可能取不到, 请使用付费专业版。
    """
    if instrument_id is None:
        instrument_id = get_futures_continuous_contract_info(from_database=True)['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    end_dt = pd.Timestamp(end_date or date.today().strftime('%Y%m%d'))
    if start_date:
        start_dt = pd.Timestamp(start_date)
    else:
        start_dt = end_dt - pd.Timedelta(days=90)

    # EDB 免费仅近 1 年, 提前预警
    if start_dt < (end_dt - pd.Timedelta(days=365)):
        log.warning(f'[1min] start_date={start_dt.date()} 早于近1年, EDB 免费接口可能取不到更早数据')

    log.info(f'[1min] 更新 {len(instrument_id)} 个合约分钟数据: {start_dt.date()} ~ {end_dt.date()}')

    for idx, ins_id in enumerate(instrument_id, 1):
        _raise_if_cancelled(cancel_event)
        root = _to_root_instrument(ins_id)
        edb_symbol = _root_to_edb_symbol(root)
        if not edb_symbol:
            log.warning(f'[1min] {ins_id} 找不到 EDB 主连代码(root={root}), 跳过')
            continue
        log.info(f'[{idx}/{len(instrument_id)}] {ins_id} EDB主连={edb_symbol}')

        # 1) 拉取 EDB 分钟线(分段, 避免单次过大)
        frames: List[pd.DataFrame] = []
        for ws, we in _gen_minute_windows(start_dt, end_dt):
            _raise_if_cancelled(cancel_event)
            df_k = _fetch_edb_kline(edb_symbol, str(ws), str(we), wait_time=wait_time)
            if df_k is not None and not df_k.empty:
                frames.append(df_k)
        if not frames:
            log.warning(f'[1min] {ins_id} 未拉到数据, 跳过')
            continue
        edb_df = pd.concat(frames, ignore_index=True)
        edb_df = edb_df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
        edb_df = edb_df.rename(columns={'close_oi': 'open_interest'})
        if 'open_interest' not in edb_df.columns:
            edb_df['open_interest'] = edb_df.get('open_oi', 0)
        edb_df['code'] = edb_symbol
        log.info(f'[1min] {ins_id} 拉到 {len(edb_df)} 根分钟bar')

        # 2) 交易日归属 + 基于聚宽分钟数据自身检测换月日 + 计算后复权因子链
        edb_df['td'] = assign_trading_day_1min(edb_df['datetime'])
        # 增量锚定: load_prev_weighted_factor=True 时接续数据库已有 wf 链; False 从 1.0 开始
        if load_prev_weighted_factor:
            initial_wf = _load_latest_wf_1min(ins_id)
        else:
            initial_wf = 1.0
        schedule = detect_rollover_from_minute_df(edb_df, initial_weighted_factor=initial_wf)
        schedule['symbol'] = edb_symbol
        out = build_minute_continuous_df_from_edb(edb_df, schedule, ins_id, symbol=edb_symbol)
        if out.empty:
            log.warning(f'[1min] {ins_id} 构建结果为空, 跳过')
            continue
        out['source'] = source
        log.info(f'[1min] {ins_id} 构建完成 {len(out)} 行')

        # 3) 分批写入(每批 5000, 可安全中断; 唯一键含 source, 与 joinquant 分钟并存)
        total = len(out)
        for i in range(0, total, 5000):
            _raise_if_cancelled(cancel_event)
            chunk = out.iloc[i:i + 5000]
            update_data(database='futures', collection='continuous_contract_price_1min',
                        df=chunk, method=method, filter_column=['time', 'instrument_id', 'source'])
            log.info(f'[1min] {ins_id} 写入 {min(i + 5000, total)}/{total}')
        log.info(f'[1min] {ins_id} 完成, 共写入 {total} 行')

    log.info('[1min] 分钟价格更新全部完成')


def _is_holiday_normal(td, trading_days) -> bool:
    """判断某交易日是否因「前一交易日是节前最后交易日(法定节假日前夜盘暂停)」而无夜盘(属正常)。"""
    try:
        import datetime as _dt
        import chinese_calendar as cc

        tds = sorted(trading_days)
        try:
            idx = tds.index(td)
        except ValueError:
            return False
        if idx == 0:
            return False
        prev = tds[idx - 1]
        cur = prev + _dt.timedelta(days=1)
        while cur < td:
            if cur.weekday() < 5 and cc.is_holiday(cur.date()):
                return True
            cur += _dt.timedelta(days=1)
        return False
    except Exception:
        return False


def aggregate_minute_to_daily_df(df_min: pd.DataFrame, trading_days=None) -> pd.DataFrame:
    """把分钟 df 按交易日聚合成日频 df。

    df_min 需含: time, open, high, low, close, volume, position, money,
                 weighted_factor, cur_weighted_factor, is_rollover, symbol。
    返回列: time(交易日), open, high, low, close, settle, volume, position, money,
            weighted_factor, cur_weighted_factor, is_rollover, symbol, bar_count。
    """
    df = df_min.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['td'] = assign_trading_day_1min(df['time'], trading_days=trading_days)
    df = df.dropna(subset=['td'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'position', 'money',
              'weighted_factor', 'cur_weighted_factor']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    g = df.groupby('td')
    out = pd.DataFrame({
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'volume': g['volume'].sum(),
        'position': g['position'].last(),
        'money': g['money'].sum(),
        'weighted_factor': g['weighted_factor'].first(),
        'cur_weighted_factor': g['cur_weighted_factor'].first(),
        'is_rollover': g['is_rollover'].any(),
        'symbol': g['symbol'].first(),
        'bar_count': g.size(),
    }).reset_index()
    # reset_index 后的索引列名是 'td'(交易日), 直接作为日频 time
    out = out.rename(columns={'td': 'time'})
    out['time'] = pd.to_datetime(out['time']).dt.normalize()
    out['settle'] = out['close']
    return out


def update_futures_continuous_contract_price_from_minute(
    instrument_id: Union[str, List[str], None] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    method: str = 'bulk_write_update',
    cancel_event=None,
    source: str = SOURCE_JOINQUANT,
) -> None:
    """把分钟频数据库中的 joinquant 数据聚合成日频, 写入日频库(source=joinquant, 与 akshare 并存)。

    - 换月日/weighted_factor/cur_weighted_factor/is_rollover 直接沿用分钟数据(由聚宽数据确定)。
    - 写入唯一键含 source, 不覆盖 akshare 日频。
    - 对「无分钟数据 / 缺夜盘 / 节假日无夜盘」的交易日输出 warning, 便于前端展示。
    """
    if instrument_id is None:
        instrument_id = get_futures_continuous_contract_info(from_database=True)['instrument_id'].tolist()
    if isinstance(instrument_id, str):
        instrument_id = [instrument_id]

    end_dt = pd.Timestamp(end_date or date.today().strftime('%Y%m%d'))
    start_dt = pd.Timestamp(start_date) if start_date else pd.Timestamp(RESEARCH_START_DATE)

    log.info(f'[min2daily] 将分钟(source={source})聚合成日频: {start_dt.date()} ~ {end_dt.date()}, instruments={instrument_id}')

    for idx, ins_id in enumerate(instrument_id, 1):
        _raise_if_cancelled(cancel_event)
        df_min = get_data('futures', 'continuous_contract_price_1min',
                          {'instrument_id': ins_id, 'source': source})
        if df_min is None or df_min.empty:
            log.warning(f'[min2daily] {ins_id} 分钟库无 source={source} 数据, 跳过')
            continue
        df_min = df_min.copy()
        df_min['time'] = pd.to_datetime(df_min['time'], errors='coerce')
        df_min = df_min.dropna(subset=['time'])
        df_min = df_min[(df_min['time'] >= start_dt) & (df_min['time'] <= end_dt + pd.Timedelta(days=1))]

        trading_days = get_trading_days(start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
        daily = aggregate_minute_to_daily_df(df_min, trading_days=trading_days)
        if daily.empty:
            log.warning(f'[min2daily] {ins_id} 聚合结果为空, 跳过')
            continue

        # 缺失 / 缺夜盘 / 节假日检查
        bar_map = dict(zip(daily['time'].dt.normalize(), daily['bar_count']))
        for t in trading_days:
            if t not in bar_map:
                log.warning(f'[min2daily] {ins_id} 交易日 {t.date()} 无 {source} 分钟数据(缺失)')
            elif int(bar_map[t]) < 300:
                if _is_holiday_normal(t, trading_days):
                    log.warning(f'[min2daily] {ins_id} 交易日 {t.date()} bar数不足({int(bar_map[t])}), '
                                f'属节假日(节前无夜盘)正常')
                else:
                    log.warning(f'[min2daily] {ins_id} 交易日 {t.date()} bar数不足({int(bar_map[t])}), 疑似缺夜盘')

        daily['instrument_id'] = ins_id
        daily['source'] = source
        cols = ['time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle',
                'volume', 'position', 'money', 'weighted_factor', 'cur_weighted_factor',
                'is_rollover', 'source']
        out = daily[cols].drop_duplicates(subset=['time', 'instrument_id', 'source'], keep='last')
        out = out.sort_values('time').reset_index(drop=True)

        total = len(out)
        for i in range(0, total, 5000):
            _raise_if_cancelled(cancel_event)
            chunk = out.iloc[i:i + 5000]
            update_data(database='futures', collection='continuous_contract_price_daily',
                        df=chunk, method=method, filter_column=['time', 'instrument_id', 'source'])
            log.info(f'[min2daily] {ins_id} 写入 {min(i + 5000, total)}/{total}')
        log.info(f'[min2daily] {ins_id} 完成, 共写入 {total} 行')
    log.info('[min2daily] 全部完成')


# ================== 分钟频 CSV 导入(聚宽/通用分钟 CSV) ==================
# 允许导入任意格式正确、含必要字段的分钟 CSV(如聚宽导出的主 CSV + 补夜盘 CSV)。
# - 优先使用 CSV 中已有的换月字段(symbol/is_rollover/weighted_factor/cur_weighted_factor,
#   例如聚宽 get_dominant_future 导出); 否则回退到基于分钟数据自身检测换月(gap+持仓量跳变)。
# - 写入唯一键含 source(默认 'joinquant'), 因此不会覆盖 tqsdk_edb 等其他来源的记录。

CSV_DATETIME_COL = 'datetime'     # CSV 中分钟时间戳列名
CSV_OI_COL = 'open_interest'      # CSV 中持仓量列名(导入时映射为 position)


def read_and_merge_1min_csvs(main_csv: str,
                             fix_csv: Optional[str] = None) -> pd.DataFrame:
    """读取主 CSV 与补夜盘 CSV, 按分钟时间戳合并去重, 返回含 datetime 列的 DataFrame。

    - 主 CSV 与补 CSV 可能重叠(补 CSV 含主 CSV 缺失的部分夜盘), 按 datetime 去重(保留后者)。
    - 至少一个文件必须存在。
    """
    frames: List[pd.DataFrame] = []
    for path in [main_csv, fix_csv]:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if CSV_DATETIME_COL not in df.columns:
                # 兼容列名为 'time' 的 CSV
                df = df.rename(columns={'time': CSV_DATETIME_COL})
            if CSV_DATETIME_COL not in df.columns:
                raise ValueError(f'CSV 缺少时间戳列, 需要包含 "{CSV_DATETIME_COL}" 或 "time": {path}')
            df[CSV_DATETIME_COL] = pd.to_datetime(df[CSV_DATETIME_COL], errors='coerce')
            df = df.dropna(subset=[CSV_DATETIME_COL])
            frames.append(df)
            log.info(f'[1min-csv] 读取 {path}: {len(df)} 行')
        else:
            log.info(f'[1min-csv] 跳过不存在的 CSV: {path}')
    if not frames:
        raise ValueError('没有可用的 CSV 文件(主 CSV 必须存在)。')

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=[CSV_DATETIME_COL], keep='last')
    merged = merged.sort_values(CSV_DATETIME_COL).reset_index(drop=True)
    log.info(f'[1min-csv] 合并后: {len(merged)} 行, '
             f'范围 {merged[CSV_DATETIME_COL].min()} ~ {merged[CSV_DATETIME_COL].max()}')
    return merged


def _to_root_instrument_or_infer(instrument_id: str) -> str:
    return _to_root_instrument(instrument_id)


def build_minute_continuous_df_from_csv(main_csv: str,
                                        fix_csv: Optional[str] = None,
                                        instrument_id: str = 'C0',
                                        source: str = 'joinquant',
                                        load_prev_weighted_factor: bool = True,
                                        mark_rollover_first_minute_only: bool = True,
                                        gap_threshold: float = 0.01,
                                        oi_chg_threshold: float = 0.15) -> pd.DataFrame:
    """从分钟 CSV(主 + 可选补)构建待入库 DataFrame(含 source 列)。

    - 若 CSV 含换月字段(is_rollover + weighted_factor), 优先使用(聚宽官方口径);
    - 否则回退到基于分钟数据自身检测换月日(gap + 持仓量跳变)。
    - load_prev_weighted_factor=True(默认): 若 CSV 起始日之前库中已有该品种数据,
      则把 CSV 的 wf 链整体锚定到库中最新 wf 上, 保证「继续后复权因子」链连续;
      =False: 从 CSV 自身链或 1.0 起算。
    - 返回列: time, instrument_id, symbol, open, high, low, close, settle,
      volume, position, money, weighted_factor, cur_weighted_factor, is_rollover, source。
    """
    df = read_and_merge_1min_csvs(main_csv, fix_csv)
    if df.empty:
        return _empty_continuous_price_df()
    # 需要的 OHLC 列校验
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'CSV 缺少必需列: {missing}')

    # 继续后复权因子: 读取 CSV 起始日之前库中该品种最新 wf 作为锚点
    anchor_wf: Optional[float] = None
    if load_prev_weighted_factor:
        csv_start_ts = df[CSV_DATETIME_COL].min()
        anchor_wf = _load_prev_wf_1min(instrument_id, csv_start_ts, source=source)
        if anchor_wf is not None:
            log.info(f'[1min-csv] 继续后复权因子: 锚定库中已有链 wf={anchor_wf:.6f}')
        else:
            log.info('[1min-csv] 继续后复权因子: 库中 CSV 起始日之前无数据, 从 CSV 自身链起算')

    df['td'] = assign_trading_day_1min(df[CSV_DATETIME_COL])

    root = _to_root_instrument_or_infer(instrument_id)
    main_symbol = f'KQ.m@{root}'

    if 'is_rollover' in df.columns and 'weighted_factor' in df.columns:
        log.info('[1min-csv] 使用 CSV 中已有的换月字段(symbol/is_rollover/weighted_factor)')
        out = df.rename(columns={CSV_DATETIME_COL: 'time', CSV_OI_COL: 'position'})
        out['instrument_id'] = instrument_id
        out['settle'] = out['close']
        out['time'] = pd.to_datetime(out['time'], errors='coerce')
        out['td'] = assign_trading_day_1min(out['time'])
        _valid = out.dropna(subset=['weighted_factor'])
        if _valid.empty:
            _map = pd.DataFrame(columns=['td', 'symbol', 'is_rollover', 'weighted_factor', 'cur_weighted_factor'])
        else:
            _map = _valid.groupby('td').agg({
                'symbol': 'last',
                'is_rollover': 'max',
                'weighted_factor': 'last',
                'cur_weighted_factor': 'last',
            }).reset_index()
        out = out.merge(_map, on='td', how='left', suffixes=('', '_m'))
        for c in ['symbol', 'is_rollover', 'weighted_factor', 'cur_weighted_factor']:
            mc = f'{c}_m'
            if mc in out.columns:
                out[c] = out[c].fillna(out[mc])
                out = out.drop(columns=[mc])
        out = out.drop(columns=['td'])
        out['is_rollover'] = out['is_rollover'].fillna(False).astype(bool)
        out['weighted_factor'] = pd.to_numeric(out['weighted_factor'], errors='coerce').fillna(1.0)
        out['cur_weighted_factor'] = pd.to_numeric(out['cur_weighted_factor'], errors='coerce').fillna(1.0)
        cols = ['time', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'settle',
                'volume', 'position', 'money', 'weighted_factor', 'cur_weighted_factor', 'is_rollover']
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0 if c in ('volume', 'position', 'money') else ('' if c == 'symbol' else np.nan)
        for c in ['open', 'high', 'low', 'close', 'settle', 'volume', 'position', 'money']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out = out[cols].dropna(subset=['time', 'open', 'high', 'low', 'close']).sort_values('time').reset_index(drop=True)
        out['settle'] = out['settle'].fillna(out['close'])
        out['volume'] = out['volume'].fillna(0.0)
        out['position'] = out['position'].fillna(0.0)
        out['money'] = out['money'].fillna(0.0)
        out['source'] = source
        out = out.drop_duplicates(subset=['time', 'instrument_id'], keep='last').reset_index(drop=True)
        if anchor_wf is not None:
            # 把 CSV 自带链整体平移到库中已有链: new_wf = csv_wf * anchor_wf
            out['weighted_factor'] = pd.to_numeric(out['weighted_factor'], errors='coerce') * anchor_wf
        return out

    log.info('[1min-csv] CSV 无换月字段, 回退到基于分钟数据自身检测换月(gap+持仓量跳变)')
    schedule = detect_rollover_from_minute_df(
        df,
        initial_weighted_factor=(anchor_wf if anchor_wf is not None else 1.0),
        gap_threshold=gap_threshold,
        oi_chg_threshold=oi_chg_threshold,
    )
    schedule['symbol'] = main_symbol
    roll_days = schedule.loc[schedule['is_rollover'], 'td'].tolist()
    log.info(f'[1min-csv] 检测到换月日: {len(roll_days)} 个')
    out = build_minute_continuous_df_from_edb(
        df, schedule, instrument_id,
        symbol=main_symbol,
        mark_first_only=mark_rollover_first_minute_only,
    )
    out['source'] = source
    return out


def find_missing_night_days_in_csv(csv_path: str,
                                   bar_threshold: int = 340,
                                   lookback_days: int = 20) -> pd.DataFrame:
    """从分钟 CSV 中识别「缺夜盘」的交易日(bar 数低于阈值)。

    排除两类正常情况: CSV 首日(前一夜盘属更早日期)与节前最后交易日(法定节假日前夜盘暂停)。
    返回列: td, bar_count, prev_td, fetch_start, fetch_end。
    """
    import chinese_calendar as cc

    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f'CSV 文件不存在: {csv_path}')
    df = pd.read_csv(csv_path)
    if CSV_DATETIME_COL not in df.columns:
        df = df.rename(columns={'time': CSV_DATETIME_COL})
    if CSV_DATETIME_COL not in df.columns:
        raise ValueError(f'CSV 缺少时间戳列: {csv_path}')
    df[CSV_DATETIME_COL] = pd.to_datetime(df[CSV_DATETIME_COL], errors='coerce')
    df = df.dropna(subset=[CSV_DATETIME_COL]).sort_values(CSV_DATETIME_COL).reset_index(drop=True)
    df['td'] = assign_trading_day_1min(df[CSV_DATETIME_COL])

    cnt = df.groupby('td').agg(bar_count=('open', 'count')).sort_index()
    tds = list(cnt.index)
    missing = cnt[cnt['bar_count'] < bar_threshold].copy()

    rows: List[Dict[str, object]] = []
    first_td = tds[0] if tds else None
    for i, t in enumerate(tds):
        if t not in missing.index:
            continue
        prev_td = tds[i - 1] if i > 0 else None
        if prev_td is None:
            continue  # 首日, 前一夜盘不在 CSV 内
        # 排除节前最后交易日(prev_td 与 t 之间有法定节假日)
        cur = prev_td + pd.Timedelta(days=1)
        holiday_between = False
        while cur < t:
            if cur.weekday() < 5 and cc.is_holiday(cur.date()):
                holiday_between = True
                break
            cur += pd.Timedelta(days=1)
        if holiday_between:
            continue
        fetch_start = prev_td.normalize().replace(hour=20, minute=59)
        fetch_end = t.normalize().replace(hour=15, minute=0)
        rows.append({
            'td': t.date().isoformat(),
            'bar_count': int(missing.loc[t, 'bar_count']),
            'prev_td': prev_td.date().isoformat(),
            'fetch_start': str(fetch_start),
            'fetch_end': str(fetch_end),
        })
    return pd.DataFrame(rows)


def import_1min_csv_to_db(main_csv: str,
                          fix_csv: Optional[str] = None,
                          instrument_id: str = 'C0',
                          source: str = 'joinquant',
                          load_prev_weighted_factor: bool = True,
                          method: str = 'bulk_write_update',
                          batch_size: int = 5000,
                          log_step: int = 100,
                          cancel_event=None) -> Dict[str, Any]:
    """从分钟 CSV(主 + 可选补)导入到分钟库, 唯一键含 source, 不覆盖其他来源。

    - load_prev_weighted_factor=True(默认): 接续库中已有后复权因子链;
      =False: 从 CSV 自身链或 1.0 起算。
    - 进度日志: 每写入 log_step(默认 100)条输出一次。
    返回: {total, start_time, end_time, rollover_count, missing_night_days, message}
    """
    out = build_minute_continuous_df_from_csv(
        main_csv, fix_csv, instrument_id, source=source,
        load_prev_weighted_factor=load_prev_weighted_factor,
    )
    if out is None or out.empty:
        raise ValueError('构建结果为空, 未写入任何数据。')

    total = len(out)
    batch_size = max(1, int(batch_size))
    log_step = max(1, int(log_step))
    n_batches = (total + batch_size - 1) // batch_size
    log.info(f'[1min-csv] 开始写入 {total} 条(source={source})到 {instrument_id}, 共 {n_batches} 批...')

    written = 0
    next_log = log_step
    for i in range(0, total, batch_size):
        _raise_if_cancelled(cancel_event)
        chunk = out.iloc[i:i + batch_size]
        update_data(database='futures',
                    collection='continuous_contract_price_1min',
                    df=chunk,
                    method=method,
                    filter_column=['time', 'instrument_id', 'source'])
        written += len(chunk)
        # 每写入 log_step 条输出一次进度日志
        while next_log <= written:
            log.info(f'[1min-csv] {instrument_id} 已写入 {min(next_log, total)}/{total}')
            next_log += log_step
    # 末尾补一条完成日志(当 total 不是 log_step 整数倍时, 如 3675/100)
    if written == total and next_log - log_step < total:
        log.info(f'[1min-csv] {instrument_id} 已写入 {total}/{total} (完成)')

    rollover_count = int(out['is_rollover'].sum()) if 'is_rollover' in out.columns else 0

    # 提示缺夜盘交易日(基于合并后的主 CSV 分析)
    missing_df = pd.DataFrame()
    try:
        if os.path.exists(main_csv):
            missing_df = find_missing_night_days_in_csv(main_csv)
    except Exception as e:
        log.warning(f'[1min-csv] 缺夜盘识别失败(不影响导入): {e}')

    if not missing_df.empty:
        log.warning(f'[1min-csv] 检测到 {len(missing_df)} 个缺夜盘交易日, 可在聚宽补拉后再次导入:')
        for _, r in missing_df.iterrows():
            log.warning(
                f'  td={r["td"]}, bar_count={r["bar_count"]}, '
                f'fetch=[{r["fetch_start"]} ~ {r["fetch_end"]}]'
            )

    return {
        'total': total,
        'start_time': str(out['time'].min()) if 'time' in out.columns else None,
        'end_time': str(out['time'].max()) if 'time' in out.columns else None,
        'rollover_count': rollover_count,
        'missing_night_days': missing_df.to_dict(orient='records') if not missing_df.empty else [],
        'message': f'成功导入 {total} 条分钟数据(source={source})到 {instrument_id}',
    }


# ================== 聚宽研究环境脚本生成 ==================
# 生成可直接粘贴到聚宽 Jupyter 运行的脚本文本(供前端展示/一键复制)。
# 复用 _FUTURES_ROOT_TO_EXCHANGE 中的交易所映射, 生成对应的 9999 主力连续代码。

# 交易所 -> 聚宽 9999 主力连续后缀
_JQ_EXCHANGE_SUFFIX: Dict[str, str] = {
    'SHFE': '.XSGE',
    'DCE': '.XDCE',
    'CZCE': '.XZCE',
    'INE': '.XINE',
    'GFEX': '.XGFEX',
}


def _to_joinquant_code(instrument_id: str) -> str:
    """把本地合约(如 C0 / RB0)转成聚宽主力连续代码(如 C9999.XDCE / RB9999.XSGE)。"""
    root = _to_root_instrument(instrument_id)
    exch = _FUTURES_ROOT_TO_EXCHANGE.get(root)
    if not exch:
        raise ValueError(f'找不到品种 {root} 的交易所映射, 无法生成聚宽代码。')
    suffix = _JQ_EXCHANGE_SUFFIX.get(exch)
    if not suffix:
        raise ValueError(f'交易所 {exch} 暂无聚宽代码后缀映射。')
    return f'{root}9999{suffix}'


def _to_dash_date(raw: Optional[str], default: str) -> str:
    if not raw or not str(raw).strip():
        return default
    try:
        return pd.Timestamp(str(raw).strip()).strftime('%Y-%m-%d')
    except Exception:
        return default


_JQ_EXPORT_SCRIPT_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
聚宽研究环境: 导出期货主力连续分钟数据 CSV(含换月字段 symbol/is_rollover/weighted_factor)。
在 https://www.joinquant.com 研究环境新建 notebook, 把本脚本粘贴到单元格运行。
运行结束后到左侧文件树 data/fut_min/ 目录下载 CSV。
"""
import os
import time
import datetime as dt

import numpy as np
import pandas as pd

# ---- 解析 get_price / get_dominant_future ----
try:
    get_price  # noqa: F821
except NameError:
    try:
        from jqdata import get_price, get_dominant_future
    except ImportError:
        from jqdatasdk import get_price, get_dominant_future
try:
    get_dominant_future  # noqa: F821
except NameError:
    try:
        from jqdata import get_dominant_future
    except ImportError:
        from jqdatasdk import get_dominant_future

# ================== 配置区 ==================
FREQ = "1m"                 # 分钟周期: 1m / 5m / 15m / 30m / 60m
START_DATE = "__START_DATE__"
END_DATE = "__END_DATE__"
FIELDS = ["open", "high", "low", "close", "volume", "money", "open_interest"]
CHUNK_DAYS = 14             # 每窗口天数(1m 下 14 天约 2800 根, 稳妥)
SLEEP_SEC = 0.3
OUT_DIR = "data/fut_min"
FUTURE_CODES = ["__JQ_CODE__"]

END_DT = dt.datetime.strptime(END_DATE, "%Y-%m-%d").replace(hour=15)
FLAG_ROOT = os.path.join(OUT_DIR, "_flags")


def _root_from_code(code):
    sym = str(code).split(".")[0]
    return sym.rstrip("0123456789")


def _get_day_open(contract, date):
    try:
        df = get_price(contract, start_date=str(date), end_date=str(date), frequency="daily", fields=["open"])
        if df is not None and not df.empty:
            df = df.reset_index()
            return float(pd.to_numeric(df["open"], errors="coerce").iloc[-1])
    except Exception:
        pass
    return None


def get_dominant_series_compat(root, trade_days):
    trade_days = sorted(set(pd.Timestamp(d) for d in trade_days))
    if not trade_days:
        return pd.Series(dtype=object)
    try:
        dom = get_dominant_future(root, str(trade_days[0].date()), end_date=str(trade_days[-1].date()))
        if dom is not None and len(dom) > 0:
            return dom
    except Exception:
        pass
    series = {}
    for d in trade_days:
        try:
            series[pd.Timestamp(d)] = str(get_dominant_future(root, str(d.date())))
        except Exception as e:
            print(f"    [警告] {d.date()} 主力获取失败: {e}")
    return pd.Series(series).sort_index()


def build_dominant_schedule(root, trade_days):
    dom = get_dominant_series_compat(root, trade_days)
    if dom is None or len(dom) == 0:
        return pd.DataFrame()
    items = dom.items() if isinstance(dom, pd.Series) else dom.iteritems()
    rows, wf, cur_cwf, prev_symbol = [], 1.0, 1.0, None
    for d, symbol in items:
        symbol = str(symbol)
        is_roll = (prev_symbol is not None and symbol != prev_symbol)
        if is_roll and prev_symbol is not None:
            old_open = _get_day_open(prev_symbol, d)
            new_open = _get_day_open(symbol, d)
            if old_open and new_open and abs(new_open) > 1e-12:
                cur_cwf = old_open / new_open
                wf *= cur_cwf
        rows.append({"td": pd.Timestamp(d).normalize(), "symbol": symbol, "is_rollover": is_roll,
                     "weighted_factor": wf, "cur_weighted_factor": cur_cwf})
        prev_symbol = symbol
    df = pd.DataFrame(rows).drop_duplicates(subset="td", keep="last").sort_values("td")
    return df


def assign_trading_day_local(datetime_series, trading_days):
    tds = np.array(sorted(pd.to_datetime(list(trading_days)).normalize()), dtype="datetime64[D]")
    tds = np.unique(tds)
    night = datetime_series.dt.hour >= 20
    cal = datetime_series.dt.normalize().values.astype("datetime64[D]")
    out = pd.Series(pd.NaT, index=datetime_series.index, dtype="datetime64[ns]")
    if tds.size == 0:
        return out
    day_pos = np.minimum(np.searchsorted(tds, cal[~night], side="left"), tds.size - 1)
    out.loc[~night] = pd.to_datetime(tds[day_pos])
    night_pos = np.minimum(np.searchsorted(tds, cal[night], side="right"), tds.size - 1)
    out.loc[night] = pd.to_datetime(tds[night_pos])
    return out


def gen_windows(start_dt, end_dt, chunk_days):
    windows, cur = [], start_dt
    while cur < end_dt:
        win_end = min(cur + dt.timedelta(days=chunk_days), end_dt).replace(hour=15)
        windows.append((cur, win_end))
        cur = win_end.replace(hour=0) + dt.timedelta(days=1)
    return windows


def fetch_chunk(code, start, end, depth=0):
    try:
        return get_price(code, start_date=str(start), end_date=str(end), frequency=FREQ, fields=FIELDS)
    except Exception:
        days = (end - start).days
        if days > 1 and depth < 4:
            mid = (start + dt.timedelta(days=days // 2)).replace(hour=15)
            left = fetch_chunk(code, start, mid, depth + 1)
            right = fetch_chunk(code, mid.replace(hour=0) + dt.timedelta(days=1), end, depth + 1)
            if left is None or right is None:
                return None
            return pd.concat([left, right])
        print(f"      [警告] 窗口 {start}~{end} 拉取失败")
        return None


def normalize(df, code):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    cols = list(df.columns)
    time_col = next((c for c in cols if str(c).lower() in ("time", "datetime", "index")), cols[0])
    df = df.rename(columns={time_col: "datetime"})
    if "code" not in df.columns:
        df.insert(1, "code", code)
    df["datetime"] = df["datetime"].astype(str)
    return df


def load_existing(code):
    path = os.path.join(OUT_DIR, f"{code}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FLAG_ROOT, exist_ok=True)
    start_dt = dt.datetime.strptime(START_DATE, "%Y-%m-%d")
    windows = gen_windows(start_dt, END_DT, CHUNK_DAYS)
    print(f"聚宽研究环境 | 导出 {len(FUTURE_CODES)} 个品种 {FREQ} 分钟数据")
    print(f"时间范围: {START_DATE} ~ {END_DATE} | 总窗口: {len(windows)}")
    for code in FUTURE_CODES:
        root = _root_from_code(code)
        flag_dir = os.path.join(FLAG_ROOT, code)
        os.makedirs(flag_dir, exist_ok=True)
        done = set(f.split(".")[0] for f in os.listdir(flag_dir))
        todo = [w for w in windows if w[0].strftime("%Y%m%d") not in done]
        parts = []
        existing = load_existing(code)
        if len(existing) > 0:
            parts.append(existing)
        for i, (ws, we) in enumerate(todo, 1):
            df = fetch_chunk(code, ws, we)
            time.sleep(SLEEP_SEC)
            if df is not None:
                nd = normalize(df, code)
                if not nd.empty:
                    parts.append(nd)
                open(os.path.join(flag_dir, ws.strftime("%Y%m%d") + ".done"), "w").close()
            if i % 10 == 0 or i == len(todo):
                print(f"    {code} 进度 {i}/{len(todo)} 窗口")
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.drop_duplicates(subset=["datetime", "code"], keep="last").sort_values("datetime").reset_index(drop=True)
            merged["_dt"] = pd.to_datetime(merged["datetime"])
            day_mask = merged["_dt"].dt.hour < 20
            trade_days = sorted(pd.unique(merged.loc[day_mask, "_dt"].dt.normalize()))
            merged = merged.drop(columns=["_dt"])
            sched = build_dominant_schedule(root, trade_days)
            if sched.empty:
                print(f"    [警告] {code} get_dominant_future 返回空, 换月字段缺失")
            else:
                merged["datetime"] = pd.to_datetime(merged["datetime"])
                merged["td"] = assign_trading_day_local(merged["datetime"], sched["td"])
                merged = merged.merge(sched.rename(columns={"td": "td"}), on="td", how="left")
                for c in ["symbol", "weighted_factor", "cur_weighted_factor"]:
                    if c in merged.columns:
                        merged[c] = merged[c].ffill()
                merged["is_rollover"] = merged["is_rollover"].fillna(False).astype(bool)
                merged = merged.drop(columns=["td"], errors="ignore")
            merged.to_csv(os.path.join(OUT_DIR, f"{code}.csv"), index=False)
            print(f"    完成: {code} | {len(merged)} 行 | 列: {list(merged.columns)}")
    print()
    print("全部完成。请到研究环境左侧文件树 data/fut_min/ 目录下载 CSV。")


if __name__ == "__main__":
    main()
'''


_JQ_FIX_SCRIPT_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
聚宽研究环境: 对「缺夜盘」的交易日补拉完整分钟数据(夜盘+日盘), 导出 CSV 供本地导入。
把下方 MISSING_RANGES 替换为本地识别出的缺夜盘交易日列表后运行。
运行结束后到左侧文件树 data/fix_night/ 目录下载 CSV。
"""
import os
import datetime as dt

import numpy as np
import pandas as pd

# ---- 解析 get_price / get_dominant_future ----
try:
    get_price  # noqa: F821
except NameError:
    try:
        from jqdata import get_price
    except ImportError:
        try:
            from jqdata import *
        except ImportError:
            from jqdatasdk import get_price
try:
    get_dominant_future  # noqa: F821
except NameError:
    try:
        from jqdata import get_dominant_future
    except ImportError:
        from jqdatasdk import get_dominant_future

# ================== 配置区 ==================
FUTURE_CODE = "__JQ_CODE__"
FREQ = "1m"
FIELDS = ["open", "high", "low", "close", "volume", "money", "open_interest"]
OUT_DIR = "data/fix_night"


def _root_from_code(code):
    sym = str(code).split(".")[0]
    return sym.rstrip("0123456789")


def _get_day_open(contract, date):
    try:
        df = get_price(contract, start_date=str(date), end_date=str(date), frequency="daily", fields=["open"])
        if df is not None and not df.empty:
            df = df.reset_index()
            return float(pd.to_numeric(df["open"], errors="coerce").iloc[-1])
    except Exception:
        pass
    return None


def get_dominant_series_compat(root, trade_days):
    trade_days = sorted(set(pd.Timestamp(d) for d in trade_days))
    if not trade_days:
        return pd.Series(dtype=object)
    try:
        dom = get_dominant_future(root, str(trade_days[0].date()), end_date=str(trade_days[-1].date()))
        if dom is not None and len(dom) > 0:
            return dom
    except Exception:
        pass
    series = {}
    for d in trade_days:
        try:
            series[pd.Timestamp(d)] = str(get_dominant_future(root, str(d.date())))
        except Exception as e:
            print(f"    [警告] {d.date()} 主力获取失败: {e}")
    return pd.Series(series).sort_index()


def build_dominant_schedule(root, trade_days):
    dom = get_dominant_series_compat(root, trade_days)
    if dom is None or len(dom) == 0:
        return pd.DataFrame()
    items = dom.items() if isinstance(dom, pd.Series) else dom.iteritems()
    rows, wf, cur_cwf, prev_symbol = [], 1.0, 1.0, None
    for d, symbol in items:
        symbol = str(symbol)
        is_roll = (prev_symbol is not None and symbol != prev_symbol)
        if is_roll and prev_symbol is not None:
            old_open = _get_day_open(prev_symbol, d)
            new_open = _get_day_open(symbol, d)
            if old_open and new_open and abs(new_open) > 1e-12:
                cur_cwf = old_open / new_open
                wf *= cur_cwf
        rows.append({"td": pd.Timestamp(d).normalize(), "symbol": symbol, "is_rollover": is_roll,
                     "weighted_factor": wf, "cur_weighted_factor": cur_cwf})
        prev_symbol = symbol
    df = pd.DataFrame(rows).drop_duplicates(subset="td", keep="last").sort_values("td")
    return df


def assign_trading_day_local(datetime_series, trading_days):
    tds = np.array(sorted(pd.to_datetime(list(trading_days)).normalize()), dtype="datetime64[D]")
    tds = np.unique(tds)
    night = datetime_series.dt.hour >= 20
    cal = datetime_series.dt.normalize().values.astype("datetime64[D]")
    out = pd.Series(pd.NaT, index=datetime_series.index, dtype="datetime64[ns]")
    if tds.size == 0:
        return out
    day_pos = np.minimum(np.searchsorted(tds, cal[~night], side="left"), tds.size - 1)
    out.loc[~night] = pd.to_datetime(tds[day_pos])
    night_pos = np.minimum(np.searchsorted(tds, cal[night], side="right"), tds.size - 1)
    out.loc[night] = pd.to_datetime(tds[night_pos])
    return out


# 把本地 find_missing_night_days 识别出的列表粘贴到这里 (td, fetch_start, fetch_end)
MISSING_RANGES = __MISSING_RANGES__


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not MISSING_RANGES:
        print("[提示] MISSING_RANGES 为空, 请先粘贴缺夜盘交易日列表。")
        return
    frames = []
    for td, start, end in MISSING_RANGES:
        print(f">>> 拉取交易日 {td}: {start} ~ {end} ...")
        try:
            df = get_price(FUTURE_CODE, start_date=start, end_date=end, frequency=FREQ, fields=FIELDS)
        except Exception as e:
            print(f"    [失败] {type(e).__name__}: {e}")
            continue
        if df is None or df.empty:
            print("    [警告] 该时段无数据")
            continue
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "datetime"})
        if "code" not in df.columns:
            df.insert(1, "code", FUTURE_CODE)
        df["datetime"] = pd.to_datetime(df["datetime"]).astype(str)
        for f in FIELDS:
            if f not in df.columns:
                df[f] = float("nan")
        df = df[["datetime", "code"] + FIELDS]
        print(f"    得到 {len(df)} 根 bar")
        frames.append(df)
    if not frames:
        print("未拉到任何数据。")
        return
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["datetime", "code"], keep="last").sort_values("datetime")
    root = _root_from_code(FUTURE_CODE)
    out["_dt"] = pd.to_datetime(out["datetime"])
    trade_days = sorted(pd.unique(out.loc[out["_dt"].dt.hour < 20, "_dt"].dt.normalize()))
    out = out.drop(columns=["_dt"])
    sched = build_dominant_schedule(root, trade_days)
    if not sched.empty:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out["td"] = assign_trading_day_local(out["datetime"], sched["td"])
        out = out.merge(sched, on="td", how="left")
        for c in ["symbol", "weighted_factor", "cur_weighted_factor"]:
            if c in out.columns:
                out[c] = out[c].ffill()
        out["is_rollover"] = out["is_rollover"].fillna(False).astype(bool)
        out = out.drop(columns=["td"], errors="ignore")
    path = f"{OUT_DIR}/{FUTURE_CODE}_fix_night.csv"
    out.to_csv(path, index=False)
    print(f"完成: 共 {len(out)} 行 -> {path}")
    print("请下载该 CSV, 然后在本地前端「分钟频价格更新」的补夜盘 CSV 路径中导入。")


if __name__ == "__main__":
    main()
'''


def _normalize_missing_ranges_text(raw: Optional[str]) -> str:
    """把用户粘贴的 MISSING_RANGES 文本规范成可嵌入脚本的 Python 列表文本。"""
    if not raw or not str(raw).strip():
        return '[]'
    text = str(raw).strip()
    # 若带了 MISSING_RANGES = [...] 整段, 提取方括号内容
    if '=' in text and text.lstrip().startswith('MISSING_RANGES'):
        text = text.split('=', 1)[1].strip()
    if not text.startswith('['):
        start = text.find('[')
        end = text.rfind(']')
        if start >= 0 and end > start:
            text = text[start:end + 1]
        else:
            return '[]'
    # 去掉可能多余的分号
    if text.endswith(';'):
        text = text[:-1]
    return text


def generate_joinquant_export_script(instrument_id: str = 'C0',
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> str:
    """生成聚宽导出主 CSV 的脚本(供前端展示/一键复制)。"""
    jq_code = _to_joinquant_code(instrument_id)
    start = _to_dash_date(start_date, '2020-01-01')
    end = _to_dash_date(end_date, date.today().strftime('%Y-%m-%d'))
    return (_JQ_EXPORT_SCRIPT_TEMPLATE
            .replace('__JQ_CODE__', jq_code)
            .replace('__START_DATE__', start)
            .replace('__END_DATE__', end))


def generate_joinquant_fix_script(instrument_id: str = 'C0',
                                  missing_ranges: Optional[str] = None) -> str:
    """生成聚宽补夜盘脚本(供前端展示/一键复制)。"""
    jq_code = _to_joinquant_code(instrument_id)
    ranges_text = _normalize_missing_ranges_text(missing_ranges)
    return (_JQ_FIX_SCRIPT_TEMPLATE
            .replace('__JQ_CODE__', jq_code)
            .replace('__MISSING_RANGES__', ranges_text))
