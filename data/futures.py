"""
This script is to get and deal with futures data based on akshare.
"""
import time
from datetime import date
from typing import Dict, List, Optional, Sequence, Union
import pandas as pd
import numpy as np

import akshare as ak
from mongo.mongify import get_data, update_data
from utils.params import (
    RESEARCH_START_DATE,
    FUTURES_FIXED_LISTING_MONTHS,
)
from utils.logging import log


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
                                          cancel_event=None):
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

    update_data(database='futures',
                collection='continuous_contract_price_daily',
                df=df_futures_price,
                method=method)

    log.info(f'Successfully update futures continuous contract daily price ({len(df_futures_price)} rows).')


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
