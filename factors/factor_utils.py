from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import get_factor_formula_records
from utils.logging import log
from .factor_ops import calc_formula_df
from data import get_risk_free_rate


def join_fc_name_and_parameter(fc_name, parameter):

    return fc_name + '_' + '_'.join([str(value) for _, value in parameter.items()])


def get_weighted_price(df: pd.DataFrame,
                       weighted_factor_col: str = 'weighted_factor',
                       price_cols: Union[str, list, None] = None) -> pd.DataFrame:
    """Apply weighted adjustment to price fields: adjusted = raw * weighted_factor."""
    out = df.copy()
    if weighted_factor_col not in out.columns:
        raise ValueError(f'Column `{weighted_factor_col}` is required for weighted-price adjustment.')

    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close', 'settle']
    if isinstance(price_cols, str):
        price_cols = [price_cols]

    wf = pd.to_numeric(out[weighted_factor_col], errors='coerce').replace(0, np.nan).fillna(1.0)
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce') * wf
    return out


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


def check_if_leakage(
    selected_fc_name_list: Sequence[str],
    load_base_data_fn: Callable[[], pd.DataFrame],
    generate_factor_df_fn: Callable[[pd.DataFrame, Optional[List[str]]], pd.DataFrame],
    check_leakage_count: int = 20,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    raise_error: bool = True,
) -> Dict[str, Any]:
    """Generic leakage check by comparing full-run vs sliced-run factor values."""
    fc_list = [str(x) for x in selected_fc_name_list if str(x).strip()]
    if not fc_list:
        raise ValueError('No factor columns available for leakage check.')

    base_df = load_base_data_fn()
    full_factor_df = generate_factor_df_fn(base_df, selected_fc_names=fc_list)

    all_time_list = sorted(full_factor_df['time'].dropna().unique().tolist())
    if not all_time_list:
        raise ValueError('No valid time points available for leakage check.')

    sample_count = min(len(all_time_list), max(1, int(check_leakage_count)))
    if sample_count < len(all_time_list):
        rng = np.random.default_rng()
        sampled_time_list = sorted(rng.choice(all_time_list, size=sample_count, replace=False).tolist())
    else:
        sampled_time_list = all_time_list

    slice_factor_list: List[pd.DataFrame] = []
    for t in sampled_time_list:
        log.info(f'Checking leakage for time slice <= {t}...')
        df_slice = base_df.loc[base_df['time'] <= t].copy()
        factor_df_slice = generate_factor_df_fn(df_slice, selected_fc_names=fc_list)
        factor_df_slice = factor_df_slice.loc[
            factor_df_slice['time'] == t,
            ['time', 'instrument_id'] + fc_list,
        ].copy()
        slice_factor_list.append(factor_df_slice)

    slice_factor_df = pd.concat(slice_factor_list, ignore_index=True)

    left = full_factor_df[['time', 'instrument_id'] + fc_list].copy()
    left = left[left['time'].isin(sampled_time_list)]
    right = slice_factor_df[['time', 'instrument_id'] + fc_list].copy()

    merged = left.merge(
        right,
        on=['time', 'instrument_id'],
        how='outer',
        suffixes=('_full', '_slice'),
        indicator=True,
    )

    missing_row_df = merged.loc[merged['_merge'] != 'both', ['time', 'instrument_id', '_merge']].copy()
    mismatch_detail: Dict[str, int] = {}
    mismatch_examples: Dict[str, List[dict]] = {}

    both_df = merged.loc[merged['_merge'] == 'both'].copy()
    for fc_name in fc_list:
        col_full = f'{fc_name}_full'
        col_slice = f'{fc_name}_slice'
        is_equal = np.isclose(
            pd.to_numeric(both_df[col_full], errors='coerce').astype(float),
            pd.to_numeric(both_df[col_slice], errors='coerce').astype(float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        mismatch_mask = ~is_equal
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count > 0:
            mismatch_detail[fc_name] = mismatch_count
            mismatch_examples[fc_name] = both_df.loc[
                mismatch_mask,
                ['time', 'instrument_id', col_full, col_slice],
            ].head(5).to_dict(orient='records')

    failed_factor_set = set(mismatch_detail.keys())
    if len(missing_row_df) > 0:
        failed_factor_set.update(fc_list)

    passed = len(failed_factor_set) == 0
    result = {
        'passed': passed,
        'checked_factor_count': len(fc_list),
        'checked_time_count': len(sampled_time_list),
        'missing_row_count': int(len(missing_row_df)),
        'mismatch_factor_count': int(len(mismatch_detail)),
        'mismatch_detail': mismatch_detail,
        'mismatch_examples': mismatch_examples,
        'failed_factor_list': sorted(list(failed_factor_set)),
    }

    if not passed:
        log.error('Leakage check failed with details:')
        log.error(f'  checked_time_count={result["checked_time_count"]}')
        log.error(f'  missing_row_count={result["missing_row_count"]}')
        if len(missing_row_df) > 0:
            log.error(f'  missing_row_samples={missing_row_df.head(10).to_dict(orient="records")}')
        for fc_name in result['failed_factor_list']:
            if fc_name in mismatch_detail:
                log.error(
                    f'  factor={fc_name}, mismatch_count={mismatch_detail[fc_name]}, '
                    f'samples={mismatch_examples.get(fc_name, [])}'
                )

    if raise_error and not passed:
        raise ValueError(f'Leakage check failed: {result}')
    return result


def filter_fc_by_db_relative_spearman(
    selected_fc_name_list: Sequence[str],
    generated_df: pd.DataFrame,
    base_df: pd.DataFrame,
    base_col_list: Sequence[str],
    relative_threshold: float = 0.7,
    relative_check_version_list: Optional[Sequence[str]] = None,
    n_jobs: Optional[int] = None,
    batch_size: int = 100,
    database: str = 'factors',
    collections: Optional[Sequence[str]] = None,
    self_compare_version: Optional[str] = None,
    self_compare_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter factors by abs Spearman correlation against factors already stored in DB."""
    fc_list = [str(x) for x in selected_fc_name_list if str(x).strip()]
    if not fc_list:
        return {
            'enabled': True,
            'selected_fc_name_list': [],
            'filtered_out_fc_name_list': [],
            'checked_db_factor_count': 0,
            'threshold': float(relative_threshold),
            'versions': None if relative_check_version_list is None else list(relative_check_version_list),
            'detail': {},
            'collection_count': 0,
        }

    candidate_df = generated_df[['time', 'instrument_id'] + fc_list].copy()
    candidate_df = candidate_df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

    raw_records = get_factor_formula_records(
        collections=None if collections is None else list(collections),
        versions=None if relative_check_version_list is None else list(relative_check_version_list),
        database=database,
    )
    if raw_records.empty:
        log.info('Relative check skipped: no existing factors found in DB for requested scope.')
        return {
            'enabled': True,
            'selected_fc_name_list': list(fc_list),
            'filtered_out_fc_name_list': [],
            'checked_db_factor_count': 0,
            'threshold': float(relative_threshold),
            'versions': None if relative_check_version_list is None else list(relative_check_version_list),
            'detail': {fc: {'max_abs_spearman': 0.0} for fc in fc_list},
            'collection_count': 0,
        }

    raw_records = raw_records.copy()
    raw_records['factor_name'] = raw_records['factor_name'].astype(str)
    raw_records['version'] = raw_records['version'].astype(str)
    raw_records['collection'] = raw_records['collection'].astype(str)

    if self_compare_version is not None and self_compare_collection is not None:
        same_run_mask = (
            (raw_records['version'] == str(self_compare_version))
            & (raw_records['collection'] == str(self_compare_collection))
        )
        raw_records = raw_records.loc[~same_run_mask].copy()

    if raw_records.empty:
        return {
            'enabled': True,
            'selected_fc_name_list': list(fc_list),
            'filtered_out_fc_name_list': [],
            'checked_db_factor_count': 0,
            'threshold': float(relative_threshold),
            'versions': None if relative_check_version_list is None else list(relative_check_version_list),
            'detail': {fc: {'max_abs_spearman': 0.0} for fc in fc_list},
            'collection_count': 0,
        }

    raw_records['db_factor_key'] = raw_records.apply(
        lambda x: f"{x['collection']}::{x['version']}::{x['factor_name']}", axis=1
    )
    raw_records = raw_records.drop_duplicates(subset=['db_factor_key'], keep='last').reset_index(drop=True)

    base_df = base_df.sort_values(['instrument_id', 'time']).reset_index(drop=True)
    key_df = candidate_df[['time', 'instrument_id']].copy()
    candidate_values_df = candidate_df[fc_list].apply(pd.to_numeric, errors='coerce')

    detail_map: Dict[str, Dict[str, Any]] = {
        fc_name: {'max_abs_spearman': 0.0, 'matched_db_factor': None}
        for fc_name in fc_list
    }

    def _eval_records_chunk(df_chunk: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        formula_map = {
            row['db_factor_key']: row['formula']
            for _, row in df_chunk.iterrows()
            if isinstance(row.get('formula'), str) and row.get('formula', '').strip()
        }
        if not formula_map:
            return {}
        try:
            existing_df = calc_formula_df(df=base_df, formula_map=formula_map, data_fields=base_col_list)
        except Exception as e:
            log.warning(f'Relative check chunk skipped due to formula evaluation error: {e}')
            return {}

        aligned_existing = key_df.merge(existing_df, on=['time', 'instrument_id'], how='left', validate='1:1')
        existing_cols = list(formula_map.keys())
        existing_values_df = aligned_existing[existing_cols].apply(pd.to_numeric, errors='coerce')

        corr_mat = pd.concat([candidate_values_df, existing_values_df], axis=1).corr(method='spearman').abs()
        cross = corr_mat.loc[fc_list, existing_cols].fillna(0.0)

        chunk_result: Dict[str, Dict[str, Any]] = {}
        for fc_name in fc_list:
            row = cross.loc[fc_name]
            if row.empty:
                continue
            max_key = row.idxmax()
            max_corr = float(row.loc[max_key])
            chunk_result[fc_name] = {'max_abs_spearman': max_corr, 'matched_db_factor': max_key}
        return chunk_result

    chunk_df_list = [raw_records.iloc[i:i + batch_size].copy() for i in range(0, len(raw_records), batch_size)]
    effective_jobs = max(1, min(len(chunk_df_list), int(n_jobs or 1)))
    if effective_jobs > 1 and len(chunk_df_list) > 1:
        chunk_result_list = Parallel(n_jobs=effective_jobs, prefer='threads')(
            delayed(_eval_records_chunk)(chunk_df) for chunk_df in chunk_df_list
        )
    else:
        chunk_result_list = [_eval_records_chunk(chunk_df) for chunk_df in chunk_df_list]

    for chunk_result in chunk_result_list:
        for fc_name, item in chunk_result.items():
            if item.get('max_abs_spearman', 0.0) > detail_map[fc_name]['max_abs_spearman']:
                detail_map[fc_name] = item

    selected_fc: List[str] = []
    filtered_out_fc: List[str] = []
    for fc_name in fc_list:
        max_corr = float(detail_map[fc_name].get('max_abs_spearman', 0.0))
        matched_db_factor = detail_map[fc_name].get('matched_db_factor')
        if max_corr < float(relative_threshold):
            selected_fc.append(fc_name)
            decision = 'keep'
        else:
            filtered_out_fc.append(fc_name)
            decision = 'remove'

        log.info(
            '[RelativeCheck][PerFactor] '
            f'factor={fc_name}, '
            f'max_abs_spearman={max_corr:.6f}, '
            f'matched_db_factor={matched_db_factor}, '
            f'threshold={relative_threshold}, '
            f'decision={decision}'
        )

    removed_detail = [
        {
            'factor_name': fc_name,
            'max_abs_spearman': float(detail_map[fc_name].get('max_abs_spearman', 0.0)),
            'matched_db_factor': detail_map[fc_name].get('matched_db_factor'),
        }
        for fc_name in filtered_out_fc
    ]
    kept_detail = [
        {
            'factor_name': fc_name,
            'max_abs_spearman': float(detail_map[fc_name].get('max_abs_spearman', 0.0)),
            'matched_db_factor': detail_map[fc_name].get('matched_db_factor'),
        }
        for fc_name in selected_fc
    ]

    if filtered_out_fc:
        log.warning(
            'Relative correlation filter removed factors: '
            f'threshold={relative_threshold}, removed={removed_detail}'
        )
    else:
        log.info(
            'Relative correlation filter removed factors: '
            f'threshold={relative_threshold}, removed=[]'
        )

    log.info(
        'Relative correlation filter kept factors: '
        f'threshold={relative_threshold}, kept={kept_detail}'
    )

    return {
        'enabled': True,
        'selected_fc_name_list': selected_fc,
        'filtered_out_fc_name_list': filtered_out_fc,
        'checked_db_factor_count': int(len(raw_records)),
        'threshold': float(relative_threshold),
        'versions': None if relative_check_version_list is None else list(relative_check_version_list),
        'detail': detail_map,
        'kept_detail': kept_detail,
        'removed_detail': removed_detail,
        'collection_count': int(raw_records['collection'].nunique()),
    }


