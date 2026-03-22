"""Utilities for factor-related database operations."""

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd

from mongo.mongify import get_data, update_one_data
from utils.logging import log


def update_factor_info(selected_fc_name_list: Sequence[str],
                       performance_summary: Optional[pd.DataFrame],
                       factor_formula_map: Dict[str, str],
                       factor_fitness_map: Dict[str, Dict[str, Dict[str, float]]],
                       instrument_id_list: Sequence[str],
                       method: str,
                       version: str,
                       start_date: Optional[str],
                       end_date: Optional[str],
                       database: str = 'factors',
                       collection: str = 'genetic_programming') -> None:
    """Upsert selected factor metadata into database.

    The default target is `factors.genetic_programming`.
    """
    if not selected_fc_name_list or performance_summary is None:
        return

    summary_df = performance_summary.copy()
    if 'year' not in summary_df.columns:
        summary_df = summary_df.reset_index()

    default_instruments = list(instrument_id_list)
    now_text = datetime.now().isoformat(timespec='seconds')
    persisted_formula_map: Dict[str, str] = {}
    persisted_record_count = 0
    skipped_fc_name_list = []

    for fc_name in selected_fc_name_list:
        formula = factor_formula_map.get(fc_name)
        fitness_dict = factor_fitness_map.get(fc_name)
        if formula is None or fitness_dict is None:
            skipped_fc_name_list.append(fc_name)
            continue

        df_fc = summary_df.loc[summary_df['Factor Name'] == fc_name].copy()
        if 'Instrument ID' in df_fc.columns:
            instrument_ids = [x for x in df_fc['Instrument ID'].dropna().unique().tolist()]
        else:
            instrument_ids = list(default_instruments)
        if not instrument_ids:
            instrument_ids = list(default_instruments)

        cleaned_fitness: Dict[str, Dict[str, float]] = {}
        for metric_name, metric_payload in fitness_dict.items():
            # Backward compatibility: legacy payload used flat numeric fitness.
            if not isinstance(metric_payload, dict):
                if metric_payload is None or pd.isna(metric_payload):
                    continue
                metric_value = float(pd.to_numeric(metric_payload, errors='coerce'))
                cleaned_fitness[metric_name] = {'value': metric_value}
                continue

            metric_cleaned: Dict[str, float] = {}
            original_val = metric_payload.get('original')
            penalized_val = metric_payload.get('penalized')
            direct_val = metric_payload.get('value')

            if original_val is not None and not pd.isna(original_val):
                metric_cleaned['original'] = float(original_val)
            if penalized_val is not None and not pd.isna(penalized_val):
                metric_cleaned['penalized'] = float(penalized_val)
            if direct_val is not None and not pd.isna(direct_val):
                metric_cleaned['value'] = float(direct_val)

            if metric_cleaned:
                cleaned_fitness[metric_name] = metric_cleaned

        for ins_id in instrument_ids:
            record = {
                'method': method,
                'version': version,
                'factor_name': fc_name,
                'instrument_id': ins_id,
                'start_date': start_date,
                'end_date': end_date,
                'fitness': cleaned_fitness,
                'formula': formula,
                'created_at': now_text,
            }
            mongo_operator = {
                'version': version,
                'factor_name': fc_name,
                'instrument_id': ins_id,
                'start_date': start_date,
                'end_date': end_date,
            }
            update_one_data(
                database=database,
                collection=collection,
                mongo_operator=mongo_operator,
                data=record,
                upsert=True,
            )
            persisted_record_count += 1

        persisted_formula_map[fc_name] = formula

    persisted_fc_name_list = list(persisted_formula_map.keys())
    log.info(
        'DB factor upsert finished: '
        f'selected_factor_count={len(selected_fc_name_list)}, '
        f'persisted_factor_count={len(persisted_fc_name_list)}, '
        f'persisted_record_count={persisted_record_count}, '
        f'skipped_factor_count={len(skipped_fc_name_list)}'
    )
    if skipped_fc_name_list:
        log.warning(f'Skipped factors due to missing formula/fitness: {skipped_fc_name_list}')
    if persisted_formula_map:
        log.info(f'Persisted factors: {persisted_formula_map.keys()}')


def get_factor_records_by_names(fc_name_list: Sequence[str],
                                collections: Optional[Sequence[str]] = None,
                                database: str = 'factors') -> pd.DataFrame:
    """Read factor metadata from factors DB across multiple collections."""
    if not fc_name_list:
        return pd.DataFrame()
    target_collections = list(collections) if collections else ['genetic_programming', 'llm_prompt']
    all_df: List[pd.DataFrame] = []
    operator = {'factor_name': {'$in': list(fc_name_list)}}
    for col in target_collections:
        try:
            df = get_data(database=database, collection=col, mongo_operator=operator)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df['collection'] = col
            all_df.append(df)
    if not all_df:
        return pd.DataFrame()
    return pd.concat(all_df, ignore_index=True)


def get_latest_factor_formula_map(fc_name_list: Sequence[str],
                                  collections: Optional[Sequence[str]] = None,
                                  database: str = 'factors') -> Dict[str, str]:
    """Resolve latest available formula for each factor name from factors DB."""
    df = get_factor_records_by_names(fc_name_list=fc_name_list, collections=collections, database=database)
    if df.empty:
        return {}

    if 'created_at' in df.columns:
        df['__created_at__'] = pd.to_datetime(df['created_at'], errors='coerce')
    else:
        df['__created_at__'] = pd.NaT
    if 'version' not in df.columns:
        df['version'] = ''

    df = df.sort_values(['factor_name', '__created_at__', 'version'], ascending=[True, False, False])
    formula_map: Dict[str, str] = {}
    for fc_name, df_i in df.groupby('factor_name', sort=False):
        formula = df_i.iloc[0].get('formula')
        if isinstance(formula, str) and formula.strip():
            formula_map[str(fc_name)] = formula.strip()
    return formula_map


