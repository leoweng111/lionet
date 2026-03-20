"""Utilities for factor-related database operations."""

from datetime import datetime
from typing import Dict, Optional, Sequence

import pandas as pd

from mongo.mongify import update_one_data
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
                metric_value = float(metric_payload)
                cleaned_fitness[metric_name] = {
                    'original': metric_value,
                    'penalized': metric_value,
                }
                continue

            metric_cleaned: Dict[str, float] = {}
            original_val = metric_payload.get('original')
            penalized_val = metric_payload.get('penalized')

            if original_val is not None and not pd.isna(original_val):
                metric_cleaned['original'] = float(original_val)
            if penalized_val is not None and not pd.isna(penalized_val):
                metric_cleaned['penalized'] = float(penalized_val)

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
        log.info(f'Persisted factors and formulas: {persisted_formula_map}')

