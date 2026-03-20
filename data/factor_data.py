"""Utilities for factor-related database operations."""

from datetime import datetime
from typing import Dict, List, Optional, Sequence

import pandas as pd

from mongo.mongify import update_one_data


def update_factor_info(selected_fc_name_list: Sequence[str],
                       performance_summary: Optional[pd.DataFrame],
                       factor_formula_map: Dict[str, str],
                       factor_fitness_map: Dict[str, Dict[str, float]],
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

    for fc_name in selected_fc_name_list:
        formula = factor_formula_map.get(fc_name)
        fitness_dict = factor_fitness_map.get(fc_name)
        if formula is None or fitness_dict is None:
            continue

        df_fc = summary_df.loc[summary_df['Factor Name'] == fc_name].copy()
        if 'Instrument ID' in df_fc.columns:
            instrument_ids = [x for x in df_fc['Instrument ID'].dropna().unique().tolist()]
        else:
            instrument_ids = list(default_instruments)
        if not instrument_ids:
            instrument_ids = list(default_instruments)

        cleaned_fitness = {
            k: float(v)
            for k, v in fitness_dict.items()
            if v is not None and not pd.isna(v)
        }

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

