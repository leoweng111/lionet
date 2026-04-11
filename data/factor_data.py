"""Utilities for factor-related database operations."""

from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from joblib import Parallel, delayed

from mongo.mongify import (
    delete_data,
    get_data,
    list_collection_names,
    update_many_data,
    update_one_data,
)
from utils.logging import log


def _resolve_factor_formula(fc_name: str, formula_map: Dict[str, str]) -> str:
    formula = formula_map.get(fc_name)
    if not isinstance(formula, str) or not formula.strip():
        available = sorted(formula_map.keys())
        raise NameError(f'Factor `{fc_name}` formula is not available in DB. Available factors: {available}')
    return formula.strip()


def _calc_one_factor_value(df: pd.DataFrame, fc_name: str, formula_map: Dict[str, str]) -> pd.DataFrame:
    """Calculate one factor column with a picklable helper for joblib."""
    from factors.factor_ops import calc_formula_series
    formula = _resolve_factor_formula(fc_name, formula_map)
    col = calc_formula_series(df=df, formula=formula)
    return pd.DataFrame({'time': df['time'], 'instrument_id': df['instrument_id'], fc_name: col.values})


def get_factor_value(Data: pd.DataFrame,
                     fc_name_list: Union[str, list],
                     version: str,
                     collection: Union[str, Sequence[str]] = 'genetic_programming',
                     n_jobs: int = 5) -> pd.DataFrame:
    """Calculate factor values and append them to input DataFrame.

    :return: original dataframe columns + factor values in fc_name_list
    """
    if isinstance(fc_name_list, str):
        fc_name_list = [fc_name_list]
    fc_name_counter = Counter(fc_name_list)
    duplicated_fc_names = sorted([x for x, cnt in fc_name_counter.items() if cnt > 1])
    if duplicated_fc_names:
        raise ValueError(f'fc_name_list contains duplicated factor names: {duplicated_fc_names}')
    for col in ['time', 'instrument_id']:
        assert col in Data.columns, f'Data does not contain column {col}.'

    df = Data.copy().sort_values(['instrument_id', 'time']).reset_index(drop=True)
    if isinstance(collection, str):
        collection_list = [collection]
    else:
        collection_list = list(collection)
    collection_list = [str(x).strip() for x in collection_list if str(x).strip()]
    if not collection_list:
        raise ValueError('`collection` cannot be empty in get_factor_value.')

    formula_map = get_factor_formula_map_by_version(
        fc_name_list=fc_name_list,
        version=version,
        collections=collection_list,
    )
    missing = [name for name in fc_name_list if name not in formula_map]
    if missing:
        raise ValueError(
            f'No formula found in DB for factors: {missing}. '
            f'version={version}, collections={collection_list}'
        )

    with Parallel(n_jobs=n_jobs) as parallel:
        mapper_list = parallel(
            delayed(_calc_one_factor_value)(df, fc_name, formula_map) for fc_name in fc_name_list
        )

    out = df.copy()
    for mapper in mapper_list:
        factor_col = [c for c in mapper.columns if c not in ['time', 'instrument_id']]
        if factor_col and factor_col[0] in out.columns:
            out = out.drop(columns=[factor_col[0]])
        out = out.merge(mapper, on=['time', 'instrument_id'], how='left', validate='1:1')
    return out


def update_factor_info(selected_fc_name_list: Sequence[str],
                       performance_summary: Optional[pd.DataFrame],
                       factor_formula_map: Dict[str, str],
                       factor_fitness_map: Dict[str, Dict[str, Dict[str, float]]],
                       instrument_id_list: Sequence[str],
                       method: str,
                       version: str,
                       start_date: Optional[str],
                       end_date: Optional[str],
                       extra_record_fields: Optional[Dict[str, Any]] = None,
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
            if extra_record_fields:
                record.update(dict(extra_record_fields))
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
                                versions: Optional[Sequence[str]] = None,
                                database: str = 'factors') -> pd.DataFrame:
    """Read factor metadata from factors DB across multiple collections."""
    if not fc_name_list:
        return pd.DataFrame()
    target_collections = list(collections) if collections else ['genetic_programming', 'llm_prompt']
    all_df: List[pd.DataFrame] = []
    operator = {'factor_name': {'$in': list(fc_name_list)}}
    if versions is not None:
        version_list = [str(x).strip() for x in versions if str(x).strip()]
        if not version_list:
            return pd.DataFrame()
        operator['version'] = {'$in': version_list}
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


def get_factor_formula_map_by_version(fc_name_list: Sequence[str],
                                      version: str,
                                      collections: Optional[Sequence[str]] = None,
                                      database: str = 'factors') -> Dict[str, str]:
    """Resolve formula for each factor name by exact version from factors DB."""
    version = str(version).strip()
    if not version:
        raise ValueError('`version` is required for get_factor_formula_map_by_version.')

    df = get_factor_records_by_names(
        fc_name_list=fc_name_list,
        collections=collections,
        versions=[version],
        database=database,
    )
    if df.empty:
        return {}

    formula_map: Dict[str, str] = {}
    for fc_name, df_i in df.groupby('factor_name', sort=False):
        formula_set = {
            str(x).strip()
            for x in df_i['formula'].tolist()
            if isinstance(x, str) and str(x).strip()
        }
        if len(formula_set) > 1:
            raise ValueError(
                f'Ambiguous formulas found for factor_name={fc_name}, version={version}, '
                f'collections={collections}. Please specify collection or fix DB records.'
            )
        if formula_set:
            formula_map[str(fc_name)] = next(iter(formula_set))
    return formula_map


def get_factor_formula_records(collections: Optional[Sequence[str]] = None,
                               versions: Optional[Sequence[str]] = None,
                               database: str = 'factors') -> pd.DataFrame:
    """Read factor formulas across collections with optional version filter.

    Returns columns including: collection, version, factor_name, formula.
    """
    if collections is None:
        try:
            target_collections = list_collection_names(database=database)
        except Exception:
            target_collections = ['genetic_programming', 'llm_prompt']
    else:
        target_collections = list(collections)

    if not target_collections:
        return pd.DataFrame()

    version_list = list(versions) if versions is not None else None
    all_df: List[pd.DataFrame] = []
    for col in target_collections:
        operator: Dict[str, object] = {}
        if version_list is not None:
            if not version_list:
                continue
            operator['version'] = {'$in': version_list}

        try:
            df = get_data(database=database, collection=col, mongo_operator=operator)
        except Exception:
            continue

        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        if 'formula' not in df.columns or 'factor_name' not in df.columns:
            continue

        df = df.copy()
        df['collection'] = col
        df['version'] = df['version'] if 'version' in df.columns else ''
        df['formula'] = df['formula'].astype(str).str.strip()
        df = df[df['formula'] != '']
        if df.empty:
            continue
        keep_cols = [c for c in ['collection', 'version', 'factor_name', 'formula', 'created_at'] if c in df.columns]
        all_df.append(df[keep_cols].copy())

    if not all_df:
        return pd.DataFrame()

    out = pd.concat(all_df, ignore_index=True)
    dedup_cols = [c for c in ['collection', 'version', 'factor_name'] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep='last').reset_index(drop=True)
    return out


def delete_factor_data(database: Optional[str] = None,
                       collection: Optional[str] = None,
                       version_list: Optional[Sequence[str]] = None,
                       version: Optional[str] = None) -> None:
    """Delete factor records by one or many versions.

    Cases:
    1) database + collection + version_list: delete in one collection.
    2) only version_list: delete from all collections in `factors` DB.

    Note:
    - `version` is kept for backward compatibility and will be merged into `version_list`.
    """
    merged_versions: List[str] = []
    if version_list is not None:
        merged_versions.extend([str(x).strip() for x in version_list if str(x).strip()])
    if version is not None and str(version).strip():
        merged_versions.append(str(version).strip())

    normalized_version_list = list(dict.fromkeys(merged_versions))
    if not normalized_version_list:
        raise ValueError('`version_list` is required for delete_factor_data and cannot be empty.')

    mongo_operator = {'version': {'$in': normalized_version_list}}

    if database and collection:
        delete_data(
            database=database,
            collection=collection,
            mongo_operator=mongo_operator,
        )
        log.warning(
            f'Deleted factor data: database={database}, collection={collection}, '
            f'version_list={normalized_version_list}'
        )
        return

    if database or collection:
        raise ValueError(
            'Please provide both `database` and `collection`, or provide only `version`.'
        )

    target_database = 'factors'
    try:
        target_collections = list_collection_names(database=target_database)
    except Exception as e:
        raise RuntimeError(f'Failed to list collections from `{target_database}`: {e}')

    for col in target_collections:
        delete_data(
            database=target_database,
            collection=col,
            mongo_operator=mongo_operator,
        )
    log.warning(
        f'Deleted factor data for version_list={normalized_version_list} '
        f'from all collections in database={target_database}: {target_collections}'
    )


def change_factor_version(database: Optional[str] = None,
                          collection: Optional[str] = None,
                          old_version: Optional[str] = None,
                          new_version: Optional[str] = None) -> None:
    """Change version field from old_version to new_version.

    Cases:
    1) database + collection + old_version + new_version: update one collection.
    2) only old_version + new_version: update all collections in `factors` DB.
    """
    if not old_version or not str(old_version).strip():
        raise ValueError('`old_version` is required for change_factor_version.')
    if not new_version or not str(new_version).strip():
        raise ValueError('`new_version` is required for change_factor_version.')

    old_version = str(old_version).strip()
    new_version = str(new_version).strip()

    if database and collection:
        result = update_many_data(
            database=database,
            collection=collection,
            mongo_operator={'version': old_version},
            update_data={'version': new_version},
        )
        log.warning(
            'Changed factor version: '
            f'database={database}, collection={collection}, '
            f'old_version={old_version}, new_version={new_version}, '
            f"matched={result['matched_count']}, modified={result['modified_count']}"
        )
        return

    if database or collection:
        raise ValueError(
            'Please provide both `database` and `collection`, or provide only old/new version.'
        )

    target_database = 'factors'
    try:
        target_collections = list_collection_names(database=target_database)
    except Exception as e:
        raise RuntimeError(f'Failed to list collections from `{target_database}`: {e}')

    matched_total = 0
    modified_total = 0
    for col in target_collections:
        result = update_many_data(
            database=target_database,
            collection=col,
            mongo_operator={'version': old_version},
            update_data={'version': new_version},
        )
        matched_total += int(result['matched_count'])
        modified_total += int(result['modified_count'])

    log.warning(
        'Changed factor version across all collections: '
        f'database={target_database}, old_version={old_version}, new_version={new_version}, '
        f'collection_count={len(target_collections)}, matched_total={matched_total}, '
        f'modified_total={modified_total}, collections={target_collections}'
    )

