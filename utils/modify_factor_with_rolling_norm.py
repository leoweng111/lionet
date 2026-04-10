"""Batch-wrap existing DB factor formulas with OpRollNorm."""

import argparse
import ast
from typing import Dict

from mongo.mongify import get_data, update_one_data
from utils.logging import log


def _is_top_level_roll_norm(formula: str) -> bool:
    text = str(formula).strip()
    if not text:
        return False
    try:
        node = ast.parse(text, mode='eval').body
    except Exception:
        return text.startswith('OpRollNorm(')
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    return node.func.id.lower() in {'oprollnorm', 'rollnorm'}


def _wrap_formula(formula: str,
                  rolling_norm_window: int,
                  rolling_norm_min_periods: int,
                  rolling_norm_eps: float,
                  rolling_norm_clip: float) -> str:
    return (
        f'OpRollNorm({formula}, {int(rolling_norm_window)}, {int(rolling_norm_min_periods)}, '
        f'{float(rolling_norm_eps):.10g}, {float(rolling_norm_clip):.10g})'
    )


def modify_factor_with_rolling_norm(
    database: str = 'factors',
    collection: str = 'genetic_programming',
    rolling_norm_window: int = 30,
    rolling_norm_min_periods: int = 20,
    rolling_norm_eps: float = 1e-8,
    rolling_norm_clip: float = 5.0,
) -> Dict[str, int]:
    """Wrap non-wrapped formulas in one collection with OpRollNorm.

    Returns a summary dict with total/updated/skipped counts.
    """
    if int(rolling_norm_window) <= 0:
        raise ValueError('rolling_norm_window must be positive.')
    if int(rolling_norm_min_periods) <= 0:
        raise ValueError('rolling_norm_min_periods must be positive.')
    if float(rolling_norm_eps) <= 0:
        raise ValueError('rolling_norm_eps must be positive.')
    if float(rolling_norm_clip) <= 0:
        raise ValueError('rolling_norm_clip must be positive.')

    df = get_data(database=database, collection=collection, mongo_operator={}, idx=True)
    if df is None or df.empty:
        log.warning(f'No records found in {database}.{collection}.')
        return {'total': 0, 'updated': 0, 'skipped': 0}
    if 'formula' not in df.columns:
        raise ValueError(f'Collection {database}.{collection} does not contain `formula` field.')

    total = int(len(df))
    updated = 0
    skipped = 0

    for _, row in df.iterrows():
        formula = str(row.get('formula', '')).strip()
        if not formula or _is_top_level_roll_norm(formula):
            skipped += 1
            continue

        new_formula = _wrap_formula(
            formula=formula,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
        )

        record = row.to_dict()
        record['formula'] = new_formula

        if '_id' in record and record['_id'] is not None:
            mongo_operator = {'_id': record['_id']}
        else:
            mongo_operator = {
                'version': record.get('version'),
                'factor_name': record.get('factor_name'),
                'instrument_id': record.get('instrument_id'),
                'start_date': record.get('start_date'),
                'end_date': record.get('end_date'),
            }

        update_one_data(
            database=database,
            collection=collection,
            mongo_operator=mongo_operator,
            data=record,
            upsert=False,
        )
        updated += 1

    summary = {'total': total, 'updated': updated, 'skipped': skipped}
    log.info(
        f'modify_factor_with_rolling_norm finished: database={database}, collection={collection}, '
        f'total={total}, updated={updated}, skipped={skipped}'
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Wrap factor formulas with OpRollNorm in one DB collection.')
    parser.add_argument('--database', type=str, default='factors')
    parser.add_argument('--collection', type=str, default='genetic_programming')
    parser.add_argument('--rolling-norm-window', type=int, default=30)
    parser.add_argument('--rolling-norm-min-periods', type=int, default=20)
    parser.add_argument('--rolling-norm-eps', type=float, default=1e-8)
    parser.add_argument('--rolling-norm-clip', type=float, default=5.0)
    args = parser.parse_args()

    summary = modify_factor_with_rolling_norm(
        database=args.database,
        collection=args.collection,
        rolling_norm_window=args.rolling_norm_window,
        rolling_norm_min_periods=args.rolling_norm_min_periods,
        rolling_norm_eps=args.rolling_norm_eps,
        rolling_norm_clip=args.rolling_norm_clip,
    )
    print(summary)


if __name__ == '__main__':
    main()

