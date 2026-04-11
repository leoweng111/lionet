import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow running this file directly: `python test/factor_fusion_demo.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import get_factor_formula_records
from factors.factor_auto_search import FactorFusioner


def _build_raw_factor_dict(collection: str, top_n: int) -> Dict[str, List[str]]:
    records = get_factor_formula_records(collections=[collection], versions=None, database='factors')
    if records.empty:
        raise ValueError(f'No factors found in DB for collection={collection}.')

    records = records.copy()
    records['version'] = records['version'].astype(str)
    records['factor_name'] = records['factor_name'].astype(str)

    latest_version = sorted(records['version'].dropna().unique().tolist())[-1]
    names = records.loc[records['version'] == latest_version, 'factor_name'].dropna().tolist()
    names = list(dict.fromkeys([str(x).strip() for x in names if str(x).strip()]))
    if not names:
        raise ValueError(f'No valid factor_name found in latest version={latest_version}.')

    return {latest_version: names[:max(1, int(top_n))]}


def main() -> None:
    parser = argparse.ArgumentParser(description='Minimal demo for FactorFusioner.')
    parser.add_argument('--collection', type=str, default='genetic_programming')
    parser.add_argument('--instrument-id', type=str, default='C0')
    parser.add_argument('--start-time', type=str, default='20240101')
    parser.add_argument('--end-time', type=str, default='20240630')
    parser.add_argument('--max-fusion-count', type=int, default=3)
    parser.add_argument('--version', type=str, required=True,
                        help='Fusion result version to persist into factors.factor_fusion.')
    parser.add_argument('--fusion-metrics', type=str, default='ic,sharpe',
                        help='Comma separated list, e.g. ic or ic,sharpe')
    parser.add_argument('--use-raw-factor-dict', action='store_true',
                        help='If set, pick latest DB version and top-n factors to build raw_factor_dict.')
    parser.add_argument('--raw-top-n', type=int, default=8)
    args = parser.parse_args()

    fusion_metrics = [x.strip() for x in args.fusion_metrics.split(',') if x.strip()]
    raw_factor_dict = None
    if args.use_raw_factor_dict:
        raw_factor_dict = _build_raw_factor_dict(collection=args.collection, top_n=args.raw_top_n)

    fusioner = FactorFusioner(
        fusion_method='avg_weight',
        raw_factor_dict=raw_factor_dict,
        collection=args.collection,
        instrument_id_list=args.instrument_id,
        start_time=args.start_time,
        end_time=args.end_time,
        max_fusion_count=args.max_fusion_count,
        fusion_metrics=fusion_metrics,
        version=args.version,
        apply_weighted_price=True,
        n_jobs=1,
    )

    result = fusioner.fuse()
    out = {
        'fusion_method': result['fusion_method'],
        'fusion_metrics': result['fusion_metrics'],
        'selected_factor_keys': result['selected_factor_keys'],
        'final_metrics': result['final_metrics'],
        'selected_factors_detail': result['selected_factors_detail'],
    }
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
