"""Daily job: update futures continuous contract info.

Example commands:
	# Use defaults: C0, overwrite-existing + insert-new
	# python daily/update_futures_info.py
	# Update one instrument
	# python daily/update_futures_info.py --instrument_id C0
	# Update multiple instruments
	# python daily/update_futures_info.py --instrument_id C0,AU0,RB0 --method bulk_write_update
"""

import argparse
import sys
from pathlib import Path

# Make `python daily/update_futures_info.py` work by adding project root to import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data import update_futures_continuous_contract_info


def _parse_instrument_id(value: str):
	if not value:
		return None
	ids = [item.strip() for item in value.split(',') if item.strip()]
	return ids if len(ids) > 1 else ids[0]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Update futures continuous contract info to MongoDB.')
	parser.add_argument('--instrument_id', type=str, default='C0',
						help='Instrument id, supports one id (C0) or comma-separated ids (C0,AU0,RB0).')
	parser.add_argument('--method', type=str, default='bulk_write_update',
						choices=['bulk_write_update', 'update_one', 'insert_many'],
						help='Update method. Default is overwrite-existing + insert-new.')
	return parser


def main():
	args = build_parser().parse_args()
	update_futures_continuous_contract_info(
		instrument_id=_parse_instrument_id(args.instrument_id),
		method=args.method,
	)


if __name__ == '__main__':
	main()

