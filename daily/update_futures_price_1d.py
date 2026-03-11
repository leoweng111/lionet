"""Daily job: update futures continuous contract daily price.

Example commands:
	# Use defaults: C0, full configured date range, overwrite-existing + insert-new
	# python daily/update_futures_price_1d.py
	# Update a date range for one instrument
	# python daily/update_futures_price_1d.py --instrument_id C0 --start_date 20200101 --end_date 20260331 \
	    --method bulk_write_update
	# Update multiple instruments
	# python daily/update_futures_price_1d.py --instrument_id C0,AU0,RB0 --wait_time 0.2
"""

import argparse
import sys
from pathlib import Path

# Make `python daily/update_futures_price_1d.py` work by adding project root to import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data import update_futures_continuous_contract_price


def _parse_instrument_id(value: str):
	if not value:
		return None
	ids = [item.strip() for item in value.split(',') if item.strip()]
	return ids if len(ids) > 1 else ids[0]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Update futures continuous contract daily price to MongoDB.')
	parser.add_argument('--instrument_id', type=str, default='C0',
						help='Instrument id, supports one id (C0) or comma-separated ids (C0,AU0,RB0).')
	parser.add_argument('--start_date', type=str, default=None, help='Start date in YYYYMMDD format.')
	parser.add_argument('--end_date', type=str, default=None, help='End date in YYYYMMDD format.')
	parser.add_argument('--wait_time', type=float, default=0.3, help='Sleep seconds between AkShare requests.')
	parser.add_argument('--method', type=str, default='bulk_write_update',
						choices=['bulk_write_update', 'update_one', 'insert_many'],
						help='Update method. Default is overwrite-existing + insert-new.')
	return parser


def main():
	args = build_parser().parse_args()
	update_futures_continuous_contract_price(
		instrument_id=_parse_instrument_id(args.instrument_id),
		start_date=args.start_date,
		end_date=args.end_date,
		wait_time=args.wait_time,
		method=args.method,
	)


if __name__ == '__main__':
	main()
