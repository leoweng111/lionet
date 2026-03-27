"""Demo: validate stitched continuous futures price vs ak.futures_main_sina.

Usage:
    python test/validate_continuous_price_demo.py
"""

from data.futures import compare_with_ak_main_continuous


def main():
    instrument = 'C'
    start_date = '20200101'
    end_date = '20241231'

    mismatch_df = compare_with_ak_main_continuous(
        instrument_id=instrument,
        start_date=start_date,
        end_date=end_date,
        wait_time=0.1,
        atol=1e-6,
    )

    if mismatch_df.empty:
        print(f'[OK] No mismatch found for {instrument} in [{start_date}, {end_date}].')
        return

    print(f'[WARN] Found {len(mismatch_df)} mismatch rows for {instrument}.')
    print('Mismatch dates (first 50):')
    print(mismatch_df['time'].dt.strftime('%Y-%m-%d').head(50).to_list())
    print('\nSample mismatch rows:')
    cols = [
        'time', 'symbol', 'is_rollover',
        'open_raw', 'open', 'high_raw', 'high', 'low_raw', 'low', 'close_raw', 'close',
    ]
    print(mismatch_df[cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()

