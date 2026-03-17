# Factor Auto Search

`factors/factor_auto_search.py` provides `FactorGenerator` for auto factor mining.

## Quick Start

```python
from factors.factor_auto_search import FactorGenerator

fg = FactorGenerator(
    method='tsfresh',
    instrument_id_list=['C0', 'FG0'],
    fc_freq='1d',
    start_time='20230101',
    end_time='20260310',
    min_window_size=30,
    max_factor_count=200,
    tsfresh_profile='minimal',
    apply_rolling_norm=True,
    rolling_norm_window=252,
    rolling_norm_min_periods=20,
    rolling_norm_clip=10.0,
    n_jobs=5,
)

generated_df = fg.generate()
fc_subset = fg.generated_fc_name_list[:20]
fg.save_fc_value(fc_subset, file_name='tsfresh_fc_subset', file_format='parquet')
bt = fg.backtest(fc_name_list=fc_subset)

# save reusable tsfresh feature package
config_path = fg.save_fc(fc_subset)  # default dir: factors/fc_from_tsfresh/

# later: reload and compute only this subset
selected_fc = FactorGenerator.load_fc(config_path)
generated_subset_df = fg.generate_with_fc(selected_fc)

# one-step: load config + generate + backtest
bt2 = fg.backtest_from_fc_config(config_path)

# one-step: mine + filter + save high-quality config
result = fg.auto_mine_select_and_save_fc(
    net_ret_threshold=0.05,
    sharpe_threshold=0.8,
    fc_package_name='tsfresh_high_quality_fc',
    require_all_instruments=False,
    method='tsfresh',
)
print(result['config_path'])
print(result['selected_fc_name_list'])

# llm prompt mode (DeepSeek)
fg_llm = FactorGenerator(
    method='llm_prompt',
    instrument_id_list=['C0'],
    fc_freq='1d',
    start_time='20230101',
    end_time='20260310',
    model_name='deepseek',
    llm_temperature=0.7,
    llm_factor_count=6,
    llm_user_requirement='Generate simple momentum and volatility factors.'
)
result_llm = fg_llm.auto_mine_select_and_save_fc(
    net_ret_threshold=0.03,
    sharpe_threshold=0.6,
    method='llm_prompt',
)
print(result_llm['config_path'])
```

## Output Data Format

Generated data is compatible with `BackTester(data=..., fc_name_list=...)`:

- `time`
- `instrument_id`
- `future_ret`
- tsfresh factor columns

Saved factor files are written to `data/factor_value/`.

Selected tsfresh feature definitions are saved under `factors/fc_from_tsfresh/`.

LLM-generated valid factor classes are persisted to `factors/factor_from_llm.py`.

## Rolling Normalization (No Leakage)

- Normalization is done per `instrument_id` and per factor column.
- At time `t`, the normalization only uses history up to `t-1` via `shift(1)` + `rolling(...)`.
- Default output is clipped to `[-10, 10]` (`rolling_norm_clip=10.0`) to avoid oversized position signals.
- Early samples with insufficient history are set to `0.0` to keep strategy behavior stable.

## Threshold Filter Rules

- Filter uses backtest summary columns: `Net Return` and `Net Sharpe`.
- A factor is kept only if all yearly rows pass thresholds.
- By default, `year='all'` row must also pass (`require_all_row=True`).
- For multi-instrument results, default is all instruments must pass (`require_all_instruments=True`).
- Set `require_all_instruments=False` to keep factors when at least one instrument passes.

## LLM Config

- Set `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` in `utils/params.py`.
- `llm_prompt` mode expects strict JSON output from LLM and validates syntax/runtime before use.

