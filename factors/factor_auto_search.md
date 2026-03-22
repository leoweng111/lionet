# Factor Auto Search

`factors/factor_auto_search.py` adopts a formula-first and DB-first workflow:

- Factor formulas are persisted directly to MongoDB (`factors` database).
- No local factor JSON package is generated.
- Supported mining methods: `llm_prompt` and `genetic_programming`.

## Quick Start

```python
from factors.factor_auto_search import FactorGenerator, LLMPromptFactorGenerator

fg = LLMPromptFactorGenerator(
    instrument_id_list=['C0'],
    fc_freq='1d',
    start_time='20230101',
    end_time='20260310',
    model_name='deepseek',
    llm_temperature=0.7,
    llm_factor_count=6,
    llm_user_requirement='生成期货日频量价因子，偏向趋势和量价共振',
    n_jobs=5,
)

result = fg.auto_mine_select_and_save_fc(
    filter_indicator_dict={
        'Net Return': (0.03, 0.03, 1),
        'Net Sharpe': (0.6, 0.6, 1),
    },
    require_all_instruments=False,
)

config_ref = result['config_ref']
selected_fc = FactorGenerator.load_fc(config_ref)
bt = fg.backtest_from_fc_config(config_ref)
print(config_ref)
print(selected_fc)
```

## DB Collections

- `method='genetic_programming'` -> `factors.genetic_programming`
- `method='llm_prompt'` -> `factors.llm_prompt`

The `save_fc()` return value is a DB reference:

- `database.collection@version`
- Example: `factors.llm_prompt@20260322_143000`

## Leakage Check

- Before DB persistence, selected factors go through leakage checking.
- A random subset of time points is checked (`check_leakage_count`).
- If failed, detailed mismatch rows/examples are logged and failed factors are excluded.

## Notes

- `config_path` is kept in return payload for backward compatibility and equals `config_ref`.
- `save_fc_value()` still supports exporting factor values (parquet/csv/pickle) for analysis only.

