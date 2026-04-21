"""Automatic factor generation utilities (formula-first, DB-first)."""

import importlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data import (
    get_factor_formula_records,
    get_factor_formula_map_by_version,
    get_futures_continuous_contract_price,
    update_factor_info,
)
from mongo.mongify import get_data
from utils.logging import log
from utils.params import (
    FusionSupportedMethods,
    FusionSupportedMetrics,
    GP_DEFAULT_FITNESS_INDICATOR_WEIGHT,
    GP_SUPPORTED_INDICATOR,
)

from .backtest import BackTester
from .factor_indicators import (
    get_annualized_ret,
    get_annualized_sharpe,
    get_annualized_ts_ic_and_t_corr,
    get_annualized_volatility,
    get_ts_ret_and_turnover,
)
from .factor_ops import available_operator_prompt_text, calc_formula_df, calc_formula_series
from .factor_utils import (
    check_if_leakage as check_if_leakage_util,
    filter_fc_by_db_relative_spearman as filter_fc_by_db_relative_spearman_util,
    get_future_ret,
    get_weighted_price,
)
from .gp_factor_engine import GPCandidate, run_gp_evolution


class FactorGenerator:
    """Generate factors, backtest, filter and persist factor formulas into DB."""

    method: str = 'base'

    def __init__(self,
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List[str]] = 'C0',
                 fc_freq: str = '1d',
                 data: Optional[pd.DataFrame] = None,
                 start_time: Optional[str] = '20200101',
                 end_time: Optional[str] = '20241231',
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'simple',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
                 apply_weighted_price: bool = True,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 50,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 5.0,
                 check_leakage_count: int = 20,
                 check_relative: bool = True,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 version: Optional[str] = '20260407_gp_test_1'):
        self.instrument_type = instrument_type
        self.instrument_id_list = [instrument_id_list] if isinstance(instrument_id_list, str) else list(instrument_id_list)
        self.fc_freq = fc_freq
        self.data = data
        self.start_time = start_time
        self.end_time = end_time
        self.portfolio_adjust_method = portfolio_adjust_method
        self.interest_method = interest_method
        self.risk_free_rate = risk_free_rate
        self.calculate_baseline = calculate_baseline
        self.apply_weighted_price = bool(apply_weighted_price)
        self.n_jobs = n_jobs

        self.base_col_list = list(base_col_list) if base_col_list else ['open', 'high', 'low', 'close', 'volume', 'position']
        self.min_window_size = min_window_size
        self.max_factor_count = max_factor_count
        self.apply_rolling_norm = apply_rolling_norm
        self.rolling_norm_window = rolling_norm_window
        self.rolling_norm_min_periods = rolling_norm_min_periods
        self.rolling_norm_eps = rolling_norm_eps
        self.rolling_norm_clip = rolling_norm_clip
        self.check_leakage_count = int(check_leakage_count)
        self.check_relative = bool(check_relative)
        self.relative_threshold = float(relative_threshold)
        self.relative_check_version_list = None if relative_check_version_list is None else list(relative_check_version_list)

        self.version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        self.generated_data: Optional[pd.DataFrame] = None
        self.generated_fc_name_list: List[str] = []
        self.bt: Optional[BackTester] = None

        self.factor_formula_map: Dict[str, str] = {}
        self.factor_fitness_map: Dict[str, Dict[str, Dict[str, float]]] = {}

        assert self.fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, got {self.fc_freq}.'
        assert self.portfolio_adjust_method in ['min', '1D', '1M', '1Q'], \
            f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, got {self.portfolio_adjust_method}.'
        assert self.method in ['base', 'llm_prompt', 'genetic_programming'], \
            f'Unsupported method: {self.method}.'
        assert 0.0 <= self.relative_threshold <= 1.0, \
            f'relative_threshold should be in [0, 1], got {self.relative_threshold}.'

    def load_base_data(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            df = self.data.copy()
        else:
            if self.instrument_type != 'futures_continuous_contract':
                raise ValueError(f'Unsupported instrument type: {self.instrument_type}.')
            df = get_futures_continuous_contract_price(
                instrument_id=self.instrument_id_list,
                start_date=self.start_time,
                end_date=self.end_time,
                from_database=True,
            )

        optional_cols = [c for c in ['weighted_factor', 'cur_weighted_factor', 'is_rollover', 'symbol'] if c in df.columns]
        required_cols = ['time', 'instrument_id'] + self.base_col_list + optional_cols
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f'Input data does not contain required column: {col}')

        df = df[required_cols].copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        if self.apply_weighted_price:
            if 'weighted_factor' not in df.columns:
                raise ValueError(
                    'apply_weighted_price=True requires `weighted_factor` in source data. '
                    'Set apply_weighted_price=False to use raw prices.'
                )
            df = get_weighted_price(df)

        available_instruments = df['instrument_id'].dropna().unique().tolist()
        invalid_instruments = [x for x in self.instrument_id_list if x not in available_instruments]
        if invalid_instruments:
            raise ValueError(
                f'Invalid instrument_id_list: {invalid_instruments}. Available instruments: {available_instruments}'
            )
        return df

    @staticmethod
    def auto_load_project_env() -> bool:
        candidate_paths = [
            Path(__file__).resolve().parents[1] / '.env',
            Path(__file__).resolve().parents[2] / '.env',
        ]
        for env_path in candidate_paths:
            if not env_path.exists():
                continue
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
                return True
            except Exception:
                try:
                    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                    return True
                except Exception:
                    continue
        return False

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError('Please implement generate_factor_df in subclasses.')

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError('Please use a concrete subclass, e.g. LLMPromptFactorGenerator or GeneticFactorGenerator.')

    def _finalize_generated_data(self,
                                 base_df: pd.DataFrame,
                                 factor_df: pd.DataFrame) -> pd.DataFrame:
        raw_cols = ['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position']
        optional_cols = [c for c in ['weighted_factor', 'cur_weighted_factor', 'is_rollover', 'symbol'] if c in base_df.columns]
        df_with_ret = get_future_ret(
            base_df[raw_cols + optional_cols].copy(),
            portfolio_adjust_method=self.portfolio_adjust_method,
            rfr=self.risk_free_rate,
        )
        df_with_ret = df_with_ret[['time', 'instrument_id', 'future_ret'] + optional_cols].copy()

        generated_data = df_with_ret.merge(factor_df, on=['time', 'instrument_id'], how='left', validate='1:1')
        generated_data = generated_data.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        self.generated_fc_name_list = [
            c for c in generated_data.columns
            if c not in ['time', 'instrument_id', 'future_ret', 'weighted_factor', 'cur_weighted_factor', 'is_rollover', 'symbol']
        ]
        self.generated_data = generated_data
        log.info(f'Generated {len(self.generated_fc_name_list)} factors by method={self.method}.')
        return generated_data

    def generate_with_fc(self,
                         fc_name_list: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]
        if not fc_name_list:
            raise ValueError('fc_name_list is empty.')
        return self.generate(selected_fc_name_list=list(fc_name_list))

    def save_fc_value(self,
                      fc_name_list: Union[str, List[str]],
                      file_name: Optional[str] = None,
                      file_format: str = 'parquet') -> Path:
        if self.generated_data is None:
            raise ValueError('Please call generate() before save_fc_value().')

        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]
        generated_data = cast(pd.DataFrame, self.generated_data)
        missing_cols = [c for c in fc_name_list if c not in generated_data.columns]
        if missing_cols:
            raise ValueError(f'Factor columns not found in generated data: {missing_cols}')

        save_dir = Path(__file__).resolve().parents[1] / 'data' / 'factor_value'
        save_dir.mkdir(parents=True, exist_ok=True)

        if not file_name:
            instrument_text = '-'.join(self.instrument_id_list)
            file_name = f'{self.method}_{instrument_text}_{self.fc_freq}_{self.start_time}_{self.end_time}'

        file_format = file_format.lower()
        save_path = save_dir / f'{file_name}.{file_format}'

        output_df = generated_data[['time', 'instrument_id', 'future_ret'] + list(fc_name_list)].copy()
        if file_format == 'parquet':
            output_df.to_parquet(save_path, index=False)
        elif file_format == 'csv':
            output_df.to_csv(save_path, index=False)
        elif file_format in ['pkl', 'pickle']:
            output_df.to_pickle(save_path)
        else:
            raise ValueError(f'Unsupported file_format: {file_format}. Use parquet/csv/pickle.')

        log.info(f'Saved factor values to {save_path}.')
        return save_path

    @classmethod
    def validate_filter_indicator_dict(cls,
                                       filter_indicator_dict: Dict[str, Tuple[Optional[float], Optional[float], int]]) -> None:
        if not filter_indicator_dict:
            raise ValueError('filter_indicator_dict is required and cannot be empty.')
        available_indicator_list = list(GP_SUPPORTED_INDICATOR)
        invalid_indicator_list = [k for k in filter_indicator_dict.keys() if k not in available_indicator_list]
        if invalid_indicator_list:
            raise ValueError(
                f'Unsupported filter indicators: {invalid_indicator_list}. '
                f'Available indicators: {available_indicator_list}'
            )

        for indicator, conf in filter_indicator_dict.items():
            if not isinstance(conf, tuple) or len(conf) != 3:
                raise ValueError(
                    f'Invalid filter config for `{indicator}`: {conf}. '
                    'Expected tuple(mean_threshold, yearly_threshold, direction).'
                )
            mean_threshold, yearly_threshold, direction = conf
            if mean_threshold is not None:
                float(mean_threshold)
            if yearly_threshold is not None:
                float(yearly_threshold)
            if direction not in [1, -1]:
                raise ValueError(f'Invalid direction for `{indicator}`: {direction}. Use 1 (>=) or -1 (<=).')

    def filter_fc_by_threshold(self,
                               performance_summary: Optional[pd.DataFrame] = None,
                               filter_indicator_dict: Dict[str, Tuple[Optional[float], Optional[float], int]] = None,
                               require_all_row: bool = True,
                               require_all_instruments: bool = True) -> List[str]:
        if performance_summary is None:
            if self.bt is None or self.bt.performance_summary is None:
                raise ValueError('No performance summary available. Please run backtest() first.')
            summary_df = self.bt.performance_summary.copy()
        else:
            summary_df = performance_summary.copy()

        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        required_cols = ['year', 'Factor Name']
        for col in required_cols:
            if col not in summary_df.columns:
                raise ValueError(f'performance_summary does not contain required column: {col}')

        if not filter_indicator_dict:
            raise ValueError('filter_indicator_dict is required and cannot be empty.')

        active_indicator_dict = {
            k: v for k, v in filter_indicator_dict.items()
            if v[0] is not None or v[1] is not None
        }
        if not active_indicator_dict:
            return summary_df['Factor Name'].dropna().astype(str).unique().tolist()

        indicator_list = list(active_indicator_dict.keys())
        for indicator in indicator_list:
            if indicator not in summary_df.columns:
                raise ValueError(f'performance_summary does not contain required indicator: {indicator}')

        summary_df = summary_df.copy()
        summary_df['__year_str__'] = summary_df['year'].astype(str)
        summary_df['__is_yearly__'] = summary_df['__year_str__'] != 'all'

        for indicator, conf in active_indicator_dict.items():
            _, yearly_threshold, direction = conf
            numeric_series = pd.to_numeric(summary_df[indicator], errors='coerce')
            pass_col = f'__pass_yearly__{indicator}'
            if yearly_threshold is None:
                summary_df[pass_col] = True
            elif direction == 1:
                summary_df[pass_col] = (~summary_df['__is_yearly__']) | (numeric_series >= float(yearly_threshold))
            else:
                summary_df[pass_col] = (~summary_df['__is_yearly__']) | (numeric_series <= float(yearly_threshold))

        group_cols = ['Factor Name'] + (['Instrument ID'] if 'Instrument ID' in summary_df.columns else [])

        def _eval_group(df_group: pd.DataFrame) -> bool:
            if require_all_row and 'all' not in df_group['__year_str__'].values:
                return False
            df_year = df_group.loc[df_group['__is_yearly__']].copy()
            if df_year.empty:
                return False

            for indicator, conf in active_indicator_dict.items():
                mean_threshold, _, direction = conf
                yearly_values = pd.to_numeric(df_year[indicator], errors='coerce').dropna()
                if mean_threshold is not None:
                    if yearly_values.empty:
                        return False
                    yearly_mean = float(yearly_values.mean())
                    if direction == 1 and not (yearly_mean >= float(mean_threshold)):
                        return False
                    if direction == -1 and not (yearly_mean <= float(mean_threshold)):
                        return False

                pass_col = f'__pass_yearly__{indicator}'
                if not bool(df_group.loc[df_group['__is_yearly__'], pass_col].all()):
                    return False
            return True

        ins_pass_records = []
        for group_key, group_df in summary_df.groupby(group_cols, sort=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            record = {k: v for k, v in zip(group_cols, group_key)}
            record['__ins_pass__'] = _eval_group(group_df)
            ins_pass_records.append(record)
        ins_pass_df = pd.DataFrame(ins_pass_records)

        if 'Instrument ID' in ins_pass_df.columns:
            fc_pass_series = ins_pass_df.groupby('Factor Name', sort=False)['__ins_pass__'].all() \
                if require_all_instruments else \
                ins_pass_df.groupby('Factor Name', sort=False)['__ins_pass__'].any()
        else:
            fc_pass_series = ins_pass_df.set_index('Factor Name')['__ins_pass__']

        return fc_pass_series[fc_pass_series].index.tolist()

    def summarize_best_failed_indicator_metrics(self,
                                                performance_summary: pd.DataFrame,
                                                filter_indicator_dict: Dict[str, Tuple[Optional[float], Optional[float], int]],
                                                selected_fc_name_list: Sequence[str],
                                                require_all_row: bool = True,
                                                require_all_instruments: bool = True) -> Dict[str, Dict[str, Any]]:
        """Summarize best indicator values among factors that did not pass threshold filter.

        Metric computation follows the same yearly-mean and yearly-threshold logic as filter_fc_by_threshold.
        """
        summary_df = performance_summary.copy()
        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        summary_df['__year_str__'] = summary_df['year'].astype(str)
        summary_df['__is_yearly__'] = summary_df['__year_str__'] != 'all'
        active_indicator_dict = {
            k: v for k, v in filter_indicator_dict.items()
            if v[0] is not None or v[1] is not None
        }
        if not active_indicator_dict:
            return {}

        for indicator, conf in active_indicator_dict.items():
            _, yearly_threshold, direction = conf
            numeric_series = pd.to_numeric(summary_df[indicator], errors='coerce')
            pass_col = f'__pass_yearly__{indicator}'
            if yearly_threshold is None:
                summary_df[pass_col] = True
            elif direction == 1:
                summary_df[pass_col] = (~summary_df['__is_yearly__']) | (numeric_series >= float(yearly_threshold))
            else:
                summary_df[pass_col] = (~summary_df['__is_yearly__']) | (numeric_series <= float(yearly_threshold))

        group_cols = ['Factor Name'] + (['Instrument ID'] if 'Instrument ID' in summary_df.columns else [])
        per_group_records: List[Dict[str, Any]] = []
        for group_key, group_df in summary_df.groupby(group_cols, sort=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            record: Dict[str, Any] = {k: v for k, v in zip(group_cols, group_key)}
            if require_all_row and 'all' not in group_df['__year_str__'].values:
                record['__group_valid__'] = False
            else:
                record['__group_valid__'] = True

            df_year = group_df.loc[group_df['__is_yearly__']].copy()
            for indicator, _ in active_indicator_dict.items():
                yearly_values = pd.to_numeric(df_year[indicator], errors='coerce').dropna()
                record[f'__mean__{indicator}'] = float(yearly_values.mean()) if not yearly_values.empty else np.nan
                pass_col = f'__pass_yearly__{indicator}'
                yearly_pass = bool(group_df.loc[group_df['__is_yearly__'], pass_col].all()) if not df_year.empty else False
                record[f'__yearly_pass__{indicator}'] = bool(record['__group_valid__'] and yearly_pass)
            per_group_records.append(record)

        if not per_group_records:
            return {}
        per_group_df = pd.DataFrame(per_group_records)

        selected_set = set(selected_fc_name_list)
        failed_factors = [x for x in per_group_df['Factor Name'].dropna().unique().tolist() if x not in selected_set]
        if not failed_factors:
            return {}

        best_metric_map: Dict[str, Dict[str, Any]] = {}
        for indicator, conf in active_indicator_dict.items():
            mean_threshold, _, direction = conf
            metric_col = f'__mean__{indicator}'
            pass_col = f'__yearly_pass__{indicator}'

            rows: List[Dict[str, Any]] = []
            for fc_name, df_fc in per_group_df[per_group_df['Factor Name'].isin(failed_factors)].groupby('Factor Name', sort=False):
                vals = pd.to_numeric(df_fc[metric_col], errors='coerce').dropna()
                if vals.empty:
                    continue
                if require_all_instruments:
                    agg_mean = float(vals.min()) if direction == 1 else float(vals.max())
                    agg_yearly_pass = bool(df_fc[pass_col].all())
                    agg_group_valid = bool(df_fc['__group_valid__'].all()) if '__group_valid__' in df_fc.columns else True
                else:
                    agg_mean = float(vals.max()) if direction == 1 else float(vals.min())
                    agg_yearly_pass = bool(df_fc[pass_col].any())
                    agg_group_valid = bool(df_fc['__group_valid__'].any()) if '__group_valid__' in df_fc.columns else True

                rows.append({
                    'factor_name': str(fc_name),
                    'yearly_mean': agg_mean,
                    'yearly_pass': agg_yearly_pass,
                    'group_valid': agg_group_valid,
                })

            if not rows:
                continue
            rows_df = pd.DataFrame(rows)
            rows_df = rows_df.sort_values('yearly_mean', ascending=(direction == -1)).reset_index(drop=True)
            best_row = rows_df.iloc[0]
            yearly_mean = float(best_row['yearly_mean'])

            df_factor = summary_df.loc[
                (summary_df['Factor Name'] == str(best_row['factor_name'])) & summary_df['__is_yearly__']
            ].copy()
            yearly_detail: List[Dict[str, Any]] = []
            if not df_factor.empty:
                if 'Instrument ID' in df_factor.columns:
                    for year_text, df_year in df_factor.groupby('__year_str__', sort=True):
                        vals = pd.to_numeric(df_year[indicator], errors='coerce').dropna()
                        if vals.empty:
                            continue
                        if require_all_instruments:
                            year_value = float(vals.min()) if direction == 1 else float(vals.max())
                        else:
                            year_value = float(vals.max()) if direction == 1 else float(vals.min())
                        year_pass = True if conf[1] is None else (
                            (year_value >= float(conf[1])) if direction == 1 else (year_value <= float(conf[1]))
                        )
                        yearly_detail.append({
                            'year': str(year_text),
                            'value': year_value,
                            'threshold': None if conf[1] is None else float(conf[1]),
                            'pass': bool(year_pass),
                        })
                else:
                    for year_text, df_year in df_factor.groupby('__year_str__', sort=True):
                        vals = pd.to_numeric(df_year[indicator], errors='coerce').dropna()
                        if vals.empty:
                            continue
                        year_value = float(vals.mean())
                        year_pass = True if conf[1] is None else (
                            (year_value >= float(conf[1])) if direction == 1 else (year_value <= float(conf[1]))
                        )
                        yearly_detail.append({
                            'year': str(year_text),
                            'value': year_value,
                            'threshold': None if conf[1] is None else float(conf[1]),
                            'pass': bool(year_pass),
                        })

            mean_pass = True if mean_threshold is None else (
                (yearly_mean >= float(mean_threshold)) if direction == 1 else (yearly_mean <= float(mean_threshold))
            )
            yearly_all_pass = bool(all(item['pass'] for item in yearly_detail)) if yearly_detail else False

            # Compute whole-filter pass/fail for this factor across all indicators.
            df_factor_group = per_group_df.loc[per_group_df['Factor Name'] == str(best_row['factor_name'])].copy()
            failed_indicators: List[str] = []
            for ind_all, conf_all in active_indicator_dict.items():
                mean_thr_all, _, direction_all = conf_all
                metric_col_all = f'__mean__{ind_all}'
                pass_col_all = f'__yearly_pass__{ind_all}'
                vals_all = pd.to_numeric(df_factor_group[metric_col_all], errors='coerce').dropna()
                if vals_all.empty:
                    failed_indicators.append(ind_all)
                    continue

                if require_all_instruments:
                    agg_mean_all = float(vals_all.min()) if direction_all == 1 else float(vals_all.max())
                    agg_yearly_pass_all = bool(df_factor_group[pass_col_all].all())
                    agg_group_valid_all = bool(df_factor_group['__group_valid__'].all())
                else:
                    agg_mean_all = float(vals_all.max()) if direction_all == 1 else float(vals_all.min())
                    agg_yearly_pass_all = bool(df_factor_group[pass_col_all].any())
                    agg_group_valid_all = bool(df_factor_group['__group_valid__'].any())

                mean_pass_all = True if mean_thr_all is None else (
                    (agg_mean_all >= float(mean_thr_all)) if direction_all == 1 else (agg_mean_all <= float(mean_thr_all))
                )
                if not (mean_pass_all and agg_yearly_pass_all and agg_group_valid_all):
                    failed_indicators.append(ind_all)
            overall_pass = len(failed_indicators) == 0

            failed_checks: List[str] = []
            if not mean_pass:
                failed_checks.append('mean_threshold_not_passed')
            if not yearly_all_pass:
                failed_checks.append('yearly_threshold_not_passed')
            if not bool(best_row.get('group_valid', True)):
                failed_checks.append('missing_or_invalid_all_row_or_group_constraint')

            best_metric_map[indicator] = {
                'factor_name': str(best_row['factor_name']),
                'yearly_mean': yearly_mean,
                'mean_threshold': None if mean_threshold is None else float(mean_threshold),
                'yearly_threshold': None if conf[1] is None else float(conf[1]),
                'direction': int(direction),
                f'gap_to_{indicator}_mean_threshold': 0.0 if mean_threshold is None else float(yearly_mean - float(mean_threshold)),
                'yearly_pass': bool(best_row['yearly_pass']),
                'group_valid': bool(best_row.get('group_valid', True)),
                'mean_pass': bool(mean_pass),
                'yearly_all_pass': bool(yearly_all_pass),
                'yearly_detail': yearly_detail,
                f'{indicator}_failed_checks': failed_checks,
                'overall_pass': bool(overall_pass),
                'failed_indicators': failed_indicators,
                'require_all_instruments': bool(require_all_instruments),
                'require_all_row': bool(require_all_row),
            }

        return best_metric_map

    def summarize_selected_indicator_metrics(self,
                                             performance_summary: pd.DataFrame,
                                             filter_indicator_dict: Dict[str, Tuple[Optional[float], Optional[float], int]],
                                             selected_fc_name_list: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Build per-factor indicator summary for logging before DB persistence."""
        if not selected_fc_name_list:
            return {}

        summary_df = performance_summary.copy()
        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()
        summary_df['__year_str__'] = summary_df['year'].astype(str)
        summary_df = summary_df.loc[summary_df['__year_str__'] != 'all'].copy()
        if summary_df.empty:
            return {}

        has_instrument = 'Instrument ID' in summary_df.columns
        out: Dict[str, Dict[str, Any]] = {}
        for fc_name in selected_fc_name_list:
            df_fc = summary_df.loc[summary_df['Factor Name'] == fc_name].copy()
            if df_fc.empty:
                continue

            indicator_payload: Dict[str, Any] = {}
            for indicator, conf in filter_indicator_dict.items():
                mean_threshold, yearly_threshold, direction = conf
                vals = pd.to_numeric(df_fc[indicator], errors='coerce').dropna()
                if vals.empty:
                    continue

                yearly_mean = float(vals.mean())
                yearly_worst = float(vals.min()) if int(direction) == 1 else float(vals.max())
                payload: Dict[str, Any] = {
                    'yearly_mean': yearly_mean,
                    'yearly_worst': yearly_worst,
                    'mean_threshold': None if mean_threshold is None else float(mean_threshold),
                    'yearly_threshold': None if yearly_threshold is None else float(yearly_threshold),
                    'direction': int(direction),
                }

                if has_instrument:
                    instrument_detail: Dict[str, Dict[str, float]] = {}
                    for ins_id, df_ins in df_fc.groupby('Instrument ID', sort=False):
                        ins_vals = pd.to_numeric(df_ins[indicator], errors='coerce').dropna()
                        if ins_vals.empty:
                            continue
                        ins_mean = float(ins_vals.mean())
                        ins_worst = float(ins_vals.min()) if int(direction) == 1 else float(ins_vals.max())
                        instrument_detail[str(ins_id)] = {
                            'yearly_mean': ins_mean,
                            'yearly_worst': ins_worst,
                        }
                    if instrument_detail:
                        payload['instrument_detail'] = instrument_detail

                indicator_payload[indicator] = payload

            out[str(fc_name)] = indicator_payload
        return out

    @staticmethod
    def _format_best_failed_indicator_metrics_log(best_failed_indicator_metrics: Dict[str, Dict[str, Any]],
                                                   factor_formula_map: Optional[Dict[str, str]] = None) -> str:
        if not best_failed_indicator_metrics:
            return 'Best failed indicator metrics: none.'

        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return float('nan')

        indicator_to_factor = {
            indicator: str(info.get('factor_name'))
            for indicator, info in best_failed_indicator_metrics.items()
        }
        unique_factors = sorted({x for x in indicator_to_factor.values() if x and x != 'None'})

        lines = [
            'Best failed indicator metrics (readable, indicator-wise best candidates):',
            '  Note: `gap_to_mean_threshold` = yearly_mean - mean_threshold. '
            'It can be positive while overall check still fails if any year violates yearly_threshold.',
            '  Note: each indicator independently selects its own best failed factor; '
            'factor names can be different across indicators.',
            f'  Indicator -> factor mapping: {indicator_to_factor}',
            f'  Unique factors in this diagnostic: {unique_factors}',
        ]
        if factor_formula_map and unique_factors:
            for uf in unique_factors:
                formula = factor_formula_map.get(uf)
                if formula:
                    lines.append(f'  Factor formula: {uf} = {formula}')
        for indicator, info in best_failed_indicator_metrics.items():
            lines.append(f'  - Indicator: {indicator}')
            lines.append(f'    factor_name: {info.get("factor_name")}')
            lines.append(
                '    mean_check: '
                f'value={_to_float(info.get("yearly_mean")):.6f}, '
                f'threshold={_to_float(info.get("mean_threshold")):.6f}, '
                f'pass={info.get("mean_pass")}'
            )
            lines.append(
                '    yearly_check: '
                f'threshold={_to_float(info.get("yearly_threshold")):.6f}, '
                f'all_pass={info.get("yearly_all_pass")}, '
                f'group_valid={info.get("group_valid")}'
            )
            for y in info.get('yearly_detail', []):
                lines.append(
                    f'      year={y.get("year")}: value={_to_float(y.get("value")):.6f}, '
                    f'threshold={_to_float(y.get("threshold")):.6f}, pass={y.get("pass")}'
                )
            failed_key = f'{indicator}_failed_checks'
            gap_key = f'gap_to_{indicator}_mean_threshold'
            lines.append(f'    {failed_key}: {info.get(failed_key, [])}')
            lines.append(f'    {gap_key}: {_to_float(info.get(gap_key, 0.0)):.6f}')
            lines.append(
                '    overall_filter: '
                f'overall_pass={info.get("overall_pass")}, '
                f'failed_indicators={info.get("failed_indicators", [])}'
            )
        return '\n'.join(lines)

    @staticmethod
    def _format_selected_indicator_metrics_log(fc_name: str,
                                               metrics: Dict[str, Any]) -> str:
        if not metrics:
            return f'[PersistFactorMetrics] factor={fc_name} (no indicator metrics found).'

        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return float('nan')

        lines = [f'[PersistFactorMetrics] factor={fc_name}']
        for indicator, info in metrics.items():
            lines.append(
                f'  - {indicator}: mean={_to_float(info.get("yearly_mean", 0.0)):.6f}, '
                f'mean_threshold={_to_float(info.get("mean_threshold", 0.0)):.6f}, '
                f'worst_year={_to_float(info.get("yearly_worst", 0.0)):.6f}, '
                f'yearly_threshold={_to_float(info.get("yearly_threshold", 0.0)):.6f}, '
                f'direction={int(info.get("direction", 1))}'
            )
        return '\n'.join(lines)

    def _build_metric_fitness_map(self,
                                  performance_summary: pd.DataFrame,
                                  selected_fc_name_list: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        summary_df = performance_summary.copy()
        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        metric_cols = [
            c for c in GP_SUPPORTED_INDICATOR
            if c in summary_df.columns
        ]

        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for fc_name in selected_fc_name_list:
            df_fc = summary_df.loc[summary_df['Factor Name'] == fc_name].copy()
            df_year = df_fc.loc[df_fc['year'].astype(str) != 'all'].copy()
            metric_map: Dict[str, Dict[str, float]] = {}
            for m in metric_cols:
                vals = pd.to_numeric(df_year[m], errors='coerce').dropna()
                if vals.empty:
                    continue
                metric_map[m] = {'value': float(vals.mean())}
            out[fc_name] = metric_map
        return out

    def _db_collection(self) -> str:
        if self.method == 'genetic_programming':
            return 'genetic_programming'
        if self.method == 'llm_prompt':
            return 'llm_prompt'
        raise ValueError(f'Unsupported method for DB persistence: {self.method}')

    def save_fc(self,
                fc_name_list: Union[str, List[str]],
                performance_summary: Optional[pd.DataFrame] = None) -> str:
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]
        if not fc_name_list:
            raise ValueError('fc_name_list is empty.')

        formula_map = getattr(self, 'factor_formula_map', {}) or {}
        missing_formula = [x for x in fc_name_list if x not in formula_map]
        if missing_formula:
            raise ValueError(f'Missing formula for factors: {missing_formula}')

        if performance_summary is None:
            if self.bt is None or self.bt.performance_summary is None:
                raise ValueError('performance_summary is required to save factor metadata.')
            performance_summary = self.bt.performance_summary

        fitness_map = getattr(self, 'factor_fitness_map', {}) or {}
        if not fitness_map:
            fitness_map = self._build_metric_fitness_map(performance_summary, fc_name_list)

        update_factor_info(
            selected_fc_name_list=fc_name_list,
            performance_summary=performance_summary,
            factor_formula_map=formula_map,
            factor_fitness_map=fitness_map,
            instrument_id_list=self.instrument_id_list,
            method=self.method,
            version=self.version,
            start_date=self.start_time,
            end_date=self.end_time,
            database='factors',
            collection=self._db_collection(),
        )

        db_ref = f'factors.{self._db_collection()}@{self.version}'
        log.info(f'Saved selected factors into DB: {db_ref}, factor_count={len(fc_name_list)}')
        return db_ref

    @staticmethod
    def _parse_db_ref(config_ref: str) -> Tuple[str, str, str]:
        if not isinstance(config_ref, str) or '@' not in config_ref or '.' not in config_ref.split('@', 1)[0]:
            raise ValueError(
                'Invalid config_ref format. Expected `database.collection@version`, '
                f'got: {config_ref}'
            )
        left, version = config_ref.split('@', 1)
        database, collection = left.split('.', 1)
        database = database.strip()
        collection = collection.strip()
        version = version.strip()
        if not database or not collection or not version:
            raise ValueError(
                'Invalid config_ref format. Expected non-empty `database.collection@version`, '
                f'got: {config_ref}'
            )
        return database, collection, version

    @classmethod
    def load_fc(cls,
                config_ref: str,
                instrument_id_list: Optional[Sequence[str]] = None) -> List[str]:
        database, collection, version = cls._parse_db_ref(config_ref)
        mongo_operator: Dict[str, Any] = {'version': version}
        if instrument_id_list:
            mongo_operator['instrument_id'] = {'$in': list(instrument_id_list)}

        df = get_data(database=database, collection=collection, mongo_operator=mongo_operator)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f'No factors found in DB for config_ref={config_ref}.')
        if 'factor_name' not in df.columns:
            raise ValueError(f'Invalid DB record: `factor_name` not found for config_ref={config_ref}.')

        fc_name_list = [str(x) for x in df['factor_name'].dropna().tolist()]
        dedup_fc_name_list = list(dict.fromkeys(fc_name_list))
        if not dedup_fc_name_list:
            raise ValueError(f'No valid factor_name found in DB for config_ref={config_ref}.')
        return dedup_fc_name_list

    def backtest_from_fc_config(self,
                                config_ref: str,
                                n_jobs: Optional[int] = None) -> BackTester:
        _, _, config_version = self._parse_db_ref(config_ref)
        fc_name_list = self.load_fc(config_ref=config_ref, instrument_id_list=self.instrument_id_list)
        # Ensure replay computes formulas from the exact saved version.
        self.version = config_version
        generated_df = self.generate_with_fc(fc_name_list=fc_name_list)
        return self.backtest(data=generated_df, fc_name_list=fc_name_list, n_jobs=n_jobs)

    def check_if_leakage(self,
                         fc_name_list: Optional[Union[str, List[str]]] = None,
                         atol: float = 1e-10,
                         rtol: float = 1e-8,
                         raise_error: bool = True) -> dict:
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]

        base_df = self.load_base_data()
        if fc_name_list:
            selected_fc_name_list = list(fc_name_list)
        else:
            factor_df_for_names = self.generate_factor_df(base_df)
            selected_fc_name_list = [c for c in factor_df_for_names.columns if c not in ['time', 'instrument_id']]

        return check_if_leakage_util(
            selected_fc_name_list=selected_fc_name_list,
            load_base_data_fn=self.load_base_data,
            generate_factor_df_fn=self.generate_factor_df,
            check_leakage_count=self.check_leakage_count,
            atol=atol,
            rtol=rtol,
            raise_error=raise_error,
        )

    def auto_mine_select_and_save_fc(self,
                                     filter_indicator_dict: Dict[str, Tuple[Optional[float], Optional[float], int]],
                                     n_jobs: Optional[int] = None,
                                     require_all_row: bool = True,
                                     require_all_instruments: bool = True) -> dict:
        self.validate_filter_indicator_dict(filter_indicator_dict)

        generated_df = self.generate()
        bt = self.backtest(data=generated_df, fc_name_list=self.generated_fc_name_list, n_jobs=n_jobs)
        selected_fc_name_list = self.filter_fc_by_threshold(
            performance_summary=bt.performance_summary,
            filter_indicator_dict=filter_indicator_dict,
            require_all_row=require_all_row,
            require_all_instruments=require_all_instruments,
        )

        if not selected_fc_name_list:
            msg = f'No factors passed thresholds: filter_indicator_dict={filter_indicator_dict}.'
            best_failed_indicator_metrics = self.summarize_best_failed_indicator_metrics(
                performance_summary=bt.performance_summary,
                filter_indicator_dict=filter_indicator_dict,
                selected_fc_name_list=selected_fc_name_list,
                require_all_row=require_all_row,
                require_all_instruments=require_all_instruments,
            )
            log.warning(msg)
            if best_failed_indicator_metrics:
                log.warning(self._format_best_failed_indicator_metrics_log(
                    best_failed_indicator_metrics,
                    factor_formula_map=getattr(self, 'factor_formula_map', None),
                ))
            return {
                'config_ref': None,
                'config_path': None,
                'selected_fc_name_list': [],
                'bt': bt,
                'message': msg,
                'best_failed_indicator_metrics': best_failed_indicator_metrics,
            }
        log.info(f'{len(selected_fc_name_list)} factors passed threshold filtering: {selected_fc_name_list}')
        leakage_result = self.check_if_leakage(fc_name_list=selected_fc_name_list, raise_error=False)
        selected_after_leakage = [
            x for x in selected_fc_name_list
            if x not in leakage_result.get('failed_factor_list', [])
        ]
        if not selected_after_leakage:
            msg = 'No factors passed leakage check after threshold filtering.'
            log.warning(msg)
            return {
                'config_ref': None,
                'config_path': None,
                'selected_fc_name_list': [],
                'bt': bt,
                'leakage_check': leakage_result,
                'message': msg,
            }

        relative_result: Optional[Dict[str, Any]] = None
        selected_after_relative = list(selected_after_leakage)
        log.info(f'{len(selected_after_leakage)} factors passed leakage check: {selected_after_leakage}')
        if self.check_relative:
            relative_result = self.filter_fc_by_db_relative_spearman(
                selected_fc_name_list=selected_after_leakage,
                generated_df=generated_df,
                n_jobs=n_jobs,
            )
            selected_after_relative = relative_result.get('selected_fc_name_list', [])

            if not selected_after_relative:
                msg = (
                    'No factors passed relative correlation check: '
                    f'threshold={self.relative_threshold}, '
                    f'versions={self.relative_check_version_list}.'
                )
                log.warning(msg)
                return {
                    'config_ref': None,
                    'config_path': None,
                    'selected_fc_name_list': [],
                    'bt': bt,
                    'leakage_check': leakage_result,
                    'relative_check': relative_result,
                    'message': msg,
                }
            log.info(f'{len(selected_after_relative)} factors passed relative correlation check.')
        config_ref = self.save_fc(
            fc_name_list=selected_after_relative,
            performance_summary=bt.performance_summary,
        )

        selected_indicator_metrics = self.summarize_selected_indicator_metrics(
            performance_summary=bt.performance_summary,
            filter_indicator_dict=filter_indicator_dict,
            selected_fc_name_list=selected_after_relative,
        )
        for fc_name in selected_after_relative:
            log.info(self._format_selected_indicator_metrics_log(
                fc_name=fc_name,
                metrics=selected_indicator_metrics.get(fc_name, {}),
            ))

        return {
            'config_ref': config_ref,
            'config_path': config_ref,
            'selected_fc_name_list': selected_after_relative,
            'bt': bt,
            'leakage_check': leakage_result,
            'relative_check': relative_result,
            'selected_indicator_metrics': selected_indicator_metrics,
        }

    def filter_fc_by_db_relative_spearman(self,
                                          selected_fc_name_list: Sequence[str],
                                          generated_df: pd.DataFrame,
                                          n_jobs: Optional[int] = None,
                                          batch_size: int = 100) -> Dict[str, Any]:
        return filter_fc_by_db_relative_spearman_util(
            selected_fc_name_list=selected_fc_name_list,
            generated_df=generated_df,
            base_df=self.load_base_data(),
            base_col_list=self.base_col_list,
            relative_threshold=self.relative_threshold,
            relative_check_version_list=self.relative_check_version_list,
            n_jobs=n_jobs or self.n_jobs,
            batch_size=batch_size,
            database='factors',
            collections=None,
            self_compare_version=self.version,
            self_compare_collection=self._db_collection(),
        )

    def backtest(self,
                 data: Optional[pd.DataFrame] = None,
                 fc_name_list: Optional[Union[str, List[str]]] = None,
                 n_jobs: Optional[int] = None) -> BackTester:
        if data is None:
            if self.generated_data is None:
                raise ValueError('Please call generate() first, or pass `data` explicitly.')
            data = self.generated_data

        if fc_name_list is None:
            fc_name_list = self.generated_fc_name_list
        if not fc_name_list:
            raise ValueError('fc_name_list is empty. Please specify factor columns for backtesting.')

        bt = BackTester(
            fc_name_list=fc_name_list,
            version=self.version,
            collection=self.method if self.method in ['genetic_programming', 'llm_prompt', 'fusion_factor'] else 'genetic_programming',
            instrument_type=self.instrument_type,
            instrument_id_list=self.instrument_id_list,
            fc_freq=self.fc_freq,
            data=data,
            start_time=self.start_time,
            end_time=self.end_time,
            portfolio_adjust_method=self.portfolio_adjust_method,
            interest_method=self.interest_method,
            risk_free_rate=self.risk_free_rate,
            calculate_baseline=self.calculate_baseline,
            # FactorGenerator data path already handles price adjustment in load_base_data.
            # Keep BackTester from applying weighted-price preprocessing again.
            # 这样设计是因为generate_factor_df需要使用到复权后的价格进行fitness计算。因此，不能只在backtest时才进行复权。
            apply_weighted_price=False,
            n_jobs=n_jobs or self.n_jobs,
        )
        bt.backtest()
        self.bt = bt
        return bt


class LLMPromptFactorGenerator(FactorGenerator):
    """LLM-based formula generator. LLM outputs formulas, not class code."""

    method: str = 'llm_prompt'

    def __init__(self,
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List[str]] = 'C0',
                 fc_freq: str = '1d',
                 data: Optional[pd.DataFrame] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'simple',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
                 apply_weighted_price: bool = True,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 50,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 5.0,
                 check_leakage_count: int = 20,
                 check_relative: bool = True,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 model_name: str = 'deepseek',
                 llm_temperature: float = 0.7,
                 llm_factor_count: int = 5,
                 llm_early_stopping_round: int = 20,
                 llm_user_requirement: str = '生成期货的日频量价因子',
                 version: Optional[str] = None):
        """LLM提示因子生成器。

        参数说明（中文）：
        - instrument_type: 数据品种类型，目前仅支持期货连续合约。
        - instrument_id_list: 合约列表（如 'C0' 或 ['C0','FG0']）。
        - fc_freq: 因子频率（1m/5m/1d）。
        - data: 直接传入的价格数据（优先使用）。
        - start_time/end_time: 回测区间起止时间。
        - portfolio_adjust_method: 收益对齐周期（min/1D/1M/1Q）。
        - interest_method: 利息计算方式（simple/compound）。
        - risk_free_rate: 是否加入无风险利率。
        - calculate_baseline: 是否计算基准表现。
        - apply_weighted_price: 是否对价格做复权处理。
        - n_jobs: 并行进程/线程数。
        - base_col_list: 可用基础字段列表（默认 OHLCV+position）。
        - min_window_size: 最小窗口长度。
        - max_factor_count: 最终返回的最大因子数量。
        - apply_rolling_norm: 是否对因子做滚动标准化（LLM生成因子默认为False）。
        - rolling_norm_window: 滚动标准化窗口长度。
        - rolling_norm_min_periods: 滚动标准化最小样本数。
        - rolling_norm_eps: 标准化数值稳定项。
        - rolling_norm_clip: 标准化后的截断范围。
        - check_leakage_count: 泄露检查抽样次数。
        - check_relative: 是否与历史库做相关性去重。
        - relative_threshold: 相对相关性阈值（Spearman abs）。
        - relative_check_version_list: 相对检查的版本白名单。
        - model_name: 使用的LLM模型名称（目前仅支持deepseek）。
        - llm_temperature: LLM生成温度，控制随机性。
        - llm_factor_count: 每轮LLM请求生成的因子数量。
        - llm_early_stopping_round: 连续多少轮无新因子则早停。
        - llm_user_requirement: 用户需求描述，用于指导LLM生成因子。
        - version: 本次生成的版本标识。
        """
        super().__init__(
            instrument_type=instrument_type,
            instrument_id_list=instrument_id_list,
            fc_freq=fc_freq,
            data=data,
            start_time=start_time,
            end_time=end_time,
            portfolio_adjust_method=portfolio_adjust_method,
            interest_method=interest_method,
            risk_free_rate=risk_free_rate,
            calculate_baseline=calculate_baseline,
            apply_weighted_price=apply_weighted_price,
            n_jobs=n_jobs,
            base_col_list=base_col_list,
            min_window_size=min_window_size,
            max_factor_count=max_factor_count,
            apply_rolling_norm=False,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            check_leakage_count=check_leakage_count,
            check_relative=check_relative,
            relative_threshold=relative_threshold,
            relative_check_version_list=relative_check_version_list,
            version=version,
        )
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        self.llm_factor_count = llm_factor_count
        self.llm_early_stopping_round = llm_early_stopping_round
        self.llm_user_requirement = llm_user_requirement

    @staticmethod
    def get_llm(temperature: float = 0.7, model_name: Optional[str] = None):
        model_name = model_name or 'deepseek'
        if model_name != 'deepseek':
            raise ValueError(f'Unsupported model_name: {model_name}.')

        FactorGenerator.auto_load_project_env()
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        if not deepseek_api_key:
            raise ValueError('Missing environment variable: DEEPSEEK_API_KEY.')
        try:
            ChatOpenAI = importlib.import_module('langchain_openai').ChatOpenAI
        except Exception as e:
            raise ImportError('Please install langchain-openai to use method=llm_prompt.') from e

        return ChatOpenAI(
            model='deepseek-chat',
            temperature=temperature,
            api_key=deepseek_api_key,
            base_url=deepseek_base_url,
        )

    def _build_llm_formula_prompt(self, factor_count: int, existing_formulas: Optional[List[str]] = None) -> str:
        ops_text = available_operator_prompt_text()
        few_shot = (
            '示例1:\n'
            '{"formula": "TsRank(Div(Sub(close, open), open), 10)", "logic": "价格强弱滚动分位"}\n'
            '示例2:\n'
            '{"formula": "TsCorr(TsPctDelta(close, 1), TsPctDelta(volume, 1), 20)", "logic": "量价共振相关"}\n'
            '示例3:\n'
            '{"formula": "Sub(TsMean(Div(close, open), 5), TsMean(Div(close, open), 20))", "logic": "短长窗口价差"}'
        )
        existing_text = ''
        if existing_formulas:
            existing_text = f"\n已生成公式（请勿重复）：{existing_formulas}"

        return (
            '你是期货量化研究助理。请只输出可执行的因子公式。\n'
            f'{ops_text}\n'
            '输出必须为严格 JSON，格式：'
            '{"factors": [{"formula": "...", "logic": "..."}, ...]}\n'
            '规则：\n'
            '1) 只能使用上述算子和字段 open/high/low/close/volume/position。\n'
            '2) 禁止未来函数，不要构造任何使用未来数据的表达式。\n'
            '3) 窗口参数必须是正整数常量。\n'
            f'4) 必须返回且仅返回 {factor_count} 条公式。\n'
            '5) 不要输出 markdown 代码块，不要输出 JSON 以外文本。\n'
            f'6) 用户需求：{self.llm_user_requirement}\n'
            f'Few-shot:\n{few_shot}{existing_text}'
        )

    @staticmethod
    def _extract_json_text(raw_text: str) -> str:
        raw_text = raw_text.strip()
        if raw_text.startswith('{') and raw_text.endswith('}'):
            return raw_text
        match = re.search(r'\{[\s\S]*}', raw_text)
        if not match:
            raise ValueError('Failed to parse JSON from LLM output.')
        return match.group(0)

    def _validate_llm_formula(self, formula: str, sample_df: pd.DataFrame) -> None:
        _ = calc_formula_series(sample_df, formula=formula, data_fields=self.base_col_list)

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        if selected_fc_names is not None:
            cached_formula_map = getattr(self, 'factor_formula_map', {}) or {}
            missing_in_cache = [x for x in selected_fc_names if x not in cached_formula_map]
            if not missing_in_cache:
                return calc_formula_df(
                    df=df,
                    formula_map={k: cached_formula_map[k] for k in selected_fc_names},
                    data_fields=self.base_col_list,
                )

            formula_map = get_factor_formula_map_by_version(
                fc_name_list=selected_fc_names,
                version=self.version,
                collections=['llm_prompt'],
                database='factors',
            )
            missing = [x for x in selected_fc_names if x not in formula_map]
            if missing:
                raise ValueError(f'LLM formulas not found in DB for factors: {missing}')
            return calc_formula_df(
                df=df,
                formula_map={k: formula_map[k] for k in selected_fc_names},
                data_fields=self.base_col_list,
            )

        llm = self.get_llm(temperature=self.llm_temperature, model_name=self.model_name)
        target_factor_count = self.max_factor_count if self.max_factor_count is not None else self.llm_factor_count
        if target_factor_count <= 0:
            raise ValueError('max_factor_count must be positive for method=llm_prompt.')

        max_rounds = max(3, int(np.ceil(target_factor_count / max(1, self.llm_factor_count))) * 3)
        sample_df = df.groupby('instrument_id', sort=False).head(max(30, self.min_window_size)).copy()
        valid_formula_list: List[str] = []
        no_growth_rounds = 0

        for round_idx in range(max_rounds):
            if len(valid_formula_list) >= target_factor_count:
                break

            remaining = target_factor_count - len(valid_formula_list)
            this_round_count = min(max(1, self.llm_factor_count), remaining)
            prompt = self._build_llm_formula_prompt(this_round_count, existing_formulas=valid_formula_list)
            log.info(
                f'LLM prompt round {round_idx + 1}/{max_rounds}, requesting {this_round_count} formulas, '
                f'{len(valid_formula_list)}/{target_factor_count} valid formulas collected so far.'
            )
            try:
                response = llm.invoke(prompt)
                content = getattr(response, 'content', '')
                payload = json.loads(self._extract_json_text(content))
            except Exception as e:
                log.warning(f'Round {round_idx + 1}/{max_rounds}: invalid LLM response, skip. {e}')
                continue

            factors = payload.get('factors', [])
            if not isinstance(factors, list) or not factors:
                no_growth_rounds += 1
                continue

            prev_count = len(valid_formula_list)
            for item in factors:
                if not isinstance(item, dict):
                    continue
                formula = item.get('formula')
                if not isinstance(formula, str):
                    continue
                formula = formula.strip()
                if not formula or formula in valid_formula_list:
                    continue
                try:
                    self._validate_llm_formula(formula, sample_df)
                except Exception as e:
                    log.warning(f'Skip invalid LLM formula `{formula}`: {e}')
                    continue
                valid_formula_list.append(formula)
                if len(valid_formula_list) >= target_factor_count:
                    break

            if len(valid_formula_list) == prev_count:
                no_growth_rounds += 1
            else:
                no_growth_rounds = 0

            if 0 < self.llm_early_stopping_round <= no_growth_rounds:
                log.warning(
                    f'LLM early stop triggered after {no_growth_rounds} non-growth rounds. '
                    f'valid_formula_count={len(valid_formula_list)} / target={target_factor_count}'
                )
                break

        if not valid_formula_list:
            raise ValueError('No valid formulas generated by LLM prompt.')

        self.factor_formula_map = {
            f'fac_llm_prompt_{idx + 1:04d}': formula
            for idx, formula in enumerate(valid_formula_list)
        }
        self.factor_fitness_map = {}
        return calc_formula_df(df=df, formula_map=self.factor_formula_map, data_fields=self.base_col_list)

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        base_df = self.load_base_data()
        factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        return self._finalize_generated_data(base_df, factor_df)


class GeneticFactorGenerator(FactorGenerator):
    """Factor generator using genetic programming AST evolution."""

    method: str = 'genetic_programming'

    def __init__(self,
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List[str]] = 'C0',
                 fc_freq: str = '1d',
                 data: Optional[pd.DataFrame] = None,
                 start_time: Optional[str] = '20200101',
                 end_time: Optional[str] = '20241231',
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'simple',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
                 apply_weighted_price: bool = True,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 50,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 5.0,
                 check_leakage_count: int = 20,
                 check_relative: bool = True,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 version: Optional[str] = '20260407_gp_test_1',
                 gp_generations: int = 20,
                 gp_population_size: int = 500,
                 gp_max_depth: int = 6,
                 gp_elite_size: int = 50,
                 gp_elite_relative_threshold: float = 0.65,
                 gp_tournament_size: int = 3,
                 gp_crossover_prob: float = 0.3,
                 gp_mutation_prob: float = 0.7,
                 gp_leaf_prob: float = 0.2,
                 gp_const_prob: float = 0.02,
                 gp_window_choices: Optional[Sequence[int]] = None,
                 fitness_indicator_dict: Optional[Dict[str, float]] = None,
                 fitness_metric: Optional[str] = None,
                 random_seed: Optional[int] = None,
                 gp_early_stopping_generation_count: int = 20,
                 gp_depth_penalty_coef: float = 0.0,
                 gp_depth_penalty_start_depth: int = 6,
                 gp_depth_penalty_linear_coef: float = 0.03,
                 gp_depth_penalty_quadratic_coef: float = 0.0,
                 gp_log_interval: int = 5,
                 gp_small_factor_penalty_coef: float = 0.0,
                 gp_assumed_initial_capital: float = 100_000.0,
                 gp_elite_stagnation_generation_count: int = 4,
                 gp_max_shock_generation: int = 3,
                 consistency_penalty_enabled: bool = False,
                 consistency_penalty_coef: float = 1.0):
        """遗传规划因子生成器。

        参数说明（中文）：
        - instrument_type: 数据品种类型，目前仅支持期货连续合约。
        - instrument_id_list: 合约列表（如 'C0' 或 ['C0','FG0']）。
        - fc_freq: 因子频率（1m/5m/1d）。
        - data: 直接传入的价格数据（优先使用）。
        - start_time/end_time: 回测区间起止时间。
        - portfolio_adjust_method: 收益对齐周期（min/1D/1M/1Q）。
        - interest_method: 利息计算方式（simple/compound）。
        - risk_free_rate: 是否加入无风险利率。
        - calculate_baseline: 是否计算基准表现。
        - apply_weighted_price: 是否对价格做复权处理。
        - n_jobs: 并行进程/线程数。
        - base_col_list: 可用基础字段列表（默认 OHLCV+position）。
        - min_window_size: 最小窗口长度。
        - max_factor_count: 最终返回的最大因子数量。
        - apply_rolling_norm: 是否对因子做滚动标准化。
        - rolling_norm_window: 滚动标准化窗口长度。
        - rolling_norm_min_periods: 滚动标准化最小样本数。
        - rolling_norm_eps: 标准化数值稳定项。
        - rolling_norm_clip: 标准化后的截断范围。
        - check_leakage_count: 泄露检查抽样次数。
        - check_relative: 是否与历史库做相关性去重。
        - relative_threshold: 相对相关性阈值（Spearman abs）。
        - relative_check_version_list: 相对检查的版本白名单。
        - version: 本次生成的版本标识。
        - gp_generations: GP 演化代数。
        - gp_population_size: 每代种群规模。
        - gp_max_depth: 树的最大深度。
        - gp_elite_size: 精英库最大容量。
        - gp_elite_relative_threshold: 精英库同流派判定阈值。
        - gp_tournament_size: 锦标赛选择规模。
        - gp_crossover_prob: 交叉概率。
        - gp_mutation_prob: 变异概率。
        - gp_leaf_prob: 叶子生成概率。
        - gp_const_prob: 常数叶子概率。
        - gp_window_choices: 时序算子可用窗口集合。
        - fitness_indicator_dict: 适应度指标权重字典，形如 {indicator: weight}。
        - random_seed: 随机种子。
        - gp_early_stopping_generation_count: 连续多少代无提升则早停。
        - gp_depth_penalty_coef: 深度惩罚系数（线性）。
        - gp_depth_penalty_start_depth: 深度惩罚起始深度。
        - gp_depth_penalty_linear_coef: 深度惩罚线性系数（超起始深度）。
        - gp_depth_penalty_quadratic_coef: 深度惩罚二次系数。
        - gp_log_interval: GP 日志输出间隔（代）。
        - gp_small_factor_penalty_coef: 小因子惩罚系数。
        - gp_assumed_initial_capital: 小因子惩罚的资金假设。
        - gp_elite_stagnation_generation_count: 精英库连续停滞触发 Shock 的代数。
        - gp_max_shock_generation: Shock 模式最长持续代数。
        """
        super().__init__(
            instrument_type=instrument_type,
            instrument_id_list=instrument_id_list,
            fc_freq=fc_freq,
            data=data,
            start_time=start_time,
            end_time=end_time,
            portfolio_adjust_method=portfolio_adjust_method,
            interest_method=interest_method,
            risk_free_rate=risk_free_rate,
            calculate_baseline=calculate_baseline,
            apply_weighted_price=apply_weighted_price,
            n_jobs=n_jobs,
            base_col_list=base_col_list,
            min_window_size=min_window_size,
            max_factor_count=max_factor_count,
            apply_rolling_norm=apply_rolling_norm,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            check_leakage_count=check_leakage_count,
            check_relative=check_relative,
            relative_threshold=relative_threshold,
            relative_check_version_list=relative_check_version_list,
            version=version,
        )

        self.gp_generations = gp_generations
        self.gp_population_size = gp_population_size
        self.gp_max_depth = gp_max_depth
        self.gp_elite_size = gp_elite_size
        self.gp_elite_relative_threshold = float(gp_elite_relative_threshold)
        self.gp_tournament_size = gp_tournament_size
        self.gp_crossover_prob = gp_crossover_prob
        self.gp_mutation_prob = gp_mutation_prob
        self.gp_leaf_prob = gp_leaf_prob
        self.gp_const_prob = gp_const_prob
        self.gp_window_choices = list(gp_window_choices) if gp_window_choices else [3, 5, 10, 20, 30]
        if fitness_indicator_dict is not None:
            raw_fitness_indicator_dict = dict(fitness_indicator_dict)
        else:
            metric_text = str(fitness_metric or '').strip().lower()
            if metric_text == 'sharpe':
                raw_fitness_indicator_dict = {'Gross Sharpe': 1.0}
            elif metric_text == 'ic' or metric_text == '':
                raw_fitness_indicator_dict = dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)
            else:
                raise ValueError(f'Unsupported fitness_metric={fitness_metric}, use ic/sharpe.')
        self.fitness_indicator_dict: Dict[str, float] = {}
        for indicator, weight in raw_fitness_indicator_dict.items():
            if indicator not in GP_SUPPORTED_INDICATOR:
                raise ValueError(
                    f'Unsupported fitness indicator: {indicator}. '
                    f'Available indicators: {GP_SUPPORTED_INDICATOR}'
                )
            w = float(weight)
            if abs(w) <= 1e-12:
                continue
            self.fitness_indicator_dict[str(indicator)] = w
        if not self.fitness_indicator_dict:
            self.fitness_indicator_dict = dict(GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)

        self.random_seed = random_seed
        self.gp_early_stopping_generation_count = int(gp_early_stopping_generation_count)
        self.gp_depth_penalty_coef = float(gp_depth_penalty_coef)
        self.gp_depth_penalty_start_depth = int(gp_depth_penalty_start_depth)
        self.gp_depth_penalty_linear_coef = float(gp_depth_penalty_linear_coef)
        self.gp_depth_penalty_quadratic_coef = float(gp_depth_penalty_quadratic_coef)
        self.gp_log_interval = gp_log_interval
        self.gp_small_factor_penalty_coef = float(gp_small_factor_penalty_coef)
        self.gp_assumed_initial_capital = float(gp_assumed_initial_capital)
        self.gp_elite_stagnation_generation_count = int(gp_elite_stagnation_generation_count)
        self.gp_max_shock_generation = int(gp_max_shock_generation)
        self.consistency_penalty_enabled = bool(consistency_penalty_enabled)
        self.consistency_penalty_coef = float(consistency_penalty_coef)
        # GP 阶段将滚动标准化固化到公式 OpRollNorm 中，避免后续流程重复标准化。
        self.apply_formula_rolling_norm = bool(apply_rolling_norm)

        self.factor_tree_map: Dict[str, Any] = {}
        self.cancel_event = None  # threading.Event, set externally to cancel GP

    def _prepare_df_for_gp(self, df: pd.DataFrame) -> pd.DataFrame:
        base_cols = ['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position']
        for col in base_cols:
            if col not in df.columns:
                raise ValueError(f'Missing required column for GP: {col}')

        df_eval = df[base_cols].copy()
        df_eval = df_eval.sort_values(['instrument_id', 'time']).reset_index(drop=True)
        df_eval = get_future_ret(
            df_eval,
            portfolio_adjust_method=self.portfolio_adjust_method,
            rfr=self.risk_free_rate,
        )
        return df_eval

    def _build_factor_df_from_candidates(self,
                                         df_eval: pd.DataFrame,
                                         candidates: List[GPCandidate]) -> pd.DataFrame:
        factor_df = df_eval[['time', 'instrument_id']].copy()
        self.factor_tree_map = {}
        self.factor_formula_map = {}
        self.factor_fitness_map = {}

        # Keep output factor naming stable with descending penalized fitness.
        candidates = sorted(
            list(candidates),
            key=lambda x: float(getattr(x, 'penalized_fitness', getattr(x, 'fitness', float('-inf')))),
            reverse=True,
        )

        for idx, cand in enumerate(candidates, start=1):
            fc_name = f'fac_gp_{idx:04d}'
            signal = cand.node.calc(df_eval)
            signal = pd.to_numeric(signal, errors='coerce')
            factor_df[fc_name] = signal.values
            self.factor_tree_map[fc_name] = cand.node
            self.factor_formula_map[fc_name] = cand.formula
            self.factor_fitness_map[fc_name] = {
                'fitness': {
                    'indicator_weight': self.fitness_indicator_dict,
                    'original': float(cand.original_fitness),
                    'penalized': float(cand.penalized_fitness),
                }
            }
        return factor_df

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        df_eval = self._prepare_df_for_gp(df)

        if selected_fc_names is not None:
            if self.factor_tree_map:
                factor_df = df_eval[['time', 'instrument_id']].copy()
                for fc_name in selected_fc_names:
                    if fc_name not in self.factor_tree_map:
                        raise ValueError(f'Factor `{fc_name}` is not available in current GP cache.')
                    signal = self.factor_tree_map[fc_name].calc(df_eval)
                    factor_df[fc_name] = pd.to_numeric(signal, errors='coerce').values
                return factor_df

            formula_map = get_factor_formula_map_by_version(
                fc_name_list=selected_fc_names,
                version=self.version,
                collections=['genetic_programming'],
                database='factors',
            )
            missing = [x for x in selected_fc_names if x not in formula_map]
            if missing:
                raise ValueError(f'GP formulas not found in DB for factors: {missing}')
            return calc_formula_df(
                df=df_eval,
                formula_map={k: formula_map[k] for k in selected_fc_names},
                data_fields=self.base_col_list,
            )

        limit = int(self.max_factor_count or 0)
        if limit <= 0:
            raise ValueError('max_factor_count must be positive for genetic programming.')

        candidates = run_gp_evolution(
            df=df_eval,
            data_fields=self.base_col_list,
            fitness_indicator_dict=self.fitness_indicator_dict,
            max_factor_count=limit,
            generations=self.gp_generations,
            population_size=self.gp_population_size,
            max_depth=self.gp_max_depth,
            elite_size=self.gp_elite_size,
            elite_relative_threshold=self.gp_elite_relative_threshold,
            tournament_size=self.gp_tournament_size,
            crossover_prob=self.gp_crossover_prob,
            mutation_prob=self.gp_mutation_prob,
            window_choices=self.gp_window_choices,
            const_prob=self.gp_const_prob,
            leaf_prob=self.gp_leaf_prob,
            random_seed=self.random_seed,
            early_stopping_generation_count=self.gp_early_stopping_generation_count,
            log_interval=self.gp_log_interval,
            depth_penalty_coef=self.gp_depth_penalty_coef,
            depth_penalty_start_depth=self.gp_depth_penalty_start_depth,
            depth_penalty_linear_coef=self.gp_depth_penalty_linear_coef,
            depth_penalty_quadratic_coef=self.gp_depth_penalty_quadratic_coef,
            apply_rolling_norm=self.apply_formula_rolling_norm,
            rolling_norm_window=self.rolling_norm_window,
            rolling_norm_min_periods=self.rolling_norm_min_periods,
            rolling_norm_eps=self.rolling_norm_eps,
            rolling_norm_clip=self.rolling_norm_clip,
            small_factor_penalty_coef=self.gp_small_factor_penalty_coef,
            assumed_initial_capital=self.gp_assumed_initial_capital,
            elite_stagnation_generation_count=self.gp_elite_stagnation_generation_count,
            max_shock_generation=self.gp_max_shock_generation,
            cancel_event=self.cancel_event,
            consistency_penalty_enabled=self.consistency_penalty_enabled,
            consistency_penalty_coef=self.consistency_penalty_coef,
        )

        if not candidates:
            raise ValueError('Genetic programming produced no valid candidates.')
        return self._build_factor_df_from_candidates(df_eval, candidates)

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        base_df = self.load_base_data()
        factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        return self._finalize_generated_data(base_df, factor_df)


class FactorFusioner:
    """Fuse existing DB factors with stepwise forward selection."""

    def __init__(
        self,
        fusion_method: str = 'avg_weight',
        raw_factor_dict: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
        collection: Union[str, Sequence[str]] = 'genetic_programming',
        instrument_type: str = 'futures_continuous_contract',
        instrument_id_list: Union[str, Sequence[str]] = 'C0',
        fc_freq: str = '1d',
        data: Optional[pd.DataFrame] = None,
        start_time: Optional[str] = '20200101',
        end_time: Optional[str] = '20241231',
        portfolio_adjust_method: str = '1D',
        interest_method: str = 'simple',
        risk_free_rate: bool = False,
        apply_weighted_price: bool = True,
        check_leakage_count: int = 20,
        check_relative: bool = True,
        relative_threshold: float = 0.7,
        relative_check_version_list: Optional[Sequence[str]] = None,
        max_fusion_count: int = 5,
        fusion_metrics: Union[str, Sequence[str]] = 'ic',
        version: Optional[str] = None,
        n_jobs: int = 5,
        database: str = 'factors',
        base_col_list: Optional[Sequence[str]] = None,
        consider_outsample: bool = False,
        outsample_start_day: Optional[str] = None,
        outsample_end_day: Optional[str] = None,
    ):
        """初始化因子融合器。

        参数说明：
        - fusion_method: 融合方式。当前仅支持 `avg_weight`（等权平均融合）。
        - raw_factor_dict: 手动指定待融合因子池。格式为 `{version: [factor_name, ...]}`；
          若为 None，则自动从 `collection` 指定的集合中拉取候选因子。
        - collection: 候选因子来源集合名，支持字符串或字符串列表。
        - instrument_type: 行情数据类型，当前仅支持期货连续合约场景。
        - instrument_id_list: 回测与评估使用的标的列表；当前实现要求仅一个标的。
        - fc_freq: 因子频率，支持 `1m`、`5m`、`1d`。
        - data: 可直接传入已准备好的行情数据；为空时自动从数据层拉取。
        - start_time: 融合评估起始日期（含），如 `20200101`。
        - end_time: 融合评估结束日期（含），如 `20241231`。
        - portfolio_adjust_method: 组合调仓频率，支持 `min`、`1D`、`1M`、`1Q`。
        - interest_method: 收益累计方式（如 `simple`/`compound`），传递给回测器。
        - risk_free_rate: 是否在绩效计算中考虑无风险利率。
        - apply_weighted_price: 是否先对价格进行复权，再计算收益与因子。
        - check_leakage_count: 泄露检查抽样次数。
        - check_relative: 是否执行与历史融合因子的相似度检查。
        - relative_threshold: 相似度阈值（Spearman abs），超过则判为过于相似。
        - relative_check_version_list: 相似度检查版本白名单，None 表示不限制版本。
        - max_fusion_count: 最多融合因子数量（包含首个基石因子）。
        - fusion_metrics: 融合优化目标，可为 `ic`、`sharpe` 或其列表。
        - version: 融合结果版本号，必填。最终落库记录 version 字段将使用该值。
        - n_jobs: 并行任务数，用于候选因子计算等并行流程。
        - database: 因子公式读取与融合结果写入的数据库名。
        - base_col_list: 计算因子时需要保留的基础字段列表（默认 OHLCV+position）。
        - consider_outsample: 是否在融合排序/筛选时优先使用样本外指标。
        - outsample_start_day: 样本外区间起始日（YYYYMMDD）。
        - outsample_end_day: 样本外区间结束日（YYYYMMDD）。
        """
        self.fusion_method = str(fusion_method).strip()
        if self.fusion_method not in FusionSupportedMethods:
            raise ValueError(f'Unsupported fusion_method={self.fusion_method}, use {sorted(FusionSupportedMethods)}.')

        self.raw_factor_dict = self._normalize_raw_factor_dict(raw_factor_dict)
        self.collection_list = self._normalize_str_or_seq(collection, name='collection')
        self.instrument_type = instrument_type
        self.instrument_id_list = self._normalize_str_or_seq(instrument_id_list, name='instrument_id_list')
        self.fc_freq = fc_freq
        self.data = data
        self.start_time = start_time
        self.end_time = end_time
        self.portfolio_adjust_method = portfolio_adjust_method
        self.interest_method = interest_method
        self.risk_free_rate = bool(risk_free_rate)
        self.apply_weighted_price = bool(apply_weighted_price)
        self.check_leakage_count = int(check_leakage_count)
        self.check_relative = bool(check_relative)
        self.relative_threshold = float(relative_threshold)
        self.relative_check_version_list = None if relative_check_version_list is None else list(relative_check_version_list)
        self.max_fusion_count = int(max_fusion_count)
        self.fusion_metrics = self._normalize_fusion_metrics(fusion_metrics)
        self.version = str(version).strip() if version is not None else ''
        self.n_jobs = int(n_jobs)
        self.database = str(database).strip() or 'factors'
        self.base_col_list = list(base_col_list) if base_col_list else ['open', 'high', 'low', 'close', 'volume', 'position']
        self.consider_outsample = bool(consider_outsample)
        self.outsample_start_day = str(outsample_start_day).strip() if outsample_start_day is not None else ''
        self.outsample_end_day = str(outsample_end_day).strip() if outsample_end_day is not None else ''

        self._effective_end_time = str(self.end_time).strip() if self.end_time is not None else ''
        if self.consider_outsample:
            if not (self.outsample_start_day and self.outsample_end_day):
                raise ValueError(
                    'consider_outsample=True requires both outsample_start_day and outsample_end_day.'
                )
            try:
                out_start = pd.to_datetime(self.outsample_start_day)
                out_end = pd.to_datetime(self.outsample_end_day)
                in_start = pd.to_datetime(self.start_time)
                in_end = pd.to_datetime(self.end_time)
            except Exception as e:
                raise ValueError(
                    'Invalid outsample date format. '
                    f'outsample_start_day={self.outsample_start_day}, outsample_end_day={self.outsample_end_day}'
                ) from e
            if out_end < out_start:
                raise ValueError(
                    f'outsample_end_day must be >= outsample_start_day, got '
                    f'{self.outsample_start_day}~{self.outsample_end_day}.'
                )
            if out_start < in_start:
                raise ValueError(
                    f'outsample_start_day must be >= start_time, got start_time={self.start_time}, '
                    f'outsample_start_day={self.outsample_start_day}.'
                )
            self._effective_end_time = max(str(self.end_time), self.outsample_end_day)

        if not self.version:
            raise ValueError('FactorFusioner requires non-empty `version` and it must be explicitly provided.')

        if self.max_fusion_count <= 0:
            raise ValueError(f'max_fusion_count must be positive, got {self.max_fusion_count}.')
        if self.fc_freq not in ['1m', '5m', '1d']:
            raise ValueError(f'Only support 1m, 5m or 1d fc_freq, got {self.fc_freq}.')
        if self.portfolio_adjust_method not in ['min', '1D', '1M', '1Q']:
            raise ValueError(
                f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, got {self.portfolio_adjust_method}.'
            )
        if len(self.instrument_id_list) != 1:
            raise ValueError(
                'FactorFusioner currently supports exactly one instrument_id for TS metric consistency, '
                f'got {self.instrument_id_list}.'
            )
        if not (0.0 <= self.relative_threshold <= 1.0):
            raise ValueError(f'relative_threshold should be in [0, 1], got {self.relative_threshold}.')

    @staticmethod
    def _normalize_str_or_seq(value: Union[str, Sequence[str]], name: str) -> List[str]:
        if isinstance(value, str):
            out = [value]
        else:
            out = list(value)
        out = [str(x).strip() for x in out if str(x).strip()]
        if not out:
            raise ValueError(f'`{name}` cannot be empty.')
        return out

    @staticmethod
    def _normalize_raw_factor_dict(
        raw_factor_dict: Optional[Dict[str, Union[str, Sequence[str]]]]
    ) -> Optional[Dict[str, List[str]]]:
        if raw_factor_dict is None:
            return None
        if not isinstance(raw_factor_dict, dict) or not raw_factor_dict:
            raise ValueError('raw_factor_dict must be a non-empty dict when provided.')

        out: Dict[str, List[str]] = {}
        for version, factor_value in raw_factor_dict.items():
            version_text = str(version).strip()
            if not version_text:
                raise ValueError(f'Invalid raw_factor_dict version key: {version}')
            if isinstance(factor_value, str):
                fc_names = [factor_value]
            else:
                fc_names = list(factor_value)
            fc_names = [str(x).strip() for x in fc_names if str(x).strip()]
            if not fc_names:
                raise ValueError(f'raw_factor_dict[{version_text}] must contain at least one factor_name.')
            duplicated = sorted([x for x in set(fc_names) if fc_names.count(x) > 1])
            if duplicated:
                raise ValueError(f'raw_factor_dict[{version_text}] contains duplicated factor_name: {duplicated}')
            out[version_text] = fc_names
        return out

    def _normalize_fusion_metrics(self, fusion_metrics: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(fusion_metrics, str):
            metrics = [fusion_metrics]
        else:
            metrics = list(fusion_metrics)
        metrics = [str(x).strip().lower() for x in metrics if str(x).strip()]
        if not metrics:
            raise ValueError('fusion_metrics cannot be empty.')
        invalid = [x for x in metrics if x not in FusionSupportedMetrics]
        if invalid:
            raise ValueError(
                f'Unsupported fusion_metrics={invalid}. Available metrics: {sorted(FusionSupportedMetrics)}.'
            )
        duplicated = sorted([x for x in set(metrics) if metrics.count(x) > 1])
        if duplicated:
            raise ValueError(f'fusion_metrics contains duplicated metric(s): {duplicated}')
        return metrics

    @staticmethod
    def _factor_key(version: str, factor_name: str) -> str:
        return f'{version}::{factor_name}'

    @staticmethod
    def _safe_metric_value(value: Any) -> float:
        try:
            f = float(value)
        except Exception:
            return float('-inf')
        if np.isnan(f):
            return float('-inf')
        return f

    def _metric_sort_key(self, metrics: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(self._safe_metric_value(metrics.get(m)) for m in self.fusion_metrics)

    def _is_strictly_better(self,
                            new_metrics: Dict[str, float],
                            base_metrics: Dict[str, float]) -> bool:
        return all(
            self._safe_metric_value(new_metrics.get(m)) > self._safe_metric_value(base_metrics.get(m))
            for m in self.fusion_metrics
        )

    def _load_base_data(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            df = self.data.copy()
        else:
            if self.instrument_type != 'futures_continuous_contract':
                raise ValueError(f'Unsupported instrument type: {self.instrument_type}.')
            df = get_futures_continuous_contract_price(
                instrument_id=self.instrument_id_list,
                start_date=self.start_time,
                end_date=self._effective_end_time,
                from_database=True,
            )

        optional_cols = [c for c in ['weighted_factor', 'cur_weighted_factor', 'is_rollover', 'symbol'] if c in df.columns]
        required_cols = ['time', 'instrument_id'] + self.base_col_list + optional_cols
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f'Input data does not contain required column: {col}')

        df = df[required_cols].copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        available_instruments = df['instrument_id'].dropna().unique().tolist()
        invalid_instruments = [x for x in self.instrument_id_list if x not in available_instruments]
        if invalid_instruments:
            raise ValueError(
                f'Invalid instrument_id_list: {invalid_instruments}. Available instruments: {available_instruments}'
            )

        if self.apply_weighted_price:
            if 'weighted_factor' not in df.columns:
                raise ValueError(
                    'apply_weighted_price=True requires `weighted_factor` in source data. '
                    'Set apply_weighted_price=False to use raw prices.'
                )
            df = get_weighted_price(df)

        df = get_future_ret(
            df,
            portfolio_adjust_method=self.portfolio_adjust_method,
            rfr=self.risk_free_rate,
        )
        return df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

    def _load_candidate_records(self) -> pd.DataFrame:
        if self.raw_factor_dict is None:
            records = get_factor_formula_records(
                collections=self.collection_list,
                versions=None,
                database=self.database,
            )
            if records.empty:
                raise ValueError(
                    f'No factor formulas found in DB for collections={self.collection_list}, database={self.database}.'
                )
            records = records[['collection', 'version', 'factor_name', 'formula']].copy()
        else:
            target_versions = sorted(self.raw_factor_dict.keys())
            records = get_factor_formula_records(
                collections=self.collection_list,
                versions=target_versions,
                database=self.database,
            )
            if records.empty:
                raise ValueError(
                    'No factor formulas found in DB for raw_factor_dict request. '
                    f'collections={self.collection_list}, versions={target_versions}.'
                )

            records = records[['collection', 'version', 'factor_name', 'formula']].copy()
            expected_pairs = {
                (version, factor_name)
                for version, fc_list in self.raw_factor_dict.items()
                for factor_name in fc_list
            }
            records = records[
                records.apply(lambda x: (str(x['version']), str(x['factor_name'])) in expected_pairs, axis=1)
            ].copy()

            found_pairs = {
                (str(x['version']), str(x['factor_name']))
                for _, x in records.iterrows()
            }
            missing_pairs = sorted(expected_pairs - found_pairs)
            if missing_pairs:
                raise ValueError(
                    'Some factors specified by raw_factor_dict are not found in DB: '
                    f'{missing_pairs}. collections={self.collection_list}, database={self.database}.'
                )

        records['version'] = records['version'].astype(str)
        records['factor_name'] = records['factor_name'].astype(str)
        records['formula'] = records['formula'].astype(str).str.strip()
        records = records[records['formula'] != ''].copy()
        if records.empty:
            raise ValueError('No non-empty factor formula found for fusion.')

        # Strictly avoid ambiguous records with same (version, factor_name) but different formulas.
        ambiguous_pairs: List[Tuple[str, str]] = []
        dedup_rows: List[pd.Series] = []
        for (_, _), group in records.groupby(['version', 'factor_name'], sort=False):
            formula_set = {str(x).strip() for x in group['formula'].tolist() if str(x).strip()}
            if len(formula_set) > 1:
                sample_row = group.iloc[0]
                ambiguous_pairs.append((str(sample_row['version']), str(sample_row['factor_name'])))
                continue
            dedup_rows.append(group.iloc[-1])

        if ambiguous_pairs:
            raise ValueError(
                'Ambiguous formulas found in DB for (version, factor_name): '
                f'{ambiguous_pairs}. Please specify a unique source or clean DB records.'
            )

        out = pd.DataFrame(dedup_rows).reset_index(drop=True)
        out['factor_key'] = out.apply(lambda x: self._factor_key(str(x['version']), str(x['factor_name'])), axis=1)
        return out[['collection', 'version', 'factor_name', 'factor_key', 'formula']].copy()

    def _calc_candidate_factor_df(self,
                                  base_df: pd.DataFrame,
                                  factor_records: pd.DataFrame) -> pd.DataFrame:
        out = base_df[['time', 'instrument_id']].copy()
        calc_df = base_df[['time', 'instrument_id'] + self.base_col_list].copy()
        for _, row in factor_records.iterrows():
            factor_key = str(row['factor_key'])
            formula = str(row['formula'])
            try:
                col = calc_formula_series(df=calc_df, formula=formula, data_fields=self.base_col_list)
            except Exception as e:
                raise ValueError(
                    f'Failed to calculate factor={factor_key} from DB formula={formula}. Error: {e}'
                ) from e
            out[factor_key] = pd.to_numeric(col, errors='coerce')

        return out.sort_values(['instrument_id', 'time']).reset_index(drop=True)

    def _evaluate_metrics(self,
                          eval_df: pd.DataFrame,
                          factor_col: str) -> Dict[str, float]:
        data_i = eval_df[['time', 'instrument_id', 'future_ret', factor_col]].copy()
        _, net_ret_df, _ = get_ts_ret_and_turnover(data_i, factor_col)
        net_ret_df = net_ret_df.reset_index()

        # Fusion Sharpe uses fee-adjusted net return series for consistency with executable performance.
        annual_ret = get_annualized_ret(net_ret_df, factor_col, self.interest_method)
        annual_vol = get_annualized_volatility(net_ret_df, factor_col)
        annual_sharpe = get_annualized_sharpe(annual_ret, annual_vol)
        ic_df, _ = get_annualized_ts_ic_and_t_corr(
            data_i,
            fc_col=factor_col,
            fc_freq=self.fc_freq,
            portfolio_adjust_method=self.portfolio_adjust_method,
        )

        ic_value = float(pd.to_numeric(ic_df.loc['all', 'TS IC'], errors='coerce')) if 'all' in ic_df.index else float('nan')
        sharpe_value = float(pd.to_numeric(annual_sharpe.loc['all', factor_col], errors='coerce')) \
            if 'all' in annual_sharpe.index else float('nan')
        return {'ic': ic_value, 'sharpe': sharpe_value}

    def check_if_leakage(self,
                         fc_name_list: Sequence[str],
                         generated_df: pd.DataFrame,
                         atol: float = 1e-10,
                         rtol: float = 1e-8,
                         raise_error: bool = True) -> Dict[str, Any]:
        def _gen(df_slice: pd.DataFrame, selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
            selected = [str(x) for x in (selected_fc_names or []) if str(x).strip()]
            need_cols = ['time', 'instrument_id'] + selected
            out = generated_df[need_cols].copy()
            key_df = df_slice[['time', 'instrument_id']].copy()
            out = key_df.merge(out, on=['time', 'instrument_id'], how='left', validate='1:1')
            return out

        return check_if_leakage_util(
            selected_fc_name_list=fc_name_list,
            load_base_data_fn=self._load_base_data,
            generate_factor_df_fn=_gen,
            check_leakage_count=self.check_leakage_count,
            atol=atol,
            rtol=rtol,
            raise_error=raise_error,
        )

    def filter_fc_by_db_relative_spearman(self,
                                          selected_fc_name_list: Sequence[str],
                                          generated_df: pd.DataFrame,
                                          base_df: pd.DataFrame,
                                          n_jobs: Optional[int] = None,
                                          batch_size: int = 100) -> Dict[str, Any]:
        return filter_fc_by_db_relative_spearman_util(
            selected_fc_name_list=selected_fc_name_list,
            generated_df=generated_df,
            base_df=base_df,
            base_col_list=self.base_col_list,
            relative_threshold=self.relative_threshold,
            relative_check_version_list=self.relative_check_version_list,
            n_jobs=n_jobs or self.n_jobs,
            batch_size=batch_size,
            database='factors',
            collections=['factor_fusion'],
            self_compare_version=self.version,
            self_compare_collection='factor_fusion',
        )

    def _run_backtest_for_signal(self,
                                 eval_df: pd.DataFrame,
                                 fc_name_list: Sequence[str],
                                 calculate_baseline: bool = False) -> BackTester:
        fc_name_list = [str(x) for x in fc_name_list if str(x).strip()]
        if not fc_name_list:
            raise ValueError('fc_name_list cannot be empty for _run_backtest_for_signal.')
        # Keep order while removing duplicates.
        fc_name_list = list(dict.fromkeys(fc_name_list))
        bt_cols = ['time', 'instrument_id', 'future_ret'] + fc_name_list
        bt = BackTester(
            fc_name_list=fc_name_list,
            version='fusion_runtime',
            collection='fusion_runtime',
            instrument_type=self.instrument_type,
            instrument_id_list=self.instrument_id_list,
            fc_freq=self.fc_freq,
            data=eval_df[bt_cols].copy(),
            start_time=self.start_time,
            end_time=self._effective_end_time,
            portfolio_adjust_method=self.portfolio_adjust_method,
            interest_method=self.interest_method,
            risk_free_rate=self.risk_free_rate,
            calculate_baseline=calculate_baseline,
            apply_weighted_price=False,
            n_jobs=max(1, self.n_jobs),
        )
        bt.backtest()
        return bt

    def _format_metrics(self,
                        metrics: Dict[str, float]) -> str:
        return ', '.join([f'{k}={self._safe_metric_value(metrics.get(k)):.6f}' for k in self.fusion_metrics])

    @staticmethod
    def _build_add_formula(formula_list: Sequence[str]) -> str:
        cleaned = [str(x).strip() for x in formula_list if str(x).strip()]
        if not cleaned:
            raise ValueError('Cannot build fusion formula from empty formula_list.')
        if len(cleaned) == 1:
            return cleaned[0]
        inner = ', '.join(cleaned)
        return f'NanMean({inner})'

    def _build_fusion_formula(self,
                              selected_factor_keys: Sequence[str],
                              factor_records: pd.DataFrame) -> str:
        formula_map = {
            str(row['factor_key']): str(row['formula'])
            for _, row in factor_records.iterrows()
        }
        selected_formulas = [formula_map.get(key, '').strip() for key in selected_factor_keys]
        if any(not f for f in selected_formulas):
            missing_keys = [k for k, f in zip(selected_factor_keys, selected_formulas) if not f]
            raise ValueError(f'Missing formulas for fusion factor keys: {missing_keys}')

        return self._build_add_formula(selected_formulas)

    def fuse(self) -> Dict[str, Any]:
        """Run stepwise forward-selection fusion and return fusion artifacts."""
        log.info(
            'Fusion started: '
            f'method={self.fusion_method}, '
            f'collections={self.collection_list}, '
            f'fusion_metrics={self.fusion_metrics}, '
            f'max_fusion_count={self.max_fusion_count}, '
            f'raw_factor_dict_provided={self.raw_factor_dict is not None}'
        )

        # 1) 准备融合评估所需基础数据（行情 + future_ret）以及候选因子元信息。
        base_df = self._load_base_data()
        factor_records = self._load_candidate_records()
        log.info(f'Fusion candidate pool size={len(factor_records)}')

        # 2) 计算候选因子的数值列，并与基础评估表按(time, instrument_id)对齐。
        factor_df = self._calc_candidate_factor_df(base_df=base_df, factor_records=factor_records)
        eval_df = base_df[['time', 'instrument_id', 'future_ret']].merge(
            factor_df,
            on=['time', 'instrument_id'],
            how='left',
            validate='1:1',
        )

        # Optional out-of-sample slice used as the primary optimization target.
        optimize_eval_df = eval_df
        outsample_eval_df: Optional[pd.DataFrame] = None
        use_outsample = bool(self.consider_outsample and self.outsample_start_day and self.outsample_end_day)
        if use_outsample:
            out_start = pd.to_datetime(self.outsample_start_day)
            out_end = pd.to_datetime(self.outsample_end_day)
            outsample_eval_df = eval_df[(eval_df['time'] >= out_start) & (eval_df['time'] <= out_end)].copy()
            if outsample_eval_df.empty:
                raise ValueError(
                    '[Fusion][Outsample] empty outsample range under consider_outsample=True. '
                    f'Please ensure end_time covers outsample_end_day and data exists in '
                    f'{self.outsample_start_day}~{self.outsample_end_day}. '
                    f'Current start_time={self.start_time}, end_time={self.end_time}, '
                    f'effective_end_time={self._effective_end_time}.'
                )
            else:
                optimize_eval_df = outsample_eval_df
                log.info(
                    '[Fusion][Outsample] enabled. '
                    f'rows={len(outsample_eval_df)}, '
                    f'start={self.outsample_start_day}, end={self.outsample_end_day}'
                )

        # factor_keys: 形如 "version::factor_name" 的候选因子唯一键列表。
        factor_keys = factor_records['factor_key'].astype(str).tolist()
        if not factor_keys:
            raise ValueError('No available factors for fusion.')

        # single_metric_map: 每个单因子的总体评估指标（例如 ic/sharpe）。
        # 这些 metrics 用于初始化排序与后续比较基线，不是逐年指标。
        single_metric_map: Dict[str, Dict[str, float]] = {}
        for factor_key in factor_keys:
            signal_df = optimize_eval_df[['time', 'instrument_id', 'future_ret', factor_key]].copy()
            metrics = self._evaluate_metrics(signal_df, factor_key)
            single_metric_map[factor_key] = metrics
            log.info(f'[Fusion][Init] factor={factor_key}, {self._format_metrics(metrics)}')

        # sorted_factor_keys: 按 fusion_metrics 优先级从高到低排序后的候选池。
        # 若 fusion_metrics=['ic','sharpe']，则先比较 ic，再比较 sharpe。
        sorted_factor_keys = sorted(
            factor_keys,
            key=lambda x: self._metric_sort_key(single_metric_map[x]),
            reverse=True,
        )
        # 排名第一的单因子作为初始基石因子（Base Factor）。
        base_factor_key = sorted_factor_keys[0]
        # selected_factor_keys: 当前已被接纳进融合组合的因子键。
        selected_factor_keys: List[str] = [base_factor_key]
        # base_metrics: 当前“基线组合”的指标；只有全指标严格提升才会被新组合替代。
        base_metrics = dict(single_metric_map[base_factor_key])
        log.info(
            '[Fusion][Init] base factor selected: '
            f'base_factor={base_factor_key}, {self._format_metrics(base_metrics)}'
        )

        fusion_col = '__fused_factor__'

        max_count = min(self.max_fusion_count, len(sorted_factor_keys))
        for round_idx in range(2, max_count + 1):
            # 每一轮从未入选因子里挑一个最好的，与当前基线组合做“加一因子”对比。
            candidate_keys = [x for x in sorted_factor_keys if x not in selected_factor_keys]
            if not candidate_keys:
                log.info(f'[Fusion][Round {round_idx}] no candidate left, stop.')
                break

            log.info(
                f'[Fusion][Round {round_idx}] start. '
                f'base_factors={selected_factor_keys}, base_metrics={self._format_metrics(base_metrics)}, '
                f'candidate_count={len(candidate_keys)}'
            )

            round_best_key: Optional[str] = None
            round_best_metrics: Optional[Dict[str, float]] = None

            for cand_key in candidate_keys:
                # trial_keys: 本轮待评估组合 = 已选组合 + 当前候选。
                trial_keys = selected_factor_keys + [cand_key]
                if self.fusion_method == 'avg_weight':
                    # 等权融合：对组合内每个因子截面值直接取均值。
                    fused_series = eval_df[trial_keys].mean(axis=1, skipna=True)
                else:
                    raise ValueError(f'Unsupported fusion_method={self.fusion_method}')

                trial_df = eval_df[['time', 'instrument_id', 'future_ret']].copy()
                if use_outsample:
                    trial_df = optimize_eval_df[['time', 'instrument_id', 'future_ret']].copy()
                trial_df[fusion_col] = pd.to_numeric(fused_series, errors='coerce')

                # trial_metrics: 该组合在整个回测区间上的总体指标。
                trial_metrics = self._evaluate_metrics(trial_df, fusion_col)

                # is_better=True 表示 fusion_metrics 中每一项都严格优于当前基线。
                is_better = self._is_strictly_better(trial_metrics, base_metrics)
                log.info(
                    f'[Fusion][Round {round_idx}] candidate={cand_key}, '
                    f'trial_factors={trial_keys}, '
                    f'{self._format_metrics(trial_metrics)}, '
                    f'better_than_base={is_better}'
                )

                if not is_better:
                    continue
                if round_best_metrics is None:
                    round_best_key = cand_key
                    round_best_metrics = trial_metrics
                    continue
                if self._metric_sort_key(trial_metrics) > self._metric_sort_key(round_best_metrics):
                    round_best_key = cand_key
                    round_best_metrics = trial_metrics

            if round_best_key is None or round_best_metrics is None:
                # 本轮没有任何“全指标严格提升”的候选，提前停止融合。
                log.info(
                    f'[Fusion][Round {round_idx}] no better candidate found. '
                    f'base_factors={selected_factor_keys}, base_metrics={self._format_metrics(base_metrics)}'
                )
                break

            # 接纳本轮最佳候选，并将其指标更新为下一轮基线。
            selected_factor_keys.append(round_best_key)
            base_metrics = round_best_metrics
            log.info(
                f'[Fusion][Round {round_idx}] improved. '
                f'new_base_factors={selected_factor_keys}, '
                f'added_factor={round_best_key}, '
                f'new_base_metrics={self._format_metrics(base_metrics)}'
            )

        # 3) 生成最终融合信号，并进行一次完整回测用于落库与返回。
        final_fused_df = eval_df[['time', 'instrument_id', 'future_ret']].copy()
        final_fused_df[fusion_col] = pd.to_numeric(eval_df[selected_factor_keys].mean(axis=1, skipna=True), errors='coerce')
        for key in selected_factor_keys:
            final_fused_df[key] = pd.to_numeric(eval_df[key], errors='coerce')
        final_bt = self._run_backtest_for_signal(
            final_fused_df,
            [fusion_col] + list(selected_factor_keys),
            calculate_baseline=True,
        )
        final_metrics = self._evaluate_metrics(final_fused_df, fusion_col)
        final_metrics_outsample: Optional[Dict[str, float]] = None
        if use_outsample and outsample_eval_df is not None:
            out_df = cast(pd.DataFrame, outsample_eval_df)
            out_start = pd.to_datetime(self.outsample_start_day)
            out_end = pd.to_datetime(self.outsample_end_day)

            # Use the same continuous fused signal path (with full warmup history)
            # to avoid mismatch between reported outsample metrics and NAV curves.
            out_fused_df = final_fused_df[
                (final_fused_df['time'] >= out_start) & (final_fused_df['time'] <= out_end)
            ].copy()
            if out_fused_df.empty:
                raise ValueError(
                    'Outsample slice is empty when computing final_metrics_outsample. '
                    f'outsample_start_day={self.outsample_start_day}, '
                    f'outsample_end_day={self.outsample_end_day}'
                )

            # IC on outsample window from continuous fused signal.
            out_ic_df, _ = get_annualized_ts_ic_and_t_corr(
                out_fused_df[['time', 'instrument_id', 'future_ret', fusion_col]].copy(),
                fc_col=fusion_col,
                fc_freq=self.fc_freq,
                portfolio_adjust_method=self.portfolio_adjust_method,
            )
            out_ic_value = float(
                pd.to_numeric(out_ic_df.loc['all', 'TS IC'], errors='coerce')
            ) if 'all' in out_ic_df.index else float('nan')

            # Sharpe from continuous net return series (keeps boundary position continuity).
            instrument_id = self.instrument_id_list[0]
            net_ret_df = final_bt.performance_dc[instrument_id][fusion_col]['daily_net_ret'].copy()
            net_ret_df = net_ret_df[
                (pd.to_datetime(net_ret_df['time']) >= out_start)
                & (pd.to_datetime(net_ret_df['time']) <= out_end)
            ].copy()
            if net_ret_df.empty:
                raise ValueError(
                    'Outsample daily_net_ret is empty when computing final_metrics_outsample. '
                    f'outsample_start_day={self.outsample_start_day}, '
                    f'outsample_end_day={self.outsample_end_day}'
                )
            out_ret = get_annualized_ret(net_ret_df, fusion_col, self.interest_method)
            out_vol = get_annualized_volatility(net_ret_df, fusion_col)
            out_sharpe_df = get_annualized_sharpe(out_ret, out_vol)
            out_sharpe_value = float(
                pd.to_numeric(out_sharpe_df.loc['all', fusion_col], errors='coerce')
            ) if 'all' in out_sharpe_df.index else float('nan')

            final_metrics_outsample = {
                'ic': out_ic_value,
                'sharpe': out_sharpe_value,
            }
        leakage_result = self.check_if_leakage(
            fc_name_list=[fusion_col],
            generated_df=final_fused_df[['time', 'instrument_id', fusion_col]].copy(),
            raise_error=False,
        )
        relative_result: Optional[Dict[str, Any]] = None
        persist_allowed = True
        if not leakage_result.get('passed', False):
            persist_allowed = False
            log.warning(
                '[Fusion][Persist][Skip] leakage check failed, skip persisting fusion factor. '
                f'failed_factor_list={leakage_result.get("failed_factor_list", [])}'
            )
        elif self.check_relative:
            relative_result = self.filter_fc_by_db_relative_spearman(
                selected_fc_name_list=[fusion_col],
                generated_df=final_fused_df[['time', 'instrument_id', fusion_col]].copy(),
                base_df=base_df,
                n_jobs=self.n_jobs,
            )
            if not relative_result.get('selected_fc_name_list', []):
                persist_allowed = False
                log.warning(
                    '[Fusion][Persist][Skip] relative correlation check failed, skip persisting fusion factor. '
                    f'threshold={self.relative_threshold}, versions={self.relative_check_version_list}'
                )

        # 4) 组装入选因子的元信息，便于后续追踪来源与公式。
        meta_df = factor_records.set_index('factor_key')
        selected_factor_keys_by_perf = sorted(
            selected_factor_keys,
            key=lambda x: self._metric_sort_key(single_metric_map.get(x, {})),
            reverse=True,
        )
        selected_factors_detail = [
            {
                'factor_key': key,
                'version': str(meta_df.loc[key, 'version']),
                'factor_name': str(meta_df.loc[key, 'factor_name']),
                'collection': str(meta_df.loc[key, 'collection']),
                'formula': str(meta_df.loc[key, 'formula']),
            }
            for key in selected_factor_keys_by_perf
        ]

        # 5) 构建融合后的公式表达式，并将结果写入 factors.fusion_factor。
        fusion_formula = self._build_fusion_formula(selected_factor_keys, factor_records)

        # 5.1) Formula consistency check: recompute from formula and compare against
        #      the pre-computed mean to catch any formula/eval mismatch early.
        if len(selected_factor_keys) > 1:
            calc_df = base_df[['time', 'instrument_id'] + self.base_col_list].copy()
            formula_series = calc_formula_series(df=calc_df, formula=fusion_formula, data_fields=self.base_col_list)
            formula_series = pd.to_numeric(formula_series, errors='coerce')
            precomputed_series = final_fused_df[fusion_col].values
            formula_values = formula_series.values

            both_valid = ~(pd.isna(precomputed_series) | pd.isna(formula_values))
            if both_valid.sum() > 0:
                max_diff = float(np.nanmax(np.abs(precomputed_series[both_valid] - formula_values[both_valid])))
                corr = float(np.corrcoef(precomputed_series[both_valid], formula_values[both_valid])[0, 1])
                nan_precomputed = int(pd.isna(precomputed_series).sum())
                nan_formula = int(pd.isna(formula_values).sum())
                log.info(
                    f'[Fusion][FormulaCheck] formula={fusion_formula[:120]}..., '
                    f'max_abs_diff={max_diff:.10f}, corr={corr:.8f}, '
                    f'nan_precomputed={nan_precomputed}, nan_formula={nan_formula}, '
                    f'valid_count={int(both_valid.sum())}'
                )
                if max_diff > 1e-6:
                    log.warning(
                        f'[Fusion][FormulaCheck] MISMATCH detected! '
                        f'Formula recomputation differs from precomputed mean. '
                        f'max_abs_diff={max_diff:.10f}. '
                        f'The persisted formula may produce different results than the fusion evaluation.'
                    )
            else:
                log.warning('[Fusion][FormulaCheck] no overlapping valid values between formula and precomputed series.')

        fusion_collection = 'factor_fusion'
        fusion_idx = max(0, len(selected_factor_keys_by_perf) - 1)
        fusion_factor_name = f'fusion_{self.fusion_method}_{fusion_idx}'
        fusion_info: Dict[str, List[str]] = {}
        for item in selected_factors_detail:
            info_key = f"{item['collection']}:{item['version']}"
            fusion_info.setdefault(info_key, []).append(str(item['factor_name']))
        fitness_payload = {
            metric: {'value': float(final_metrics.get(metric, float('nan')))}
            for metric in self.fusion_metrics
        }

        perf_summary = final_bt.performance_summary.copy()
        if 'Factor Name' in perf_summary.columns:
            perf_summary = perf_summary[perf_summary['Factor Name'] == fusion_col].copy()
            perf_summary['Factor Name'] = fusion_factor_name

        if len(selected_factor_keys) <= 1:
            log.info(
                '[Fusion][Persist][Skip] '
                'selected_factor_count<=1, skip persisting to factor_fusion because no real fusion happened. '
                f'selected_factor_keys={selected_factor_keys}, '
                f'base_formula={fusion_formula}'
            )
        elif not persist_allowed:
            log.info('[Fusion][Persist][Skip] persistence disabled by leakage/relative checks.')
        else:
            update_factor_info(
                selected_fc_name_list=[fusion_factor_name],
                performance_summary=perf_summary,
                factor_formula_map={fusion_factor_name: fusion_formula},
                factor_fitness_map={fusion_factor_name: fitness_payload},
                instrument_id_list=self.instrument_id_list,
                method='fusion',
                version=self.version,
                start_date=self.start_time,
                end_date=self.end_time,
                extra_record_fields={'fusion_info': fusion_info},
                database='factors',
                collection=fusion_collection,
            )
            log.info(
                '[Fusion][Persist] '
                f'database=factors, collection={fusion_collection}, '
                f'version={self.version}, factor_name={fusion_factor_name}, '
                f'selected_factor_keys={selected_factor_keys}, '
                f'formula={fusion_formula}, '
                f'fusion_info={fusion_info}, '
                f'fitness={fitness_payload}'
            )

        persisted = bool(len(selected_factor_keys) > 1 and persist_allowed)

        log.info(
            'Fusion finished: '
            f'selected_count={len(selected_factor_keys)}, '
            f'selected_factor_keys={selected_factor_keys}, '
            f'final_metrics={self._format_metrics(final_metrics)}'
        )
        return {
            'fusion_method': self.fusion_method,
            'fusion_metrics': list(self.fusion_metrics),
            'selected_factor_keys': selected_factor_keys,
            'selected_factors_detail': selected_factors_detail,
            'final_metrics': final_metrics,
            'final_metrics_outsample': final_metrics_outsample,
            'single_factor_metrics': single_metric_map,
            'leakage_check': leakage_result,
            'relative_check': relative_result,
            'fused_col': fusion_col,
            'fused_data': final_fused_df,
            'bt': final_bt,
            'round_base_metrics': base_metrics,
            'consider_outsample': bool(use_outsample),
            'outsample_start_day': self.outsample_start_day,
            'outsample_end_day': self.outsample_end_day,
            'persisted': persisted,
            'fusion_collection': fusion_collection,
            'fusion_factor_name': fusion_factor_name,
            'fusion_formula': fusion_formula,
            'fusion_info': fusion_info,
        }


