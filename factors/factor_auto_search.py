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
    get_futures_continuous_contract_price,
    get_latest_factor_formula_map,
    update_factor_info,
)
from mongo.mongify import get_data
from utils.logging import log
from utils.params import SUPPORTED_FILTER_INDICATORS

from .backtest import BackTester
from .factor_ops import available_operator_prompt_text, calc_formula_df, calc_formula_series
from .factor_utils import get_future_ret, rolling_normalize_features
from .gp_factor_engine import GPCandidate, run_gp_evolution


class FactorGenerator:
    """Generate factors, backtest, filter and persist factor formulas into DB."""

    method: str = 'base'

    def __init__(self,
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List[str]] = 'C0',
                 fc_freq: str = '1d',
                 data: Optional[pd.DataFrame] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'compound',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = False,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 200,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 10.0,
                 check_leakage_count: int = 20,
                 check_relative: bool = False,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 version: Optional[str] = None):
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

        required_cols = ['time', 'instrument_id'] + self.base_col_list
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
        if self.apply_rolling_norm:
            factor_cols = [c for c in factor_df.columns if c not in ['time', 'instrument_id']]
            factor_df = rolling_normalize_features(
                df=factor_df,
                factor_cols=list(factor_cols),
                rolling_norm_window=self.rolling_norm_window,
                rolling_norm_min_periods=self.rolling_norm_min_periods,
                rolling_norm_eps=self.rolling_norm_eps,
                rolling_norm_clip=self.rolling_norm_clip,
                instrument_col='instrument_id',
            )

        df_with_ret = get_future_ret(
            base_df[['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position']].copy(),
            portfolio_adjust_method=self.portfolio_adjust_method,
            rfr=self.risk_free_rate,
        )
        df_with_ret = df_with_ret[['time', 'instrument_id', 'future_ret']].copy()

        generated_data = df_with_ret.merge(factor_df, on=['time', 'instrument_id'], how='left', validate='1:1')
        generated_data = generated_data.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        self.generated_fc_name_list = [c for c in generated_data.columns if c not in ['time', 'instrument_id', 'future_ret']]
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
                                       filter_indicator_dict: Dict[str, Tuple[float, float, int]]) -> None:
        if not filter_indicator_dict:
            raise ValueError('filter_indicator_dict is required and cannot be empty.')
        available_indicator_list = list(SUPPORTED_FILTER_INDICATORS)
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
            _, _, direction = conf
            if direction not in [1, -1]:
                raise ValueError(f'Invalid direction for `{indicator}`: {direction}. Use 1 (>=) or -1 (<=).')

    def filter_fc_by_threshold(self,
                               performance_summary: Optional[pd.DataFrame] = None,
                               filter_indicator_dict: Dict[str, Tuple[float, float, int]] = None,
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

        indicator_list = list(filter_indicator_dict.keys())
        for indicator in indicator_list:
            if indicator not in summary_df.columns:
                raise ValueError(f'performance_summary does not contain required indicator: {indicator}')

        summary_df = summary_df.copy()
        summary_df['__year_str__'] = summary_df['year'].astype(str)
        summary_df['__is_yearly__'] = summary_df['__year_str__'] != 'all'

        for indicator, conf in filter_indicator_dict.items():
            _, yearly_threshold, direction = conf
            numeric_series = pd.to_numeric(summary_df[indicator], errors='coerce')
            pass_col = f'__pass_yearly__{indicator}'
            if direction == 1:
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

            for indicator, conf in filter_indicator_dict.items():
                mean_threshold, _, direction = conf
                yearly_values = pd.to_numeric(df_year[indicator], errors='coerce').dropna()
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

    def _build_metric_fitness_map(self,
                                  performance_summary: pd.DataFrame,
                                  selected_fc_name_list: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        summary_df = performance_summary.copy()
        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        metric_cols = [
            c for c in SUPPORTED_FILTER_INDICATORS
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
        fc_name_list = self.load_fc(config_ref=config_ref, instrument_id_list=self.instrument_id_list)
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
            selected_fc_name_list = [
                c for c in factor_df_for_names.columns if c not in ['time', 'instrument_id']
            ]
        if not selected_fc_name_list:
            raise ValueError('No factor columns available for leakage check.')

        full_factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        if self.apply_rolling_norm:
            full_factor_df = rolling_normalize_features(
                df=full_factor_df,
                factor_cols=list(selected_fc_name_list),
                rolling_norm_window=self.rolling_norm_window,
                rolling_norm_min_periods=self.rolling_norm_min_periods,
                rolling_norm_eps=self.rolling_norm_eps,
                rolling_norm_clip=self.rolling_norm_clip,
                instrument_col='instrument_id',
            )

        all_time_list = sorted(full_factor_df['time'].dropna().unique().tolist())
        if not all_time_list:
            raise ValueError('No valid time points available for leakage check.')

        sample_count = min(len(all_time_list), max(1, self.check_leakage_count))
        if sample_count < len(all_time_list):
            rng = np.random.default_rng()
            sampled_time_list = sorted(rng.choice(all_time_list, size=sample_count, replace=False).tolist())
        else:
            sampled_time_list = all_time_list

        slice_factor_list = []
        for t in sampled_time_list:
            log.info(f'Checking leakage for time slice <= {t}...')
            df_slice = base_df.loc[base_df['time'] <= t].copy()
            factor_df_slice = self.generate_factor_df(df_slice, selected_fc_names=selected_fc_name_list)
            if self.apply_rolling_norm:
                factor_df_slice = rolling_normalize_features(
                    df=factor_df_slice,
                    factor_cols=list(selected_fc_name_list),
                    rolling_norm_window=self.rolling_norm_window,
                    rolling_norm_min_periods=self.rolling_norm_min_periods,
                    rolling_norm_eps=self.rolling_norm_eps,
                    rolling_norm_clip=self.rolling_norm_clip,
                    instrument_col='instrument_id',
                )
            factor_df_slice = factor_df_slice.loc[
                factor_df_slice['time'] == t,
                ['time', 'instrument_id'] + selected_fc_name_list,
            ].copy()
            slice_factor_list.append(factor_df_slice)

        slice_factor_df = pd.concat(slice_factor_list, ignore_index=True)

        left = full_factor_df[['time', 'instrument_id'] + selected_fc_name_list].copy()
        left = left[left['time'].isin(sampled_time_list)]
        right = slice_factor_df[['time', 'instrument_id'] + selected_fc_name_list].copy()

        merged = left.merge(
            right,
            on=['time', 'instrument_id'],
            how='outer',
            suffixes=('_full', '_slice'),
            indicator=True,
        )

        missing_row_df = merged.loc[merged['_merge'] != 'both', ['time', 'instrument_id', '_merge']].copy()
        mismatch_detail: Dict[str, int] = {}
        mismatch_examples: Dict[str, List[dict]] = {}

        both_df = merged.loc[merged['_merge'] == 'both'].copy()
        for fc_name in selected_fc_name_list:
            col_full = f'{fc_name}_full'
            col_slice = f'{fc_name}_slice'
            is_equal = np.isclose(
                pd.to_numeric(both_df[col_full], errors='coerce').astype(float),
                pd.to_numeric(both_df[col_slice], errors='coerce').astype(float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            mismatch_mask = ~is_equal
            mismatch_count = int(mismatch_mask.sum())
            if mismatch_count > 0:
                mismatch_detail[fc_name] = mismatch_count
                mismatch_examples[fc_name] = both_df.loc[
                    mismatch_mask,
                    ['time', 'instrument_id', col_full, col_slice],
                ].head(5).to_dict(orient='records')

        failed_factor_set = set(mismatch_detail.keys())
        if len(missing_row_df) > 0:
            failed_factor_set.update(selected_fc_name_list)

        passed = len(failed_factor_set) == 0
        result = {
            'passed': passed,
            'checked_factor_count': len(selected_fc_name_list),
            'checked_time_count': len(sampled_time_list),
            'missing_row_count': int(len(missing_row_df)),
            'mismatch_factor_count': int(len(mismatch_detail)),
            'mismatch_detail': mismatch_detail,
            'mismatch_examples': mismatch_examples,
            'failed_factor_list': sorted(list(failed_factor_set)),
        }

        if not passed:
            log.error('Leakage check failed with details:')
            log.error(f'  checked_time_count={result["checked_time_count"]}')
            log.error(f'  missing_row_count={result["missing_row_count"]}')
            if len(missing_row_df) > 0:
                log.error(f'  missing_row_samples={missing_row_df.head(10).to_dict(orient="records")}')
            for fc_name in result['failed_factor_list']:
                if fc_name in mismatch_detail:
                    log.error(
                        f'  factor={fc_name}, mismatch_count={mismatch_detail[fc_name]}, '
                        f'samples={mismatch_examples.get(fc_name, [])}'
                    )

        if raise_error and not passed:
            raise ValueError(f'Leakage check failed: {result}')
        return result

    def auto_mine_select_and_save_fc(self,
                                     filter_indicator_dict: Dict[str, Tuple[float, float, int]],
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
            log.warning(msg)
            return {
                'config_ref': None,
                'config_path': None,
                'selected_fc_name_list': [],
                'bt': bt,
                'message': msg,
            }

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

        config_ref = self.save_fc(
            fc_name_list=selected_after_relative,
            performance_summary=bt.performance_summary,
        )

        return {
            'config_ref': config_ref,
            'config_path': config_ref,
            'selected_fc_name_list': selected_after_relative,
            'bt': bt,
            'leakage_check': leakage_result,
            'relative_check': relative_result,
        }

    def filter_fc_by_db_relative_spearman(self,
                                          selected_fc_name_list: Sequence[str],
                                          generated_df: pd.DataFrame,
                                          n_jobs: Optional[int] = None,
                                          batch_size: int = 100) -> Dict[str, Any]:
        """Filter factors by absolute Spearman correlation vs factors already in DB."""
        if not selected_fc_name_list:
            return {
                'enabled': True,
                'selected_fc_name_list': [],
                'filtered_out_fc_name_list': [],
                'checked_db_factor_count': 0,
                'threshold': self.relative_threshold,
                'versions': self.relative_check_version_list,
                'detail': {},
                'collection_count': 0,
            }

        candidate_df = generated_df[['time', 'instrument_id'] + list(selected_fc_name_list)].copy()
        candidate_df = candidate_df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        raw_records = get_factor_formula_records(
            collections=None,
            versions=self.relative_check_version_list,
            database='factors',
        )
        if raw_records.empty:
            log.info(f'Relative check skipped: no existing factors found in DB with version '
                     f'{self.relative_check_version_list}.')
            return {
                'enabled': True,
                'selected_fc_name_list': list(selected_fc_name_list),
                'filtered_out_fc_name_list': [],
                'checked_db_factor_count': 0,
                'threshold': self.relative_threshold,
                'versions': self.relative_check_version_list,
                'detail': {fc: {'max_abs_spearman': 0.0} for fc in selected_fc_name_list},
                'collection_count': 0,
            }

        raw_records = raw_records.copy()
        raw_records['factor_name'] = raw_records['factor_name'].astype(str)
        raw_records['version'] = raw_records['version'].astype(str)
        raw_records['collection'] = raw_records['collection'].astype(str)

        # Avoid self-comparison when current version already exists in target collection.
        same_run_mask = (raw_records['version'] == str(self.version)) & (raw_records['collection'] == self._db_collection())
        raw_records = raw_records.loc[~same_run_mask].copy()
        if raw_records.empty:
            return {
                'enabled': True,
                'selected_fc_name_list': list(selected_fc_name_list),
                'filtered_out_fc_name_list': [],
                'checked_db_factor_count': 0,
                'threshold': self.relative_threshold,
                'versions': self.relative_check_version_list,
                'detail': {fc: {'max_abs_spearman': 0.0} for fc in selected_fc_name_list},
                'collection_count': 0,
            }

        raw_records['db_factor_key'] = raw_records.apply(
            lambda x: f"{x['collection']}::{x['version']}::{x['factor_name']}", axis=1
        )
        raw_records = raw_records.drop_duplicates(subset=['db_factor_key'], keep='last').reset_index(drop=True)

        base_df = self.load_base_data()
        base_df = base_df.sort_values(['instrument_id', 'time']).reset_index(drop=True)
        key_df = candidate_df[['time', 'instrument_id']].copy()
        candidate_col_list = list(selected_fc_name_list)
        candidate_values_df = candidate_df[candidate_col_list].apply(pd.to_numeric, errors='coerce')

        detail_map: Dict[str, Dict[str, Any]] = {
            fc_name: {
                'max_abs_spearman': 0.0,
                'matched_db_factor': None,
            }
            for fc_name in candidate_col_list
        }

        def _eval_records_chunk(df_chunk: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
            formula_map = {
                row['db_factor_key']: row['formula']
                for _, row in df_chunk.iterrows()
                if isinstance(row.get('formula'), str) and row.get('formula', '').strip()
            }
            if not formula_map:
                return {}

            try:
                existing_df = calc_formula_df(df=base_df, formula_map=formula_map, data_fields=self.base_col_list)
            except Exception as e:
                log.warning(f'Relative check chunk skipped due to formula evaluation error: {e}')
                return {}

            if self.apply_rolling_norm:
                existing_df = rolling_normalize_features(
                    df=existing_df,
                    factor_cols=list(formula_map.keys()),
                    rolling_norm_window=self.rolling_norm_window,
                    rolling_norm_min_periods=self.rolling_norm_min_periods,
                    rolling_norm_eps=self.rolling_norm_eps,
                    rolling_norm_clip=self.rolling_norm_clip,
                    instrument_col='instrument_id',
                )

            aligned_existing = key_df.merge(existing_df, on=['time', 'instrument_id'], how='left', validate='1:1')
            existing_cols = list(formula_map.keys())
            existing_values_df = aligned_existing[existing_cols].apply(pd.to_numeric, errors='coerce')

            corr_mat = pd.concat([candidate_values_df, existing_values_df], axis=1).corr(method='spearman').abs()
            cross = corr_mat.loc[candidate_col_list, existing_cols].fillna(0.0)

            chunk_result: Dict[str, Dict[str, Any]] = {}
            for fc_name in candidate_col_list:
                row = cross.loc[fc_name]
                if row.empty:
                    continue
                max_key = row.idxmax()
                max_corr = float(row.loc[max_key])
                chunk_result[fc_name] = {
                    'max_abs_spearman': max_corr,
                    'matched_db_factor': max_key,
                }
            return chunk_result

        chunk_df_list = [
            raw_records.iloc[i:i + batch_size].copy()
            for i in range(0, len(raw_records), batch_size)
        ]
        effective_jobs = max(1, min(len(chunk_df_list), (n_jobs or self.n_jobs or 1)))
        if effective_jobs > 1 and len(chunk_df_list) > 1:
            chunk_result_list = Parallel(n_jobs=effective_jobs, prefer='threads')(
                delayed(_eval_records_chunk)(chunk_df)
                for chunk_df in chunk_df_list
            )
        else:
            chunk_result_list = [_eval_records_chunk(chunk_df) for chunk_df in chunk_df_list]

        for chunk_result in chunk_result_list:
            for fc_name, item in chunk_result.items():
                if item.get('max_abs_spearman', 0.0) > detail_map[fc_name]['max_abs_spearman']:
                    detail_map[fc_name] = item

        selected_fc = []
        filtered_out_fc = []
        for fc_name in candidate_col_list:
            max_corr = float(detail_map[fc_name].get('max_abs_spearman', 0.0))
            if max_corr < self.relative_threshold:
                selected_fc.append(fc_name)
            else:
                filtered_out_fc.append(fc_name)

        if filtered_out_fc:
            detail_text = {
                x: detail_map[x] for x in filtered_out_fc
            }
            log.warning(
                'Relative correlation filter removed factors: '
                f'threshold={self.relative_threshold}, removed={detail_text}'
            )

        return {
            'enabled': True,
            'selected_fc_name_list': selected_fc,
            'filtered_out_fc_name_list': filtered_out_fc,
            'checked_db_factor_count': int(len(raw_records)),
            'threshold': self.relative_threshold,
            'versions': self.relative_check_version_list,
            'detail': detail_map,
            'collection_count': int(raw_records['collection'].nunique()),
        }

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
                 interest_method: str = 'compound',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
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
                 check_relative: bool = False,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 model_name: str = 'deepseek',
                 llm_temperature: float = 0.7,
                 llm_factor_count: int = 5,
                 llm_early_stopping_round: int = 20,
                 llm_user_requirement: str = '生成期货的日频量价因子',
                 version: Optional[str] = None):
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

            formula_map = get_latest_factor_formula_map(
                fc_name_list=selected_fc_names,
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
            log.info(f'LLM prompt round {round_idx + 1}/{max_rounds}, requesting {this_round_count} formulas, ',
                     f'{len(valid_formula_list)}/{target_factor_count} valid formulas collected so far.')
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
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'compound',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 200,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 5.0,
                 check_leakage_count: int = 20,
                 check_relative: bool = False,
                 relative_threshold: float = 0.7,
                 relative_check_version_list: Optional[Sequence[str]] = None,
                 version: Optional[str] = None,
                 gp_generations: int = 50,
                 gp_population_size: int = 200,
                 gp_max_depth: int = 4,
                 gp_elite_size: int = 20,
                 gp_tournament_size: int = 6,
                 gp_crossover_prob: float = 0.7,
                 gp_mutation_prob: float = 0.25,
                 gp_leaf_prob: float = 0.2,
                 gp_const_prob: float = 0.02,
                 gp_window_choices: Optional[Sequence[int]] = None,
                 fitness_metric: str = 'ic',
                 random_seed: Optional[int] = None,
                 gp_early_stopping_generation_count: int = 8,
                 gp_depth_penalty_coef: float = 0.0,
                 gp_depth_penalty_start_depth: int = 3,
                 gp_depth_penalty_linear_coef: float = 0.0,
                 gp_depth_penalty_quadratic_coef: float = 0.0,
                 gp_log_interval: int = 5):
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
        self.gp_tournament_size = gp_tournament_size
        self.gp_crossover_prob = gp_crossover_prob
        self.gp_mutation_prob = gp_mutation_prob
        self.gp_leaf_prob = gp_leaf_prob
        self.gp_const_prob = gp_const_prob
        self.gp_window_choices = list(gp_window_choices) if gp_window_choices else [5, 10, 20]
        self.fitness_metric = fitness_metric
        self.random_seed = random_seed
        self.gp_early_stopping_generation_count = int(gp_early_stopping_generation_count)
        self.gp_depth_penalty_coef = float(gp_depth_penalty_coef)
        self.gp_depth_penalty_start_depth = int(gp_depth_penalty_start_depth)
        self.gp_depth_penalty_linear_coef = float(gp_depth_penalty_linear_coef)
        self.gp_depth_penalty_quadratic_coef = float(gp_depth_penalty_quadratic_coef)
        self.gp_log_interval = gp_log_interval

        self.factor_tree_map: Dict[str, Any] = {}

        assert self.fitness_metric in ['ic', 'sharpe'], \
            f'Unsupported fitness_metric={self.fitness_metric}, use ic/sharpe.'

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

        for idx, cand in enumerate(candidates, start=1):
            fc_name = f'fac_gp_{idx:04d}'
            signal = cand.node.calc(df_eval)
            signal = pd.to_numeric(signal, errors='coerce')
            factor_df[fc_name] = signal.values
            self.factor_tree_map[fc_name] = cand.node
            self.factor_formula_map[fc_name] = cand.formula
            self.factor_fitness_map[fc_name] = {
                self.fitness_metric: {
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

            formula_map = get_latest_factor_formula_map(
                fc_name_list=selected_fc_names,
                collections=['genetic_programming'],
                database='factors',
            )
            missing = [x for x in selected_fc_names if x not in formula_map]
            if missing:
                raise ValueError(f'GP formulas not found in DB for factors: {missing}')
            return calc_formula_df(df=df_eval, formula_map={k: formula_map[k] for k in selected_fc_names}, data_fields=self.base_col_list)

        limit = int(self.max_factor_count or 0)
        if limit <= 0:
            raise ValueError('max_factor_count must be positive for genetic programming.')

        candidates = run_gp_evolution(
            df=df_eval,
            data_fields=self.base_col_list,
            fitness_metric=self.fitness_metric,
            max_factor_count=limit,
            generations=self.gp_generations,
            population_size=self.gp_population_size,
            max_depth=self.gp_max_depth,
            elite_size=self.gp_elite_size,
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
            apply_rolling_norm=self.apply_rolling_norm,
            rolling_norm_window=self.rolling_norm_window,
            rolling_norm_min_periods=self.rolling_norm_min_periods,
            rolling_norm_eps=self.rolling_norm_eps,
            rolling_norm_clip=self.rolling_norm_clip,
        )

        if not candidates:
            raise ValueError('Genetic programming produced no valid candidates.')
        return self._build_factor_df_from_candidates(df_eval, candidates)

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        base_df = self.load_base_data()
        factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        return self._finalize_generated_data(base_df, factor_df)


