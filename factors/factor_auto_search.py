"""
Automatic factor generation utilities.
"""
import json
import re
import ast
import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

from data import get_futures_continuous_contract_price
from utils.logging import log

from .backtest import BackTester
from .factor_utils import get_future_ret, join_fc_name_and_parameter, rolling_normalize_features
from .gp_factor_engine import run_gp_evolution, GPCandidate
from stats import iterdict
from mongo.mongify import update_one_data


class FactorGenerator:
    """
    Generate factor values automatically and run backtests.

    The generated data is aligned with BackTester external-data format:
    `time`, `instrument_id`, `future_ret`, and factor columns.
    """

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

        self.version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        self.generated_data: Optional[pd.DataFrame] = None
        self.generated_fc_name_list: List[str] = []
        self.bt: Optional[BackTester] = None
        self.generated_factor_code_map: Dict[str, str] = {}

        assert self.fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, got {self.fc_freq}.'
        assert self.portfolio_adjust_method in ['min', '1D', '1M', '1Q'], \
            f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, got {self.portfolio_adjust_method}.'
        assert self.method in ['base', 'tsfresh', 'llm_prompt', 'genetic_programming'], \
            f'Unsupported method: {self.method}.'

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
                from_database=True
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
        """Best-effort load of .env from common project-root locations."""
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
                # Fallback parser for simple KEY=VALUE lines.
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
        """Subclass hook: build factor dataframe with columns [time, instrument_id, ...factors]."""
        raise NotImplementedError('Please implement generate_factor_df in subclasses.')

    @staticmethod
    def _expand_requested_factor_names(requested_names: Sequence[str],
                                       available_names: Sequence[str],
                                       separators: Sequence[str] = ('__', '_')) -> List[str]:
        """
        Expand base names to all parameterized columns.

        Example:
        - requested: ["fac_volatility_ratio"]
        - available: ["fac_volatility_ratio_a_5", "fac_volatility_ratio_a_10"]
        -> expanded to both parameterized names.
        """
        available_list = list(available_names)
        available_set = set(available_list)

        expanded: List[str] = []
        for req in requested_names:
            if req in available_set:
                if req not in expanded:
                    expanded.append(req)
                continue

            matches: List[str] = []
            for name in available_list:
                if any(name.startswith(f'{req}{sep}') for sep in separators):
                    matches.append(name)

            if not matches:
                preview = available_list[:20]
                raise ValueError(
                    f'Factor `{req}` is not available and cannot be expanded from existing factors. '
                    f'Available preview: {preview}'
                )

            for name in matches:
                if name not in expanded:
                    expanded.append(name)

        return expanded

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate factor values for subclass-specific method.
        """
        raise NotImplementedError('Please use a concrete subclass, e.g. TsfreshFactorGenerator or LLMPromptFactorGenerator.')

    def _finalize_generated_data(self,
                                 base_df: pd.DataFrame,
                                 factor_df: pd.DataFrame) -> pd.DataFrame:
        """Merge factor dataframe with return label and update object state."""
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
        """
        Generate factor values only for the given tsfresh feature names.

        This supports reusing previously saved feature names without extracting
        the full tsfresh feature set.
        """
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]
        if not fc_name_list:
            raise ValueError('fc_name_list is empty.')

        return self.generate(selected_fc_name_list=list(fc_name_list))

    def save_fc_value(self,
                      fc_name_list: Union[str, List[str]],
                      file_name: Optional[str] = None,
                      file_format: str = 'parquet') -> Path:
        """
        Save selected factor columns from latest generated data.
        """
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

    def save_fc(self,
                fc_name_list: Union[str, List[str]],
                save_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Save selected feature definitions for future reuse.

        Notes:
        - This method only writes one JSON config file.
        - tsfresh reuse is achieved by storing feature names and converting them
          back to tsfresh settings via `from_columns`.
        """
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]
        if not fc_name_list:
            raise ValueError('fc_name_list is empty.')

        if self.generated_fc_name_list:
            missing = [x for x in fc_name_list if x not in self.generated_fc_name_list]
            if missing:
                raise ValueError(f'fc_name_list contains unknown features: {missing}')

        if save_dir is None:
            if self.method == 'llm_prompt':
                default_dir_name = 'fc_from_llm'
            elif self.method == 'tsfresh':
                default_dir_name = 'fc_from_tsfresh'
            elif self.method == 'genetic_programming':
                default_dir_name = 'fc_from_genetic_programming'
            else:
                raise ValueError(f'save_fc does not support method={self.method}.')
            save_dir = Path(__file__).resolve().parent / default_dir_name
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.method == 'llm_prompt':
            package_prefix = 'llm_fc'
        elif self.method == 'tsfresh':
            package_prefix = 'tsfresh_fc'
        elif self.method == 'genetic_programming':
            package_prefix = 'gp_fc'
        else:
            raise ValueError(f'save_fc does not support method={self.method}.')
        fc_package_name = f'{package_prefix}_{self.version}'

        fc_name_list = list(fc_name_list)

        if self.method not in ['tsfresh', 'llm_prompt', 'genetic_programming']:
            raise ValueError(f'save_fc does not support method={self.method}.')

        def _build_payload(selected_fc_name_list: List[str], created_at: Optional[str] = None) -> dict:
            payload_i = {
                'method': self.method,
                'created_at': created_at or datetime.now().isoformat(timespec='seconds'),
                'version': self.version,
                'fc_name_list': list(selected_fc_name_list),
                # Keep only strict_meta related fields and basic traceability fields.
                'meta': {
                    'instrument_type': self.instrument_type,
                    'instrument_id_list': self.instrument_id_list,
                    'fc_freq': self.fc_freq,
                    'base_col_list': self.base_col_list,
                }
            }
            if self.method == 'tsfresh':
                from tsfresh.feature_extraction.settings import from_columns
                payload_i['kind_to_fc_parameters'] = from_columns(selected_fc_name_list)
            elif self.method == 'genetic_programming':
                formula_map = getattr(self, 'factor_formula_map', {})
                fitness_map = getattr(self, 'factor_fitness_map', {})
                payload_i['formula_map'] = {
                    k: formula_map[k] for k in selected_fc_name_list if isinstance(formula_map, dict) and k in formula_map
                }
                payload_i['fitness_map'] = {
                    k: fitness_map[k] for k in selected_fc_name_list if isinstance(fitness_map, dict) and k in fitness_map
                }
            return payload_i

        payload = _build_payload(fc_name_list)

        config_path = save_dir / f'{fc_package_name}.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_payload = json.load(f)

            existing_method = existing_payload.get('method')
            if existing_method and existing_method != self.method:
                raise ValueError(
                    f'Existing config method={existing_method} mismatches current method={self.method}: {config_path}'
                )

            existing_fc_name_list = existing_payload.get('fc_name_list', [])
            if not isinstance(existing_fc_name_list, list):
                existing_fc_name_list = []

            merged_fc_name_list = list(existing_fc_name_list)
            for fc_name in fc_name_list:
                if fc_name not in merged_fc_name_list:
                    merged_fc_name_list.append(fc_name)

            payload = _build_payload(
                merged_fc_name_list,
                created_at=existing_payload.get('created_at') if isinstance(existing_payload, dict) else None,
            )
            payload['updated_at'] = datetime.now().isoformat(timespec='seconds')

            added_count = len(merged_fc_name_list) - len(existing_fc_name_list)
            log.info(f'Existing config found, merged {added_count} new factors into {config_path}.')

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        log.info(f'Saved {self.method} feature config to {config_path}.')
        return config_path

    @staticmethod
    def load_fc(config_path: Union[str, Path]) -> List[str]:
        """
        Load saved tsfresh feature names from `save_fc` config.
        """
        config_path = Path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        fc_name_list = payload.get('fc_name_list', [])
        if not isinstance(fc_name_list, list) or not fc_name_list:
            raise ValueError(f'Invalid config file: {config_path}.')
        return fc_name_list

    def backtest_from_fc_config(self,
                                config_path: Union[str, Path],
                                n_jobs: Optional[int] = None,
                                strict_meta: bool = False) -> BackTester:
        """
        One-step workflow:
        load saved feature config -> generate selected factors -> run backtest.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with open(config_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        config_method = payload.get('method')
        if config_method and config_method != self.method:
            raise ValueError(
                f'Config method={config_method} does not match current generator method={self.method}. '
                'Please use the matching subclass.'
            )

        fc_name_list = payload.get('fc_name_list', [])
        if not isinstance(fc_name_list, list) or not fc_name_list:
            raise ValueError(f'Invalid config file: {config_path}. Missing non-empty fc_name_list.')

        if strict_meta:
            meta = payload.get('meta', {})
            if meta.get('fc_freq') and meta.get('fc_freq') != self.fc_freq:
                raise ValueError(
                    f'Config fc_freq={meta.get("fc_freq")} does not match current fc_freq={self.fc_freq}.'
                )
            if meta.get('instrument_type') and meta.get('instrument_type') != self.instrument_type:
                raise ValueError(
                    f'Config instrument_type={meta.get("instrument_type")} '
                    f'does not match current instrument_type={self.instrument_type}.'
                )
            if meta.get('base_col_list') and list(meta.get('base_col_list')) != list(self.base_col_list):
                raise ValueError(
                    'Config base_col_list does not match current base_col_list when strict_meta=True.'
                )

        self.generate_with_fc(fc_name_list)
        return self.backtest(fc_name_list=fc_name_list, n_jobs=n_jobs)

    def filter_fc_by_threshold(self,
                               performance_summary: Optional[pd.DataFrame] = None,
                               net_ret_threshold: float = 0.0,
                               sharpe_threshold: float = 0.5,
                               require_all_row: bool = True,
                               require_all_instruments: bool = True) -> List[str]:
        """
        Filter factors by net return and net sharpe thresholds.

        A factor is kept only if ALL required rows satisfy thresholds:
        - every year row
        - and row `year == 'all'` when require_all_row is True

        For multi-instrument backtest summary:
        - require_all_instruments=True: all instruments must pass
        - require_all_instruments=False: at least one instrument must pass
        """
        if performance_summary is None:
            if self.bt is None or self.bt.performance_summary is None:
                raise ValueError('No performance summary available. Please run backtest() first.')
            summary_df = self.bt.performance_summary.copy()
        else:
            summary_df = performance_summary.copy()

        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        required_cols = ['year', 'Factor Name', 'Net Return', 'Net Sharpe']
        for col in required_cols:
            if col not in summary_df.columns:
                raise ValueError(f'performance_summary does not contain required column: {col}')

        mask = (
            (summary_df['Net Return'] >= net_ret_threshold) &
            (summary_df['Net Sharpe'] >= sharpe_threshold)
        )
        summary_df = summary_df.assign(_pass=mask)

        selected_fc_name_list = []
        for fc_name, df_fc in summary_df.groupby('Factor Name', sort=False):
            if 'Instrument ID' in df_fc.columns:
                instrument_pass_list = []
                for _, df_ins in df_fc.groupby('Instrument ID', sort=False):
                    if require_all_row and 'all' not in df_ins['year'].astype(str).values:
                        instrument_pass_list.append(False)
                    else:
                        instrument_pass_list.append(bool(df_ins['_pass'].all()))

                if not instrument_pass_list:
                    continue
                fc_pass = all(instrument_pass_list) if require_all_instruments else any(instrument_pass_list)
            else:
                if require_all_row and 'all' not in df_fc['year'].astype(str).values:
                    continue
                fc_pass = bool(df_fc['_pass'].all())

            if fc_pass:
                selected_fc_name_list.append(fc_name)

        return selected_fc_name_list

    def auto_mine_select_and_save_fc(self,
                                     net_ret_threshold: float,
                                     sharpe_threshold: float,
                                     save_dir: Optional[Union[str, Path]] = None,
                                     n_jobs: Optional[int] = None,
                                     require_all_row: bool = True,
                                     require_all_instruments: bool = True) -> dict:
        """
        One-step automation:
        generate factors -> backtest -> threshold filter -> save selected config.
        """
        generated_df = self.generate()
        bt = self.backtest(data=generated_df, fc_name_list=self.generated_fc_name_list, n_jobs=n_jobs)
        selected_fc_name_list = self.filter_fc_by_threshold(
            performance_summary=bt.performance_summary,
            net_ret_threshold=net_ret_threshold,
            sharpe_threshold=sharpe_threshold,
            require_all_row=require_all_row,
            require_all_instruments=require_all_instruments,
        )

        if not selected_fc_name_list:
            msg = (
                f'No factors passed thresholds: net_ret_threshold={net_ret_threshold}, '
                f'sharpe_threshold={sharpe_threshold}.'
            )
            print(msg)
            log.warning(msg)
            return {
                'config_path': None,
                'selected_fc_name_list': [],
                'bt': bt,
                'message': msg,
            }

        config_path = self.save_fc(
            fc_name_list=selected_fc_name_list,
            save_dir=save_dir,
        )

        return {
            'config_path': config_path,
            'selected_fc_name_list': selected_fc_name_list,
            'bt': bt,
        }

    def check_if_leakage(self,
                         fc_name_list: Optional[Union[str, List[str]]] = None,
                         atol: float = 1e-10,
                         rtol: float = 1e-8,
                         raise_error: bool = True) -> dict:
        """
        Strict leakage check by expanding-time-slice recomputation.

        Procedure:
        1) Compute full-sample factor series.
        2) For each day t, recompute factors using only data <= t, and take factor values at t.
        3) Compare the two series day-by-day for each factor.
        """
        if isinstance(fc_name_list, str):
            fc_name_list = [fc_name_list]

        base_df = self.load_base_data()

        # Resolve feature list to check.
        if fc_name_list:
            selected_fc_name_list = list(fc_name_list)
        else:
            factor_df_for_names = self.generate_factor_df(base_df)
            selected_fc_name_list = [
                c for c in factor_df_for_names.columns if c not in ['time', 'instrument_id']
            ]

        if not selected_fc_name_list:
            raise ValueError('No factor columns available for leakage check.')

        # Full-sample factor series.
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

        check_time_list = sorted(full_factor_df['time'].dropna().unique().tolist())
        if not check_time_list:
            raise ValueError('No valid time points available for leakage check.')

        # Expanding-slice factor series: for each day, only use data up to that day.
        slice_factor_list = []
        for t in check_time_list:
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
                ['time', 'instrument_id'] + selected_fc_name_list
            ].copy()
            slice_factor_list.append(factor_df_slice)

        slice_factor_df = pd.concat(slice_factor_list, ignore_index=True)

        left = full_factor_df[['time', 'instrument_id'] + selected_fc_name_list].copy()
        right = slice_factor_df[['time', 'instrument_id'] + selected_fc_name_list].copy()

        merged = left.merge(
            right,
            on=['time', 'instrument_id'],
            how='outer',
            suffixes=('_full', '_slice'),
            indicator=True,
        )

        missing_row_df = merged.loc[merged['_merge'] != 'both', ['time', 'instrument_id', '_merge']].copy()
        mismatch_detail = {}

        both_df = merged.loc[merged['_merge'] == 'both'].copy()
        for fc_name in selected_fc_name_list:
            col_full = f'{fc_name}_full'
            col_slice = f'{fc_name}_slice'
            is_equal = np.isclose(
                both_df[col_full].astype(float),
                both_df[col_slice].astype(float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            mismatch_count = int((~is_equal).sum())
            if mismatch_count > 0:
                mismatch_detail[fc_name] = mismatch_count

        passed = (len(missing_row_df) == 0) and (len(mismatch_detail) == 0)
        result = {
            'passed': passed,
            'checked_factor_count': len(selected_fc_name_list),
            'checked_time_count': len(check_time_list),
            'missing_row_count': int(len(missing_row_df)),
            'mismatch_factor_count': int(len(mismatch_detail)),
            'mismatch_detail': mismatch_detail,
        }

        if raise_error and not passed:
            raise ValueError(f'Leakage check failed: {result}')

        return result

    def backtest(self,
                 data: Optional[pd.DataFrame] = None,
                 fc_name_list: Optional[Union[str, List[str]]] = None,
                 n_jobs: Optional[int] = None) -> BackTester:
        """
        Run backtest with generated or external factor dataframe.
        """
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


class TsfreshFactorGenerator(FactorGenerator):
    """Factor generator using tsfresh feature extraction."""

    method: str = 'tsfresh'

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
                 tsfresh_profile: str = 'minimal',
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 10.0,
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
            version=version,
        )
        self.tsfresh_profile = tsfresh_profile

    @staticmethod
    def _get_tsfresh_fc_parameters(profile: str):
        from tsfresh.feature_extraction import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )

        profile = profile.lower()
        if profile == 'minimal':
            return MinimalFCParameters()
        if profile == 'efficient':
            return EfficientFCParameters()
        if profile == 'comprehensive':
            return ComprehensiveFCParameters()
        raise ValueError(f'Unsupported tsfresh_profile: {profile}. Use minimal/efficient/comprehensive.')

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        from tsfresh import extract_features
        from tsfresh.feature_extraction.settings import from_columns

        fc_parameters = self._get_tsfresh_fc_parameters(self.tsfresh_profile)

        window_meta_list = []
        long_df_list = []

        for instrument_id, df_i in df.groupby('instrument_id', sort=False):
            df_i = df_i.sort_values('time').reset_index(drop=True)
            if len(df_i) < self.min_window_size:
                log.warning(f'Skip instrument {instrument_id}: insufficient rows ({len(df_i)} < {self.min_window_size}).')
                continue

            for end_idx in range(self.min_window_size - 1, len(df_i)):
                window_id = f'{instrument_id}__{int(df_i.loc[end_idx, "time"].value)}'
                window_df = df_i.iloc[end_idx - self.min_window_size + 1: end_idx + 1]
                window_meta_list.append({
                    'window_id': window_id,
                    'time': df_i.loc[end_idx, 'time'],
                    'instrument_id': instrument_id,
                })

                for col in self.base_col_list:
                    long_df = pd.DataFrame({
                        'id': window_id,
                        'time': range(self.min_window_size),
                        'kind': col,
                        'value': window_df[col].values,
                    })
                    long_df_list.append(long_df)

        if not long_df_list:
            raise ValueError('No rolling windows generated for tsfresh extraction. Please check input size.')

        def _normalize_extracted_df(extracted_df: pd.DataFrame) -> pd.DataFrame:
            extracted_df = extracted_df.reset_index()
            if 'id' in extracted_df.columns:
                extracted_df = extracted_df.rename(columns={'id': 'window_id'})
            elif 'index' in extracted_df.columns:
                extracted_df = extracted_df.rename(columns={'index': 'window_id'})
            else:
                extracted_df = extracted_df.rename(columns={extracted_df.columns[0]: 'window_id'})
            return extracted_df

        tsfresh_input = pd.concat(long_df_list, ignore_index=True)
        selected_extraction_failed = False
        if selected_fc_names:
            try:
                extracted = extract_features(
                    tsfresh_input,
                    column_id='id',
                    column_sort='time',
                    column_kind='kind',
                    column_value='value',
                    kind_to_fc_parameters=from_columns(selected_fc_names),
                    disable_progressbar=True,
                    n_jobs=self.n_jobs,
                )
            except Exception:
                selected_extraction_failed = True
                extracted = pd.DataFrame(columns=['window_id'])
        else:
            extracted = extract_features(
                tsfresh_input,
                column_id='id',
                column_sort='time',
                column_kind='kind',
                column_value='value',
                default_fc_parameters=dict(fc_parameters),
                disable_progressbar=True,
                n_jobs=self.n_jobs,
            )
        extracted = _normalize_extracted_df(extracted)

        df_meta = pd.DataFrame(window_meta_list)
        df_feature = df_meta.merge(extracted, on='window_id', how='left', validate='1:1')

        factor_cols = [c for c in df_feature.columns if c not in ['window_id', 'time', 'instrument_id']]
        if not factor_cols:
            raise ValueError('No factor columns were generated by tsfresh.')

        if selected_fc_names is not None:
            requested_fc_names = list(selected_fc_names)
            unresolved = selected_extraction_failed or any(col not in df_feature.columns for col in requested_fc_names)
            if unresolved:
                # Fallback to full extraction so base names (without explicit params)
                # can be expanded into all matching parameter combinations.
                extracted_full = extract_features(
                    tsfresh_input,
                    column_id='id',
                    column_sort='time',
                    column_kind='kind',
                    column_value='value',
                    default_fc_parameters=dict(fc_parameters),
                    disable_progressbar=True,
                    n_jobs=self.n_jobs,
                )
                extracted_full = _normalize_extracted_df(extracted_full)
                df_feature = df_meta.merge(extracted_full, on='window_id', how='left', validate='1:1')

            available_cols = [c for c in df_feature.columns if c not in ['window_id', 'time', 'instrument_id']]
            resolved_fc_names = self._expand_requested_factor_names(
                requested_fc_names,
                available_cols,
                separators=('__', '_'),
            )

            df_feature = df_feature[['time', 'instrument_id'] + resolved_fc_names].copy()
            df_feature = df_feature.sort_values(['instrument_id', 'time']).reset_index(drop=True)
            return df_feature

        valid_cols = []
        for col in factor_cols:
            series = df_feature[col]
            if series.isna().all():
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            valid_cols.append(col)

        if not valid_cols:
            raise ValueError('All generated tsfresh features are empty/constant after filtering.')

        if self.max_factor_count is not None and len(valid_cols) > self.max_factor_count:
            valid_cols = valid_cols[:self.max_factor_count]

        df_feature = df_feature[['time', 'instrument_id'] + valid_cols].copy()
        df_feature = df_feature.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        return df_feature

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        base_df = self.load_base_data()
        factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        return self._finalize_generated_data(base_df, factor_df)


class LLMPromptFactorGenerator(FactorGenerator):
    """Factor generator using LLM prompt-based factor synthesis."""

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
                 calculate_baseline: bool = False,
                 n_jobs: int = 5,
                 base_col_list: Optional[Sequence[str]] = None,
                 min_window_size: int = 30,
                 max_factor_count: Optional[int] = 50,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 10.0,
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
            version=version,
        )
        self.model_name = model_name
        self.llm_temperature = llm_temperature
        self.llm_factor_count = llm_factor_count
        self.llm_early_stopping_round = llm_early_stopping_round
        self.llm_user_requirement = llm_user_requirement

    @staticmethod
    def get_llm(temperature: float = 0.7, model_name: Optional[str] = None):
        """Load chat LLM instance for prompt-based factor generation."""
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

    def _build_llm_few_shot_prompt(self, factor_count: int) -> str:
        example_code = (
            "class fac_winrate:\n"
            "    param_range = {'a': [10, 20, 30]}\n\n"
            "    @staticmethod\n"
            "    def operate(Data: pd.DataFrame, **kwargs):\n"
            "        hash_tb = {chr(i): 0 for i in range(97, 123)}\n"
            "        for key, value in kwargs.items():\n"
            "            hash_tb[key] = value\n"
            "        a = int(hash_tb['a'])\n\n"
            "        df = Data.copy()\n"
            "        df['ret'] = df['close'].pct_change()\n"
            "        df['win_ratio'] = (df['ret'] > 0) * 1\n"
            "        df['win_ratio'] = df['win_ratio'].rolling(a).mean()\n\n"
            "        return df['win_ratio']\n"
        )
        return (
            '你是量化研究助理。'
            '请用 Python 生成简单的期货日频因子类。'
            '输出必须是严格 JSON，格式如下：'
            '{"factors": [{"fc_name": "fac_xxx", "code": "class fac_xxx: ..."}, ...]}。'
            '规则：'
            '1）fc_name 必须以 fac_ 开头。'
            '2）class 代码必须包含 param_range(dict) 和 @staticmethod operate(Data: pd.DataFrame, **kwargs)。'
            '3）operate 只能使用历史/当前行，禁止未来函数（例如 shift(-k)）。'
            '4）因子逻辑基于 open/high/low/close/volume/position这些量价字段。'
            f'5）必须返回且仅返回 {factor_count} 个因子。'
            '6）不要输出 markdown 代码块或额外解释。'
            f'Few-shot 示例：\n{example_code}\n'
            f'额外需求：{self.llm_user_requirement}'
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

    @staticmethod
    def _first_parameter(param_range: dict) -> dict:
        if not param_range:
            return {}
        first_param = {}
        for k, v in param_range.items():
            if isinstance(v, list) and len(v) > 0:
                first_param[k] = v[0]
            else:
                raise ValueError(f'Invalid param_range for key `{k}`: {v}')
        return first_param

    def _validate_llm_factor_class(self, fc_name: str, code: str, sample_df: pd.DataFrame):
        ast.parse(code)
        local_ns: Dict[str, object] = {}
        exec(code, {'pd': pd, 'np': np}, local_ns)
        fc_class = cast(Any, local_ns.get(fc_name))
        if fc_class is None or not isinstance(fc_class, type):
            raise ValueError(f'LLM code does not define class `{fc_name}`.')
        if not hasattr(fc_class, 'param_range'):
            raise ValueError(f'Factor class `{fc_name}` missing param_range.')
        if not hasattr(fc_class, 'operate'):
            raise ValueError(f'Factor class `{fc_name}` missing operate method.')

        first_param = self._first_parameter(fc_class.param_range)
        output = fc_class.operate(sample_df.copy(), **first_param)
        if output is None:
            raise ValueError(f'Factor class `{fc_name}` returned None.')
        output_series = pd.Series(output)
        if len(output_series) != len(sample_df):
            raise ValueError(
                f'Factor class `{fc_name}` output length mismatch: {len(output_series)} vs {len(sample_df)}'
            )
        return fc_class

    @staticmethod
    def _save_valid_llm_factor_code(valid_factor_items: List[Tuple[str, str]]):
        file_path = Path(__file__).resolve().parent / 'factor_from_llm.py'
        if file_path.exists():
            existing = file_path.read_text(encoding='utf-8')
        else:
            existing = '"""LLM-generated factors."""\nimport pandas as pd\nimport numpy as np\n\n'

        append_blocks = []
        for fc_name, code in valid_factor_items:
            if f'class {fc_name}' in existing:
                continue
            append_blocks.append(f'\n\n{code.strip()}\n')

        if append_blocks:
            file_path.write_text(existing + ''.join(append_blocks), encoding='utf-8')
        return file_path

    @staticmethod
    def _load_all_llm_factor_classes() -> Dict[str, Any]:
        module = importlib.import_module('factors.factor_from_llm')
        module = importlib.reload(module)
        fc_class_map = {}
        for name, obj in vars(module).items():
            if name.startswith('fac_') and isinstance(obj, type) and hasattr(obj, 'param_range') and hasattr(obj, 'operate'):
                fc_class_map[name] = obj
        if not fc_class_map:
            raise ValueError('No valid factor classes found in factors/factor_from_llm.py.')
        return fc_class_map

    def _compute_factor_df_from_classes(self,
                                        df: pd.DataFrame,
                                        fc_class_map: Dict[str, Any],
                                        selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        out_df = df[['time', 'instrument_id']].copy().reset_index(drop=True)

        column_mapping: Dict[str, Tuple[str, dict]] = {}
        for fc_name, fc_class in fc_class_map.items():
            for parameter in iterdict(fc_class.param_range):
                col_name = join_fc_name_and_parameter(fc_name, parameter)
                column_mapping[col_name] = (fc_name, parameter)

        if selected_fc_names is not None:
            target_columns = self._expand_requested_factor_names(
                selected_fc_names,
                list(column_mapping.keys()),
                separators=('_',),
            )
        else:
            target_columns = list(column_mapping.keys())

        grouped_indices = df.groupby('instrument_id', sort=False).groups
        for col_name in target_columns:
            if col_name not in column_mapping:
                raise ValueError(f'Factor column `{col_name}` is not defined by available LLM factors.')
            fc_name, parameter = column_mapping[col_name]
            fc_class = fc_class_map[fc_name]
            col_values = pd.Series(np.nan, index=df.index, dtype=float)
            for _, idx in grouped_indices.items():
                idx_list = list(idx)
                df_i = df.loc[idx_list].copy().reset_index(drop=True)
                signal = fc_class.operate(df_i, **parameter)
                signal = pd.Series(signal).reset_index(drop=True)
                if len(signal) != len(df_i):
                    raise ValueError(f'Factor `{fc_name}` generated invalid length for parameter {parameter}.')
                col_values.loc[idx_list] = signal.values
            out_df[col_name] = col_values.values

        return out_df.sort_values(['instrument_id', 'time']).reset_index(drop=True)

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        if selected_fc_names is not None:
            fc_class_map = self._load_all_llm_factor_classes()
            return self._compute_factor_df_from_classes(df, fc_class_map, selected_fc_names=selected_fc_names)

        llm = self.get_llm(temperature=self.llm_temperature, model_name=self.model_name)
        target_factor_count = self.max_factor_count if self.max_factor_count is not None else self.llm_factor_count
        if target_factor_count <= 0:
            raise ValueError('max_factor_count must be positive for method=llm_prompt.')

        max_rounds = max(3, int(np.ceil(target_factor_count / max(1, self.llm_factor_count))) * 3)
        sample_df = df.groupby('instrument_id', sort=False).head(max(30, self.min_window_size)).copy()
        valid_code_map: Dict[str, str] = {}
        fc_class_map: Dict[str, Any] = {}
        no_growth_rounds = 0
        early_stopped = False

        for round_idx in range(max_rounds):
            if len(fc_class_map) >= target_factor_count:
                break

            remaining = target_factor_count - len(fc_class_map)
            this_round_count = min(max(1, self.llm_factor_count), remaining)
            prompt = self._build_llm_few_shot_prompt(this_round_count)
            if valid_code_map:
                existing_names = ', '.join(valid_code_map.keys())
                prompt += f'\n已生成的因子名（请勿重复）：{existing_names}'

            try:
                response = llm.invoke(prompt)
                content = getattr(response, 'content', '')
                payload = json.loads(self._extract_json_text(content))
            except Exception as e:
                log.warning(f'Round {round_idx + 1}/{max_rounds}: invalid LLM response, skip. {e}')
                continue

            factors = payload.get('factors', [])
            if not isinstance(factors, list) or not factors:
                log.warning(f'Round {round_idx + 1}/{max_rounds}: empty factors in LLM output.')
                continue

            prev_valid_count = len(fc_class_map)
            for item in factors:
                if not isinstance(item, dict):
                    log.warning(
                        f'Round {round_idx + 1}/{max_rounds}: skip factor item because it is not a dict: {type(item).__name__}'
                    )
                    continue

                fc_name = item.get('fc_name')
                code = item.get('code')
                if not isinstance(fc_name, str) or not fc_name.startswith('fac_'):
                    log.warning(
                        f'Round {round_idx + 1}/{max_rounds}: skip factor due to invalid fc_name={fc_name!r}. '
                        'Expected string starting with `fac_`.'
                    )
                    continue
                if fc_name in valid_code_map:
                    log.warning(
                        f'Round {round_idx + 1}/{max_rounds}: skip duplicated factor name `{fc_name}`.'
                    )
                    continue
                if not isinstance(code, str) or f'class {fc_name}' not in code:
                    log.warning(
                        f'Round {round_idx + 1}/{max_rounds}: skip `{fc_name}` due to invalid code content. '
                        f'Need string containing `class {fc_name}`.'
                    )
                    continue
                try:
                    fc_class = self._validate_llm_factor_class(fc_name, code, sample_df)
                except Exception as e:
                    log.warning(f'Skip invalid LLM factor `{fc_name}`: {e}')
                    continue
                valid_code_map[fc_name] = code
                fc_class_map[fc_name] = fc_class
                if len(fc_class_map) >= target_factor_count:
                    break

            added_count_this_round = len(fc_class_map) - prev_valid_count
            if added_count_this_round <= 0:
                no_growth_rounds += 1
            else:
                no_growth_rounds = 0

            log.info(f'Round {round_idx + 1}/{max_rounds}: got {len(factors)} factors, {len(fc_class_map)} valid so far.')

            if 0 < self.llm_early_stopping_round <= no_growth_rounds:
                early_stopped = True
                log.warning(
                    f'LLM early stop triggered: no valid-factor growth for {no_growth_rounds} consecutive rounds. '
                    f'Current valid factors: {len(fc_class_map)} / target {target_factor_count}.'
                )
                break

        if not fc_class_map:
            raise ValueError('No valid factors generated by LLM prompt.')
        if len(fc_class_map) < target_factor_count:
            stop_reason = 'early_stop' if early_stopped else 'max_rounds_reached'
            log.warning(
                f'LLM generation ended early: got {len(fc_class_map)} valid factors, '
                f'less than target {target_factor_count} after {max_rounds} rounds. '
                f'stop_reason={stop_reason}.'
            )

        valid_items = [(name, valid_code_map[name]) for name in fc_class_map.keys()]
        self.generated_factor_code_map = {name: code for name, code in valid_items}
        self._save_valid_llm_factor_code(valid_items)
        return self._compute_factor_df_from_classes(df, fc_class_map)

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
                 gp_depth_penalty_coef: float = 0.0,
                 gp_depth_penalty_start_depth: int = 3,
                 gp_depth_penalty_linear_coef: float = 0.0,
                 gp_depth_penalty_quadratic_coef: float = 0.0,
                 gp_log_interval: int = 5):
        """
        Genetic programming key controls:

        - `max_factor_count`: keep top-N candidates after full evolution.
        - `gp_generations`: number of evolution iterations.
        - `gp_population_size`: number of individuals per generation.
        - `gp_max_depth`: max AST depth to limit formula complexity.
        - `gp_elite_size`: top-K individuals copied directly to next generation.
        - `gp_tournament_size`: candidate pool size in tournament selection.
        - `gp_crossover_prob` / `gp_mutation_prob`: probabilities for genetic operators.
        - `gp_leaf_prob`: probability to stop expanding and create a leaf node.
        - `gp_const_prob`: probability of using constant leaves vs data-field leaves.
        - `gp_window_choices`: allowed rolling windows for time-series operators.
        - `fitness_metric`: objective for evolution, currently `ic` or `sharpe`.
        - `random_seed`: random seed for reproducible evolution.
        - `gp_depth_penalty_coef`: base depth regularization coefficient.
          base_penalty = gp_depth_penalty_coef * tree_depth.
        - `gp_depth_penalty_start_depth`: dynamic penalty starts after this depth.
        - `gp_depth_penalty_linear_coef`: linear dynamic penalty slope.
        - `gp_depth_penalty_quadratic_coef`: quadratic dynamic penalty slope.
          total_penalty = base_penalty + linear_coef * extra_depth + quadratic_coef * extra_depth^2,
          where extra_depth = max(tree_depth - start_depth, 0).
        - `gp_log_interval`: log progress every N generations.
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
            n_jobs=n_jobs,
            base_col_list=base_col_list,
            min_window_size=min_window_size,
            max_factor_count=max_factor_count,
            apply_rolling_norm=apply_rolling_norm,
            rolling_norm_window=rolling_norm_window,
            rolling_norm_min_periods=rolling_norm_min_periods,
            rolling_norm_eps=rolling_norm_eps,
            rolling_norm_clip=rolling_norm_clip,
            version=version,
        )

        # Evolution loop controls
        self.gp_generations = gp_generations
        self.gp_population_size = gp_population_size
        self.gp_max_depth = gp_max_depth

        # Selection and reproduction controls
        self.gp_elite_size = gp_elite_size
        self.gp_tournament_size = gp_tournament_size
        self.gp_crossover_prob = gp_crossover_prob
        self.gp_mutation_prob = gp_mutation_prob

        # Tree-shape / primitive sampling controls
        self.gp_leaf_prob = gp_leaf_prob
        self.gp_const_prob = gp_const_prob
        self.gp_window_choices = list(gp_window_choices) if gp_window_choices else [5, 10, 20]

        # Objective and reproducibility controls
        self.fitness_metric = fitness_metric
        self.random_seed = random_seed
        self.gp_depth_penalty_coef = float(gp_depth_penalty_coef)
        self.gp_depth_penalty_start_depth = int(gp_depth_penalty_start_depth)
        self.gp_depth_penalty_linear_coef = float(gp_depth_penalty_linear_coef)
        self.gp_depth_penalty_quadratic_coef = float(gp_depth_penalty_quadratic_coef)
        self.gp_log_interval = gp_log_interval

        self.factor_tree_map: Dict[str, Any] = {}
        self.factor_formula_map: Dict[str, str] = {}
        self.factor_fitness_map: Dict[str, Dict[str, float]] = {}

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
            self.factor_fitness_map[fc_name] = {self.fitness_metric: float(cand.fitness)}

        return factor_df

    def generate_factor_df(self,
                           df: pd.DataFrame,
                           selected_fc_names: Optional[List[str]] = None) -> pd.DataFrame:
        df_eval = self._prepare_df_for_gp(df)

        if selected_fc_names is not None:
            if not self.factor_tree_map:
                raise ValueError('No GP factor tree cache found. Please run generate() first for this object.')
            resolved_names = self._expand_requested_factor_names(
                selected_fc_names,
                list(self.factor_tree_map.keys()),
                separators=('_',),
            )
            factor_df = df_eval[['time', 'instrument_id']].copy()
            for fc_name in resolved_names:
                signal = self.factor_tree_map[fc_name].calc(df_eval)
                factor_df[fc_name] = pd.to_numeric(signal, errors='coerce').values
            return factor_df

        limit = int(self.max_factor_count or 0)
        if limit <= 0:
            raise ValueError('max_factor_count must be positive for genetic programming.')

        log.info(
            f'GeneticFactorGenerator generate start: instrument_id_list={self.instrument_id_list}, '
            f'start_time={self.start_time}, end_time={self.end_time}, max_factor_count={limit}, '
            f'fitness_metric={self.fitness_metric}, gp_generations={self.gp_generations}, '
            f'gp_population_size={self.gp_population_size}, '
            f'gp_depth_penalty_coef={self.gp_depth_penalty_coef}, '
            f'gp_depth_penalty_start_depth={self.gp_depth_penalty_start_depth}, '
            f'gp_depth_penalty_linear_coef={self.gp_depth_penalty_linear_coef}, '
            f'gp_depth_penalty_quadratic_coef={self.gp_depth_penalty_quadratic_coef}'
        )

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
        log.info(f'GeneticFactorGenerator generate finished: candidate_count={len(candidates)}')
        return self._build_factor_df_from_candidates(df_eval, candidates)

    def generate(self,
                 selected_fc_name_list: Optional[List[str]] = None) -> pd.DataFrame:
        base_df = self.load_base_data()
        factor_df = self.generate_factor_df(base_df, selected_fc_names=selected_fc_name_list)
        return self._finalize_generated_data(base_df, factor_df)

    def _save_selected_factors_to_database(self,
                                           selected_fc_name_list: List[str],
                                           performance_summary: Optional[pd.DataFrame]) -> None:
        if not selected_fc_name_list:
            return
        if performance_summary is None:
            return

        summary_df = performance_summary.copy()
        if 'year' not in summary_df.columns:
            summary_df = summary_df.reset_index()

        for fc_name in selected_fc_name_list:
            formula = self.factor_formula_map.get(fc_name)
            fitness_dict = self.factor_fitness_map.get(fc_name)
            if formula is None or fitness_dict is None:
                continue

            df_fc = summary_df.loc[summary_df['Factor Name'] == fc_name].copy()
            if 'Instrument ID' in df_fc.columns:
                instrument_ids = [x for x in df_fc['Instrument ID'].dropna().unique().tolist()]
            else:
                instrument_ids = list(self.instrument_id_list)
            if not instrument_ids:
                instrument_ids = list(self.instrument_id_list)

            for ins_id in instrument_ids:
                record = {
                    'method': self.method,
                    'version': self.version,
                    'factor_name': fc_name,
                    'instrument_id': ins_id,
                    'start_date': self.start_time,
                    'end_date': self.end_time,
                    'fitness': {k: float(v) for k, v in fitness_dict.items() if v is not None and not pd.isna(v)},
                    'formula': formula,
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                }
                mongo_operator = {
                    'version': self.version,
                    'factor_name': fc_name,
                    'instrument_id': ins_id,
                    'start_date': self.start_time,
                    'end_date': self.end_time,
                }
                update_one_data(
                    database='factors',
                    collection='genetic_programming',
                    mongo_operator=mongo_operator,
                    data=record,
                    upsert=True,
                )

    def auto_mine_select_and_save_fc(self,
                                     net_ret_threshold: float,
                                     sharpe_threshold: float,
                                     save_dir: Optional[Union[str, Path]] = None,
                                     n_jobs: Optional[int] = None,
                                     require_all_row: bool = True,
                                     require_all_instruments: bool = True) -> dict:
        result = super().auto_mine_select_and_save_fc(
            net_ret_threshold=net_ret_threshold,
            sharpe_threshold=sharpe_threshold,
            save_dir=save_dir,
            n_jobs=n_jobs,
            require_all_row=require_all_row,
            require_all_instruments=require_all_instruments,
        )
        self._save_selected_factors_to_database(
            selected_fc_name_list=result.get('selected_fc_name_list', []),
            performance_summary=result.get('bt').performance_summary if result.get('bt') is not None else None,
        )
        return result


class FactorFusioner:
    """
    Fuse factors from saved configs (tsfresh + llm_prompt) by version.

    Current supported fusion strategy:
    - average_weight: equal-weight average of all loaded factor columns.
    """

    def __init__(self,
                 version_list: Union[str, List[str]],
                 fusion_strategy: str = 'average_weight',
                 fused_fc_name_suffix: Optional[str] = None,
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
                 apply_rolling_norm: bool = True,  # 融合前的每个因子是否进行rolling norm
                 apply_rolling_norm_after_fusion: bool = False,  # 融合后的因子是否再次rolling norm
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 10.0):
        self.version_list = [version_list] if isinstance(version_list, str) else list(version_list)
        self.fusion_strategy = fusion_strategy
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

        # Keep generator-related knobs so fused run stays consistent with source factor generation.
        self.base_col_list = list(base_col_list) if base_col_list else ['open', 'high', 'low', 'close', 'volume', 'position']
        self.min_window_size = min_window_size
        self.apply_rolling_norm = apply_rolling_norm
        self.apply_rolling_norm_after_fusion = apply_rolling_norm_after_fusion
        self.rolling_norm_window = rolling_norm_window
        self.rolling_norm_min_periods = rolling_norm_min_periods
        self.rolling_norm_eps = rolling_norm_eps
        self.rolling_norm_clip = rolling_norm_clip

        self.generated_data: Optional[pd.DataFrame] = None
        self.generated_fc_name_list: List[str] = []
        self.bt: Optional[BackTester] = None
        self.loaded_config_info_list: List[Dict[str, Any]] = []
        self.fused_fc_name_suffix = fused_fc_name_suffix
        self.raw_fc_value = None
        if self.fused_fc_name_suffix:
            self.fused_fc_name = f'fac_fusion_{self.fusion_strategy}_{self.fused_fc_name_suffix}'
        else:
            now_text = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_text, time_text = now_text.split('_')
            self.fused_fc_name = f'fac_fusion_{self.fusion_strategy}_{date_text}_{time_text}'

        if not self.version_list:
            raise ValueError('version_list is empty.')
        supported_fusion_strategy = {'average_weight'}
        if self.fusion_strategy not in supported_fusion_strategy:
            raise ValueError(
                f'Unsupported fusion_strategy: {self.fusion_strategy}. '
                f'Supported: {sorted(supported_fusion_strategy)}'
            )

    def _apply_fusion_strategy(self,
                               merged_factor_df: pd.DataFrame,
                               factor_cols: List[str]) -> pd.Series:
        """Apply selected fusion strategy on source factor columns."""
        if self.fusion_strategy == 'average_weight':
            return merged_factor_df[factor_cols].mean(axis=1, skipna=True)

        raise ValueError(f'Unsupported fusion_strategy: {self.fusion_strategy}.')

    @staticmethod
    def _config_dir_by_method(method: str) -> Path:
        if method == 'tsfresh':
            return Path(__file__).resolve().parent / 'fc_from_tsfresh'
        if method == 'llm_prompt':
            return Path(__file__).resolve().parent / 'fc_from_llm'
        raise ValueError(f'Unsupported method: {method}')

    @staticmethod
    def _config_stem_by_method(method: str) -> str:
        if method == 'tsfresh':
            return 'tsfresh_fc'
        if method == 'llm_prompt':
            return 'llm_fc'
        raise ValueError(f'Unsupported method: {method}')

    def _load_config_payload(self, config_path: Path) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f'Invalid config format: {config_path}')
        fc_name_list = payload.get('fc_name_list', [])
        if not isinstance(fc_name_list, list) or not fc_name_list:
            raise ValueError(f'Invalid config file: {config_path}. Missing non-empty fc_name_list.')
        return payload

    def _load_configs_by_versions(self) -> List[Dict[str, Any]]:
        config_info_list: List[Dict[str, Any]] = []
        for version in self.version_list:
            missing_methods = []
            for method in ['tsfresh', 'llm_prompt']:
                config_dir = self._config_dir_by_method(method)
                config_stem = self._config_stem_by_method(method)
                config_path = config_dir / f'{config_stem}_{version}.json'
                if not config_path.exists():
                    missing_methods.append(method)
                    continue
                payload = self._load_config_payload(config_path)
                config_info_list.append({
                    'version': version,
                    'method': method,
                    'config_path': config_path,
                    'fc_name_list': list(payload['fc_name_list']),
                })

            if missing_methods:
                expected_tsfresh_path = (
                    self._config_dir_by_method('tsfresh') /
                    f'{self._config_stem_by_method("tsfresh")}_{version}.json'
                )
                expected_llm_path = (
                    self._config_dir_by_method('llm_prompt') /
                    f'{self._config_stem_by_method("llm_prompt")}_{version}.json'
                )
                raise FileNotFoundError(
                    f'Incomplete config set for version={version}. Missing methods: {missing_methods}. '
                    f'Expected files: {expected_tsfresh_path} and {expected_llm_path}.'
                )
        return config_info_list

    def _build_generator(self, method: str) -> FactorGenerator:
        common_kwargs = {
            'instrument_type': self.instrument_type,
            'instrument_id_list': self.instrument_id_list,
            'fc_freq': self.fc_freq,
            'data': self.data,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'portfolio_adjust_method': self.portfolio_adjust_method,
            'interest_method': self.interest_method,
            'risk_free_rate': self.risk_free_rate,
            'calculate_baseline': self.calculate_baseline,
            'n_jobs': self.n_jobs,
            'base_col_list': self.base_col_list,
            'min_window_size': self.min_window_size,
            'apply_rolling_norm': self.apply_rolling_norm,
            'rolling_norm_window': self.rolling_norm_window,
            'rolling_norm_min_periods': self.rolling_norm_min_periods,
            'rolling_norm_eps': self.rolling_norm_eps,
            'rolling_norm_clip': self.rolling_norm_clip,
        }
        if method == 'tsfresh':
            return TsfreshFactorGenerator(**common_kwargs)
        if method == 'llm_prompt':
            return LLMPromptFactorGenerator(**common_kwargs)
        raise ValueError(f'Unsupported method in fusion: {method}')

    def generate(self) -> pd.DataFrame:
        """
        Generate fused factor values from all factors referenced by version_list.
        """
        config_info_list = self._load_configs_by_versions()
        self.loaded_config_info_list = config_info_list

        # 用于得到基准的base_df
        data_generator = self._build_generator(method='tsfresh')
        base_df = data_generator.load_base_data()
        df_with_ret = get_future_ret(
            base_df[['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position']].copy(),
            portfolio_adjust_method=self.portfolio_adjust_method,
            rfr=self.risk_free_rate,
        )
        df_with_ret = df_with_ret[['time', 'instrument_id', 'future_ret']].copy()

        merged_factor_df = base_df[['time', 'instrument_id']].copy()
        all_factor_cols: List[str] = []

        for config_info in config_info_list:
            method = cast(str, config_info['method'])
            fc_name_list = cast(List[str], config_info['fc_name_list'])

            generator = self._build_generator(method=method)
            factor_df_i = generator.generate_factor_df(base_df, selected_fc_names=fc_name_list)

            if generator.apply_rolling_norm:
                factor_df_i = rolling_normalize_features(
                    df=factor_df_i,
                    factor_cols=list(fc_name_list),
                    rolling_norm_window=generator.rolling_norm_window,
                    rolling_norm_min_periods=generator.rolling_norm_min_periods,
                    rolling_norm_eps=generator.rolling_norm_eps,
                    rolling_norm_clip=generator.rolling_norm_clip,
                    instrument_col='instrument_id',
                )

            # Avoid collisions across methods/versions by adding stable suffix.
            version = cast(str, config_info['version'])
            renamed_cols = {
                c: f'{c}__{method}__{version}'
                for c in fc_name_list
                if c in factor_df_i.columns
            }
            factor_df_i = factor_df_i[['time', 'instrument_id'] + list(renamed_cols.keys())].rename(columns=renamed_cols)
            all_factor_cols.extend(list(renamed_cols.values()))
            merged_factor_df = merged_factor_df.merge(factor_df_i, on=['time', 'instrument_id'], how='left')

        if not all_factor_cols:
            raise ValueError('No factor columns loaded from version configs for fusion.')

        # Fuse source factors according to selected strategy.
        merged_factor_df[self.fused_fc_name] = self._apply_fusion_strategy(merged_factor_df, all_factor_cols)
        merged_factor_df[self.fused_fc_name] = merged_factor_df[self.fused_fc_name].fillna(0.0)

        if self.apply_rolling_norm_after_fusion:
            # Reuse the same rolling normalization implementation for post-fusion processing.
            norm_helper = self._build_generator(method='tsfresh')
            merged_factor_df = rolling_normalize_features(
                df=merged_factor_df,
                factor_cols=[self.fused_fc_name],
                rolling_norm_window=norm_helper.rolling_norm_window,
                rolling_norm_min_periods=norm_helper.rolling_norm_min_periods,
                rolling_norm_eps=norm_helper.rolling_norm_eps,
                rolling_norm_clip=norm_helper.rolling_norm_clip,
                instrument_col='instrument_id',
            )

        self.raw_fc_value = merged_factor_df
        generated_data = df_with_ret.merge(
            merged_factor_df[['time', 'instrument_id', self.fused_fc_name]],
            on=['time', 'instrument_id'],
            how='left',
            validate='1:1',
        )
        generated_data = generated_data.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        self.generated_data = generated_data
        self.generated_fc_name_list = [self.fused_fc_name]
        log.info(
            f'Generated fused factor `{self.fused_fc_name}` from {len(all_factor_cols)} source factors '
            f'across {len(config_info_list)} configs.'
        )
        return generated_data

    def backtest(self,
                 data: Optional[pd.DataFrame] = None,
                 fc_name_list: Optional[Union[str, List[str]]] = None,
                 n_jobs: Optional[int] = None) -> BackTester:
        """
        Run backtest for fused factor result.
        """
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

