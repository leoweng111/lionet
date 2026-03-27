"""Daily open-to-open futures trading simulation based on one stored factor formula."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from math import floor
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import get_data, get_futures_continuous_contract_price
from factors.factor_indicators import get_performance
from factors.factor_ops import calc_formula_series
from factors.factor_utils import get_weighted_price, rolling_normalize_features
from utils.logging import log
from utils.params import FUTURES_CONTRACT_MULTIPLIER


class Strategy:
    """Simulate day-session open-to-open futures trading from one stored factor formula."""

    def __init__(self,
                 version: str,
                 factor_name: str,
                 instrument_id: str,
                 start_time: str,
                 end_time: str,
                 database: str = 'factors',
                 collection: str = 'genetic_programming',
                 initial_capital: float = 1_000_000.0,
                 margin_rate: float = 0.1,
                 fee_per_lot: float = 2.0,
                 slippage: float = 1.0,
                 apply_rolling_norm: bool = True,
                 rolling_norm_window: int = 30,
                 rolling_norm_min_periods: int = 20,
                 rolling_norm_eps: float = 1e-8,
                 rolling_norm_clip: float = 5.0,
                 signal_delay_days: int = 1):
        self.database = database
        self.collection = collection
        self.version = version
        self.factor_name = factor_name
        self.instrument_id = instrument_id
        self.start_time = start_time
        self.end_time = end_time

        self.initial_capital = float(initial_capital)
        self.margin_rate = float(margin_rate)
        self.fee_per_lot = float(fee_per_lot)
        self.slippage = float(slippage)
        self.apply_rolling_norm = bool(apply_rolling_norm)
        self.rolling_norm_window = int(rolling_norm_window)
        self.rolling_norm_min_periods = int(rolling_norm_min_periods)
        self.rolling_norm_eps = float(rolling_norm_eps)
        self.rolling_norm_clip = float(rolling_norm_clip)
        self.signal_delay_days = int(signal_delay_days)

        self.performance_detail: Optional[pd.DataFrame] = None
        self.performance_summary: Optional[pd.DataFrame] = None
        self.formula: Optional[str] = None

        if self.initial_capital <= 0:
            raise ValueError('initial_capital must be positive.')
        if self.margin_rate <= 0:
            raise ValueError('margin_rate must be positive.')
        if self.signal_delay_days < 0:
            raise ValueError('signal_delay_days must be non-negative.')

    @staticmethod
    def _root_instrument(instrument_id: str) -> str:
        ins = str(instrument_id).upper().strip()
        if not ins:
            raise ValueError('instrument_id is empty.')
        return ins[:-1] if ins.endswith('0') else ins

    @staticmethod
    def _main_instrument(instrument_id: str) -> str:
        ins = str(instrument_id).upper().strip()
        if not ins:
            raise ValueError('instrument_id is empty.')
        return ins if ins.endswith('0') else f'{ins}0'

    @staticmethod
    def _calc_target_lots(initial_capital: float,
                          factor_value: float,
                          open_price: float,
                          multiplier: int) -> int:
        if not np.isfinite(factor_value) or not np.isfinite(open_price) or open_price <= 0:
            return 0
        raw_target = (initial_capital * factor_value) / (open_price * float(multiplier))
        # keep truncation-toward-zero behavior for positive/negative factors
        return int(raw_target)

    def _load_formula(self) -> str:
        operator = {
            '$and': [
                {'version': self.version},
                {'factor_name': self.factor_name},
            ]
        }
        df = get_data(database=self.database, collection=self.collection, mongo_operator=operator)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                f'No factor record found in {self.database}.{self.collection} for '
                f'version={self.version}, factor_name={self.factor_name}.'
            )
        formula_list = [
            str(x).strip()
            for x in df['formula'].tolist()
            if isinstance(x, str) and str(x).strip()
        ]
        formula_set = sorted(set(formula_list))
        if not formula_set:
            raise ValueError('Factor formula is empty in DB records.')
        if len(formula_set) > 1:
            raise ValueError(
                f'Multiple formulas found for one factor record: {formula_set}. '
                'Please keep one unique formula per (version, factor_name).'
            )
        return formula_set[0]

    def _load_price_df(self) -> pd.DataFrame:
        main_id = self._main_instrument(self.instrument_id)
        df = get_futures_continuous_contract_price(
            instrument_id=main_id,
            start_date=self.start_time,
            end_date=self.end_time,
            from_database=True,
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                f'No continuous price data in DB for instrument={main_id}, '
                f'range=[{self.start_time}, {self.end_time}].'
            )

        required = ['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position', 'weighted_factor']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Price data missing required columns: {missing}')

        keep_cols = required + [c for c in ['symbol', 'is_rollover'] if c in df.columns]
        out = df[keep_cols].copy()
        out['time'] = pd.to_datetime(out['time'])
        out = out.sort_values(['instrument_id', 'time']).reset_index(drop=True)
        return out

    def backtest(self) -> pd.DataFrame:
        root = self._root_instrument(self.instrument_id)
        multiplier = FUTURES_CONTRACT_MULTIPLIER.get(root)
        if multiplier is None:
            raise ValueError(
                f'Missing contract multiplier for root instrument `{root}`. '
                'Please update utils/params.py FUTURES_CONTRACT_MULTIPLIER.'
            )
        multiplier = int(multiplier)

        self.formula = self._load_formula()
        raw_df = self._load_price_df()

        # Factor must be calculated on weighted(adjusted) prices.
        factor_input = get_weighted_price(raw_df)
        factor_input[self.factor_name] = calc_formula_series(df=factor_input, formula=self.formula)
        if self.apply_rolling_norm:
            factor_input = rolling_normalize_features(
                df=factor_input,
                factor_cols=[self.factor_name],
                rolling_norm_window=self.rolling_norm_window,
                rolling_norm_min_periods=self.rolling_norm_min_periods,
                rolling_norm_eps=self.rolling_norm_eps,
                rolling_norm_clip=self.rolling_norm_clip,
                instrument_col='instrument_id',
            )
        sim_df = raw_df[['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume', 'position', 'weighted_factor'] +
                        [c for c in ['symbol', 'is_rollover'] if c in raw_df.columns]].copy()
        sim_df = sim_df.merge(
            factor_input[['time', 'instrument_id', self.factor_name]],
            on=['time', 'instrument_id'],
            how='left',
            validate='1:1',
        )
        # Align execution timing with T-signal -> T+1 open trade convention.
        if self.signal_delay_days > 0:
            sim_df[self.factor_name] = sim_df.groupby('instrument_id')[self.factor_name].shift(self.signal_delay_days)
        sim_df['next_open'] = sim_df.groupby('instrument_id')['open'].shift(-1)

        equity = float(self.initial_capital)
        position_lots = 0

        baseline_lots = 0
        baseline_equity = float(self.initial_capital)

        rows = []
        terminated = False
        terminate_reason = ''

        for i, row in sim_df.iterrows():
            t = pd.Timestamp(row['time'])
            open_px = float(pd.to_numeric(row['open'], errors='coerce'))
            close_px = float(pd.to_numeric(row['close'], errors='coerce'))
            next_open_px = float(pd.to_numeric(row['next_open'], errors='coerce')) if pd.notna(row['next_open']) else np.nan
            factor_val = float(pd.to_numeric(row[self.factor_name], errors='coerce')) if pd.notna(row[self.factor_name]) else 0.0
            symbol = row['symbol'] if 'symbol' in row.index else None

            # -------- start daily log --------
            log.info('=' * 50)
            log.info(
                f'[Strategy][{t.strftime("%Y-%m-%d")}] instrument={self.instrument_id}, symbol={symbol}, '
                f'open={open_px:.6f}, next_open={next_open_px if np.isfinite(next_open_px) else float("nan"):.6f}, '
                f'factor={factor_val:.6f}'
            )

            if i == 0:
                baseline_lots = self._calc_target_lots(self.initial_capital, 1.0, open_px, multiplier)

            # At today open, compute target lots from fixed initial capital (simple-interest style).
            target_lots = self._calc_target_lots(self.initial_capital, factor_val, open_px, multiplier)

            max_lots_by_margin = floor(max(equity, 0.0) / (open_px * multiplier * self.margin_rate))
            warning_text = ''
            if abs(target_lots) > max_lots_by_margin:
                old_target = target_lots
                target_lots = int(np.sign(target_lots)) * int(max_lots_by_margin)
                warning_text = (
                    f'target_lots_downgraded_by_margin old={old_target}, new={target_lots}, '
                    f'max_lots_by_margin={max_lots_by_margin}'
                )
                log.warning(f'[Strategy] {warning_text}')

            if abs(target_lots) == 0 and abs(factor_val) > 0:
                msg = (
                    'cannot_open_min_one_lot_under_current_equity_or_factor '
                    f'(factor={factor_val:.6f}, equity={equity:.6f})'
                )
                warning_text = f'{warning_text}; {msg}' if warning_text else msg
                log.warning(f'[Strategy] {msg}')

            delta_lots = int(target_lots - position_lots)
            is_rebalanced = abs(delta_lots) >= 1

            fee = 0.0
            slippage_cost = 0.0
            if is_rebalanced:
                if position_lots == 0:
                    close_lots = 0
                    open_lots = abs(target_lots)
                elif target_lots == 0:
                    close_lots = abs(position_lots)
                    open_lots = 0
                elif np.sign(position_lots) == np.sign(target_lots):
                    if abs(target_lots) >= abs(position_lots):
                        close_lots = 0
                        open_lots = abs(target_lots) - abs(position_lots)
                    else:
                        close_lots = abs(position_lots) - abs(target_lots)
                        open_lots = 0
                else:
                    close_lots = abs(position_lots)
                    open_lots = abs(target_lots)

                trade_lots = close_lots + open_lots
                fee = float(trade_lots) * float(self.fee_per_lot)
                slippage_cost = float(trade_lots) * float(self.slippage) * float(multiplier)
                equity -= (fee + slippage_cost)
                position_lots = target_lots

            if equity <= 0:
                terminated = True
                terminate_reason = 'equity_le_zero_after_trade_cost'
                log.warning(f'[Strategy] Terminated: equity <= 0 after fee/slippage. equity={equity:.6f}')
                rows.append({
                    'time': t,
                    'instrument_id': self.instrument_id,
                    'symbol': symbol,
                    'factor_value': factor_val,
                    'position_lots': position_lots,
                    'target_lots': target_lots,
                    'delta_lots': delta_lots,
                    'open': open_px,
                    'high': float(pd.to_numeric(row['high'], errors='coerce')),
                    'low': float(pd.to_numeric(row['low'], errors='coerce')),
                    'close': close_px,
                    'volume': float(pd.to_numeric(row['volume'], errors='coerce')),
                    'position': float(pd.to_numeric(row['position'], errors='coerce')),
                    'weighted_factor': float(pd.to_numeric(row['weighted_factor'], errors='coerce')),
                    'next_open': next_open_px,
                    'gap_pnl': 0.0,
                    'intraday_pnl': 0.0,
                    'open_to_open_pnl': 0.0,
                    'daily_gross_pnl': 0.0,
                    'daily_net_pnl': -fee - slippage_cost,
                    'fee': fee,
                    'slippage_cost': slippage_cost,
                    'equity': equity,
                    'required_margin': 0.0,
                    'available_cash': equity,
                    'is_rebalanced': is_rebalanced,
                    'warning': terminate_reason if not warning_text else f'{warning_text}; {terminate_reason}',
                    'baseline_gap_pnl': 0.0,
                    'baseline_intraday_pnl': 0.0,
                    'baseline_open_to_open_pnl': 0.0,
                    'baseline_equity': baseline_equity,
                })
                break

            # Open-to-open holding PnL: today open -> next trading day's open.
            if np.isfinite(next_open_px) and next_open_px > 0:
                open_to_open_pnl = position_lots * (next_open_px - open_px) * multiplier
                baseline_open_to_open_pnl = baseline_lots * (next_open_px - open_px) * multiplier
            else:
                open_to_open_pnl = 0.0
                baseline_open_to_open_pnl = 0.0

            equity += open_to_open_pnl
            baseline_equity += baseline_open_to_open_pnl

            required_margin = abs(position_lots) * open_px * multiplier * self.margin_rate
            available_cash = equity - required_margin

            if equity <= 0:
                terminated = True
                terminate_reason = 'equity_le_zero_after_intraday'
                warning_text = terminate_reason if not warning_text else f'{warning_text}; {terminate_reason}'
                log.warning(f'[Strategy] Terminated: equity <= 0 after intraday pnl. equity={equity:.6f}')

            log.info(
                f'[Strategy][DailySummary] lots={position_lots}, target={target_lots}, delta={delta_lots}, '
                f'open_to_open_pnl={open_to_open_pnl:.6f}, fee={fee:.6f}, '
                f'slippage_cost={slippage_cost:.6f}, equity={equity:.6f}, '
                f'required_margin={required_margin:.6f}, available_cash={available_cash:.6f}'
            )

            rows.append({
                'time': t,
                'instrument_id': self.instrument_id,
                'symbol': symbol,
                'factor_value': factor_val,
                'position_lots': position_lots,
                'target_lots': target_lots,
                'delta_lots': delta_lots,
                'open': open_px,
                'high': float(pd.to_numeric(row['high'], errors='coerce')),
                'low': float(pd.to_numeric(row['low'], errors='coerce')),
                'close': close_px,
                'volume': float(pd.to_numeric(row['volume'], errors='coerce')),
                'position': float(pd.to_numeric(row['position'], errors='coerce')),
                'weighted_factor': float(pd.to_numeric(row['weighted_factor'], errors='coerce')),
                'next_open': next_open_px,
                'gap_pnl': 0.0,
                'intraday_pnl': 0.0,
                'open_to_open_pnl': open_to_open_pnl,
                'daily_gross_pnl': open_to_open_pnl,
                'daily_net_pnl': open_to_open_pnl - fee - slippage_cost,
                'fee': fee,
                'slippage_cost': slippage_cost,
                'equity': equity,
                'required_margin': required_margin,
                'available_cash': available_cash,
                'is_rebalanced': is_rebalanced,
                'warning': warning_text,
                'baseline_gap_pnl': 0.0,
                'baseline_intraday_pnl': 0.0,
                'baseline_open_to_open_pnl': baseline_open_to_open_pnl,
                'baseline_equity': baseline_equity,
            })

            if terminated:
                break

        detail = pd.DataFrame(rows)
        if detail.empty:
            raise ValueError('No simulation result generated.')

        detail = detail.sort_values('time').reset_index(drop=True)
        detail['daily_gross_ret'] = detail['daily_gross_pnl'] / float(self.initial_capital)
        detail['daily_net_ret'] = detail['daily_net_pnl'] / float(self.initial_capital)
        detail['daily_turnover'] = detail['delta_lots'].abs()
        detail['daily_gross_nav'] = 1.0 + detail['daily_gross_ret'].fillna(0.0).cumsum()
        detail['daily_net_nav'] = 1.0 + detail['daily_net_ret'].fillna(0.0).cumsum()
        detail['factor_name'] = self.factor_name

        detail['nav'] = detail['equity'] / float(self.initial_capital)
        detail['baseline_nav'] = detail['baseline_equity'] / float(self.initial_capital)
        detail['terminated'] = False
        if terminated:
            detail.loc[detail.index[-1], 'terminated'] = True
            detail.loc[detail.index[-1], 'warning'] = (
                (str(detail.loc[detail.index[-1], 'warning']) + '; ' if detail.loc[detail.index[-1], 'warning'] else '')
                + f'terminated={terminate_reason}'
            )

        summary_input = detail[['time']].merge(
            sim_df[['time', 'instrument_id', self.factor_name]],
            on='time',
            how='left',
            validate='1:1',
        )
        summary_input['future_ret'] = 0.0
        nonzero_mask = summary_input[self.factor_name].abs() > 1e-12
        summary_input.loc[nonzero_mask, 'future_ret'] = (
            detail.loc[nonzero_mask, 'daily_gross_ret'].to_numpy() /
            summary_input.loc[nonzero_mask, self.factor_name].to_numpy()
        )
        if 'is_rollover' in sim_df.columns:
            summary_input = summary_input.merge(
                sim_df[['time', 'is_rollover']],
                on='time',
                how='left',
                validate='1:1',
            )

        _, strategy_summary = get_performance(
            Data=summary_input,
            fc_name_list=[self.factor_name],
            fc_freq='1d',
            portfolio_adjust_method='1D',
            interest_method='simple',
            calculate_baseline=True,
            n_jobs=1,
        )
        strategy_summary = strategy_summary.copy()
        strategy_summary['Instrument ID'] = self.instrument_id

        self.performance_detail = detail
        self.performance_summary = strategy_summary
        return detail

    def plot_nav(self,
                 show_baseline: bool = False,
                 x_tick_rotation: int = 45,
                 auto_layout: bool = True,
                 close_fig: bool = True):
        if self.performance_detail is None or self.performance_detail.empty:
            raise ValueError('Please call backtest() first.')
        df = self.performance_detail.copy().sort_values('time')

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        # Strategy NAV is shown in simple-interest form: 1 + cumulative daily net return.
        simple_nav = 1.0 + pd.to_numeric(df['daily_net_ret'], errors='coerce').fillna(0.0).cumsum()
        ax.plot(df['time'], simple_nav, label='Strategy')

        if show_baseline and 'baseline_nav' in df.columns:
            baseline_simple_nav = df['baseline_equity'] / float(self.initial_capital)
            ax.plot(df['time'], baseline_simple_nav, linestyle='--', label='Baseline Long Hold')

        ax.set_title(f'Strategy NAV ({self.instrument_id}, factor={self.factor_name})')
        ax.set_xlabel('time')
        ax.set_ylabel('NAV')
        ax.tick_params(axis='x', labelrotation=x_tick_rotation)
        ax.legend(loc='best')

        if auto_layout:
            fig.tight_layout()
        plt.show()
        if close_fig:
            plt.close(fig)

