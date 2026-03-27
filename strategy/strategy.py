"""Daily open-to-open futures trading simulation based on one stored factor formula."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass
from math import floor
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import get_data, get_futures_continuous_contract_price
from factors.factor_ops import calc_formula_series
from factors.factor_utils import get_weighted_price
from utils.logging import log
from utils.params import FUTURES_CONTRACT_MULTIPLIER


@dataclass
class StrategyBacktestView:
    """Lightweight plotting view for strategy simulation results."""

    performance_detail: pd.DataFrame
    initial_capital: float
    factor_name: str
    instrument_id: str

    def plot_nav(self,
                 show_baseline: bool = False,
                 x_tick_rotation: int = 45,
                 auto_layout: bool = True,
                 close_fig: bool = True):
        if self.performance_detail is None or self.performance_detail.empty:
            raise ValueError('No performance_detail available. Please run Strategy.backtest() first.')

        df = self.performance_detail.copy().sort_values('time')
        if 'equity' not in df.columns:
            raise ValueError('performance_detail missing `equity` column.')

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        nav = df['equity'] / float(self.initial_capital)
        ax.plot(df['time'], nav, label='Strategy')

        if show_baseline and 'baseline_equity' in df.columns:
            base_nav = df['baseline_equity'] / float(self.initial_capital)
            ax.plot(df['time'], base_nav, linestyle='--', label='Baseline Long Hold')

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
                 slippage: float = 1.0):
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

        self.performance_detail: Optional[pd.DataFrame] = None
        self.bt: Optional[StrategyBacktestView] = None
        self.formula: Optional[str] = None

        if self.initial_capital <= 0:
            raise ValueError('initial_capital must be positive.')
        if self.margin_rate <= 0:
            raise ValueError('margin_rate must be positive.')

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

        sim_df = raw_df[['time', 'instrument_id', 'open', 'close', 'weighted_factor'] +
                        [c for c in ['symbol', 'is_rollover'] if c in raw_df.columns]].copy()
        sim_df = sim_df.merge(
            factor_input[['time', 'instrument_id', self.factor_name]],
            on=['time', 'instrument_id'],
            how='left',
            validate='1:1',
        )

        equity = float(self.initial_capital)
        position_lots = 0
        prev_close = None

        baseline_lots = 0
        baseline_equity = float(self.initial_capital)

        rows = []
        terminated = False
        terminate_reason = ''

        for i, row in sim_df.iterrows():
            t = pd.Timestamp(row['time'])
            open_px = float(pd.to_numeric(row['open'], errors='coerce'))
            close_px = float(pd.to_numeric(row['close'], errors='coerce'))
            factor_val = float(pd.to_numeric(row[self.factor_name], errors='coerce')) if pd.notna(row[self.factor_name]) else 0.0
            symbol = row['symbol'] if 'symbol' in row.index else None

            # -------- start daily log --------
            log.info('=' * 50)
            log.info(
                f'[Strategy][{t.strftime("%Y-%m-%d")}] instrument={self.instrument_id}, symbol={symbol}, '
                f'open={open_px:.6f}, close={close_px:.6f}, factor={factor_val:.6f}'
            )

            # 1) Gap PnL: previous close -> today open on carried overnight position.
            gap_pnl = 0.0
            if prev_close is not None:
                gap_pnl = position_lots * (open_px - prev_close) * multiplier
            equity += gap_pnl

            if i == 0:
                baseline_lots = self._calc_target_lots(self.initial_capital, 1.0, open_px, multiplier)
            baseline_gap_pnl = 0.0 if prev_close is None else baseline_lots * (open_px - prev_close) * multiplier
            baseline_equity += baseline_gap_pnl

            if equity <= 0:
                terminated = True
                terminate_reason = 'equity_le_zero_after_gap'
                log.warning(f'[Strategy] Terminated: equity <= 0 after gap pnl. equity={equity:.6f}')
                rows.append({
                    'time': t,
                    'instrument_id': self.instrument_id,
                    'symbol': symbol,
                    'factor_value': factor_val,
                    'position_lots': position_lots,
                    'target_lots': position_lots,
                    'delta_lots': 0,
                    'open': open_px,
                    'close': close_px,
                    'gap_pnl': gap_pnl,
                    'intraday_pnl': 0.0,
                    'fee': 0.0,
                    'slippage_cost': 0.0,
                    'equity': equity,
                    'required_margin': 0.0,
                    'available_cash': equity,
                    'is_rebalanced': False,
                    'warning': terminate_reason,
                    'baseline_gap_pnl': baseline_gap_pnl,
                    'baseline_intraday_pnl': 0.0,
                    'baseline_equity': baseline_equity,
                })
                break

            # 2) At today open, compute target lots from fixed initial capital (simple-interest style).
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
                    'close': close_px,
                    'gap_pnl': gap_pnl,
                    'intraday_pnl': 0.0,
                    'fee': fee,
                    'slippage_cost': slippage_cost,
                    'equity': equity,
                    'required_margin': 0.0,
                    'available_cash': equity,
                    'is_rebalanced': is_rebalanced,
                    'warning': terminate_reason if not warning_text else f'{warning_text}; {terminate_reason}',
                    'baseline_gap_pnl': baseline_gap_pnl,
                    'baseline_intraday_pnl': 0.0,
                    'baseline_equity': baseline_equity,
                })
                break

            # 3) Intraday PnL: today open -> today close after rebalance.
            intraday_pnl = position_lots * (close_px - open_px) * multiplier
            equity += intraday_pnl

            baseline_intraday_pnl = baseline_lots * (close_px - open_px) * multiplier
            baseline_equity += baseline_intraday_pnl

            required_margin = abs(position_lots) * open_px * multiplier * self.margin_rate
            available_cash = equity - required_margin

            if equity <= 0:
                terminated = True
                terminate_reason = 'equity_le_zero_after_intraday'
                warning_text = terminate_reason if not warning_text else f'{warning_text}; {terminate_reason}'
                log.warning(f'[Strategy] Terminated: equity <= 0 after intraday pnl. equity={equity:.6f}')

            log.info(
                f'[Strategy][DailySummary] lots={position_lots}, target={target_lots}, delta={delta_lots}, '
                f'gap_pnl={gap_pnl:.6f}, intraday_pnl={intraday_pnl:.6f}, fee={fee:.6f}, '
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
                'close': close_px,
                'gap_pnl': gap_pnl,
                'intraday_pnl': intraday_pnl,
                'fee': fee,
                'slippage_cost': slippage_cost,
                'equity': equity,
                'required_margin': required_margin,
                'available_cash': available_cash,
                'is_rebalanced': is_rebalanced,
                'warning': warning_text,
                'baseline_gap_pnl': baseline_gap_pnl,
                'baseline_intraday_pnl': baseline_intraday_pnl,
                'baseline_equity': baseline_equity,
            })

            prev_close = close_px
            if terminated:
                break

        detail = pd.DataFrame(rows)
        if detail.empty:
            raise ValueError('No simulation result generated.')

        detail['nav'] = detail['equity'] / float(self.initial_capital)
        detail['baseline_nav'] = detail['baseline_equity'] / float(self.initial_capital)
        detail['terminated'] = False
        if terminated:
            detail.loc[detail.index[-1], 'terminated'] = True
            detail.loc[detail.index[-1], 'warning'] = (
                (str(detail.loc[detail.index[-1], 'warning']) + '; ' if detail.loc[detail.index[-1], 'warning'] else '')
                + f'terminated={terminate_reason}'
            )

        self.performance_detail = detail
        self.bt = StrategyBacktestView(
            performance_detail=detail,
            initial_capital=self.initial_capital,
            factor_name=self.factor_name,
            instrument_id=self.instrument_id,
        )
        return detail

    def plot_nav(self,
                 show_baseline: bool = False,
                 x_tick_rotation: int = 45,
                 auto_layout: bool = True,
                 close_fig: bool = True):
        if self.bt is None:
            raise ValueError('Please call backtest() first.')
        self.bt.plot_nav(
            show_baseline=show_baseline,
            x_tick_rotation=x_tick_rotation,
            auto_layout=auto_layout,
            close_fig=close_fig,
        )

