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

from data import get_futures_continuous_contract_price, get_factor_value
from factors.factor_indicators import get_performance
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
                 signal_delay_days: int = 1,
                 min_open_ratio: float = 1.0):
        """Initialize strategy simulation settings.

        :param version: 因子版本号，用于在因子库中唯一定位公式。
        :param factor_name: 因子名称，对应 DB 中的 `factor_name` 字段。
        :param instrument_id: 回测标的，例如 `C0`。内部会统一映射为主力连续合约代码。
        :param start_time: 回测开始日期，格式建议 `YYYYMMDD`。
        :param end_time: 回测结束日期，格式建议 `YYYYMMDD`。
        :param database: 因子公式所在数据库名，默认 `factors`。
        :param collection: 因子公式所在集合名，默认 `genetic_programming`。
        :param initial_capital: 初始资金。当前仓位目标按“单利口径”始终基于该初始资金计算。
        :param margin_rate: 保证金比例，用于约束最大可开手数。
            可开仓上限约为 equity / (open * multiplier * margin_rate)。
        :param fee_per_lot: 每手固定手续费（开平合计按实际交易手数累计）。
        :param slippage: 每手滑点（价格点数）。实际成本为 slippage * multiplier * trade_lots。
        :param apply_rolling_norm: 是否对因子做滚动标准化，默认 True。
            建议与 BackTester 保持一致，避免信号分布口径不一致。
        :param rolling_norm_window: 滚动标准化窗口长度。
        :param rolling_norm_min_periods: 滚动标准化最小样本数。
        :param rolling_norm_eps: 标准化分母平滑项，防止除零。
        :param rolling_norm_clip: 标准化后截断阈值（对 zscore 做 clip）。
        :param signal_delay_days: 信号执行延迟天数。
            例如 1 表示使用 T 日信号，在 T+1 开盘执行交易，
            与“信号生成后下一交易日开盘执行”的常见研究口径一致。
        :param min_open_ratio: 开仓“激进程度”阈值，范围 [0, 1]。
            令 raw_target 为理论手数，trunc_target 为向0截断后的手数。
            若 |raw_target - trunc_target| >= min_open_ratio，则向远离0方向多开1手；
            否则保持 trunc_target。
            e.g.
            - min_open_ratio=1: 完全向0截断（当前保守模式）
            - min_open_ratio=0.6: 大于等于0.6则加一手，到更激进的总手数
            - min_open_ratio=0: 只要非整数就向远离0方向取整
        """
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
        self.min_open_ratio = float(min_open_ratio)

        self.performance_detail: Optional[pd.DataFrame] = None
        self.performance_summary: Optional[pd.DataFrame] = None

        if self.initial_capital <= 0:
            raise ValueError('initial_capital must be positive.')
        if self.margin_rate <= 0:
            raise ValueError('margin_rate must be positive.')
        if self.signal_delay_days < 0:
            raise ValueError('signal_delay_days must be non-negative.')
        if not (0.0 <= self.min_open_ratio <= 1.0):
            raise ValueError('min_open_ratio must be in [0, 1].')

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
                          multiplier: int,
                          min_open_ratio: float = 1.0) -> int:
        if not np.isfinite(factor_value) or not np.isfinite(open_price) or open_price <= 0:
            return 0
        raw_target = (initial_capital * factor_value) / (open_price * float(multiplier))
        trunc_target = int(raw_target)
        frac = raw_target - trunc_target

        # Aggressive opening rule controlled by min_open_ratio.
        if abs(frac) >= float(min_open_ratio) and abs(frac) > 0:
            return trunc_target + (1 if raw_target > 0 else -1)
        return trunc_target

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

        raw_df = self._load_price_df()

        # Factor must be calculated on weighted(adjusted) prices.
        factor_input = get_weighted_price(raw_df)
        factor_input = get_factor_value(
            Data=factor_input,
            fc_name_list=[self.factor_name],
            version=self.version,
            collection=self.collection,
            n_jobs=1,
        )
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
        # this is corresponding with the Backtester procedure
        if self.signal_delay_days > 0:
            sim_df[self.factor_name] = sim_df.groupby('instrument_id')[self.factor_name].shift(self.signal_delay_days)
        sim_df['next_open'] = sim_df.groupby('instrument_id')['open'].shift(-1)

        equity = float(self.initial_capital)
        position_lots = 0

        baseline_long_lots = 0
        baseline_short_lots = 0
        baseline_long_equity = float(self.initial_capital)
        baseline_short_equity = float(self.initial_capital)

        rows = []
        terminated = False
        terminate_reason = ''

        # 对于T日的模拟
        for i, row in sim_df.iterrows():
            t = pd.Timestamp(row['time'])  # T日
            open_px = float(pd.to_numeric(row['open'], errors='coerce'))
            close_px = float(pd.to_numeric(row['close'], errors='coerce'))
            next_open_px = float(pd.to_numeric(row['next_open'], errors='coerce')) if pd.notna(row['next_open']) else np.nan
            future_ret_t = ((next_open_px - open_px) / open_px) if (np.isfinite(next_open_px) and open_px > 0) else 0.0
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
                baseline_long_lots = self._calc_target_lots(
                    self.initial_capital, 1.0, open_px, multiplier, self.min_open_ratio
                )
                baseline_short_lots = self._calc_target_lots(
                    self.initial_capital, -1.0, open_px, multiplier, self.min_open_ratio
                )

            # At today open, compute target lots from fixed initial capital (simple-interest style).
            target_lots = self._calc_target_lots(
                self.initial_capital, factor_val, open_px, multiplier, self.min_open_ratio
            )

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
            is_rebalanced = abs(delta_lots) >= 1  # 是否需要调仓

            fee = 0.0
            slippage_cost = 0.0
            if is_rebalanced:
                # 开仓
                if position_lots == 0:
                    close_lots = 0
                    open_lots = abs(target_lots)
                # 平仓
                elif target_lots == 0:
                    close_lots = abs(position_lots)
                    open_lots = 0
                # 同向加仓
                elif np.sign(position_lots) == np.sign(target_lots):
                    if abs(target_lots) >= abs(position_lots):
                        close_lots = 0
                        open_lots = abs(target_lots) - abs(position_lots)
                    else:
                        close_lots = abs(position_lots) - abs(target_lots)
                        open_lots = 0
                # 先平仓后反向开仓
                else:
                    close_lots = abs(position_lots)
                    open_lots = abs(target_lots)

                trade_lots = close_lots + open_lots
                fee = float(trade_lots) * float(self.fee_per_lot)
                # 滑点成本 = 每价格滑点 * 合约乘数 * 交易手数
                # 这里假设每手交易都会存在价格滑点（默认为一点）
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
                    'future_ret': future_ret_t,
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
                    'baseline_long_open_to_open_pnl': 0.0,
                    'baseline_short_open_to_open_pnl': 0.0,
                    'baseline_open_to_open_pnl': 0.0,
                    'baseline_long_equity': baseline_long_equity,
                    'baseline_short_equity': baseline_short_equity,
                    'baseline_equity': baseline_long_equity,
                })
                break
            
            """
            “equity 里包含期货头寸，不全是现金”，在股票模型里是对的；
            但在期货逐日盯市模型里，头寸不是按“资产市值”记账，而是保证金占用 + 每日盈亏结算到权益，所以：
            equity 可理解为账户净值（近似现金权益）
            required_margin 是冻结保证金
            available_cash = equity - required_margin 常用于表示可再开仓资金
            """
            
            # Equity right after T-open rebalancing and transaction costs.
            equity_t_open = equity

            # T-open margin snapshot (used for available margin before T->T+1 holding PnL).
            required_margin = abs(position_lots) * open_px * multiplier * self.margin_rate
            available_cash = equity_t_open - required_margin

            # Open-to-open holding PnL: today open -> next trading day's open.
            if np.isfinite(next_open_px) and next_open_px > 0:
                # T日根据T-1日的因子，开盘后交易使得持仓手数为position_lots，
                # 那么T日open到T+1日open区间的open_to_open_pnl就等于position_lots * (T+1日open - T日open) * multiplier

                open_to_open_pnl = position_lots * (next_open_px - open_px) * multiplier
                baseline_long_open_to_open_pnl = baseline_long_lots * (next_open_px - open_px) * multiplier
                baseline_short_open_to_open_pnl = baseline_short_lots * (next_open_px - open_px) * multiplier
            else:
                open_to_open_pnl = 0.0
                baseline_long_open_to_open_pnl = 0.0
                baseline_short_open_to_open_pnl = 0.0

            equity += open_to_open_pnl  # 这里的equity指的是T+1日开盘后的瞬间，根据T+1日开盘价计算的总净值
            baseline_long_equity += baseline_long_open_to_open_pnl
            baseline_short_equity += baseline_short_open_to_open_pnl

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
                'time': t,  # T日
                'instrument_id': self.instrument_id,
                'symbol': symbol,
                'factor_value': factor_val,  # T-1日的因子值（假设signal_delay_days=1）
                'position_lots': position_lots,  # T日根据T-1的因子，得到的T日开盘后交易后，保持的持仓手数
                'target_lots': target_lots,  # T日根据T-1的因子，得到的理论目标持仓手数（一般就等于position_lots）
                'delta_lots': delta_lots,  # T日根据T-1的因子，得到的T日开盘后需要交易的手数
                'open': open_px,  # T日开盘价
                'high': float(pd.to_numeric(row['high'], errors='coerce')),  # T日最高价
                'low': float(pd.to_numeric(row['low'], errors='coerce')),  # T日最低价
                'close': close_px,  # T日收盘价
                'volume': float(pd.to_numeric(row['volume'], errors='coerce')),  # T日市场成交量
                'position': float(pd.to_numeric(row['position'], errors='coerce')),  # T日市场持仓量
                'weighted_factor': float(pd.to_numeric(row['weighted_factor'], errors='coerce')),
                'next_open': next_open_px,  # T+1日开盘价
                'future_ret': future_ret_t,  # (next_open_px - open_px) / open_px)
                'open_to_open_pnl': open_to_open_pnl,  # T日open到T+1日open这段时间内的收益。即T日根据T-1的因子，在T日开盘后交易，持续到T+1日开盘的这段时间内的收益。注意这里是为了保证和Backtester的逻辑一致。
                'daily_gross_pnl': open_to_open_pnl,  # 目前按照open-to-open的逻辑模拟交易，每日的净pnl就等于open_to_open_pnl
                'daily_net_pnl': open_to_open_pnl - fee - slippage_cost,  # 每日净pnl还要扣除手续费和滑点成本
                'fee': fee,
                'slippage_cost': slippage_cost,
                'equity': equity,  # 这里的equity指的是T+1日开盘后的瞬间，根据T+1日开盘价计算的账户总净值。注意这里是为了保证和Backtester的逻辑一致。
                'required_margin': required_margin,  # 根据T日开盘价、T日开盘交易后的持仓position_lots计算的保证金占用
                'available_cash': available_cash,  # T日开盘后，根据T日开盘价计算的可用cash
                'is_rebalanced': is_rebalanced,  # T日是否需要调仓
                'warning': warning_text,
                'baseline_long_open_to_open_pnl': baseline_long_open_to_open_pnl,
                'baseline_short_open_to_open_pnl': baseline_short_open_to_open_pnl,
                # Keep legacy field as long-only baseline for backward compatibility.
                'baseline_open_to_open_pnl': baseline_long_open_to_open_pnl,
                'baseline_long_equity': baseline_long_equity,
                'baseline_short_equity': baseline_short_equity,
                'baseline_equity': baseline_long_equity,
            })

            if terminated:
                break

        detail = pd.DataFrame(rows)
        if detail.empty:
            raise ValueError('No simulation result generated.')

        detail = detail.sort_values('time').reset_index(drop=True)
        # 单利计算
        detail['daily_gross_ret'] = detail['daily_gross_pnl'] / float(self.initial_capital)
        detail['daily_net_ret'] = detail['daily_net_pnl'] / float(self.initial_capital)
        detail['daily_turnover'] = detail['delta_lots'].abs()
        detail['daily_gross_nav'] = 1.0 + detail['daily_gross_ret'].fillna(0.0).cumsum()
        detail['daily_net_nav'] = 1.0 + detail['daily_net_ret'].fillna(0.0).cumsum()
        detail['factor_name'] = self.factor_name

        detail['nav'] = detail['equity'] / float(self.initial_capital)  # 单利计算
        detail['baseline_nav'] = detail['baseline_equity'] / float(self.initial_capital)
        detail['baseline_nav_long'] = detail['baseline_long_equity'] / float(self.initial_capital)
        detail['baseline_nav_short'] = detail['baseline_short_equity'] / float(self.initial_capital)
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
        summary_input = summary_input.merge(
            detail[['time', 'future_ret']],
            on='time',
            how='left',
            validate='1:1',
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

        if show_baseline and 'baseline_nav_long' in df.columns and 'baseline_nav_short' in df.columns:
            ax.plot(
                df['time'],
                df['baseline_nav_long'],
                color='tab:red',
                linestyle='--',
                label='Baseline Long',
            )
            ax.plot(
                df['time'],
                df['baseline_nav_short'],
                color='tab:green',
                linestyle='--',
                label='Baseline Short',
            )

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

