"""
This script is for the backtesting of signals.
"""
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Union
import pandas as pd
from joblib import Parallel, delayed

from .factor_indicators import get_performance
from .factor_utils import get_weighted_price, get_future_ret
from data import get_futures_continuous_contract_price, get_factor_value
from utils.logging import log
from error.errors import NotBackTestingError


class BackTester:
    """
    A class for single-instrument, multi-signal time-series backtesting.
    """

    def __init__(self,
                 fc_name_list: Union[str, List] = None,
                 version: str = None,
                 collection: Union[str, List] = 'genetic_programming',
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List] = 'C0',
                 fc_freq: str = '1d',
                 data: Union[pd.DataFrame, None] = None,
                 start_time: str = None,
                 end_time: str = None,
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'simple',
                 risk_free_rate: bool = False,
                 calculate_baseline: bool = True,
                 apply_weighted_price: bool = True,
                 n_jobs: int = 5,
                 formula: str = None):
        # todo: 数据频率，调仓频率和收益率计算频率三者相关
        # 对数据频率1min，调仓频率为1day情况，计算IC需要先将一分钟的数据聚合成一天的，然后计算日频收益率，再计算日频因子值和日频收益率
        # 之间的相关系数。
        # 时序收益率
        """
        Some params need to be initialized for backtesting.
        We assume that the signal is generated at the open time of every bar, and we immediately strategy at this bar
        with average transaction price (open + high + low + close) / 4 of this bar, and will complete the transaction
        at the close time of the next `transaction_period` bar

        :param data: data should be a dataframe with high, open, low, close, volume, position columns 
            for one instrument in each bar.
        :param fc_name_list: factor names persisted in factors DB, or existing columns when data is preprocessed.
        :param collection: factor collection(s) in factors DB used with `version` + `fc_name_list` to locate formulas.
        :param start_time: backtesting start time,
            default is the earliest price data that can be found on database
        :param end_time: backtesting end time,
            default is the latest price data that can be found on database
        :param portfolio_adjust_method: 多种调仓方式。定义fac(t)为使用直到t日的信息计算出的因子值
               1. 分钟频率调仓，只针对1min和5min数据，在每一分钟或者每五分钟调仓。
               2. 日频调仓：T日收盘利用fac(T-1)开仓，然后在T+1日收盘平T日的仓位，再开T+1的仓位，如此反复
               3. 月度调仓：在每个月最后一个交易日调仓，使用这个月（不包含最后一个交易日）的因子值的平均值调仓。
               4. 季度调仓：在每个季度最后一个交易日调仓，使用这个季度（不包含最后一个交易日）的因子值的平均值调仓。
        :param fc_freq: the frequency of factor, 1m, 5m or 1d.
        :param n_jobs: Parallel's n_job param. Default is to parallelize 5 jobs.
        """
        self.data = data
        self.instrument_type = instrument_type
        self.fc_name_list = fc_name_list
        self.version = version
        self.collection = collection
        self.instrument_id_list = instrument_id_list
        self.start_time = start_time
        self.end_time = end_time
        self.portfolio_adjust_method = portfolio_adjust_method
        self.fc_freq = fc_freq
        self.interest_method = interest_method
        self.rfr = risk_free_rate
        self.calculate_baseline = calculate_baseline
        self.apply_weighted_price = apply_weighted_price
        self.n_jobs = n_jobs
        self.formula = formula

        self.fc_name_with_param_list = None
        self.performance_dc = None
        self.performance_summary = None
        self.performance_detail = None
        self.is_backtested = False
        self.is_preprocessed = isinstance(self.data, pd.DataFrame)
        self._did_preprocess = False

        # Formula mode: derive fc_name_list and relax version/collection checks
        if self.formula:
            self.fc_name_list = ['formula_factor']
            if not self.version:
                self.version = '__formula__'
        else:
            if not isinstance(self.version, str) or not self.version.strip():
                raise ValueError('BackTester requires a non-empty `version` to resolve factor formulas precisely.')

        if isinstance(self.collection, str):
            self.collection = [self.collection]
        self.collection = [str(x).strip() for x in self.collection if str(x).strip()]
        if not self.formula and not self.collection:
            raise ValueError('BackTester requires non-empty `collection` to resolve factor formulas precisely.')

        assert self.fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, but got {fc_freq} instead.'
        assert self.portfolio_adjust_method in ['min', '1D', '1M', '1Q'], \
            f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, but got {portfolio_adjust_method} instead.'
        if self.fc_freq == '1d':
            assert self.portfolio_adjust_method != 'min', 'portfolio adjust period should be equal to or longer ' \
                                                          'than factor frequency'
        if isinstance(self.fc_name_list, str):
            self.fc_name_list = [self.fc_name_list]
        if not self.formula:
            fc_name_counter = Counter(self.fc_name_list)
            duplicated_fc_names = sorted([x for x, cnt in fc_name_counter.items() if cnt > 1])
            if duplicated_fc_names:
                raise ValueError(f'fc_name_list contains duplicated factor names: {duplicated_fc_names}')
        if isinstance(self.instrument_id_list, str):
            self.instrument_id_list = [self.instrument_id_list]

        # if data is provided outside
        if isinstance(self.data, pd.DataFrame):
            # when data is provided, it should have been preprocessed, and we will not preprocess it here.
            if not self.formula:
                for col in ['time', 'instrument_id', 'future_ret'] + self.fc_name_list:
                    assert col in self.data.columns, f'Provided data does not contain column {col}.'

            if self.apply_weighted_price:
                for col in ['weighted_factor', 'open', 'high', 'low', 'close']:
                    if col not in self.data.columns:
                        raise ValueError(
                            f'apply_weighted_price=True requires external `data` to contain column `{col}`.'
                        )

        # otherwise we load data from local database
        else:
            if self.instrument_type == 'futures_continuous_contract':
                self.data = get_futures_continuous_contract_price(instrument_id=self.instrument_id_list,
                                                                  start_date=self.start_time,
                                                                  end_date=self.end_time,
                                                                  from_database=True)
            else:
                raise ValueError(f'Does not support instrument type {self.instrument_type}.')
            self.is_preprocessed = False

        available_instruments = self.data['instrument_id'].dropna().unique().tolist()
        invalid_instruments = [x for x in self.instrument_id_list if x not in available_instruments]
        if invalid_instruments:
            raise ValueError(
                f'Invalid instrument_id_list: {invalid_instruments}. '
                f'Available instruments include: {available_instruments}'
            )

        # For external input data, still apply selection to keep behavior consistent with DB path.
        if self.is_preprocessed:
            self.data = self.data[self.data['instrument_id'].isin(self.instrument_id_list)].copy()

        self.fc_name_with_param_list = list(self.fc_name_list)

    def _preprocess_data(self):
        """
        Preprocess futures data for time-series backtesting.
        """
        if self._did_preprocess:
            return

        if self.is_preprocessed:
            # if the data is provided outside, still we may need to apply the weighted factor to price.
            if self.apply_weighted_price:
                self.data = get_weighted_price(self.data)
                # Recompute future_ret from adjusted prices so single BackTester usage can fully reproduce
                # weighted-price logic even with externally prepared data.
                self.data = get_future_ret(
                    self.data,
                    portfolio_adjust_method=self.portfolio_adjust_method,
                    rfr=self.rfr,
                )
            self._did_preprocess = True
            return

        # get factor value
        if self.apply_weighted_price:
            self.data = get_weighted_price(self.data)

        # Ensure consistent sort order for rolling/groupby operations.
        # Fusion pipeline sorts by ['instrument_id', 'time'] before computing factors;
        # BackTester must do the same to guarantee identical results.
        self.data = self.data.sort_values(['instrument_id', 'time']).reset_index(drop=True)

        if self.formula:
            # Compute factor values from formula expression
            from factors.factor_ops import calc_formula_series
            data_fields = ['open', 'high', 'low', 'close', 'volume', 'position']
            fc_name = self.fc_name_list[0]
            self.data[fc_name] = calc_formula_series(
                self.data, formula=self.formula, data_fields=data_fields,
            )
        else:
            self.data = get_factor_value(
                self.data,
                self.fc_name_list,
                version=self.version,
                collection=self.collection,
                n_jobs=self.n_jobs,
            )


        # get return as label
        self.data = get_future_ret(
            self.data,
            self.portfolio_adjust_method,
            self.rfr,
        )
        self._did_preprocess = True

        # For single-instrument TS strategy we keep raw factor values.

    def plot_nav(self,
                 fc_name: Union[str, list, None] = None,
                 instrument_id_list: Union[str, list, None] = None,
                 start_time: Union[str, pd.Timestamp, None] = None,
                 end_time: Union[str, pd.Timestamp, None] = None,
                 show_baseline: bool = False,
                 x_tick_rotation: int = 45,
                 auto_layout: bool = True,
                 close_fig: bool = True):
        """
        Plot the cumulative return(i.e. nav) graph for factor.
        e.g. bt.plot_nav()

        :return: None
        """

        if not self.is_backtested:
            raise NotBackTestingError('Need to backtest first before plotting nav.')
        if fc_name is None:
            fc_name = self.fc_name_with_param_list
        if instrument_id_list is None:
            instrument_id_list = self.instrument_id_list
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        start_ts = pd.to_datetime(start_time) if start_time is not None else None
        end_ts = pd.to_datetime(end_time) if end_time is not None else None
        if isinstance(fc_name, str):
            fc_name = [fc_name]
        if isinstance(instrument_id_list, str):
            instrument_id_list = [instrument_id_list]

        def _build_nav(ret_series: pd.Series) -> pd.Series:
            ret = pd.to_numeric(ret_series, errors='coerce').fillna(0.0)
            if self.interest_method == 'simple':
                return 1.0 + ret.cumsum()
            if self.interest_method == 'compound':
                return (1.0 + ret).cumprod()
            raise ValueError(f'Unsupported interest_method={self.interest_method}.')

        legend_drawn = False
        for instrument_id in instrument_id_list:
            for fac in fc_name:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
                # gross ret
                gross_ret = self.performance_dc[instrument_id][fac]['daily_gross_ret'].copy()
                if start_ts is not None:
                    gross_ret = gross_ret.loc[gross_ret['time'] >= start_ts]
                if end_ts is not None:
                    gross_ret = gross_ret.loc[gross_ret['time'] <= end_ts]
                gross_ret = gross_ret.set_index('time')
                cum_gross_ret = _build_nav(gross_ret[fac])
                line_gross_strategy, = ax1.plot(cum_gross_ret.index, cum_gross_ret, label='Strategy')
                ax1.set_title(f'Cumulative Gross Return of factor {fac} ({instrument_id})')
                ax1.set_xlabel('time')

                # net ret
                net_ret = self.performance_dc[instrument_id][fac]['daily_net_ret'].copy()
                if start_ts is not None:
                    net_ret = net_ret.loc[net_ret['time'] >= start_ts]
                if end_ts is not None:
                    net_ret = net_ret.loc[net_ret['time'] <= end_ts]
                net_ret = net_ret.set_index('time')
                cum_net_ret = _build_nav(net_ret[fac])
                line_net_strategy, = ax2.plot(cum_net_ret.index, cum_net_ret, label='Strategy')
                ax2.set_title(f'Cumulative Net Return of factor {fac} ({instrument_id})')
                ax2.set_xlabel('time')

                legend_handles = [line_gross_strategy]
                legend_labels = ['Strategy']

                if show_baseline:
                    baseline_key = 'daily_gross_ret_baseline'
                    if baseline_key not in self.performance_dc[instrument_id][fac]:
                        raise ValueError(
                            'Baseline series not found. Please run backtest with calculate_baseline=True.'
                        )

                    gross_baseline = self.performance_dc[instrument_id][fac]['daily_gross_ret_baseline'].copy()
                    net_baseline = self.performance_dc[instrument_id][fac]['daily_net_ret_baseline'].copy()

                    if start_ts is not None:
                        gross_baseline = gross_baseline.loc[gross_baseline['time'] >= start_ts]
                        net_baseline = net_baseline.loc[net_baseline['time'] >= start_ts]
                    if end_ts is not None:
                        gross_baseline = gross_baseline.loc[gross_baseline['time'] <= end_ts]
                        net_baseline = net_baseline.loc[net_baseline['time'] <= end_ts]

                    gross_baseline = gross_baseline.set_index('time')
                    net_baseline = net_baseline.set_index('time')

                    line_gross_long, = ax1.plot(
                        gross_baseline.index,
                        _build_nav(gross_baseline['__baseline_long__']),
                        color='tab:red',
                        linestyle='--',
                        label='Baseline Long',
                    )
                    line_gross_short, = ax1.plot(
                        gross_baseline.index,
                        _build_nav(gross_baseline['__baseline_short__']),
                        color='tab:green',
                        linestyle='--',
                        label='Baseline Short',
                    )
                    ax2.plot(
                        net_baseline.index,
                        _build_nav(net_baseline['__baseline_long__']),
                        color='tab:red',
                        linestyle='--',
                        label='Baseline Long',
                    )
                    ax2.plot(
                        net_baseline.index,
                        _build_nav(net_baseline['__baseline_short__']),
                        color='tab:green',
                        linestyle='--',
                        label='Baseline Short',
                    )

                    legend_handles = [line_gross_strategy, line_gross_long, line_gross_short]
                    legend_labels = ['Strategy', 'Baseline Long', 'Baseline Short']

                for ax in [ax1, ax2]:
                    ax.tick_params(axis='x', labelrotation=x_tick_rotation)
                    for label in ax.get_xticklabels():
                        label.set_horizontalalignment('right')

                should_show_legend = show_baseline and (not legend_drawn)
                if should_show_legend:
                    # Put legend above axes to avoid covering subplot titles.
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, 1.0),
                        ncol=3,
                        frameon=False,
                    )
                    legend_drawn = True

                if auto_layout:
                    # Reserve top margin for figure-level legend when shown.
                    top_margin = 0.90 if should_show_legend else 0.98
                    fig.tight_layout(rect=(0, 0, 1, top_margin))

                plt.show()
                if close_fig:
                    plt.close(fig)

    def backtest(self):
        """
        Use this method for backtesting.
        e.g. bt = BackTester([1, 2])
             bt.backtest()
        Then the performance will be stored on bt.performance.
        bt.ai is a dict which stores the performance of `all factors`. The key is the factor number.
        For example. bt.ai[1] stores all indicators of factor1. Using bt.ai[1].turnover to get turnover.
        """

        self._preprocess_data()

        def _run_one_instrument(instrument_id: str, factor_n_jobs: int):
            df_one = self.data[self.data['instrument_id'] == instrument_id].copy()
            _result = get_performance(df_one,
                                      self.fc_name_with_param_list,
                                      self.fc_freq,
                                      self.portfolio_adjust_method,
                                      self.interest_method,
                                      calculate_baseline=self.calculate_baseline,
                                     n_jobs=factor_n_jobs)
            return instrument_id, _result

        # Parallel strategy:
        # - single instrument: parallelize inside get_performance (across factors)
        # - multi instrument: parallelize across instruments and keep inner factor jobs to 1
        result_list = []
        if len(self.instrument_id_list) == 1:
            result_list = [_run_one_instrument(self.instrument_id_list[0], max(1, self.n_jobs))]
        else:
            parallel_jobs = min(self.n_jobs, len(self.instrument_id_list))
            with Parallel(n_jobs=parallel_jobs) as parallel:
                result_list = parallel(delayed(_run_one_instrument)(instrument_id, 1)
                                       for instrument_id in self.instrument_id_list)

        self.performance_dc = {}
        performance_summary_list = []
        performance_detail_list = []

        for instrument_id, result in result_list:
            performance_dc_i, performance_summary_i = result
            self.performance_dc[instrument_id] = performance_dc_i

            performance_summary_i = performance_summary_i.copy()
            performance_summary_i['Instrument ID'] = instrument_id
            performance_summary_list.append(performance_summary_i)

            data_i = self.data[self.data['instrument_id'] == instrument_id].copy()
            for fac in self.fc_name_with_param_list:
                fac_perf = performance_dc_i[fac]

                daily_gross = fac_perf['daily_gross_ret'][['time', fac]].rename(columns={fac: 'daily_gross_ret'})
                daily_net = fac_perf['daily_net_ret'][['time', fac]].rename(columns={fac: 'daily_net_ret'})
                daily_turnover = fac_perf['daily_turnover'][['time', fac]].rename(columns={fac: 'daily_turnover'})

                detail_i = data_i.merge(daily_gross, on='time', how='left', validate='1:1')
                detail_i = detail_i.merge(daily_net, on='time', how='left', validate='1:1')
                detail_i = detail_i.merge(daily_turnover, on='time', how='left', validate='1:1')
                detail_i = detail_i.sort_values('time').reset_index(drop=True)
                gross_ret = pd.to_numeric(detail_i['daily_gross_ret'], errors='coerce').fillna(0.0)
                net_ret = pd.to_numeric(detail_i['daily_net_ret'], errors='coerce').fillna(0.0)
                if self.interest_method == 'simple':
                    detail_i['daily_gross_nav'] = 1.0 + gross_ret.cumsum()
                    detail_i['daily_net_nav'] = 1.0 + net_ret.cumsum()
                elif self.interest_method == 'compound':
                    detail_i['daily_gross_nav'] = (1.0 + gross_ret).cumprod()
                    detail_i['daily_net_nav'] = (1.0 + net_ret).cumprod()
                else:
                    raise ValueError(f'Unsupported interest_method={self.interest_method}.')
                detail_i['factor_name'] = fac

                if fac in detail_i.columns:
                    detail_i = detail_i.rename(columns={fac: 'factor_value'})

                if 'daily_gross_ret_baseline' in fac_perf and 'daily_net_ret_baseline' in fac_perf:
                    gross_baseline = fac_perf['daily_gross_ret_baseline'][
                        ['time', '__baseline_long__', '__baseline_short__']
                    ].rename(columns={
                        '__baseline_long__': 'daily_gross_ret_baseline_long',
                        '__baseline_short__': 'daily_gross_ret_baseline_short',
                    })
                    net_baseline = fac_perf['daily_net_ret_baseline'][
                        ['time', '__baseline_long__', '__baseline_short__']
                    ].rename(columns={
                        '__baseline_long__': 'daily_net_ret_baseline_long',
                        '__baseline_short__': 'daily_net_ret_baseline_short',
                    })
                    detail_i = detail_i.merge(gross_baseline, on='time', how='left', validate='1:1')
                    detail_i = detail_i.merge(net_baseline, on='time', how='left', validate='1:1')
                    gross_base_long = pd.to_numeric(detail_i['daily_gross_ret_baseline_long'], errors='coerce').fillna(0.0)
                    gross_base_short = pd.to_numeric(detail_i['daily_gross_ret_baseline_short'], errors='coerce').fillna(0.0)
                    net_base_long = pd.to_numeric(detail_i['daily_net_ret_baseline_long'], errors='coerce').fillna(0.0)
                    net_base_short = pd.to_numeric(detail_i['daily_net_ret_baseline_short'], errors='coerce').fillna(0.0)
                    if self.interest_method == 'simple':
                        detail_i['daily_gross_nav_baseline_long'] = 1.0 + gross_base_long.cumsum()
                        detail_i['daily_gross_nav_baseline_short'] = 1.0 + gross_base_short.cumsum()
                        detail_i['daily_net_nav_baseline_long'] = 1.0 + net_base_long.cumsum()
                        detail_i['daily_net_nav_baseline_short'] = 1.0 + net_base_short.cumsum()
                    else:
                        detail_i['daily_gross_nav_baseline_long'] = (1.0 + gross_base_long).cumprod()
                        detail_i['daily_gross_nav_baseline_short'] = (1.0 + gross_base_short).cumprod()
                        detail_i['daily_net_nav_baseline_long'] = (1.0 + net_base_long).cumprod()
                        detail_i['daily_net_nav_baseline_short'] = (1.0 + net_base_short).cumprod()

                performance_detail_list.append(detail_i)

        self.performance_summary = pd.concat(performance_summary_list)
        self.performance_detail = pd.concat(performance_detail_list).sort_values(
            ['instrument_id', 'factor_name', 'time']
        ).reset_index(drop=True)

        self.is_backtested = True
        self.is_preprocessed = True
        log.info(f'Successfully generate per-instrument backtesting result for instruments '
                 f'{self.instrument_id_list} on data with frequency {self.fc_freq}')
