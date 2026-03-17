"""
This script is for the backtesting of signals.
"""
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Union
import pandas as pd
from joblib import Parallel, delayed

from .factor_indicators import get_performance
from .factor_utils import get_factor_value, get_future_ret, join_fc_name_and_parameter, resolve_factor_class
from data import get_futures_continuous_contract_price
from stats import iterdict
from utils.logging import log
from error.errors import NotBackTestingError


class BackTester:
    """
    A class for single-instrument, multi-signal time-series backtesting.
    """

    def __init__(self,
                 fc_name_list: Union[str, List],
                 instrument_type: str = 'futures_continuous_contract',
                 instrument_id_list: Union[str, List] = 'C0',
                 fc_freq: str = '1d',
                 data: Union[pd.DataFrame, None] = None,
                 start_time: str = None,
                 end_time: str = None,
                 portfolio_adjust_method: str = '1D',
                 interest_method: str = 'compound',
                 risk_free_rate: bool = False,
                 n_jobs: int = 5):
        # todo: 数据频率，调仓频率和收益率计算频率三者相关
        # 对数据频率1min，调仓频率为1day情况，计算IC需要先将一分钟的数据聚合成一天的，然后计算日频收益率，再计算日频因子值和日频收益率
        # 之间的相关系数。
        # 时序收益率
        """
        Some params need to be initialized for backtesting.
        We assume that the signal is generated at the open time of every bar, and we immediately trade at this bar
        with average transaction price (open + high + low + close) / 4 of this bar, and will complete the transaction
        at the close time of the next `transaction_period` bar

        :param data: data should be a dataframe with high, open, low, close, volume, position columns 
            for one instrument in each bar.
        :param fc_name_list: When not given data, fc_name_list is factor class names in factor.py;
                             When given data, fc_name_list must be factor/signal columns in data.
                             So fc_name_list must be provided.
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
        self.instrument_id_list = instrument_id_list
        self.start_time = start_time
        self.end_time = end_time
        self.portfolio_adjust_method = portfolio_adjust_method
        self.fc_freq = fc_freq
        self.interest_method = interest_method
        self.rfr = risk_free_rate
        self.n_jobs = n_jobs

        self.fc_name_with_param_list = None
        self.performance_dc = None
        self.performance_summary = None
        self.ts_performance_summary = None
        self.is_backtested = False
        self.is_preprocessed = isinstance(self.data, pd.DataFrame)

        assert self.fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, but got {fc_freq} instead.'
        assert self.portfolio_adjust_method in ['min', '1D', '1M', '1Q'], \
            f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, but got {portfolio_adjust_method} instead.'
        if self.fc_freq == '1d':
            assert self.portfolio_adjust_method != 'min', 'portfolio adjust period should be equal to or longer ' \
                                                          'than factor frequency'
        if isinstance(self.fc_name_list, str):
            self.fc_name_list = [self.fc_name_list]
        fc_name_counter = Counter(self.fc_name_list)
        duplicated_fc_names = sorted([x for x, cnt in fc_name_counter.items() if cnt > 1])
        if duplicated_fc_names:
            raise ValueError(f'fc_name_list contains duplicated factor names: {duplicated_fc_names}')
        if isinstance(self.instrument_id_list, str):
            self.instrument_id_list = [self.instrument_id_list]

        # if data is provided outside
        if isinstance(self.data, pd.DataFrame):
            # when data is provided, it should have been preprocessed, and we will not preprocess it here.
            for col in ['time', 'instrument_id', 'future_ret'] + self.fc_name_list:
                assert col in self.data.columns

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

        self.fc_name_with_param_list = []
        for fc_name in self.fc_name_list:
            if self.is_preprocessed and fc_name in self.data.columns:
                self.fc_name_with_param_list.append(fc_name)
                continue

            parameters = resolve_factor_class(fc_name).param_range
            self.fc_name_with_param_list += [
                join_fc_name_and_parameter(fc_name, parameter) for parameter in iterdict(parameters)
            ]

    def _preprocess_data(self):
        """
        Preprocess futures data for time-series backtesting.
        """
        if self.is_preprocessed:
            return

        # get factor value
        self.data = get_factor_value(self.data, self.fc_name_list, self.n_jobs)

        # get return as label
        self.data = get_future_ret(self.data, self.portfolio_adjust_method, self.rfr)

        # For single-instrument TS strategy we keep raw factor values.

    def plot_nav(self,
                 fc_name: Union[str, list, None] = None,
                 instrument_id_list: Union[str, list, None] = None,
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
        if isinstance(fc_name, str):
            fc_name = [fc_name]
        if isinstance(instrument_id_list, str):
            instrument_id_list = [instrument_id_list]

        for instrument_id in instrument_id_list:
            for fac in fc_name:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
                # gross ret
                gross_ret = self.performance_dc[instrument_id][fac]['daily_gross_ret'].copy()
                gross_ret = gross_ret.set_index('time')
                cum_gross_ret = (1 + gross_ret[fac]).cumprod()
                ax1.plot(cum_gross_ret.index, cum_gross_ret)
                ax1.set_title(f'Cumulative Gross Return of factor {fac} ({instrument_id})')
                ax1.set_xlabel('time')

                # net ret
                net_ret = self.performance_dc[instrument_id][fac]['daily_net_ret'].copy()
                net_ret = net_ret.set_index('time')
                cum_net_ret = (1 + net_ret[fac]).cumprod()
                ax2.plot(cum_net_ret.index, cum_net_ret)
                ax2.set_title(f'Cumulative Net Return of factor {fac} ({instrument_id})')
                ax2.set_xlabel('time')

                for ax in [ax1, ax2]:
                    ax.tick_params(axis='x', labelrotation=x_tick_rotation)
                    for label in ax.get_xticklabels():
                        label.set_horizontalalignment('right')

                if auto_layout:
                    fig.tight_layout()

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

        if not self.is_preprocessed:
            self._preprocess_data()

        def _run_one_instrument(instrument_id: str, factor_n_jobs: int):
            df_one = self.data[self.data['instrument_id'] == instrument_id].copy()
            _result = get_performance(df_one,
                                      self.fc_name_with_param_list,
                                      self.fc_freq,
                                      self.portfolio_adjust_method,
                                      self.interest_method,
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
        ts_performance_summary_list = []

        for instrument_id, result in result_list:
            performance_dc_i, performance_summary_i = result
            self.performance_dc[instrument_id] = performance_dc_i

            performance_summary_i = performance_summary_i.copy()
            performance_summary_i['Instrument ID'] = instrument_id
            performance_summary_list.append(performance_summary_i)

            ts_performance_summary_i = performance_summary_i.copy()
            ts_performance_summary_i['Instrument ID'] = instrument_id
            ts_performance_summary_list.append(ts_performance_summary_i)

        self.performance_summary = pd.concat(performance_summary_list)
        self.ts_performance_summary = pd.concat(ts_performance_summary_list)

        self.is_backtested = True
        self.is_preprocessed = True
        log.info(f'Successfully generate per-instrument backtesting result for instruments '
                 f'{self.instrument_id_list} on data with frequency {self.fc_freq}')
