"""
This script is for the backtesting of signals.
"""
import matplotlib.pyplot as plt
from typing import List, Union
import pandas as pd

from .factor import *
from .factor_indicators import get_performance
from .factor_utils import get_factor_value, get_future_ret, cross_sectional_norm, join_fc_name_and_parameter
from data import get_futures_continuous_contract_price
from stats import iterdict
from utils.logging import log
from error.errors import NotBackTestingError


class BackTester:
    """
    A class supports multi-signal backtesting using Parallel.
    """

    def __init__(self,
                 fc_name_list: Union[str, List],
                 fc_freq: str = '1d',
                 data: Union[pd.DataFrame, None] = None,
                 instrument_type: str = 'futures_continuous_contract',
                 start_time: str = None,
                 end_time: str = None,
                 portfolio_adjust_method: str = '1D',
                 portfolio_number: int = 10,
                 portfolio_method: str = 'longshort',
                 interest_method: str = 'simple',
                 fee: float = 0.00025,
                 risk_free_rate: bool = True,
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

        :param data: data should be a dataframe with factors of all instrument in every bar. The factor values
            need to be preprocessed. If given data, there will only be one factor.
        :param fc_name_list: When not given data, fc_name_list is the name of factors in factor.py;
                             When given data, fc_name_list must be the columns of factor value in data.
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
        :param portfolio_method: the method for calculating sharpe, can be 'longshort' or 'long_only',
            default is 'longshort'
        :param fee: the cost of trade.
        :param portfolio_number: number of grouped portfolio
            value
        :param n_jobs: Parallel's n_job param. Default is to parallelize 5 jobs.
        """
        self.data = data
        self.instrument_type = instrument_type
        self.fc_name_list = fc_name_list
        self.start_time = start_time
        self.end_time = end_time
        self.portfolio_adjust_method = portfolio_adjust_method
        self.fc_freq = fc_freq
        self.portfolio_number = portfolio_number
        self.portfolio_method = portfolio_method
        self.interest_method = interest_method
        self.fee = fee
        self.rfr = risk_free_rate
        self.n_jobs = n_jobs

        self.fc_name_with_param_list = None
        self.performance_dc = None
        self.performance_summary = None
        self.ts_performance_dc = None
        self.ts_performance_summary = None
        self.is_backtested = False

        assert self.fc_freq in ['1m', '5m', '1d'], f'Only support 1m, 5m or 1d fc_freq, but got {fc_freq} instead.'
        assert self.portfolio_adjust_method in ['min', '1D', '1M', '1Q'], \
            f'Only support min, 1D, 1M or 1Q portfolio_adjust_method, but got {portfolio_adjust_method} instead.'
        if self.fc_freq == '1d':
            assert self.portfolio_adjust_method != 'min', 'portfolio adjust period should be equal to or longer ' \
                                                          'than factor frequency'
        if isinstance(self.fc_name_list, str):
            self.fc_name_list = [self.fc_name_list]

        # if data is provided outside
        if isinstance(self.data, pd.DataFrame):
            # when data is provided, it should have been preprocessed, and we will not preprocess it here.
            for col in ['time', 'instrument_id', 'future_ret'] + self.fc_name_list:
                assert col in self.data.columns
            self.is_preprocessed = True

        # otherwise we load data from local database
        else:
            if self.instrument_type == 'futures_continuous_contract':
                self.data = get_futures_continuous_contract_price(start_date=self.start_time,
                                                                  end_date=self.end_time,
                                                                  from_database=True)
            else:
                raise ValueError(f'Does not support instrument type {self.instrument_type}.')

            self.is_preprocessed = False

        self.fc_name_with_param_list = []
        for fc_name in fc_name_list:
            parameters = eval(fc_name).param_range
            self.fc_name_with_param_list += \
                [join_fc_name_and_parameter(fc_name, parameter) for parameter in iterdict(parameters)]

    def _preprocess_data(self):
        """
        Preprocess stock data for backtesting.
        """
        if self.is_preprocessed:
            return

        # get factor value
        self.data = get_factor_value(self.data, self.fc_name_list, self.n_jobs)

        # get return as label
        self.data = get_future_ret(self.data, self.portfolio_adjust_method, self.rfr)

        # standardize factor values to zscore
        self.data = cross_sectional_norm(self.data, self.fc_name_with_param_list)

    def plot_nav(self, fc_name: Union[str, list, None] = None):
        """
        Plot the cumulative return(i.e. nav) graph for factor.
        e.g. bt.plot_nav()

        :return: None
        """

        if not self.is_backtested:
            raise NotBackTestingError('Need to backtest first before plotting nav.')
        if fc_name is None:
            fc_name = self.fc_name_list
        if isinstance(fc_name, str):
            fc_name = [fc_name]

        for fac in fc_name:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            # gross ret
            gross_ret = self.performance_dc[fac]['daily_gross_ret'].copy()
            gross_ret = gross_ret.set_index('time')
            cum_gross_ret = (1 + gross_ret['ret']).cumprod()
            ax1.plot(cum_gross_ret.index, cum_gross_ret)
            ax1.set_title(f'Cumulative Gross Return of factor {fac}')
            ax1.set_xlabel('time')

            # net ret
            net_ret = self.performance_dc[fac]['daily_net_ret'].copy()
            net_ret = net_ret.set_index('time')
            cum_net_ret = (1 + net_ret['ret']).cumprod()
            ax2.plot(cum_net_ret.index, cum_net_ret)
            ax2.set_title(f'Cumulative Net Return of factor {fac}')
            ax2.set_xlabel('time')

    def plot_longshort(self, fc_name: Union[str, list, None] = None):
        """
        Plot the ret for n-largest and n-smallest signal values, with n equals to self.longshort_instrument_number.
        e.g. the factor value from ret_short5, ret_short_4, ..., ret_short_1, ret_long_1, ret_long_2, ..., ret_long_5
            is becoming larger.
        e.g. bt.plot_longshort()
        """
        if not self.is_backtested:
            raise NotBackTestingError('Need to backtest first before plotting nav.')
        if fc_name is None:
            fc_name = self.fc_name_list
        if isinstance(fc_name, str):
            fc_name = [fc_name]

        for fac in fc_name:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            ret_cols = [fac + ' ' + str(i) for i in range(1, self.portfolio_number + 1)]

            # gross ret
            gross_ret = self.performance_dc[fac]['daily_gross_ret'].set_index('time')[ret_cols]
            cum_gross_ret = (1 + gross_ret).cumprod()
            for col in cum_gross_ret.columns:
                ax1.plot(cum_gross_ret.index, cum_gross_ret[col], label=col)
            ax1.legend()
            ax1.set_title(f'Gross Longshort ret of factor {fac}')
            ax1.set_xlabel('time')

            # net ret
            net_ret = self.performance_dc[fac]['daily_net_ret'].set_index('time')[ret_cols]
            cum_net_ret = (1 + net_ret).cumprod()
            for col in cum_net_ret.columns:
                ax2.plot(cum_net_ret.index, cum_net_ret[col], label=col)
            ax2.legend()
            ax2.set_title(f'Gross Longshort ret of factor {fac}')
            ax2.set_xlabel('time')

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

        self.performance_dc, self.performance_summary, \
            self.ts_performance_dc, self.ts_performance_summary = get_performance(self.data,
                                                                                  self.fc_name_with_param_list,
                                                                                  self.fc_freq,
                                                                                  self.portfolio_adjust_method,
                                                                                  self.portfolio_number,
                                                                                  self.portfolio_method,
                                                                                  self.interest_method,
                                                                                  self.fee,
                                                                                  self.n_jobs)

        self.is_backtested = True
        self.is_preprocessed = True
        log.info(f'Successfully generate backtesting result for all factors '
                 f'on data with frequency {self.fc_freq}')
