"""
Models for factor generation.

Sklearn models: https://scikit-learn.org/stable/user_guide.html

XGBoost: https://xgboost.readthedocs.io/en/stable/parameter.html
** How to customize eval_metric on XGBoost: https://blog.csdn.net/weixin_38100489/article/details/78714251
** XGboost params understanding: https://blog.csdn.net/VariableX/article/details/107238137

Light-Gradient Boosting Machine (LGBM)
** official doc: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
** detailed blog: https://blog.csdn.net/phasorhand/article/details/123336615
                      https://blog.csdn.net/VariableX/article/details/107256149
"""
import os
import warnings
import datetime
import pandas as pd
import numpy as np
from typing import Union

import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from .params import RegressorParamGrid, DefaultParamDict
from factors import get_future_ret, BackTester
from data import get_futures_continuous_contract_price, get_factor_value
from utils import generate_date_strings, OUTPUT_PATH, get_attribute_value

warnings.filterwarnings("ignore")


class BaseModel:
    """
    The basic model object.
    ** For boosting models:
        1. search: early stopping based on eval_metric on each search, using eval_metric(e.g. ic) to search for the
                   best param
            (1) get best params using search. When searching, we do not operate early stopping on each param
                combination,and we use eval_metric on validate dataset as the criterion for search.
            (2) use best params from the search result and early stopping to get the best model.
                * param `scoring` of GridSearchCV or RandomizedSearchCV indicates eval_metric

        2. non-search: early stopping on given params
            * param_dc contains param indicating early stopping rounds (only for lightgbm)
            * fit method of the model contains param `eval_metric`
    ** For non-boosting models:
        no need to early stop, so no need to specify validate set (eval_set)
        1. search: get best model among all searches, using eval_metric to search for the best param
        2. non-search: get the model on given params, do not use eval_metric
    """
    validate_index = None
    test_index = None
    model_type = 'basemodel'

    def __init__(self,
                 fc_name_list: Union[str, list],
                 signal_name: str = None,
                 param_dc: Union[DefaultParamDict, None] = None,
                 param_grid: Union[RegressorParamGrid, None] = None,
                 search_type: Union[str, bool] = False,
                 search_number: int = 10,
                 early_stopping_rounds: int = 50,
                 eval_metric: str = 'ic',
                 fc_freq: str = '1d',
                 version: Union[str, None] = None,
                 start_time: Union[str, datetime.date, datetime.datetime, pd.Timestamp, None] = None,
                 end_time: Union[str, datetime.date, datetime.datetime, pd.Timestamp, None] = None,
                 transaction_period: Union[int, None] = None,
                 ret_freq: int = 30,
                 rfr: bool = True,
                 train_range: int = 18,
                 validate_range: int = 6,
                 test_range: int = 6,
                 rolling_range: int = 6,
                 save: bool = False,
                 backtest: bool = False,
                 interest_method: str = 'simple',
                 fee: float = 0.00025,
                 n_jobs: int = 5):
        """
        Params to be initialized for models.

        :param fc_name_list: the factors (i.e. features) for models
        :param signal_name: name of the signal which generated from model
        :param param_dc: param dict for the model, if not provided, using default dict.
        :param param_grid: param gird for the model, this is for grid or random search. if not provided,
            using default grid
        :param search_type: str, can be 'grid' for grid search, or 'random' for random search or False for not search
        :param search_number: param for random search, meaning the number of param combination of random search
        :param early_stopping_rounds: param for boosting models, operating early stopping
        :param eval_metric: eval_metric for early stopping for boosting models
        :param fc_freq: the frequency of factor, 1m, 5m or 1d.
        :param start_time: backtesting start time, default is START_DATE.
        :param end_time: backtesting start time, default is END_DATE.
        :param transaction_period: assumed period (number of bars) on average that a transaction lasts.
        :param ret_freq: we calculate return as the return between current bar and the next `ret_freq` bars
        :param rfr: whether to consider risk-free rate when calculating excess return
        :param train_range: data range for train dataset, unit is month.
        :param validate_range: data range for validate dataset, unit is month.
        :param test_range: data range for test dataset, unit is month.
        :param rolling_range: data range for rolling training, unit is month.
        :param save: If True, saving the predict result of the model.
        :param backtest: If True, backtesting the performance of the predict result (i.e. factor value) of the model.
        """

        self.best_model_name = None
        self.fc_name_list = fc_name_list
        self.param_dc = param_dc
        self.param_grid = param_grid
        self.search_type = search_type
        self.search_number = search_number
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.fc_name_list = fc_name_list
        self.signal_name = signal_name
        self.fc_freq = fc_freq
        self.version = version
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_period = transaction_period
        self.ret_freq = ret_freq
        self.rfr = rfr
        self.train_range = train_range
        self.validate_range = validate_range
        self.test_range = test_range
        self.rolling_range = rolling_range
        self.all_range = train_range + validate_range + test_range
        self.save = save
        self.backtest = backtest
        self.interest_method = interest_method
        self.fee = fee
        self.n_jobs = n_jobs

        if self.search_type:
            assert self.search_type in ['grid', 'random']
        if not self.signal_name:
            self.signal_name = self.model_type
        self.split_date_list = None
        self.data = get_futures_continuous_contract_price(start_date=self.start_time,
                                                          end_date=self.end_time,
                                                          from_database=True)
        self.data_cols = ['open', 'high', 'low', 'close', 'volume']

        self.is_preprocessed = False

        # rolling train generates multiple models (model of one type but with different params)
        self.model = None
        self.model_list = []

        # prediction result on test dataset
        self.predict_result = None
        self.is_trained = False

        self._preprocess_data()
        self.get_split_date_list()

        self.bt = None
        self.performance_dc = None
        self.performance_summary = None
        self.ts_performance_dc = None
        self.ts_performance_summary = None

        # three attributes:
        # 1. eval_metric: str, the eval metric
        # 2. scoring: dict, using for param `scoring` of search
        # 3. eval_metric_func: func return eval_metric
        score_ic = make_scorer(self.eval_metric_ic_for_search, greater_is_better=True)
        score_top_ret = make_scorer(self.eval_metric_top_ret_for_search, greater_is_better=True)
        if self.eval_metric == 'ic':
            self.scoring = {'ic': score_ic}
            self.eval_metric_func = self.eval_metric_ic
        elif self.eval_metric == 'top_ret':
            self.scoring = {'top_ret': score_top_ret}
            self.eval_metric_func = self.eval_metric_top_ret
        else:
            raise NotImplementedError

        if self.search_type:
            assert self.param_dc is None, 'When using search, should not specify any params.'
            if self.param_grid is None:
                self.param_grid = get_attribute_value(RegressorParamGrid, self.model_type)
        else:
            assert self.param_grid is None, 'When not using search, should not specify param grid.'
            if self.param_dc is None:
                self.param_dc = get_attribute_value(DefaultParamDict, self.model_type)

        self.is_boosting_model = None

    def _preprocess_data(self):
        for col in ['time', 'instrument_id']:
            assert col in self.data.columns, f'self.data does not contain column {col}.'
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError('BaseModel requires non-empty `version` to load factor formulas from DB.')
        # get factor value
        self.data = get_factor_value(self.data, self.fc_name_list, version=self.version, n_jobs=self.n_jobs)
        # get return as label
        self.data = get_future_ret(self.data, portfolio_adjust_method='1D', rfr=self.rfr)

        self.data = self.data.sort_values(by='time', ascending=True)
        self.data['date'] = self.data['time'].dt.strftime('%Y%m%d')
        assert all(self.data['date'].str.len() == 8)
        self.data['year_month_str'] = self.data['date'].str[:6]
        self.data = self.data.dropna(how='any', axis=0)
        self.data = self.data.set_index(['time', 'instrument_id'])

        self.is_preprocessed = True

    def get_split_date_list(self):
        start_year = int(self.data['year_month_str'].iloc[0][:4])
        start_month = int(self.data['year_month_str'].iloc[0][4:])
        end_year = int(self.data['year_month_str'].iloc[-1][:4])
        end_month = int(self.data['year_month_str'].iloc[-1][4:])
        date_list = generate_date_strings(start_year, start_month, end_year, end_month)
        all_date_list = \
            [date_list[start: start + self.all_range] for start in range(0, len(date_list), self.rolling_range)]
        all_date_list = [i for i in all_date_list if len(i) == self.all_range]
        self.split_date_list = [[t[:self.train_range],
                                 t[self.train_range: self.train_range + self.validate_range],
                                 t[self.train_range + self.validate_range:]] for t in all_date_list]

    @property
    def model_data_generator(self):
        assert self.is_preprocessed, f'Need to preprocess data first!'
        self.get_split_date_list()
        df = self.data.copy()
        assert 'year_month_str' in df.columns
        for split_date in self.split_date_list:
            X_train = df.loc[df['year_month_str'].isin(split_date[0])][self.fc_name_list].reset_index(drop=True)
            y_train = df.loc[df['year_month_str'].isin(split_date[0])]['future_ret'].reset_index(drop=True)
            X_validate = df.loc[df['year_month_str'].isin(split_date[1])][self.fc_name_list]
            y_validate = df.loc[df['year_month_str'].isin(split_date[1])]['future_ret']
            X_test = df.loc[df['year_month_str'].isin(split_date[2])][self.fc_name_list].reset_index(drop=True)
            y_test = df.loc[df['year_month_str'].isin(split_date[2])]['future_ret']
            BaseModel.validate_index = X_validate.index
            BaseModel.test_index = X_test.index
            X_validate.reset_index(drop=True)
            y_validate.reset_index(drop=True)

            print('\n\n', '-' * 100)
            print(f'train dataset from {split_date[0][0]} to {split_date[0][-1]}, '
                  f'validate dataset from {split_date[1][0]} to {split_date[1][-1]}, '
                  f'test dataset from {split_date[2][0]} to {split_date[2][-1]}.')

            yield X_train, y_train, X_validate, y_validate, X_test, y_test

    def train_for_given_params(self):
        """
        train method for models which given params
        """
        assert not self.search_type
        if self.model_type == 'basemodel':
            raise NotImplementedError(f'Should not use train method in the BaseModel class.')
        # add early stopping condition
        if self.model_type == 'LGBMRegressor':
            self.param_dc['early_stopping_rounds'] = self.early_stopping_rounds
            self.param_dc['metric'] = None
        elif self.search_type == 'GradientBoostingRegressor':
            self.param_dc['n_iter_no_change'] = self.early_stopping_rounds
        else:
            pass

        model = eval(self.model_type)(**self.param_dc)
        # df_pred_result is the prediction result of model on test dataset
        df_pred_result_list = []
        for X_train, y_train, X_validate, y_validate, X_test, y_test in self.model_data_generator:

            if self.is_boosting_model:
                model.fit(X_train, y_train,
                          eval_set=[(X_validate, y_validate)],
                          eval_metric=self.eval_metric_func)
            else:
                model.fit(X_train, y_train)

            self.model_list.append(model)
            # prediction on test dataset
            y_pred = model.predict(X_test)
            df_pred_result_list.append(pd.DataFrame(data={self.signal_name: y_pred, 'future_ret': y_test}, index=y_test.index))

        self.predict_result = pd.concat(df_pred_result_list)
        # Keep raw signal values for single-instrument TS strategy.
        self.predict_result = self.predict_result.reset_index()

        if self.save:
            self.save_predict_result()

        if self.backtest:
            self.backtest_predict_result()

        self.is_trained = True

    def train_for_search(self):
        """
        train method for grid or random search
        """
        assert self.search_type
        if self.model_type == 'basemodel':
            raise NotImplementedError(f'Should not use train method in the BaseModel class.')
        estimator = eval(self.model_type)()

        df_pred_result_list = []
        for X_train, y_train, X_validate, y_validate, X_test, y_test in self.model_data_generator:
            X_train_val = np.concatenate((X_train.values, X_validate.values), axis=0)
            y_train_val = np.concatenate((y_train.values, y_validate.values), axis=0)
            assert X_train_val.shape[0] == y_train_val.shape[0]

            test_fold = np.zeros(X_train_val.shape[0])
            # set the index of data belong to train set to -1, meaning that they will never be regarded as validate set.
            test_fold[:X_train.shape[0]] = -1
            ps = PredefinedSplit(test_fold=test_fold)

            ## First, search for the best param based on eval_metric
            if self.search_type == 'grid':  # gird search
                search = GridSearchCV(estimator=estimator,
                                      param_grid=self.param_grid,
                                      scoring=self.scoring,
                                      refit=self.eval_metric,
                                      cv=ps)
            else:  # random search
                search = RandomizedSearchCV(estimator=estimator,
                                            param_distributions=self.param_grid,
                                            n_iter=self.search_number,
                                            scoring=self.scoring,
                                            refit=self.eval_metric,
                                            cv=ps)
            search.fit(X_train_val, y_train_val)

            ## Then, using best params and early stopping to fit the best model.
            self.param_dc = search.best_params_
            if self.model_type == 'LGBMRegressor':
                self.param_dc['early_stopping_rounds'] = self.early_stopping_rounds
                self.param_dc['metric'] = None
            elif self.search_type == 'GradientBoostingRegressor':
                self.param_dc['n_iter_no_change'] = self.early_stopping_rounds
            else:
                pass

            model = eval(self.model_type)(**self.param_dc)

            if self.is_boosting_model:
                model.fit(X_train, y_train,
                          eval_set=[(X_validate, y_validate)],
                          eval_metric=self.eval_metric_func)
            else:
                model.fit(X_train, y_train)

            self.model_list.append(model)
            y_pred = model.predict(X_test)
            df_pred_result_list.append(pd.DataFrame(data={self.signal_name: y_pred, 'future_ret': y_test}, index=y_test.index))

        self.predict_result = pd.concat(df_pred_result_list)
        # Keep raw signal values for single-instrument TS strategy.
        self.predict_result = self.predict_result.reset_index()

        self.is_trained = True

        if self.save:
            self.save_predict_result()

        if self.backtest:
            self.backtest_predict_result()

    def train(self):
        if self.search_type:
            self.train_for_search()
        else:
            self.train_for_given_params()

    @staticmethod
    def eval_metric_ic(y_true, y_pred):
        """
        customized eval_metric, based on ic

        :param y_true: true label
        :param y_pred: predict label
        :return:
        """
        pass

    @staticmethod
    def eval_metric_top_ret(y_true, y_pred):
        """
        customized eval_metric, based on top ret

        :param y_true: true label
        :param y_pred: predict label
        :return:
        """
        pass

    @staticmethod
    def eval_metric_ic_for_search(y_true, y_pred):
        """
        customized eval_metric for grid or random search, based on ic

        :param y_true: true label
        :param y_pred: predict label
        :return:
        """
        df = pd.DataFrame(data={'signal': y_pred, 'ret': y_true}, index=BaseModel.validate_index)
        ic = df.groupby('time').corr('spearman').loc[(slice(None), 'signal'), ['ret']].droplevel(1).mean().values[0]

        return ic

    @staticmethod
    def eval_metric_top_ret_for_search(y_true, y_pred):
        """
        customized eval_metric for grid or random search, based on top ret

        :param y_true: true label
        :param y_pred: predict label
        :return:
        """
        df = pd.DataFrame(data={'signal': y_pred, 'ret': y_true}, index=BaseModel.validate_index)
        top_ret = df.groupby('time').apply(lambda x: x.nlargest(10, 'signal')['ret'].mean()).mean()

        return top_ret

    def save_predict_result(self):
        self.get_best_model_name()
        assert self.predict_result is not None
        assert self.best_model_name is not None

        factor_name = self.best_model_name + '.pkl'
        file_path = os.path.join(OUTPUT_PATH, factor_name)
        self.predict_result.to_pickle(file_path)

    def get_best_model_name(self):
        if not self.search_type:
            self.best_model_name = '&'.join([key + '-' + str(val) for key, val in self.param_dc.items()])
        else:
            # todo: find a good name for the model on gird and random search
            assert self.is_trained
            self.best_model_name = self.search_type + '_search_' + datetime.date.today().strftime('%Y%m%d')

    def backtest_predict_result(self):
        instrument_id_list = sorted(self.predict_result['instrument_id'].dropna().unique().tolist())
        bt = BackTester(fc_name_list=[self.signal_name],
                        version=self.version or 'external_input',
                        instrument_id_list=instrument_id_list,
                        data=self.predict_result,
                        fc_freq=self.fc_freq,
                        interest_method=self.interest_method)

        bt.backtest()
        self.bt = bt
        self.performance_dc = bt.performance_dc
        self.performance_summary = bt.performance_summary
        self.ts_performance_dc = bt.performance_dc
        self.ts_performance_summary = bt.performance_summary

    def plot_nav(self, net: bool = True):
        self.bt.plot_nav(self.signal_name)


class LinearRegressor(BaseModel):
    model_type = 'LinearRegression'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = False


class SupportedVectorMachine(BaseModel):
    model_type = 'SVR'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = False


class DecisionTree(BaseModel):
    model_type = 'DecisionTreeRegressor'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = False


class RandomForest(BaseModel):
    model_type = 'RandomForestRegressor'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = False


class GradientBoostingDecisionTree(BaseModel):
    """
    Currently no use.
    ** Maybe useful if we do not implement early stopping.
    """
    model_type = 'GradientBoostingRegressor'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = True

    def train(self):
        if self.search_type:
            raise NotImplementedError('GBDT does not support customize validate set, so it is unable to implement '
                                      'early stopping, so we do not use GBDT in any case. '
                                      'Try to switch to LightGBM or XGboost instead.')
        else:
            raise NotImplementedError('When using GradientBoostingRegressor, should only use grid or random search. '
                                      'This is because gbdt does not support customized eval metric and eval set.')


class XGradientBoosting(BaseModel):
    model_type = 'XGBoost'

    def __init__(self, num_boost_round: int = 200, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_boost_round = num_boost_round
        self.is_boosting_model = True

    def train_for_given_params(self):
        assert not self.search_type

        self.param_dc['maximize'] = True  # meaning that we need to maximize the eval metric (e.g. ic)
        self.param_dc['disable_default_eval_metric'] = True  # do not use default eval_metric

        # df_pred_result is the prediction result of model on test dataset
        df_pred_result_list = []
        for X_train, y_train, X_validate, y_validate, X_test, y_test in self.model_data_generator:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_validate, label=y_validate)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(params=self.param_dc,
                              dtrain=dtrain,
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping_rounds,
                              evals=[(dval, 'validation')],
                              feval=self.eval_metric_ic)
            self.model_list.append(model)

            y_pred = model.predict(dtest)
            df_pred_result_list.append(pd.DataFrame(data={'signal': y_pred, 'future_ret': y_test}, index=y_test.index))

        self.predict_result = pd.concat(df_pred_result_list)

        self.is_trained = True

        if self.save:
            self.save_predict_result()

        if self.backtest:
            self.backtest_predict_result()

    def train_for_search(self):
        assert self.search_type

        estimator = xgb.XGBRegressor()

        df_pred_result_list = []
        for X_train, y_train, X_validate, y_validate, X_test, y_test in self.model_data_generator:
            X_train_val = np.concatenate((X_train.values, X_validate.values), axis=0)
            y_train_val = np.concatenate((y_train.values, y_validate.values), axis=0)
            assert X_train_val.shape[0] == y_train_val.shape[0]

            test_fold = np.zeros(X_train_val.shape[0])
            # set the index of data belong to train set to -1, meaning that they will never be regarded as validate set.
            test_fold[:X_train.shape[0]] = -1
            ps = PredefinedSplit(test_fold=test_fold)

            ## First, search for the best param based on eval_metric
            if self.search_type == 'grid':  # gird search
                search = GridSearchCV(estimator=estimator,
                                      param_grid=self.param_grid,
                                      scoring=self.scoring,
                                      refit=self.eval_metric,
                                      cv=ps)
            else:  # random search
                search = RandomizedSearchCV(estimator=estimator,
                                            param_distributions=self.param_grid,
                                            n_iter=self.search_number,
                                            scoring=self.scoring,
                                            refit=self.eval_metric,
                                            cv=ps)
            search.fit(X_train_val, y_train_val)

            ## Then, using best params and early stopping to fit the best model.
            self.param_dc = search.best_params_
            self.param_dc['maximize'] = True  # meaning that we need to maximize the eval metric (e.g. ic)
            self.param_dc['disable_default_eval_metric'] = True  # do not use default eval_metric

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_validate, label=y_validate)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(params=self.param_dc,
                              dtrain=dtrain,
                              num_boost_round=self.num_boost_round,
                              early_stopping_rounds=self.early_stopping_rounds,
                              evals=[(dval, 'validation')],
                              feval=self.eval_metric_ic)

            self.model_list.append(model)
            y_pred = model.predict(dtest)
            df_pred_result_list.append(pd.DataFrame(data={'signal': y_pred, 'future_ret': y_test}, index=y_test.index))

        self.predict_result = pd.concat(df_pred_result_list)

        self.is_trained = True

        if self.save:
            self.save_predict_result()

        if self.backtest:
            self.backtest_predict_result()

    def train(self):
        if self.search_type:
            self.train_for_search()
        else:
            self.train_for_given_params()

    @staticmethod
    def eval_metric_ic(y_true, y_pred):
        y = y_pred.get_label()
        df = pd.DataFrame(data={'signal': y, 'ret': y_true}, index=BaseModel.validate_index)
        ic = df.groupby('time').corr('spearman').loc[(slice(None), 'signal'), ['ret']].droplevel(1).mean().values[0]
        return 'ic', ic


class LightGBM(BaseModel):
    model_type = 'LGBMRegressor'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_boosting_model = True

    @staticmethod
    def eval_metric_ic(y_true, y_pred):
        df = pd.DataFrame(data={'signal': y_pred, 'ret': y_true}, index=BaseModel.validate_index)
        ic = df.groupby('time').corr('spearman').loc[(slice(None), 'signal'), ['ret']].droplevel(1).mean().values[0]
        # True means the bigger, the better
        return 'ic', ic, True

    @staticmethod
    def eval_metric_top_ret(y_true, y_pred):
        df = pd.DataFrame(data={'signal': y_pred, 'ret': y_true}, index=BaseModel.validate_index)
        top_ret = df.groupby('time').apply(lambda x: x.nlargest(10, 'signal')['ret'].mean())
        # True means the bigger, the better
        return 'top_ret', top_ret.mean(), True
