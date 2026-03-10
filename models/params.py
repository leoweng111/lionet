"""
The default param for models and params for grid and random search of models.
"""
import numpy as np


class RegressorParamGrid:
    """
    Param gird for regressors.
    """
    # Linear Regression
    LinearRegression = {'fit_intercept': [True, False]}

    # SVM
    SVR = {'C': np.arange(0, 5, 0.2),
           'kernel': ['linear', 'poly', 'rbf']}
    # Decision Tree
    DecisionTreeRegressor = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                             'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                             'min_samples_split': np.arange(0, 1, 0.1),
                             'min_samples_leaf': np.arange(0, 1, 0.1),
                             'max_features': np.arange(0.5, 1, 0.1)}
    # Random Forest
    RandomForestRegressor = {'n_estimators': [20, 50, ],
                             'criterion': ['squared_error', 'absolute_error'],
                             'max_depth': [5, 10],
                             'min_samples_split': [0.1, 0.5, 0.8],
                             'min_samples_leaf': np.arange(0.1, 1, 0.3),
                             'max_features': np.arange(0.5, 1, 0.2)}
    # GBDT
    GradientBoostingRegressor = {'learning_rate': np.arange(0.01, 0.5, 0.01),
                                 'n_estimators': [20, 50, 80, 100, 120, 150, 180, 200],
                                 'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                                 'subsample': [0.7, 0.8, 0.9, 1.0],
                                 'min_samples_split': np.arange(0, 1, 0.1),
                                 'min_samples_leaf': np.arange(0, 1, 0.1),
                                 'loss': ['deviance', 'exponential'],
                                 'max_features': np.arange(0.5, 1, 0.1)}

    XGBoost = {'objective': ['reg:squarederror'],
               'booster': ['gbtree'],
               'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
               'gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
               'alpha': [0, 0.5, 1],
               'max_depth': [3, 5, 6, 7, 9, 12, 15, 17, 25],
               'min_child_weight': [1, 3, 5, 7],
               'subsample': [0.6, 0.7, 0.8, 0.9, 1]}

    LGBMRegressor = {'n_estimators': [100, 200, 300, 400, 500],
                     'min_child_samples': [64, 128, 256]}


class DefaultParamDict:
    LinearRegression = {'fit_intercept': False}

    SVR = {'C': 3,
           'kernel': 'linear'}

    DecisionTreeRegressor = {'criterion': 'gini',
                             'max_depth': 100,
                             'min_samples_split': 0.5,
                             'min_samples_leaf': 0.5,
                             'max_features': 0.5}
    # Random Forest
    RandomForestRegressor = {'n_estimators': [20, 50, ],
                             'criterion': ['squared_error', 'absolute_error'],
                             'max_depth': [5, 10],
                             'min_samples_split': [0.1, 0.5, 0.8],
                             'min_samples_leaf': np.arange(0.1, 1, 0.3),
                             'max_features': np.arange(0.5, 1, 0.2)}

    GradientBoostingRegressor = {'learning_rate': 0.05,
                                 'n_estimators': 300,
                                 'max_depth': 50,
                                 'subsample': 1,
                                 'min_samples_split': 0.6,
                                 'min_samples_leaf': 0.6,
                                 'loss': 'exponential',
                                 'max_features': 0.6,
                                 'n_iter_no_change': 50}

    XGBoost = {'objective': 'reg:squarederror',
               'booster': 'gbtree',
               'learning_rate': 0.05,
               'gamma': 0.1}

    LGBMRegressor = {'seed': 29,
                     'boosting_type': 'gbdt',
                     'n_jobs': -1,
                     'objective': 'regression',
                     'n_estimators': 300,
                     'first_metric_only': True,
                     'max_depth': -1,
                     'metric': 'None',
                     'learning_rate': 0.05,
                     'num_leaves': 31,
                     'min_child_samples': 256,
                     'subsample': 0.9,
                     'subsample_freq': 2,
                     'colsample_bytree': 0.9,
                     'reg_alpha': 0,
                     'reg_lambda': 0.0001}
