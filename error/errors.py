"""
Error classes.
"""


class Error(Exception):
    pass


class DataBaseError(Error):
    pass


class BackTestError(Error):
    pass


class StockDataError(DataBaseError):
    pass


class NotBackTestingError(BackTestError):
    pass
