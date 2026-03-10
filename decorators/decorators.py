"""
Useful decorators.
"""


class FormatCheck:
    def __init__(self, arg1, arg2):  # init()方法里面的参数都是装饰器的参数
        print('执行类Decorator的__init__()方法')
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, func):  # 因为装饰器带了参数，所以接收传入函数变量的位置是这里
        print('执行类Decorator的__call__()方法')

        def wrapper(*args, **kwargs):  # 这里装饰器的函数名字可以随便命名，只要跟return的函数名相同即可
            print('执行wrap()')
            print('装饰器参数：', self.arg1, self.arg2)
            print('执行' + func.__name__ + '()')
            func(*args, **kwargs)
            print(func.__name__ + '()执行完毕')

        return wrapper


def column_check(df: pd.DataFrame, cols: list):
    for col in cols:
        assert col in df.columns, f'df does not contain columns {col}.'
