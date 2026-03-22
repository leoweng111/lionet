"""Factor operator definitions and formula parser/evaluator.

This module centralizes all GP/LLM formula primitives so both mining and
backtest can share the same operator space.
"""

import ast
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd


class FactorNode:
    """Base class for AST nodes."""

    def calc(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def to_formula(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_formula()


class DataNode(FactorNode):
    def __init__(self, field: str):
        self.field = field

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[self.field], errors='coerce')

    def to_formula(self) -> str:
        return self.field


class ConstNode(FactorNode):
    def __init__(self, value: float):
        self.value = float(value)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)

    def to_formula(self) -> str:
        return f"{self.value:.6g}"


def _safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _group_apply_series(df: pd.DataFrame, series: pd.Series, fn) -> pd.Series:
    if 'instrument_id' not in df.columns:
        return fn(_safe_series(series))
    out = pd.Series(np.nan, index=df.index, dtype=float)
    grouped = df.groupby('instrument_id', sort=False).groups
    for _, idx in grouped.items():
        idx_list = list(idx)
        out.loc[idx_list] = fn(_safe_series(series.loc[idx_list])).values
    return out


def _rolling_argext_distance(arr: np.ndarray, mode: str) -> float:
    values = np.asarray(arr, dtype=float)
    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return np.nan

    target = np.nanmax(values) if mode == 'max' else np.nanmin(values)
    hit_idx = np.flatnonzero(values == target)
    if len(hit_idx) == 0:
        return np.nan

    return float((len(values) - 1) - int(hit_idx[-1]))


class OpAdd(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) + self.right.calc(df)

    def to_formula(self) -> str:
        return f"Add({self.left}, {self.right})"


class OpSub(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) - self.right.calc(df)

    def to_formula(self) -> str:
        return f"Sub({self.left}, {self.right})"


class OpMul(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return self.left.calc(df) * self.right.calc(df)

    def to_formula(self) -> str:
        return f"Mul({self.left}, {self.right})"


class OpDiv(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        denominator = self.right.calc(df).replace(0, np.nan)
        return self.left.calc(df) / denominator

    def to_formula(self) -> str:
        return f"Div({self.left}, {self.right})"


class OpMax(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return np.maximum(self.left.calc(df), self.right.calc(df))

    def to_formula(self) -> str:
        return f"Max({self.left}, {self.right})"


class OpMin(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return np.minimum(self.left.calc(df), self.right.calc(df))

    def to_formula(self) -> str:
        return f"Min({self.left}, {self.right})"


class OpNeg(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return -self.child.calc(df)

    def to_formula(self) -> str:
        return f"Neg({self.child})"


class OpSqrtAbs(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.child.calc(df)).abs()
        return pd.Series(np.sqrt(x), index=x.index)

    def to_formula(self) -> str:
        return f"SqrtAbs({self.child})"


class OpAbs(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _safe_series(self.child.calc(df)).abs()

    def to_formula(self) -> str:
        return f"Abs({self.child})"


class OpInv(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.child.calc(df)).replace(0, np.nan)
        return 1.0 / x

    def to_formula(self) -> str:
        return f"Inv({self.child})"


class OpSig(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.child.calc(df)).clip(-50, 50)
        return pd.Series(1.0 / (1.0 + np.exp(-x)), index=x.index)

    def to_formula(self) -> str:
        return f"Sig({self.child})"


class OpSign(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.child.calc(df))
        return pd.Series(np.sign(x), index=x.index)

    def to_formula(self) -> str:
        return f"Sign({self.child})"


class OpLt(FactorNode):
    # As requested: lt(X, Y) => 1 if X > Y else 0
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return (_safe_series(self.left.calc(df)) > _safe_series(self.right.calc(df))).astype(float)

    def to_formula(self) -> str:
        return f"Lt({self.left}, {self.right})"


class OpGt(FactorNode):
    # As requested: gt(X, Y) => 1 if X < Y else 0
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return (_safe_series(self.left.calc(df)) < _safe_series(self.right.calc(df))).astype(float)

    def to_formula(self) -> str:
        return f"Gt({self.left}, {self.right})"


class OpDelta(FactorNode):
    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.diff(1))

    def to_formula(self) -> str:
        return f"Delta({self.child})"


class OpTsMean(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.calc(df)
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).mean())
        return s.rolling(self.window).mean()

    def to_formula(self) -> str:
        return f"TsMean({self.child}, {self.window})"


class OpTsStd(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = self.child.calc(df)
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).std())
        return s.rolling(self.window).std()

    def to_formula(self) -> str:
        return f"TsStd({self.child}, {self.window})"


class OpTsDelta(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.diff(self.window))

    def to_formula(self) -> str:
        return f"TsDelta({self.child}, {self.window})"


class OpTsPctDelta(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.pct_change(self.window))

    def to_formula(self) -> str:
        return f"TsPctDelta({self.child}, {self.window})"


class OpTsDelay(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.shift(self.window))

    def to_formula(self) -> str:
        return f"TsDelay({self.child}, {self.window})"


class OpTsSum(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.rolling(self.window).sum())

    def to_formula(self) -> str:
        return f"TsSum({self.child}, {self.window})"


class OpTsMax(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.rolling(self.window).max())

    def to_formula(self) -> str:
        return f"TsMax({self.child}, {self.window})"


class OpTsMin(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.rolling(self.window).min())

    def to_formula(self) -> str:
        return f"TsMin({self.child}, {self.window})"


class OpTsArgmax(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(
            df,
            self.child.calc(df),
            lambda x: x.rolling(self.window).apply(lambda a: _rolling_argext_distance(np.asarray(a), mode='max'), raw=True),
        )

    def to_formula(self) -> str:
        return f"TsArgmax({self.child}, {self.window})"


class OpTsArgmin(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(
            df,
            self.child.calc(df),
            lambda x: x.rolling(self.window).apply(lambda a: _rolling_argext_distance(np.asarray(a), mode='min'), raw=True),
        )

    def to_formula(self) -> str:
        return f"TsArgmin({self.child}, {self.window})"


class OpTsTimeWeightedMean(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    @staticmethod
    def _weighted_mean(arr: np.ndarray) -> float:
        weights = np.arange(1, len(arr) + 1, dtype=float)
        return float(np.dot(arr, weights) / weights.sum())

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(
            df,
            self.child.calc(df),
            lambda x: x.rolling(self.window).apply(lambda a: self._weighted_mean(np.asarray(a)), raw=True),
        )

    def to_formula(self) -> str:
        return f"TsTimeWeightedMean({self.child}, {self.window})"


class OpTsRank(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    @staticmethod
    def _last_rank(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        return float(s.rank(method='average').iloc[-1] / len(s))

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(
            df,
            self.child.calc(df),
            lambda x: x.rolling(self.window).apply(lambda a: self._last_rank(np.asarray(a)), raw=True),
        )

    def to_formula(self) -> str:
        return f"TsRank({self.child}, {self.window})"


class OpTsCorr(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return x.rolling(self.window).corr(y)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.left.calc(df))
        y = _safe_series(self.right.calc(df))
        if 'instrument_id' not in df.columns:
            return self._calc_one(x, y)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        grouped = df.groupby('instrument_id', sort=False).groups
        for _, idx in grouped.items():
            idx_list = list(idx)
            out.loc[idx_list] = self._calc_one(x.loc[idx_list], y.loc[idx_list]).values
        return out

    def to_formula(self) -> str:
        return f"TsCorr({self.left}, {self.right}, {self.window})"


class OpTsRankCorr(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, x: pd.Series, y: pd.Series) -> pd.Series:
        x = _safe_series(x)
        y = _safe_series(y)
        out = np.full(len(x), np.nan, dtype=float)
        for i in range(self.window - 1, len(x)):
            xr = x.iloc[i - self.window + 1:i + 1]
            yr = y.iloc[i - self.window + 1:i + 1]
            tmp = pd.DataFrame({'x': xr, 'y': yr}).dropna()
            if len(tmp) < 2:
                continue
            if tmp['x'].nunique(dropna=True) <= 1 or tmp['y'].nunique(dropna=True) <= 1:
                continue
            out[i] = tmp['x'].corr(tmp['y'], method='spearman')
        return pd.Series(out, index=x.index)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.left.calc(df))
        y = _safe_series(self.right.calc(df))
        if 'instrument_id' not in df.columns:
            return self._calc_one(x, y)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        grouped = df.groupby('instrument_id', sort=False).groups
        for _, idx in grouped.items():
            idx_list = list(idx)
            out.loc[idx_list] = self._calc_one(x.loc[idx_list], y.loc[idx_list]).values
        return out

    def to_formula(self) -> str:
        return f"TsRankCorr({self.left}, {self.right}, {self.window})"


BINARY_OPS = [OpAdd, OpSub, OpMul, OpDiv, OpMax, OpMin, OpLt, OpGt]
UNARY_OPS = [OpSqrtAbs, OpAbs, OpInv, OpSig, OpSign, OpDelta]
UNARY_TS_OPS = [
    OpTsMean,
    OpTsStd,
    OpTsDelta,
    OpTsPctDelta,
    OpTsDelay,
    OpTsSum,
    OpTsMax,
    OpTsMin,
    OpTsArgmax,
    OpTsArgmin,
    OpTsTimeWeightedMean,
    OpTsRank,
]
BINARY_TS_OPS = [OpTsCorr, OpTsRankCorr]

UNARY_CHILD_OPS = tuple(UNARY_OPS + UNARY_TS_OPS + [OpNeg])
BINARY_CHILD_OPS = tuple(BINARY_OPS + BINARY_TS_OPS)

OP_CLASS_BY_NAME: Dict[str, Type[Any]] = {
    cls.__name__.replace('Op', ''): cls
    for cls in (BINARY_OPS + UNARY_OPS + UNARY_TS_OPS + BINARY_TS_OPS + [OpNeg])
}


def available_operator_prompt_text() -> str:
    binary = ', '.join(cls.__name__.replace('Op', '') for cls in BINARY_OPS)
    unary = ', '.join(cls.__name__.replace('Op', '') for cls in UNARY_OPS + [OpNeg])
    unary_ts = ', '.join(cls.__name__.replace('Op', '') for cls in UNARY_TS_OPS)
    binary_ts = ', '.join(cls.__name__.replace('Op', '') for cls in BINARY_TS_OPS)
    return (
        f"可用二元算子(2参数): {binary}\n"
        f"可用一元算子(1参数): {unary}\n"
        f"可用一元时序算子(2参数, 第二个为窗口整数N): {unary_ts}\n"
        f"可用二元时序算子(3参数, 第三个为窗口整数N): {binary_ts}"
    )


def _parse_window_arg(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        w = int(node.value)
    else:
        raise ValueError('Window argument must be an integer literal.')
    if w <= 0:
        raise ValueError(f'Window argument must be positive, got {w}.')
    return w


def parse_formula_to_node(formula: str,
                          data_fields: Optional[Sequence[str]] = None) -> FactorNode:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError('formula must be a non-empty string.')

    fields = set(data_fields or ['open', 'high', 'low', 'close', 'volume', 'position'])
    expr = ast.parse(formula.strip(), mode='eval').body

    def _build(node: ast.AST) -> FactorNode:
        if isinstance(node, ast.Name):
            if node.id in fields:
                return DataNode(node.id)
            raise ValueError(f'Unknown field `{node.id}` in formula.')

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return ConstNode(float(node.value))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            val = node.operand.value
            if isinstance(val, (int, float)):
                return ConstNode(float(-val))

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            op_name = node.func.id
            op_cls = OP_CLASS_BY_NAME.get(op_name)
            if op_cls is None:
                raise ValueError(f'Unsupported operator `{op_name}`.')

            if op_cls in BINARY_OPS:
                if len(node.args) != 2:
                    raise ValueError(f'{op_name} expects 2 args.')
                return op_cls(_build(node.args[0]), _build(node.args[1]))
            if op_cls in UNARY_OPS or op_cls is OpNeg:
                if len(node.args) != 1:
                    raise ValueError(f'{op_name} expects 1 arg.')
                return op_cls(_build(node.args[0]))
            if op_cls in UNARY_TS_OPS:
                if len(node.args) != 2:
                    raise ValueError(f'{op_name} expects 2 args (X, N).')
                return op_cls(_build(node.args[0]), _parse_window_arg(node.args[1]))
            if op_cls in BINARY_TS_OPS:
                if len(node.args) != 3:
                    raise ValueError(f'{op_name} expects 3 args (X, Y, N).')
                return op_cls(_build(node.args[0]), _build(node.args[1]), _parse_window_arg(node.args[2]))

        raise ValueError(f'Unsupported formula node: {ast.dump(node)}')

    return _build(expr)


def calc_formula_series(df: pd.DataFrame,
                        formula: str,
                        data_fields: Optional[Sequence[str]] = None) -> pd.Series:
    node = parse_formula_to_node(formula=formula, data_fields=data_fields)
    return pd.to_numeric(node.calc(df), errors='coerce')


def calc_formula_df(df: pd.DataFrame,
                    formula_map: Dict[str, str],
                    data_fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
    out = df[['time', 'instrument_id']].copy()
    for fc_name, formula in formula_map.items():
        out[fc_name] = calc_formula_series(df=df, formula=formula, data_fields=data_fields).values
    return out


