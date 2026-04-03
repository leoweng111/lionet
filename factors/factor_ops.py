"""Factor operator definitions and formula parser/evaluator.

This module centralizes all GP/LLM formula primitives so both mining and
backtest can share the same operator space.
"""

import ast
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd


class FactorDataType(str, Enum):
    """Semantic types for GP nodes to prevent invalid operator mixes."""

    PRICE = 'price'
    VOLUME = 'volume'
    OI = 'oi'
    RETURN = 'return'
    VOLATILITY = 'volatility'
    RATIO = 'ratio'
    BOOLEAN = 'boolean'
    GENERIC = 'generic'


_BASE_FIELD_TYPE_MAP: Dict[str, FactorDataType] = {
    'open': FactorDataType.PRICE,
    'high': FactorDataType.PRICE,
    'low': FactorDataType.PRICE,
    'close': FactorDataType.PRICE,
    'volume': FactorDataType.VOLUME,
    'position': FactorDataType.OI,
    'oi': FactorDataType.OI,
}


def infer_field_type(field: str) -> FactorDataType:
    """Infer semantic type from raw or derived field name."""
    key = str(field).strip().lower()
    if key in _BASE_FIELD_TYPE_MAP:
        return _BASE_FIELD_TYPE_MAP[key]
    if key.startswith('ret_') or key.startswith('log_ret_') or key.startswith('oi_chg_'):
        return FactorDataType.RETURN
    if key.startswith('volatility_') or key.startswith('volume_std_'):
        return FactorDataType.VOLATILITY
    if key.endswith('_ratio') or key.startswith('volume_zscore_') or key.startswith('turnover_shock_'):
        return FactorDataType.RATIO
    return FactorDataType.GENERIC


def _is_valid_mul(left_type: FactorDataType, right_type: FactorDataType) -> bool:
    # Only allow scaling by unitless ratio to avoid meaningless mixes like Price * Volume.
    return left_type == FactorDataType.RATIO or right_type == FactorDataType.RATIO


def _infer_mul_type(left_type: FactorDataType, right_type: FactorDataType) -> FactorDataType:
    if left_type == FactorDataType.RATIO:
        return right_type
    if right_type == FactorDataType.RATIO:
        return left_type
    return FactorDataType.RATIO


def _is_valid_div(left_type: FactorDataType, right_type: FactorDataType) -> bool:
    # Division is valid for same-unit ratio or dividing by ratio.
    return left_type == right_type or right_type == FactorDataType.RATIO


def _infer_div_type(left_type: FactorDataType, right_type: FactorDataType) -> FactorDataType:
    if right_type == FactorDataType.RATIO:
        return left_type
    return FactorDataType.RATIO


class FactorNode:
    """Base class for AST nodes."""

    data_type: FactorDataType = FactorDataType.GENERIC

    def calc(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def to_formula(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_formula()


class DataNode(FactorNode):
    def __init__(self, field: str):
        self.field = field
        self.data_type = infer_field_type(field)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        if self.field in df.columns:
            return pd.to_numeric(df[self.field], errors='coerce')
        # Alias support for futures open interest naming.
        if self.field == 'oi' and 'position' in df.columns:
            return pd.to_numeric(df['position'], errors='coerce')
        if self.field == 'position' and 'oi' in df.columns:
            return pd.to_numeric(df['oi'], errors='coerce')
        raise KeyError(f'Field `{self.field}` is not available in input dataframe.')

    def to_formula(self) -> str:
        return self.field


class ConstNode(FactorNode):
    def __init__(self, value: float):
        self.value = float(value)
        # Constant is treated as unitless scaling factor by default.
        self.data_type = FactorDataType.RATIO

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



def infer_node_type(node: FactorNode) -> FactorDataType:
    """Infer and validate semantic type for AST recursively.

    规则摘要（类型约束）：
    - Add/Sub/Max/Min：左右必须同类型，且不能是 BOOLEAN
    - Mul：只允许与 RATIO 相乘（防止 price * volume 等无意义组合）
    - Div：同类型相除，或除以 RATIO
    - Lt/Gt：左右必须同类型，输出 BOOLEAN
    - TsRank/TsArgmax/TsArgmin/Sig/Inv：输入任意，输出 RATIO
    - Return/LogReturn：输入必须是 PRICE，输出 RETURN
    - Body/Upper/Lower/Shadows/StochasticK：输入 PRICE，输出 RATIO
    - TypicalPrice：输入 PRICE，输出 PRICE
    - Volatility：输入 PRICE/RETURN，输出 VOLATILITY
    - VolumeStd/VolumeZScore/TurnoverShock：输入 VOLUME/OI，输出 VOLATILITY/RATIO
    - Bias/RangePosition/PriceAcceleration/TrueAmplitude：输入 PRICE，输出 RATIO/RETURN
    - TsCorr/TsRankCorr/TsCov/TsBeta：左右必须同类型，输出 RATIO（TsCov 为 GENERIC）
    - OiTrendConviction：输入 (PRICE, OI)，输出 RETURN
    - Amihud：输入 (PRICE, VOLUME)，输出 RATIO
    - MaRibbon：输入 PRICE，输出 RATIO
    """
    if isinstance(node, (DataNode, ConstNode)):
        return node.data_type

    if isinstance(node, (OpAdd, OpSub, OpMax, OpMin)):
        # 同类型加减/极值，禁止 BOOLEAN
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt != rt:
            raise TypeError(f'{node.__class__.__name__} requires same type, got {lt} and {rt}.')
        if lt == FactorDataType.BOOLEAN:
            raise TypeError(f'{node.__class__.__name__} does not accept boolean operands.')
        node.data_type = lt
        return node.data_type

    if isinstance(node, OpMul):
        # 只允许与 RATIO 相乘（unitless scaling）
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if not _is_valid_mul(lt, rt):
            raise TypeError(f'OpMul forbids multiplying incompatible types: {lt} * {rt}.')
        node.data_type = _infer_mul_type(lt, rt)
        return node.data_type

    if isinstance(node, OpDiv):
        # 同类型相除或除以 RATIO
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if not _is_valid_div(lt, rt):
            raise TypeError(f'OpDiv forbids dividing incompatible types: {lt} / {rt}.')
        node.data_type = _infer_div_type(lt, rt)
        return node.data_type

    if isinstance(node, (OpLt, OpGt)):
        # 同类型比较，输出 BOOLEAN
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt != rt:
            raise TypeError(f'{node.__class__.__name__} comparison requires same type, got {lt} and {rt}.')
        node.data_type = FactorDataType.BOOLEAN
        return node.data_type

    if isinstance(node, OpNeg):
        node.data_type = infer_node_type(node.child)
        return node.data_type

    if isinstance(node, (OpSqrtAbs, OpAbs, OpDelta, OpTsMean, OpTsDelta, OpTsDelay, OpTsSum, OpTsMax, OpTsMin,
                         OpTsTimeWeightedMean, OpEma, OpTsDecayExp)):
        node.data_type = infer_node_type(node.child)
        return node.data_type

    if isinstance(node, OpTsStd):
        _ = infer_node_type(node.child)
        node.data_type = FactorDataType.VOLATILITY
        return node.data_type

    if isinstance(node, OpTsPctDelta):
        _ = infer_node_type(node.child)
        node.data_type = FactorDataType.RETURN
        return node.data_type

    if isinstance(node, (OpReturn, OpLogReturn)):
        # 价格序列 -> 收益
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'{node.__class__.__name__} requires price input, got {t}.')
        node.data_type = FactorDataType.RETURN
        return node.data_type

    if isinstance(node, (OpBodyRatio, OpUpperShadowRatio, OpLowerShadowRatio, OpStochasticK)):
        # K线形态类：输入价格，输出无量纲比例
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'{node.__class__.__name__} requires price input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpTypicalPrice):
        # 典型价格：输入价格，输出仍为价格类型
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpTypicalPrice requires price input, got {t}.')
        node.data_type = FactorDataType.PRICE
        return node.data_type

    if isinstance(node, OpVolatility):
        # 价格/收益 -> 波动率
        t = infer_node_type(node.child)
        if t not in {FactorDataType.RETURN, FactorDataType.PRICE}:
            raise TypeError(f'OpVolatility requires return/price input, got {t}.')
        node.data_type = FactorDataType.VOLATILITY
        return node.data_type

    if isinstance(node, OpBias):
        # 价格 -> 偏离度比例
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpBias requires price input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpRangePosition):
        # 价格 -> 区间位置比例
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpRangePosition requires price input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpPriceAcceleration):
        # 价格 -> 加速度（类似收益变化）
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpPriceAcceleration requires price input, got {t}.')
        node.data_type = FactorDataType.RETURN
        return node.data_type

    if isinstance(node, OpTrueAmplitude):
        # 价格 -> 真实振幅比例
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpTrueAmplitude requires price input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpVolumeStd):
        # 成交量/持仓量 -> 波动率
        t = infer_node_type(node.child)
        if t not in {FactorDataType.VOLUME, FactorDataType.OI}:
            raise TypeError(f'OpVolumeStd requires volume/oi input, got {t}.')
        node.data_type = FactorDataType.VOLATILITY
        return node.data_type

    if isinstance(node, (OpVolumeZScore, OpTurnoverShock)):
        # 成交量/持仓量 -> 比例类
        t = infer_node_type(node.child)
        if t not in {FactorDataType.VOLUME, FactorDataType.OI}:
            raise TypeError(f'{node.__class__.__name__} requires volume/oi input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, (OpTsArgmax, OpTsArgmin, OpTsRank, OpSig, OpSign, OpInv)):
        # 任意输入，输出 RATIO
        _ = infer_node_type(node.child)
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, (OpTsCorr, OpTsRankCorr, OpTsCov, OpTsBeta)):
        # 同类型相关/协方差/贝塔
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt != rt:
            raise TypeError(f'{node.__class__.__name__} requires same type inputs, got {lt} and {rt}.')
        node.data_type = FactorDataType.RATIO if not isinstance(node, OpTsCov) else FactorDataType.GENERIC
        return node.data_type

    if isinstance(node, OpVpDivergence):
        # (price/return/ratio, volume/oi)
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt not in {FactorDataType.PRICE, FactorDataType.RETURN, FactorDataType.RATIO}:
            raise TypeError(f'OpVpDivergence left input must be price/return/ratio, got {lt}.')
        if rt not in {FactorDataType.VOLUME, FactorDataType.OI}:
            raise TypeError(f'OpVpDivergence right input must be volume/oi, got {rt}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpAmihud):
        # (price, volume)
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt != FactorDataType.PRICE or rt != FactorDataType.VOLUME:
            raise TypeError(f'OpAmihud requires (price, volume), got ({lt}, {rt}).')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    if isinstance(node, OpOiTrendConviction):
        # (price, oi)
        lt = infer_node_type(node.left)
        rt = infer_node_type(node.right)
        if lt != FactorDataType.PRICE or rt != FactorDataType.OI:
            raise TypeError(f'OpOiTrendConviction requires (price, oi), got ({lt}, {rt}).')
        node.data_type = FactorDataType.RETURN
        return node.data_type

    if isinstance(node, OpMaRibbon):
        # price -> ratio
        t = infer_node_type(node.child)
        if t != FactorDataType.PRICE:
            raise TypeError(f'OpMaRibbon requires price input, got {t}.')
        node.data_type = FactorDataType.RATIO
        return node.data_type

    raise TypeError(f'Unsupported node type for semantic typing: {type(node).__name__}')


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


class OpBodyRatio(FactorNode):
    """K线实体比例：(close-open)/(high-low+eps)，输出 [-1,1] 的无量纲比例。"""

    def __init__(self, child: FactorNode):
        # child 仅用于类型约束与公式一致性，实际计算使用 OHLC。
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        open_ = pd.to_numeric(df['open'], errors='coerce')
        denom = (high - low + 1e-8)
        return (close - open_) / denom

    def to_formula(self) -> str:
        return f"BodyRatio({self.child})"


class OpUpperShadowRatio(FactorNode):
    """K线上影线比例：(high-max(open,close))/(high-low+eps)。"""

    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        open_ = pd.to_numeric(df['open'], errors='coerce')
        denom = (high - low + 1e-8)
        return (high - np.maximum(open_, close)) / denom

    def to_formula(self) -> str:
        return f"UpperShadowRatio({self.child})"


class OpLowerShadowRatio(FactorNode):
    """K线下影线比例：(min(open,close)-low)/(high-low+eps)。"""

    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        open_ = pd.to_numeric(df['open'], errors='coerce')
        denom = (high - low + 1e-8)
        return (np.minimum(open_, close) - low) / denom

    def to_formula(self) -> str:
        return f"LowerShadowRatio({self.child})"


class OpStochasticK(FactorNode):
    """日内相对位置：(close-low)/(high-low+eps)。"""

    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        denom = (high - low + 1e-8)
        return (close - low) / denom

    def to_formula(self) -> str:
        return f"StochasticK({self.child})"


class OpTypicalPrice(FactorNode):
    """典型价格：(high+low+close)/3。"""

    def __init__(self, child: FactorNode):
        self.child = child

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        return (high + low + close) / 3.0

    def to_formula(self) -> str:
        return f"TypicalPrice({self.child})"


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


class OpBias(FactorNode):
    """均线偏离度：close / TsMean(close, d) - 1。"""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        close = _safe_series(self.child.calc(df))
        if 'instrument_id' in df.columns:
            mean = close.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).mean())
        else:
            mean = close.rolling(self.window).mean()
        return close / mean.replace(0, np.nan) - 1.0

    def to_formula(self) -> str:
        return f"Bias({self.child}, {self.window})"


class OpRangePosition(FactorNode):
    """区间位置：(close - TsMin(low, d)) / (TsMax(high, d) - TsMin(low, d) + eps)。"""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        close = _safe_series(self.child.calc(df))
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        if 'instrument_id' in df.columns:
            high_max = high.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).max())
            low_min = low.groupby(df['instrument_id']).transform(lambda x: x.rolling(self.window).min())
        else:
            high_max = high.rolling(self.window).max()
            low_min = low.rolling(self.window).min()
        denom = (high_max - low_min + 1e-8)
        return (close - low_min) / denom

    def to_formula(self) -> str:
        return f"RangePosition({self.child}, {self.window})"


class OpPriceAcceleration(FactorNode):
    """价格加速度：Delta(close, d) - Delay(Delta(close, d), d)。"""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        def _acc(x: pd.Series) -> pd.Series:
            delta = x.diff(self.window)
            return delta - delta.shift(self.window)

        return _group_apply_series(df, self.child.calc(df), _acc)

    def to_formula(self) -> str:
        return f"PriceAcceleration({self.child}, {self.window})"


class OpTrueAmplitude(FactorNode):
    """真实中心振幅：(high-low)/Delay(close, 1)。"""

    def __init__(self, child: FactorNode, window: int = 1):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = _safe_series(self.child.calc(df))
        if 'instrument_id' in df.columns:
            prev_close = close.groupby(df['instrument_id']).shift(self.window)
        else:
            prev_close = close.shift(self.window)
        return (high - low) / prev_close.replace(0, np.nan)

    def to_formula(self) -> str:
        return f"TrueAmplitude({self.child}, {self.window})"


class OpReturn(FactorNode):
    """Rolling return over window (pct change)."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.pct_change(self.window))

    def to_formula(self) -> str:
        return f"Return({self.child}, {self.window})"


class OpLogReturn(FactorNode):
    """Rolling log-return over window."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        def _logret(x: pd.Series) -> pd.Series:
            s = pd.to_numeric(x, errors='coerce')
            s = s.mask(s <= 0, np.nan)
            return np.log(s).diff(self.window)

        return _group_apply_series(df, self.child.calc(df), _logret)

    def to_formula(self) -> str:
        return f"LogReturn({self.child}, {self.window})"


class OpVolatility(FactorNode):
    """Rolling volatility (std) over window."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        def _vol(x: pd.Series) -> pd.Series:
            s = pd.to_numeric(x, errors='coerce')
            if getattr(self.child, 'data_type', None) == FactorDataType.PRICE:
                s = s.pct_change(1)
            return s.rolling(self.window).std()

        return _group_apply_series(df, self.child.calc(df), _vol)

    def to_formula(self) -> str:
        return f"Volatility({self.child}, {self.window})"


class OpVolumeStd(FactorNode):
    """Rolling std of volume/open interest."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        return _group_apply_series(df, self.child.calc(df), lambda x: x.rolling(self.window).std())

    def to_formula(self) -> str:
        return f"VolumeStd({self.child}, {self.window})"


class OpVolumeZScore(FactorNode):
    """Rolling z-score of volume/open interest."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        def _zscore(x: pd.Series) -> pd.Series:
            s = pd.to_numeric(x, errors='coerce')
            mean = s.rolling(self.window).mean()
            std = s.rolling(self.window).std().replace(0, np.nan)
            return (s - mean) / std

        return _group_apply_series(df, self.child.calc(df), _zscore)

    def to_formula(self) -> str:
        return f"VolumeZScore({self.child}, {self.window})"


class OpTurnoverShock(FactorNode):
    """Rolling turnover shock: volume / rolling mean - 1."""

    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        def _shock(x: pd.Series) -> pd.Series:
            s = pd.to_numeric(x, errors='coerce')
            mean = s.rolling(self.window).mean().replace(0, np.nan)
            return s / mean - 1.0

        return _group_apply_series(df, self.child.calc(df), _shock)

    def to_formula(self) -> str:
        return f"TurnoverShock({self.child}, {self.window})"


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


class OpEma(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = _safe_series(self.child.calc(df))
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.ewm(span=self.window, adjust=False).mean())
        return s.ewm(span=self.window, adjust=False).mean()

    def to_formula(self) -> str:
        return f"Ema({self.child}, {self.window})"


class OpTsDecayExp(FactorNode):
    def __init__(self, child: FactorNode, window: int):
        self.child = child
        self.window = int(window)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        s = _safe_series(self.child.calc(df))
        if 'instrument_id' in df.columns:
            return s.groupby(df['instrument_id']).transform(lambda x: x.ewm(span=self.window, adjust=False).mean())
        return s.ewm(span=self.window, adjust=False).mean()

    def to_formula(self) -> str:
        return f"TsDecayExp({self.child}, {self.window})"


class OpTsCov(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return x.rolling(self.window).cov(y)

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
        return f"TsCov({self.left}, {self.right}, {self.window})"


class OpTsBeta(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, x: pd.Series, y: pd.Series) -> pd.Series:
        cov_xy = x.rolling(self.window).cov(y)
        var_y = y.rolling(self.window).var().replace(0, np.nan)
        return cov_xy / var_y

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
        return f"TsBeta({self.left}, {self.right}, {self.window})"


class OpVpDivergence(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, x: pd.Series, y: pd.Series) -> pd.Series:
        return x.diff(1).rolling(self.window).corr(y.diff(1))

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
        return f"VpDivergence({self.left}, {self.right}, {self.window})"


class OpAmihud(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode, window: int):
        self.left = left
        self.right = right
        self.window = int(window)

    def _calc_one(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        close_safe = close.replace(0, np.nan)
        volume_safe = volume.replace(0, np.nan)
        shock = (close.diff(1).abs() / close_safe) / volume_safe
        return shock.rolling(self.window).mean()

    def calc(self, df: pd.DataFrame) -> pd.Series:
        close = _safe_series(self.left.calc(df))
        volume = _safe_series(self.right.calc(df))
        if 'instrument_id' not in df.columns:
            return self._calc_one(close, volume)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        grouped = df.groupby('instrument_id', sort=False).groups
        for _, idx in grouped.items():
            idx_list = list(idx)
            out.loc[idx_list] = self._calc_one(close.loc[idx_list], volume.loc[idx_list]).values
        return out

    def to_formula(self) -> str:
        return f"Amihud({self.left}, {self.right}, {self.window})"


class OpOiTrendConviction(FactorNode):
    def __init__(self, left: FactorNode, right: FactorNode):
        self.left = left
        self.right = right

    def _calc_one(self, close: pd.Series, oi: pd.Series) -> pd.Series:
        return pd.Series(np.sign(close.diff(1)) * oi.diff(1), index=close.index)

    def calc(self, df: pd.DataFrame) -> pd.Series:
        close = _safe_series(self.left.calc(df))
        oi = _safe_series(self.right.calc(df))
        if 'instrument_id' not in df.columns:
            return self._calc_one(close, oi)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        grouped = df.groupby('instrument_id', sort=False).groups
        for _, idx in grouped.items():
            idx_list = list(idx)
            out.loc[idx_list] = self._calc_one(close.loc[idx_list], oi.loc[idx_list]).values
        return out

    def to_formula(self) -> str:
        return f"OiTrendConviction({self.left}, {self.right})"


class OpMaRibbon(FactorNode):
    def __init__(self, child: FactorNode, window1: int, window2: int, window3: int):
        self.child = child
        self.window1 = int(window1)
        self.window2 = int(window2)
        self.window3 = int(window3)

    def _calc_one(self, x: pd.Series) -> pd.Series:
        ma1 = x.rolling(self.window1).mean()
        ma2 = x.rolling(self.window2).mean()
        ma3 = x.rolling(self.window3).mean()
        ma_df = pd.DataFrame({'ma1': ma1, 'ma2': ma2, 'ma3': ma3}, index=x.index)
        spread = ma_df.max(axis=1) - ma_df.min(axis=1)
        center = ma_df.mean(axis=1).abs().replace(0, np.nan)
        return spread / center

    def calc(self, df: pd.DataFrame) -> pd.Series:
        x = _safe_series(self.child.calc(df))
        if 'instrument_id' not in df.columns:
            return self._calc_one(x)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        grouped = df.groupby('instrument_id', sort=False).groups
        for _, idx in grouped.items():
            idx_list = list(idx)
            out.loc[idx_list] = self._calc_one(x.loc[idx_list]).values
        return out

    def to_formula(self) -> str:
        return f"MaRibbon({self.child}, {self.window1}, {self.window2}, {self.window3})"


BINARY_OPS = [OpAdd, OpSub, OpMul, OpDiv, OpMax, OpMin, OpLt, OpGt, OpOiTrendConviction]
UNARY_OPS = [
    OpSqrtAbs,
    OpAbs,
    OpInv,
    OpSig,
    OpSign,
    OpDelta,
    OpBodyRatio,
    OpUpperShadowRatio,
    OpLowerShadowRatio,
    OpStochasticK,
    OpTypicalPrice,
]
UNARY_TS_OPS = [
    OpEma,
    OpTsDecayExp,
    OpTsMean,
    OpTsStd,
    OpTsDelta,
    OpTsPctDelta,
    OpReturn,
    OpLogReturn,
    OpVolatility,
    OpVolumeStd,
    OpVolumeZScore,
    OpTurnoverShock,
    OpBias,
    OpRangePosition,
    OpPriceAcceleration,
    OpTrueAmplitude,
    OpTsDelay,
    OpTsSum,
    OpTsMax,
    OpTsMin,
    OpTsArgmax,
    OpTsArgmin,
    OpTsTimeWeightedMean,
    OpTsRank,
]
BINARY_TS_OPS = [OpTsCorr, OpTsRankCorr, OpTsCov, OpTsBeta, OpVpDivergence, OpAmihud]
TERNARY_WINDOW_TS_OPS = [OpMaRibbon]

UNARY_CHILD_OPS = tuple(UNARY_OPS + UNARY_TS_OPS + [OpNeg])
BINARY_CHILD_OPS = tuple(BINARY_OPS + BINARY_TS_OPS)

OP_CLASS_BY_NAME: Dict[str, Type[Any]] = {
    cls.__name__.replace('Op', ''): cls
    for cls in (BINARY_OPS + UNARY_OPS + UNARY_TS_OPS + BINARY_TS_OPS + TERNARY_WINDOW_TS_OPS + [OpNeg])
}

OP_ALIAS_BY_NAME: Dict[str, Type[Any]] = {
    # Case-insensitive aliases.
    **{k.lower(): v for k, v in OP_CLASS_BY_NAME.items()},
    # Snake-case aliases expected from quant literature.
    'return': OpReturn,
    'log_return': OpLogReturn,
    'volatility': OpVolatility,
    'volume_std': OpVolumeStd,
    'volume_zscore': OpVolumeZScore,
    'turnover_shock': OpTurnoverShock,
    'body_ratio': OpBodyRatio,
    'upper_shadow_ratio': OpUpperShadowRatio,
    'lower_shadow_ratio': OpLowerShadowRatio,
    'stochastic_k': OpStochasticK,
    'typical_price': OpTypicalPrice,
    'bias': OpBias,
    'range_position': OpRangePosition,
    'price_acceleration': OpPriceAcceleration,
    'true_amplitude': OpTrueAmplitude,
    'ema': OpEma,
    'ts_decay_exp': OpTsDecayExp,
    'ts_cov': OpTsCov,
    'ts_beta': OpTsBeta,
    'vp_divergence': OpVpDivergence,
    'amihud': OpAmihud,
    'oi_trend_conviction': OpOiTrendConviction,
    'ma_ribbon': OpMaRibbon,
}


def available_operator_prompt_text() -> str:
    binary = ', '.join(cls.__name__.replace('Op', '') for cls in BINARY_OPS)
    unary = ', '.join(cls.__name__.replace('Op', '') for cls in UNARY_OPS + [OpNeg])
    unary_ts = ', '.join(cls.__name__.replace('Op', '') for cls in UNARY_TS_OPS)
    binary_ts = ', '.join(cls.__name__.replace('Op', '') for cls in BINARY_TS_OPS)
    ternary_ts = ', '.join(cls.__name__.replace('Op', '') for cls in TERNARY_WINDOW_TS_OPS)
    alias_text = 'ema, ts_decay_exp, ts_cov, ts_beta, vp_divergence, amihud, oi_trend_conviction, ma_ribbon'
    return (
        f"可用二元算子(2参数): {binary}\n"
        f"可用一元算子(1参数): {unary}\n"
        f"可用一元时序算子(2参数, 第二个为窗口整数N): {unary_ts}\n"
        f"可用二元时序算子(3参数, 第三个为窗口整数N): {binary_ts}\n"
        f"可用多窗口时序算子(4参数, 后三个为窗口整数N): {ternary_ts}\n"
        f"同义别名(大小写不敏感): {alias_text}"
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
    if 'position' in fields:
        fields.add('oi')
    if 'oi' in fields:
        fields.add('position')
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
                op_cls = OP_ALIAS_BY_NAME.get(op_name.lower())
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
            if op_cls in TERNARY_WINDOW_TS_OPS:
                if len(node.args) != 4:
                    raise ValueError(f'{op_name} expects 4 args (X, N1, N2, N3).')
                return op_cls(
                    _build(node.args[0]),
                    _parse_window_arg(node.args[1]),
                    _parse_window_arg(node.args[2]),
                    _parse_window_arg(node.args[3]),
                )

        raise ValueError(f'Unsupported formula node: {ast.dump(node)}')

    root = _build(expr)
    try:
        infer_node_type(root)
    except TypeError as exc:
        raise ValueError(f'Formula type violation: {exc}') from exc
    return root


def calc_formula_series(df: pd.DataFrame,
                        formula: str,
                        data_fields: Optional[Sequence[str]] = None) -> pd.Series:
    node = parse_formula_to_node(formula=formula, data_fields=data_fields)
    return pd.to_numeric(node.calc(df), errors='coerce')


def calc_formula_df(df: pd.DataFrame,
                    formula_map: Dict[str, str],
                    data_fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
    base = df[['time', 'instrument_id']].copy()
    value_map: Dict[str, pd.Series] = {}
    for fc_name, formula in formula_map.items():
        value_map[fc_name] = calc_formula_series(df=df, formula=formula, data_fields=data_fields)
    if not value_map:
        return base
    value_df = pd.DataFrame(value_map, index=df.index)
    return pd.concat([base, value_df], axis=1)


