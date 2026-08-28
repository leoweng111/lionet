"""Compute daily-frequency features from minute bars on-the-fly.

All features are aggregated per trading day (td), where night session
bars (hour >= 20) are attributed to the next trading day, consistent with
data.futures.assign_trading_day_1min.

Feature families:
  A. Realized volatility & return moments
     rv, bpv, jump, rv_neg, rv_pos, rskew, rkurt, medrv
  B. Range-based volatility estimators
     pk, gk, rs_vol, yz
  C. Intraday momentum & time-segment returns
     ret_open30, ret_close30, ret_overnight, ret_intraday
  D. Microstructure & order flow
     oi_flow, cvd, vwap_dev, amihud_min, vr_k, kyle_lambda
"""

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from data.futures import assign_trading_day_1min

# ── Normal CDF for BVC ───────────────────────────────────────────
try:
    from scipy.stats import norm as _norm
    _norm_cdf = _norm.cdf
except ImportError:  # pragma: no cover - fallback
    _SQRT2 = math.sqrt(2.0)

    def _norm_cdf(x):
        """Approximate standard normal CDF via tanh (close to erf(x/sqrt2))."""
        return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=float) / _SQRT2))


# ── Feature registry ─────────────────────────────────────────────

# feature_name -> FactorDataType string (used by factor_ops.infer_field_type)
FEATURE_TYPE_MAP: Dict[str, str] = {
    # A. Realized volatility & return moments
    'rv': 'volatility',
    'bpv': 'volatility',
    'jump': 'volatility',
    'rv_neg': 'volatility',
    'rv_pos': 'volatility',
    'rskew': 'ratio',
    'rkurt': 'ratio',
    'medrv': 'volatility',
    # B. Range-based volatility
    'pk': 'volatility',
    'gk': 'volatility',
    'rs_vol': 'volatility',
    'yz': 'volatility',
    # C. Intraday momentum & time-segment (ret_* auto-typed as 'return')
    # D. Microstructure & order flow
    'oi_flow': 'return',
    'cvd': 'return',
    'vwap_dev': 'ratio',
    'amihud_min': 'ratio',
    'vr_k': 'ratio',
    'kyle_lambda': 'ratio',
}

# C-family features (names start with ret_ so infer_field_type auto-assigns RETURN)
_C_FEATURES: List[str] = ['ret_open30', 'ret_close30', 'ret_overnight', 'ret_intraday']

ALL_FEATURE_NAMES: List[str] = list(FEATURE_TYPE_MAP.keys()) + _C_FEATURES

FEATURE_CATEGORIES: Dict[str, List[str]] = {
    'A. 已实现波动率与日内收益矩': ['rv', 'bpv', 'jump', 'rv_neg', 'rv_pos', 'rskew', 'rkurt', 'medrv'],
    'B. 区间型波动率估计': ['pk', 'gk', 'rs_vol', 'yz'],
    'C. 日内动量与时段结构': _C_FEATURES,
    'D. 微观结构与订单流': ['oi_flow', 'cvd', 'vwap_dev', 'amihud_min', 'vr_k', 'kyle_lambda'],
}

# Variance-ratio aggregation window (k-minute returns)
_VR_K = 5


# ── Data loading is in data/futures.py ────────────────────────────
# Use get_futures_continuous_contract_price_1min() from data.futures to
# load minute bars from the DB.


# ── Per-group feature computation ─────────────────────────────────

def _safe_log(x: pd.Series) -> pd.Series:
    """Safe natural log: replaces <=0 with NaN."""
    s = pd.to_numeric(x, errors='coerce')
    return np.log(s.mask(s <= 0))


def _safe_log_scalar(x) -> float:
    """Safe natural log for a scalar value."""
    if x is None or not np.isfinite(x) or x <= 0:
        return float('nan')
    return float(np.log(x))


def _compute_group_features(g: pd.DataFrame) -> pd.Series:
    """Compute all intraday features for one (instrument, td) group.

    *g* must be sorted by time and contain at least: time, open, high, low,
    close, volume, position, weighted_factor.
    """
    g = g.sort_values('time')
    n = len(g)
    features: Dict[str, float] = {}

    # ── Weighted (back-adjusted) prices for return computations ──
    if 'weighted_factor' in g.columns:
        wf = pd.to_numeric(g['weighted_factor'], errors='coerce').fillna(1.0)
    else:
        wf = pd.Series(1.0, index=g.index)
    w_open = pd.to_numeric(g['open'], errors='coerce') * wf
    w_high = pd.to_numeric(g['high'], errors='coerce') * wf
    w_low = pd.to_numeric(g['low'], errors='coerce') * wf
    w_close = pd.to_numeric(g['close'], errors='coerce') * wf
    volume = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0)
    if 'position' in g.columns:
        position = pd.to_numeric(g['position'], errors='coerce').fillna(0.0)
    elif 'oi' in g.columns:
        position = pd.to_numeric(g['oi'], errors='coerce').fillna(0.0)
    else:
        position = pd.Series(0.0, index=g.index)

    # ── Minute log returns (close-to-close; first bar open→close) ──
    prev_close = w_close.shift(1)
    rets = _safe_log(w_close / prev_close)
    first_mask = prev_close.isna()
    if first_mask.any():
        rets[first_mask] = _safe_log(w_close[first_mask] / w_open[first_mask])
    rets = rets.replace([np.inf, -np.inf], np.nan)
    abs_r = rets.abs()
    valid_r = rets.dropna()
    nv = len(valid_r)
    rv = float((rets ** 2).sum())

    # ══════════════════════════════════════════════════════════════
    # A. Realized volatility & return moments
    # ══════════════════════════════════════════════════════════════
    features['rv'] = rv
    bpv = float((np.pi / 2.0) * (abs_r.iloc[1:].values * abs_r.iloc[:-1].values).sum()) if n >= 2 else 0.0
    features['bpv'] = bpv
    features['jump'] = max(rv - bpv, 0.0)
    features['rv_neg'] = float((rets[rets < 0] ** 2).sum())
    features['rv_pos'] = float((rets[rets > 0] ** 2).sum())

    if nv >= 3 and rv > 0:
        features['rskew'] = float((nv ** 0.5) * (valid_r ** 3).sum()) / (rv ** 1.5)
    else:
        features['rskew'] = np.nan

    if nv >= 4 and rv > 0:
        features['rkurt'] = float(nv * (valid_r ** 4).sum()) / (rv ** 2)
    else:
        features['rkurt'] = np.nan

    # medRV: [π/(4√3-6)] · [N/(N-2)] · Σ med(|r_{i-1}|,|r_i|,|r_{i+1}|)²
    # Note: denominator is (4√3 - 6) ≈ 0.928, NOT (6 - 4√3) which would be negative.
    if n >= 3:
        med = abs_r.rolling(3, center=True).median()
        coef = np.pi / (4.0 * math.sqrt(3.0) - 6.0)
        features['medrv'] = float(coef * (n / (n - 2)) * (med ** 2).sum())
    else:
        features['medrv'] = np.nan

    # ══════════════════════════════════════════════════════════════
    # B. Range-based volatility (per-minute summed)
    # ══════════════════════════════════════════════════════════════
    ln_hl = _safe_log(w_high / w_low)
    ln_ho = _safe_log(w_high / w_open)
    ln_co = _safe_log(w_close / w_open)
    ln_hc = _safe_log(w_high / w_close)
    ln_lc = _safe_log(w_low / w_close)
    ln_lo = _safe_log(w_low / w_open)

    features['pk'] = float((ln_hl ** 2).sum() / (4.0 * math.log(2.0)))
    features['gk'] = float((0.5 * ln_ho ** 2 - (2.0 * math.log(2.0) - 1.0) * ln_co ** 2).sum())
    features['rs_vol'] = float((ln_hc * ln_ho + ln_lc * ln_lo).sum())
    # yz is computed in post-processing (needs prev-td close)

    # ══════════════════════════════════════════════════════════════
    # C. Intraday momentum & time-segment returns
    # ══════════════════════════════════════════════════════════════
    hours = g['time'].dt.hour
    day_mask = hours < 20
    night_mask = ~day_mask

    day_bars = g[day_mask]
    night_bars = g[night_mask]

    # ret_overnight: night session return (night_open → night_close)
    if len(night_bars) >= 2:
        night_o = w_open[night_bars.index[0]]
        night_c = w_close[night_bars.index[-1]]
        features['ret_overnight'] = _safe_log_scalar(night_c / night_o)
    else:
        features['ret_overnight'] = np.nan

    # ret_intraday: day session return (day_open → day_close)
    if len(day_bars) >= 2:
        day_o = w_open[day_bars.index[0]]
        day_c = w_close[day_bars.index[-1]]
        features['ret_intraday'] = _safe_log_scalar(day_c / day_o)
    else:
        features['ret_intraday'] = np.nan

    # ret_open30: first 30 minutes of day session
    features['ret_open30'] = np.nan
    if len(day_bars) > 0:
        t0 = day_bars['time'].iloc[0]
        t30 = t0 + pd.Timedelta(minutes=30)
        open30 = day_bars[day_bars['time'] <= t30]
        if len(open30) > 0:
            features['ret_open30'] = _safe_log_scalar(
                w_close[open30.index[-1]] / w_open[open30.index[0]])

    # ret_close30: last 30 minutes of day session
    features['ret_close30'] = np.nan
    if len(day_bars) > 0:
        tN = day_bars['time'].iloc[-1]
        t30b = tN - pd.Timedelta(minutes=30)
        close30 = day_bars[day_bars['time'] >= t30b]
        if len(close30) > 0:
            features['ret_close30'] = _safe_log_scalar(
                w_close[close30.index[-1]] / w_open[close30.index[0]])

    # ══════════════════════════════════════════════════════════════
    # D. Microstructure & order flow
    # ══════════════════════════════════════════════════════════════
    # oi_flow: Σ ΔOI_i · sign(r_i)
    delta_oi = position.diff().fillna(0.0)
    features['oi_flow'] = float((delta_oi * np.sign(rets.fillna(0.0))).sum())

    # cvd: BVC-based cumulative volume delta
    #   buyV_i = V_i · Φ(r_i / σ_r),  delta_i = 2·buyV_i - V_i
    sigma_r = float(rets.std()) if nv >= 2 else 0.0
    if sigma_r > 0:
        z = (rets.fillna(0.0) / sigma_r).clip(-8, 8)
        buy_v = volume * _norm_cdf(z.values)
        delta_v = 2.0 * buy_v - volume
        features['cvd'] = float(delta_v.sum())
    else:
        features['cvd'] = 0.0

    # vwap_dev: (day_close - VWAP) / VWAP
    typical = (w_high + w_low + w_close) / 3.0
    total_v = volume.sum()
    if total_v > 0:
        vwap = float((typical * volume).sum() / total_v)
        day_c = w_close.iloc[-1]
        features['vwap_dev'] = _safe_log_scalar(day_c / vwap)
    else:
        features['vwap_dev'] = np.nan

    # amihud_min: (1/N) · Σ |r_i| / DV_i,  DV_i = V_i · C_i
    dv = (volume * w_close.abs()).replace(0, np.nan)
    amihud_vals = (rets.abs() / dv).replace([np.inf, -np.inf], np.nan)
    features['amihud_min'] = float(amihud_vals.mean()) if amihud_vals.notna().any() else np.nan

    # vr_k: variance ratio VR(k) = Var(r^(k)) / (k · Var(r^(1)))
    if nv >= _VR_K * 2:
        r_arr = valid_r.values
        # k-minute aggregated returns (non-overlapping)
        n_k = len(r_arr) // _VR_K
        if n_k >= 2:
            r_k = r_arr[:n_k * _VR_K].reshape(n_k, _VR_K).sum(axis=1)
            var1 = float(np.var(r_arr, ddof=1))
            var_k = float(np.var(r_k, ddof=1))
            if var1 > 0:
                features['vr_k'] = var_k / (_VR_K * var1)
            else:
                features['vr_k'] = np.nan
        else:
            features['vr_k'] = np.nan
    else:
        features['vr_k'] = np.nan

    # kyle_lambda: regression slope of r_i on signed volume
    #   r_i = α + λ·SV_i + ε,  SV_i = V_i · sign(r_i) (tick-rule proxy)
    sv = (volume * np.sign(rets.fillna(0.0))).values
    r_vals = rets.fillna(0.0).values
    mask = (sv != 0) & np.isfinite(r_vals)
    if mask.sum() >= 5:
        x = sv[mask].astype(float)
        y = r_vals[mask].astype(float)
        var_x = float(np.var(x, ddof=0))
        if var_x > 0:
            features['kyle_lambda'] = float(np.cov(y, x, ddof=0)[0, 1] / var_x)
        else:
            features['kyle_lambda'] = np.nan
    else:
        features['kyle_lambda'] = np.nan

    return pd.Series(features)


# ── Yang-Zhang post-processing (needs cross-td data) ──────────────

def _compute_yz(
    daily_ohlc: pd.DataFrame,
    yz_k: float = 0.96,
) -> pd.Series:
    """Compute Yang-Zhang volatility from daily aggregated OHLC.

    daily_ohlc must have columns: td, open, high, low, close  (weighted prices).
    Returns a Series indexed like daily_ohlc with yz values.
    """
    df = daily_ohlc.copy().sort_values('td').reset_index(drop=True)
    o = pd.to_numeric(df['open'], errors='coerce')
    h = pd.to_numeric(df['high'], errors='coerce')
    l = pd.to_numeric(df['low'], errors='coerce')
    c = pd.to_numeric(df['close'], errors='coerce')

    prev_c = c.shift(1)
    # Overnight variance: (ln(O_t / C_{t-1}))²
    overnight = (_safe_log(o / prev_c) ** 2)
    # Daily RS (Rogers-Satchell on aggregated OHLC)
    rs_daily = (_safe_log(h / c) * _safe_log(h / o)
                + _safe_log(l / c) * _safe_log(l / o))
    # Drift: (ln(C_t / O_t))²
    drift = (_safe_log(c / o) ** 2)

    yz = overnight + yz_k * rs_daily + (1.0 - yz_k) * drift
    yz.index = df.index
    return yz


# ── Main entry point ──────────────────────────────────────────────

def compute_intraday_daily_features(
    minute_df: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    trading_days=None,
    instrument_id_list: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute daily-frequency features from minute bars.

    Parameters
    ----------
    minute_df : pd.DataFrame
        Minute bars with columns: time, instrument_id, open, high, low,
        close, volume, position (or oi), weighted_factor.
    feature_names : list of str, optional
        Which features to compute.  None means **all** features.
    trading_days : list, optional
        Trading-day list for assign_trading_day_1min.
    instrument_id_list : list, optional
        Instruments to keep (None = all in minute_df).

    Returns
    -------
    pd.DataFrame
        Columns: time (trading day), instrument_id, <feature columns>.
        Only the requested features are included.
    """
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()

    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES
    else:
        feature_names = list(feature_names)

    df = minute_df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])

    if instrument_id_list is not None:
        df = df[df['instrument_id'].isin(list(instrument_id_list))]

    # Assign trading day (night session → next trading day)
    df['td'] = assign_trading_day_1min(df['time'], trading_days=trading_days)
    df = df.dropna(subset=['td'])

    # Group by (instrument_id, td) and compute all features
    results = []
    for ins_id, ins_df in df.groupby('instrument_id', sort=False):
        ins_df = ins_df.sort_values('time')
        grouped = ins_df.groupby('td')

        # Select only columns needed by _compute_group_features to avoid
        # the pandas deprecation warning about grouping columns in apply.
        apply_cols = [c for c in
                      ['time', 'open', 'high', 'low', 'close', 'volume',
                       'position', 'oi', 'weighted_factor']
                      if c in ins_df.columns]

        # Compute per-td features via apply
        feat_df = grouped[apply_cols].apply(_compute_group_features)
        if feat_df is None or feat_df.empty:
            continue

        # groupby().apply() returns DataFrame with td as index.
        feat_df = feat_df.reset_index()
        # Ensure the group-key column is named 'td'
        if 'td' not in feat_df.columns:
            # In some pandas versions the index col gets named 'index' or 'level_0'
            first_col = feat_df.columns[0]
            feat_df = feat_df.rename(columns={first_col: 'td'})

        # Compute YZ (needs cross-td daily OHLC + prev close)
        if 'yz' in feature_names:
            daily_ohlc = grouped.agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
            ).reset_index()
            # Apply weighted factor for consistent prices
            if 'weighted_factor' in ins_df.columns:
                wf = ins_df.groupby('td')['weighted_factor'].first().reset_index()
                wf = wf.rename(columns={'weighted_factor': '_wf'})
                daily_ohlc = daily_ohlc.merge(wf, on='td', how='left')
                for col in ['open', 'high', 'low', 'close']:
                    daily_ohlc[col] = (
                        pd.to_numeric(daily_ohlc[col], errors='coerce')
                        * daily_ohlc['_wf']
                    )
            yz_series = _compute_yz(daily_ohlc)
            # Align YZ to feat_df by td
            yz_map = dict(zip(daily_ohlc['td'], yz_series.values))
            feat_df['yz'] = feat_df['td'].map(yz_map)

        feat_df['instrument_id'] = ins_id
        # Normalize td to date
        feat_df['time'] = pd.to_datetime(feat_df['td']).dt.normalize()
        feat_df = feat_df.drop(columns=['td'], errors='ignore')

        results.append(feat_df)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    # Select only requested features + time + instrument_id
    keep = ['time', 'instrument_id'] + [f for f in feature_names if f in out.columns]
    out = out[keep].sort_values(['instrument_id', 'time']).reset_index(drop=True)
    return out
