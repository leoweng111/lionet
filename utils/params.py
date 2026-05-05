# This script contains some important universal paths.
from pathlib import Path
import os

from .configuration import get_config
from .config_json import load_json_config

PREPROCESS_PHOTO_PATH = Path('/Volumes/Leo Weng/photos/整理前的照片')
POSTPROCESS_PHOTO_PATH = Path('/Volumes/Leo Weng/photos/整理完成的照片')

# album_folder to album map
FOLDER_TO_ALBUM_MAP = {'Camera': '相机',
                       'Screenshots': '截屏',
                       'bing': 'bing',
                       'CoolMarket': '酷安',
                       'bili': 'bilibili',
                       'WeiXin': '微信',
                       '知乎': '知乎',
                       '小红书': '小红书',
                       'news_article': '今日头条',
                       'others': 'others'}

OUTPUT_PATH = Path('/Users/wenglongao/output')
PREP_PATH = Path('/Users/wenglongao/prep')

# MongoDB
USER_NAME = 'leo'
MONGODB_PASSWORD = get_config('password', 'mongodb_password')


# Futures back-adjustment anchor date:
# adjusted price equals raw price on this date, then rolls apply cumulatively afterwards.
RESEARCH_START_DATE = '20200101'

# Futures fixed listing-month mapping.
# Key is root instrument id (without trailing 0), value is listed month numbers.
# Example: C -> Jan/Mar/May/Jul/Sep/Nov.
FUTURES_FIXED_LISTING_MONTHS = {
    'C': [1, 3, 5, 7, 9, 11],
}

# Instrument-level trading fee config.
FEE = {"C0": "0.0002", "FG0": "0.0002"}

# Futures contract multiplier config, keyed by root symbol (e.g. C, RB, FG).
FUTURES_CONTRACT_MULTIPLIER = {
    'C': 10,
}

# LLM (DeepSeek)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Unified indicator names for both GP fitness construction and threshold filtering.
# NOTE:
# - `count` (annualized instrument count) is intentionally excluded.
_DEFAULT_GP_SUPPORTED_INDICATOR = [
    'Gross Return', 'Net Return',
    'Gross Volatility', 'Net Volatility',
    'Gross Sharpe', 'Net Sharpe',
    'Gross Sortino', 'Net Sortino',
    'Gross MaxDD', 'Net MaxDD',
    'Gross Calmar', 'Net Calmar',
    'Gross Win Rate', 'Net Win Rate',
    'Turnover', 'TS IC', 'TS RankIC', 'TS ICIR', 'TS RankICIR', 'T-corr',
]

# Direction for threshold comparison:
#  1: larger is better, -1: smaller is better.
_DEFAULT_GP_INDICATOR_DIRECTION = {
    'Gross Return': 1,
    'Net Return': 1,
    'Gross Volatility': -1,
    'Net Volatility': -1,
    'Gross Sharpe': 1,
    'Net Sharpe': 1,
    'Gross Sortino': 1,
    'Net Sortino': 1,
    'Gross MaxDD': -1,
    'Net MaxDD': -1,
    'Gross Calmar': 1,
    'Net Calmar': 1,
    'Gross Win Rate': 1,
    'Net Win Rate': 1,
    'Turnover': -1,
    'TS IC': 1,
    'TS RankIC': 1,
    'TS ICIR': 1,
    'TS RankICIR': 1,
    'T-corr': 1,
}

# Default fitness weights: keep current behavior (IC-driven) by default.
_DEFAULT_GP_DEFAULT_FITNESS_INDICATOR_WEIGHT = {
    'TS IC': 1.0,
}

# Default filter threshold config:
# indicator -> {mean_threshold, yearly_threshold, direction}
_DEFAULT_GP_DEFAULT_FILTER_INDICATOR_DICT = {
    'Net Return': {
        'mean_threshold': 0.05,
        'yearly_threshold': 0.03,
        'direction': 1,
    },
    'Net Sharpe': {
        'mean_threshold': 0.5,
        'yearly_threshold': 0.3,
        'direction': 1,
    },
}


def _normalize_indicator_list(raw_list):
    if not isinstance(raw_list, list):
        return list(_DEFAULT_GP_SUPPORTED_INDICATOR)
    out = [str(x).strip() for x in raw_list if str(x).strip()]
    return out or list(_DEFAULT_GP_SUPPORTED_INDICATOR)


def _normalize_indicator_direction(raw_direction, supported_indicator):
    out = {}
    if isinstance(raw_direction, dict):
        for indicator in supported_indicator:
            raw_val = raw_direction.get(indicator, _DEFAULT_GP_INDICATOR_DIRECTION.get(indicator, 1))
            try:
                direction = int(raw_val)
            except (TypeError, ValueError):
                direction = int(_DEFAULT_GP_INDICATOR_DIRECTION.get(indicator, 1))
            out[indicator] = direction if direction in (1, -1) else int(_DEFAULT_GP_INDICATOR_DIRECTION.get(indicator, 1))
    if not out:
        out = dict(_DEFAULT_GP_INDICATOR_DIRECTION)
    return out


def _normalize_fitness_weight(raw_weight, supported_indicator):
    out = {}
    if isinstance(raw_weight, dict):
        for indicator in supported_indicator:
            if indicator not in raw_weight:
                continue
            try:
                out[indicator] = float(raw_weight[indicator])
            except (TypeError, ValueError):
                continue
    if not out:
        return dict(_DEFAULT_GP_DEFAULT_FITNESS_INDICATOR_WEIGHT)
    return out


def _normalize_filter_indicator_dict(raw_filter, supported_indicator, direction_map):
    out = {}
    if isinstance(raw_filter, dict):
        for indicator, conf in raw_filter.items():
            if indicator not in supported_indicator:
                continue
            if not isinstance(conf, dict):
                continue
            mean_threshold = conf.get('mean_threshold')
            yearly_threshold = conf.get('yearly_threshold')
            direction_raw = conf.get('direction', direction_map.get(indicator, 1))

            try:
                direction = int(direction_raw)
            except (TypeError, ValueError):
                direction = int(direction_map.get(indicator, 1))
            direction = direction if direction in (1, -1) else int(direction_map.get(indicator, 1))

            mean_val = None
            yearly_val = None
            if mean_threshold is not None:
                try:
                    mean_val = float(mean_threshold)
                except (TypeError, ValueError):
                    mean_val = None
            if yearly_threshold is not None:
                try:
                    yearly_val = float(yearly_threshold)
                except (TypeError, ValueError):
                    yearly_val = None

            out[indicator] = (mean_val, yearly_val, direction)

    if not out:
        for indicator, conf in _DEFAULT_GP_DEFAULT_FILTER_INDICATOR_DICT.items():
            out[indicator] = (
                conf.get('mean_threshold'),
                conf.get('yearly_threshold'),
                int(conf.get('direction', 1)),
            )
    return out


GP_SUPPORTED_INDICATOR = _normalize_indicator_list(
    load_json_config('gp_supported_indicator.json', _DEFAULT_GP_SUPPORTED_INDICATOR)
)

GP_INDICATOR_DIRECTION = _normalize_indicator_direction(
    load_json_config('gp_indicator_direction.json', _DEFAULT_GP_INDICATOR_DIRECTION),
    GP_SUPPORTED_INDICATOR,
)

GP_DEFAULT_FITNESS_INDICATOR_WEIGHT = _normalize_fitness_weight(
    load_json_config('gp_default_fitness_indicator_weight.json', _DEFAULT_GP_DEFAULT_FITNESS_INDICATOR_WEIGHT),
    GP_SUPPORTED_INDICATOR,
)

GP_DEFAULT_FILTER_INDICATOR_DICT = _normalize_filter_indicator_dict(
    load_json_config('gp_default_filter_indicator_dict.json', _DEFAULT_GP_DEFAULT_FILTER_INDICATOR_DICT),
    GP_SUPPORTED_INDICATOR,
    GP_INDICATOR_DIRECTION,
)

# Supported options for factor fusion.
FusionSupportedMethods = {'avg_weight'}
FusionSupportedMetrics = {'ic', 'sharpe'}

