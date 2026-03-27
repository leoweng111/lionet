# This script contains some important universal paths.
from pathlib import Path
import datetime
import os

from .configuration import get_config

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

# time
START_TIME = datetime.datetime(2001, 1, 1, 9, 30)
START_DATE = datetime.date(2001, 1, 1)
START_DATE_STR = '20010101'

END_DATE = datetime.date.today()
END_TIME = datetime.datetime.now()
END_DATE_STR = datetime.date.today().strftime('%Y%m%d')

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

# LLM (DeepSeek)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Supported indicator names for factor filtering.
SUPPORTED_FILTER_INDICATORS = [
    'Gross Return', 'Net Return',
    'Gross Volatility', 'Net Volatility',
    'Gross Sharpe', 'Net Sharpe',
    'Gross Sortino', 'Net Sortino',
    'Gross MaxDD', 'Net MaxDD',
    'Gross Calmar', 'Net Calmar',
    'Gross Win Rate', 'Net Win Rate',
    'Turnover', 'TS IC', 'TS RankIC', 'T-corr', 'count',
]

