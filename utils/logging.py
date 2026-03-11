import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Union


LOG_PATH = Path('/Users/wenglongao/log')


def initialize_log(log_name, path: Union[str, Path] = LOG_PATH):
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime('%Y%m%d')
    file_path = base_path / f'{log_name}_{date_str}.log'

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_path_str = str(file_path)

    # Avoid adding duplicated handlers when module is imported multiple times.
    has_handler = any(
        isinstance(handler, logging.FileHandler) and
        getattr(handler, 'baseFilename', None) == file_path_str
        for handler in logger.handlers
    )
    if not has_handler:
        handler = logging.FileHandler(file_path_str, encoding='utf-8')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = initialize_log('log')
