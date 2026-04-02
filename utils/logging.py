import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Union


LOG_PATH = Path('/Users/wenglongao/log')


class DailyFileHandler(logging.FileHandler):
    """File handler that rolls to log_YYYYMMDD.log when the date changes."""

    def __init__(self, log_name: str, base_path: Path):
        self.log_name = str(log_name)
        self.base_path = Path(base_path)
        self.current_date = datetime.now().strftime('%Y%m%d')
        file_path = self.base_path / f'{self.log_name}_{self.current_date}.log'
        super().__init__(str(file_path), encoding='utf-8')

    def emit(self, record):
        new_date = datetime.now().strftime('%Y%m%d')
        if new_date != self.current_date:
            self.current_date = new_date
            self.close()
            self.baseFilename = str(self.base_path / f'{self.log_name}_{self.current_date}.log')
            self.stream = self._open()
        return super().emit(record)


def initialize_log(log_name, path: Union[str, Path] = LOG_PATH):
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Avoid adding duplicated handlers when module is imported multiple times.
    has_handler = any(isinstance(handler, DailyFileHandler) for handler in logger.handlers)
    if not has_handler:
        handler = DailyFileHandler(log_name=log_name, base_path=base_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


log = initialize_log('log')
