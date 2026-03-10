import os
import logging
from pathlib import Path
from typing import Union


LOG_PATH = Path('/Users/wenglongao/log')


def initialize_log(log_name, path: Union[str, Path] = LOG_PATH):
    log_path = os.path.join(path, log_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO, filename=f'{log_path}.log')

    return logging.getLogger(log_path)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO, filename=f'{LOG_PATH}/log.log')

log = logging.getLogger('log')
