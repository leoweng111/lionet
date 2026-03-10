"""
Basic configuration.
"""
import os
from pathlib import Path
from configparser import ConfigParser

from .logging import log


CONFIG_PATH = Path('/Users/wenglongao/configuration')
CONFIG_FILE_NAME = os.path.join(CONFIG_PATH, 'config.ini')


def _init_config():
    """
    Init configuration file.
    """
    config = ConfigParser()
    with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as file:
        config.write(file)
    log.info(f'Successfully init config {CONFIG_FILE_NAME}.')


def update_config(section: str, option: str, value: str):
    """
    Update configuration on specific option of specific section.

    :param section: config section
    :param option: config option
    :param value: updating value
    :return: None
    """
    config = ConfigParser()
    config.read(CONFIG_FILE_NAME)
    if section not in config:
        config.add_section(section)
    config.set(section, option, value)
    with open(CONFIG_FILE_NAME, 'w', encoding='utf-8') as file:
        config.write(file)

    log.info(f'Successfully update config {CONFIG_FILE_NAME}.')


def get_config(section: str, option: str):
    """
    Get configuration on specific option of specific section.

    :param section: config section
    :param option: config option
    :return: None
    """
    config = ConfigParser()
    config.read(CONFIG_FILE_NAME)
    config_dc = dict(config.items(section))

    return config_dc[option]
