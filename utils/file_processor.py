# This package contains some file utils.

import os
import shutil
from pathlib import Path
from typing import Union, Literal

from utils.logging import log


def file_type(file_path: Union[str, Path]):
    """
    e.g.
    os.path.isfile(PRE_PROCESS_PATH/'Expert RAW') -- folder
    os.path.isfile(PRE_PROCESS_PATH/'20190616_145743_IMG_0106.HEIC') -- HEIC
    instead of using third-party packages(e.g. magic), simply return 'folder' or the suffix of file.

    :param file_path: file path
    :return: file type, 'folder' or suffix
    """
    if not os.path.exists(file_path):
        file = file_path.split('.')
        assert len(file) > 1, f'Wrong file type {file}!'
        return file[-1]

    if os.path.isdir(file_path):
        return 'folder'
    elif os.path.isfile(file_path):
        file = os.path.basename(file_path).split('.')
        assert len(file) > 1, f'Wrong file type {file}!'
        return file[-1]
    else:
        raise Exception(f'Unknown file type for {file_path}.')


def check_file_exists(check_path: Union[str, Path], file_name: str):
    """
    check whether file with file_name exists under check_path
    :param check_path: check path
    :param file_name: file name of checked file
    :return: True if exists else False
    """
    file_path = os.path.join(check_path, file_name)
    return os.path.exists(file_path)


def make_path(path: Union[str, Path], folder_name: str = None):
    """
    Make a new path, with optional folder_name. if folder_name, then it will make path/folder_name;
    otherwise, it will simply make a path if path does not exist.
    The benefit of using this func is mostly based on the attribute that it will not raise error when making an already
    existed path.

    :param path: path
    :param folder_name: folder name
    :return: None
    """

    if not folder_name:
        if os.path.exists(path):
            log.info(f'{path} already exists.')

        else:
            os.makedirs(path)
            log.info(f'Make path{path}.')

    else:
        if check_file_exists(path, folder_name):
            log.info(f'{os.path.join(path, folder_name)} already exists.')

        else:
            os.makedirs(os.path.join(path, folder_name))
            log.info(f'Make path{os.path.join(path, folder_name)}.')


def move_file(file_old: Union[str, Path],
              file_new: Union[str, Path],
              method: Literal['skip', 'overwrite'],
              create_new: bool = False):
    """
    Move file from file_old to file_new, attention that both file_old and file_new are file path including file name as
    os.path.basename.


    :param file_old: old file path
    :param file_new: new file path
    :param method: skip or overwrite
    :param create_new: if True, then create a new path if os.path.dirname of file_new does not exist. Default is False.
    :return: True if overwrite or file_new does not exist, meaning that the file is newly-added to the target folder.
    """

    assert method in ['skip', 'overwrite'], "Only supports 'skip' or 'overwrite' method."
    assert os.path.exists(file_old), f'Old file {file_old} does not exist.'

    assert os.path.isfile(file_old), f'Old file {file_old} is not a file name.'

    if create_new:
        make_path(os.path.dirname(file_new))
    else:
        assert os.path.exists(os.path.dirname(file_new)), f'New Path{os.path.dirname(file_new)} does not exist.'

    if method == 'overwrite':
        shutil.copy(file_old, file_new)
        log.info(f'Successfully overwrite {file_old} to {file_new}.')
        return True
    else:
        if os.path.exists(file_new):
            log.info(f'File {file_new} already exists, skip copying.')
            return False
        else:
            shutil.copy(file_old, file_new)
            return True


def clear_path(path: Union[str, Path]):
    """
    Delete all files on path, leaving path as an empty folder.

    :param path: path
    :return: None
    """

    shutil.rmtree(path)
    os.mkdir(path)
    log.info(f'Successfully clear {path}.')
