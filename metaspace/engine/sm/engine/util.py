import os
import json
from datetime import datetime
from subprocess import check_call, call
import logging
from logging.config import dictConfig
from pathlib import Path
import re
from fnmatch import translate
from copy import deepcopy

import boto3


def proj_root():
    return os.getcwd()


def init_loggers(config=None):
    """ Init logger using config file, 'logs' section of the sm config
    """
    if not config:
        SMConfig.set_path('conf/config.json')
        config = SMConfig.get_conf()['logs']

    logs_dir = Path(proj_root()).joinpath('logs')
    if not logs_dir.exists():
        logs_dir.mkdir()

    log_level_codes = {
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    def convert_levels(orig_d):
        d = orig_d.copy()
        for k, v in d.items():
            if k == 'level':
                d[k] = log_level_codes[d[k]]
            elif type(v) == dict:
                d[k] = convert_levels(v)
        return d

    log_config = convert_levels(config)
    dictConfig(log_config)


class SMConfig(object):
    """ Engine configuration manager """

    _path = 'conf/config.json'
    _config_dict = {}

    @classmethod
    def set_path(cls, path):
        """ Set path for a SM configuration file

        Parameters
        ----------
        path : String
        """
        cls._path = os.path.realpath(str(path))

    @classmethod
    def get_conf(cls, update=False):
        """
        Returns
        -------
        : dict
            SM engine configuration
        """
        assert cls._path
        if update or not cls._config_dict:
            try:
                with open(cls._path) as f:
                    cls._config_dict = json.load(f)
            except IOError as e:
                logging.getLogger('engine').warning(e)
        return deepcopy(cls._config_dict)

    @classmethod
    def get_ms_file_handler(cls, ms_file_path):
        """
        Parameters
        ----------
        ms_file_path : String

        Returns
        -------
        : dict
            SM configuration for handling specific type of MS data
        """
        conf = cls.get_conf()
        ms_file_extension = Path(ms_file_path).suffix[1:].lower()  # skip the leading "."
        return next((h for h in conf['ms_file_handlers'] if ms_file_extension in h['extensions']), None)


def _cmd(template, call_func, *args):
    cmd_str = template.format(*args)
    logging.getLogger('engine').info('Call "%s"', cmd_str)
    return call_func(cmd_str.split())


def cmd_check(template, *args):
    return _cmd(template, check_call, *args)


def cmd(template, *args):
    return _cmd(template, call, *args)


def read_json(path):
    res = {}
    try:
        with open(path) as f:
            res = json.load(f)
    except IOError as e:
        logging.getLogger('engine').warning("Couldn't find %s file", path)
    finally:
        return res


def create_ds_from_files(ds_id, ds_name, ds_input_path, config_path, meta_path):
    if Path(meta_path).exists():
        metadata = json.load(open(str(meta_path)))
    else:
        raise Exception('meta.json not found')
    ds_config = json.load(open(str(config_path)))

    from sm.engine.dataset import Dataset
    return Dataset(id=ds_id,
                   name=ds_name,
                   input_path=str(ds_input_path),
                   upload_dt=datetime.now(),
                   metadata=metadata,
                   is_public=True,
                   mol_dbs=ds_config['databases'],
                   adducts=ds_config['isotope_generation']['adducts'])


def split_s3_path(path):
    """
    Returns
    ---
        tuple[string, string]
    Returns a pair of (bucket, key)
    """
    return str(path).split('s3a://')[-1].split('/', 1)


def create_s3_client(aws_config):
    session = boto3.session.Session(aws_access_key_id=aws_config['aws_access_key_id'],
                                    aws_secret_access_key=aws_config['aws_secret_access_key'])
    return session.resource('s3', region_name=aws_config['aws_region'])


def upload_dir_to_s3(s3, dir_path, s3_path):
    logger = logging.getLogger('engine')
    logger.debug(f'Uploading directory {dir_path} to {s3_path}')
    bucket_name, prefix = split_s3_path(s3_path)
    for local_path in dir_path.iterdir():
        key = f'{prefix}/{dir_path.name}/{local_path.name}'
        s3.Object(bucket_name, key).upload_file(str(local_path))


def download_file_from_s3(s3, s3_path, local_path):
    logger = logging.getLogger('engine')
    bucket_name, key = split_s3_path(s3_path)
    if not local_path.exists():
        logger.debug(f'Downloading file {s3_path} to {local_path}')
        s3.Object(bucket_name, key).download_file(str(local_path))


def download_prefix_from_s3(aws_config, s3_path, dir_path):
    logger = logging.getLogger('engine')
    logger.debug(f'Downloading S3 path {s3_path} to {dir_path}')

    s3 = create_s3_client(aws_config)
    bucket_name, key = split_s3_path(s3_path)
    for obj_sum in (s3.Bucket(bucket_name)
                    .objects.filter(Prefix=key)):
        local_file = str(dir_path / Path(obj_sum.key).name)
        obj_sum.Object().download_file(local_file)
