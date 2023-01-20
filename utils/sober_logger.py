# coding=utf-8
# @Project  ：SoberLogTools 
# @FileName ：sober_logger.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/11/4 8:39 下午

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import os.path as osp
from datetime import datetime

from utils.util import get_project_root

DEFAULT_LOGFILE_LEVEL = 'debug'
DEFAULT_STDOUT_LEVEL = 'info'
DEFAULT_LOG_FILE_CHARACTER = 'default_%s.log'
DEFAULT_LOG_FORMAT =  "%(asctime)s %(levelname)-7s %(message)s"

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class SoberLogger(object):
    """
    Args:
      Log level: CRITICAL>ERROR>WARNING>INFO>DEBUG.
      Log file: The file that stores the logging info.
      rewrite: Clear the log file.
      log format: The format of log messages.
      stdout level: The log level to print on the screen.
    """
    logfile_level = None
    log_file = None
    log_format = None
    rewrite = None
    stdout_level = None
    logger = None

    _caches = {}

    @staticmethod
    def init(logfile_level=DEFAULT_LOGFILE_LEVEL,
             log_file=None,
             log_format=DEFAULT_LOG_FORMAT,
             rewrite=False,
             stdout_level='debug', default_name = None):
        '''

        :param logfile_level:
        :param log_file:
        :param log_format:
        :param rewrite:
        :param stdout_level:
        :return:
        '''
        if log_file is None:
            SoberLogger.log_file = osp.join(get_project_root(), 'logs', DEFAULT_LOG_FILE_CHARACTER % (datetime.now().strftime(
                "%Y%d%m_%H-%M-%S")))
        elif '.' not in osp.split(log_file)[-1]:
            SoberLogger.log_file = osp.join(log_file, DEFAULT_LOG_FILE_CHARACTER % (datetime.now().strftime(
                "%Y%d%m_%H-%M-%S")))
        else:
            SoberLogger.log_file = log_file

        SoberLogger.logfile_level = logfile_level
        SoberLogger.log_format = log_format if log_format is not None else DEFAULT_LOG_FORMAT
        SoberLogger.rewrite = rewrite
        SoberLogger.stdout_level = stdout_level

        SoberLogger.logger = logging.getLogger(__name__ if default_name == None else default_name)
        SoberLogger.logger.handlers = []

        fmt = logging.Formatter(SoberLogger.log_format)
        SoberLogger.logger.setLevel(logging.DEBUG)
        if SoberLogger.logfile_level is not None:
            filemode = 'w'
            if not SoberLogger.rewrite:
                filemode = 'a'

            dir_name = os.path.dirname(os.path.abspath(SoberLogger.log_file))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if SoberLogger.logfile_level not in LOG_LEVEL_DICT:
                print('Invalid logging level: {}'.format(SoberLogger.logfile_level))
                SoberLogger.logfile_level = DEFAULT_LOGFILE_LEVEL


            fh = logging.FileHandler(SoberLogger.log_file, mode=filemode)
            fh.setLevel(LOG_LEVEL_DICT[SoberLogger.logfile_level])
            fh.setFormatter(fmt)
            SoberLogger.logger.addHandler(fh)

        if stdout_level is not None:
            if SoberLogger.logfile_level is None:
                SoberLogger.logger.setLevel(LOG_LEVEL_DICT[SoberLogger.stdout_level])

            if SoberLogger.stdout_level not in LOG_LEVEL_DICT:
                print('Invalid logging level: {}'.format(SoberLogger.stdout_level))
                return

            console = logging.StreamHandler()
            console.setLevel(LOG_LEVEL_DICT[SoberLogger.stdout_level])
            console.setFormatter(fmt)
            SoberLogger.logger.addHandler(console)

        SoberLogger.logger.propagate = False


    @staticmethod
    def set_log_file(file_path):
        SoberLogger.log_file = file_path
        SoberLogger.init(log_file=file_path)

    @staticmethod
    def set_logfile_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return

        SoberLogger.init(logfile_level=log_level)

    @staticmethod
    def clear_log_file():
        SoberLogger.rewrite = True
        SoberLogger.init(rewrite=True)

    @staticmethod
    def check_logger():
        if SoberLogger.logger is None:
            SoberLogger.init(logfile_level=None, stdout_level=DEFAULT_STDOUT_LEVEL)

    @staticmethod
    def __obtain_filename_info_prefix():
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        return prefix

    @staticmethod
    def set_stdout_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return

        SoberLogger.init(stdout_level=log_level)

    @staticmethod
    def debug(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        SoberLogger.logger.debug('{} {}'.format(prefix, ''.join(map(str,
                                                                    message))))

    @staticmethod
    def info(*message):
        SoberLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)
        string = '{} {}'.format(prefix, ''.join(map(str, message)))
        SoberLogger.logger.info('{} {}'.format(prefix, ''.join(map(str, message))))

    @staticmethod
    def info_once(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        if SoberLogger._caches.get((prefix, message)) is not None:
            return

        SoberLogger.logger.info('{} {}'.format(prefix, ''.join(map(str, message))))
        SoberLogger._caches[(prefix, ''.join(message))] = True


    @staticmethod
    def warn(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        SoberLogger.logger.warning('{} {}'.format(prefix, ''.join(map(str, message))))

    @staticmethod
    def warning(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        SoberLogger.logger.warning('{} {}'.format(prefix, ''.join(map(str, message))))

    @staticmethod
    def error(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        SoberLogger.logger.error('{} {}'.format(prefix, ''.join(map(str, message))))

    @staticmethod
    def critical(*message):
        SoberLogger.check_logger()

        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename,lineno)

        SoberLogger.logger.critical('{} {}'.format(prefix, ''.join(map(str, message))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile_level', default="info", type=str,
                        dest='logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default='debug', type=str,
                        dest='stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default="./default.log", type=str,
                        dest='log_file', help='The path of log files.')
    parser.add_argument('--log_format', default="%(asctime)s %(levelname)-7s %(message)s",
                        type=str, dest='log_format', help='The format of log messages.')
    parser.add_argument('--rewrite', default=False, type=bool,
                        dest='rewrite', help='Clear the log files existed.')

    args = parser.parse_args()
    # "%(asctime)s %(levelname)-7s %(message)s"
    SoberLogger.init(logfile_level=args.logfile_level, stdout_level=args.stdout_level,
                     log_file=None, log_format=None, rewrite=args.rewrite)

    SoberLogger.info("info test.", 'abcd')
    SoberLogger.debug("debug test.")
    SoberLogger.warn("warn test.")
    SoberLogger.error("error test.")
    SoberLogger.debug("debug test.")
    SoberLogger.critical('critical !!!')
