import logging
import os
from pathlib import Path


class LoggerFactory(object):

    @staticmethod
    def create(name: str, log_type = None):
        LOG_FOLDER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..', '..', '..', 'log')

        if log_type is None:
            log_type = 'debug'

        logger = logging.getLogger('logger.%s' % name)

        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            if not Path(LOG_FOLDER_PATH).exists():
                Path(LOG_FOLDER_PATH).mkdir()

            log_file = os.path.join(LOG_FOLDER_PATH, '%s.log' % log_type)
            file_handler = logging.FileHandler(log_file)

            format = logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s')
            file_handler.setFormatter(format)

            file_handler.setLevel(logging.DEBUG)

            logger.addHandler(file_handler)

        return logger

