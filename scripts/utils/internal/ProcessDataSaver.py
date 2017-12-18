from pathlib import Path
import numpy as np

from scripts.utils.internal.LoggerFactory import LoggerFactory


class ProcessDataSaver:
    def __init__(self, file_path: str, file_name: str, log_level="debug"):
        """Loob salvestaja. 'File_path' on koht kuhu fail salvestatake (tavaliselt
        FolderConstants.SAVE_FOLDER) ja 'file_name' on faili nimi (tavaliselt klassimuutja, et
        oleks lihtsam pärast salvestatut laadida).

        Loggeri jaoks, 'log_type'. Kui logida ei taha pole vaja siis muutuja None'iks.
        Tavaline väärtus sellel on 'debug'.

        Selleks, et salvestatut laadida kasuta np.load*i. Selle jaoks klassi ei ole."""

        def check_param(param):
            return param is None or param == ''

        if check_param(file_path):
            raise AttributeError("Check file path")
        elif check_param(file_name):
            raise AttributeError("Check file name")

        self.file_path_with_name = Path(file_path, file_name)

        if log_level is not None or log_level != '':
            self.__logger = self.make_logger(file_name)
            self.__logger.info("ProcessDataSaver created. File path " + file_path
                               + ", file_name " + file_name)

        if not Path(file_path).exists():
            Path(file_path).mkdir(parents=True)
            self.__logger.info("Folders created")

    def save_data(self, **data):
        np.savez(str(self.file_path_with_name.absolute()), **data)

        if self.__logger is not None:
            self.__logger.debug(data)

    # noinspection PyMethodMayBeStatic
    def make_logger(self, file_name):
        logger_name = "LoggerFactory." + file_name
        return LoggerFactory.create(logger_name)
