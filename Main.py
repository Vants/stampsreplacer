import os

import sys

from scripts import RESOURCES_PATH
from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PhaseCorrection import PhaseCorrection
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.processes.PsWeed import PsWeed
from scripts.utils.internal.ConfigUtils import ConfigUtils
from scripts.utils.internal.LoggerFactory import LoggerFactory
from scripts.utils.internal.ProcessHandler import ProcessHandler


class Main:
    processes = [CreateLonLat, PsFiles, PsEstGamma, PsSelect, PsWeed, PhaseCorrection]

    def __init__(self) -> None:
        self.__logger = LoggerFactory.create("Main")

        self.__process_factory = self.__make_process_factory()

    def run(self, start=0, end=8):
        """
        The parameters start and end indicate the index of the process start and when to end
        processing. The smallest number is 0 and the greatest 5 (see len(self.processes)).

        You can run one process at the time when you set start and end to equal value
        (:e.g. run(0, 0)).

        :param start: step index where to start processing
        :param end: step index where to end processing
        :return: saves result(s) to save_load_path that is configuration
        """

        step = -1
        try:
            for step in range(len(self.processes)):
                if (step - 1) == end:
                    break
                elif step < start:
                    self.__load_saved(step)
                else:
                    self.__start_process(step)
        except Exception:
            self.__logger.error("Main process run error", exc_info=True)
            end = (step - 1)
        finally:
            self.__save_results(start, end)

    def __load_saved(self, step: int):
        self.__process_factory.load_results(self.processes[step])

    def __start_process(self, step: int):
        self.__process_factory.start_process(self.processes[step])

    def __save_results(self, start: int, end: int):
        for step in range(len(self.processes)):
            if step < start:
                continue
            elif (step - 1) == end:
                break
            elif step >= start:
                self.__process_factory.save_process(self.processes[step])

    def __assert_params(self, start: int, stop: int):
        if start < 0:
            raise AttributeError("Start less than 0")
        elif stop > len(self.processes):
            raise AttributeError(
                "Stop more than than {0} or len(self.processes)".format(len(self.processes)))

    def __make_process_factory(self) -> ProcessHandler:
        path, geo_file_path, save_load_path, rand_dist_cached = self.__get_from_config()
        return ProcessHandler(path, geo_file_path, save_load_path, rand_dist_cached)

    # noinspection PyMethodMayBeStatic
    def __get_from_config(self) -> (str, str, str, bool):
        self.__logger.info("Loading params form {0}".format(RESOURCES_PATH))

        config = ConfigUtils(RESOURCES_PATH)
        initial_path = config.get_default_section('path')
        patch_folder = config.get_default_section('patch_folder')
        # Stamps or SNAP files/folder(not mandatory)/PATCH_1
        path = os.path.join(initial_path, patch_folder)

        geo_file = config.get_default_section('geo_file')
        geo_file_path = os.path.join(initial_path, geo_file)

        save_load_path = config.get_default_section('save_load_path')

        rand_dist_cached = config.get_default_section('rand_dist_cached') == 'True'

        self.__logger.info("Loaded params. path {0}, geo_file_path {1}, save_load_path {2},"
                           " rand_dist_cached {3}".format(path, geo_file_path, save_load_path,
                                                          rand_dist_cached))
        return path, geo_file_path, save_load_path, rand_dist_cached


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print("Use Main <start> <end>. Params are not mandatory")
    elif len(sys.argv) == 1:
        main = Main()
        main.run()
    else:
        main = Main()
        main.run(start=int(sys.argv[1]), end=int(sys.argv[2]))
