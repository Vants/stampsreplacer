import os

from scripts import RESOURCES_PATH
from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PhaseCorrection import PhaseCorrection
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.processes.PsWeed import PsWeed
from scripts.utils.ConfigUtils import ConfigUtils
from scripts.utils.ProcessFactory import ProcessFactory


class Main:
    processes = [CreateLonLat, PsFiles, PsEstGamma, PsSelect, PsWeed, PhaseCorrection]

    def __init__(self) -> None:
        self.__path, self.__geo_file_path, self.__save_load_path = self.__get_from_config()

        self.__process_factory = ProcessFactory(self.__path, self.__geo_file_path,
                                                self.__save_load_path)

    def run(self, start=0, end=8):
        """Parameetrid start ja stop näitavad mitmendast protsessist alusustatakse (start) ja
        lõpetada (stop). Väikseim parameeter on 0 ja suurim 5 (tulenev len(self.processes)).

        Ühte ainsat protsessi saab kävitada kui panna start ja end võrdseks. Näiteks run(0, 0)."""



        for step in range(len(self.processes)):
            if (step - 1) == end:
                self.__save_results(start, end)
                break
            elif step > start:
                self.__load_saved(step)
            else:
                self.__start_process(step)

    def __load_saved(self, step: int):
        if step == 0:
            self.__process_factory.load_lonlat(self.processes[0], self.__path,
                                               self.__geo_file_path)
        else:
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
                if step == 0:
                    self.__process_factory.save_lonlat()
                else:
                    self.__process_factory.save_process(self.processes[step])

    def __assert_params(self, start: int, stop: int):
        if start < 0:
            raise AttributeError("Start less than 0")
        elif stop > len(self.processes):
            raise AttributeError("Stop more than than {0} or len(self.processes)".format(len(self.processes)))

    # noinspection PyMethodMayBeStatic
    def __get_from_config(self) -> (str, str, str):
        config = ConfigUtils(RESOURCES_PATH)
        initial_path = config.get_default_section('path')
        patch_folder = config.get_default_section('patch_folder')
        # Stamps'i või SNAP'i failid / vahekaust(mida ei pea olema) / PATCH_1
        path = os.path.join(initial_path, patch_folder)

        geo_file = config.get_default_section('geo_file')
        geo_file_path = os.path.join(initial_path, geo_file)

        save_load_path = config.get_default_section('save_load_path')

        return path, geo_file_path, save_load_path