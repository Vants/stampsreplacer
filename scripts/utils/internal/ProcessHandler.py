import numpy as np

from typing import Type

from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PhaseCorrection import PhaseCorrection
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.processes.PsWeed import PsWeed


class ProcessHandler:
    process_obj_dict = {}
    lonlat = np.array([])

    def __init__(self, path: str, geo_file_path: str, save_load_path: str, rand_dist_cached: bool):
        self.__path = path
        self.__geo_file_path = geo_file_path
        self.__save_load_path = save_load_path
        self.__rand_dist_cached = rand_dist_cached

    def load_results(self, process: Type[MetaSubProcess]):
        process_obj = self.__init_process(process)
        process_obj.load_results(self.__save_load_path)
        self.__set_process_to_dict(process_obj)

    def start_process(self, process_type: Type[MetaSubProcess]):
        process_obj = self.__init_process(process_type)
        process_obj.start_process()

        self.__set_process_to_dict(process_obj)

    def save_process(self, process_type: Type[MetaSubProcess]):
        process_obj = self.__get_process_from_dict(process_type)
        process_obj.save_results(self.__save_load_path)

    def __set_process_to_dict(self, process_obj: MetaSubProcess):
        process_type = type(process_obj)

        if process_type is CreateLonLat:
            self.process_obj_dict['LonLat'] = process_obj
        elif process_type is PsFiles:
            self.process_obj_dict['PsFiles'] = process_obj
        elif process_type is PsEstGamma:
            self.process_obj_dict['PsEstGamma'] = process_obj
        elif process_type is PsSelect:
            self.process_obj_dict['PsSelect'] = process_obj
        elif process_type is PsWeed:
            self.process_obj_dict['PsWeed'] = process_obj
        elif process_type is PhaseCorrection:
            self.process_obj_dict['PhaseCorrection'] = process_obj

    def __get_process_from_dict(self, process: Type[MetaSubProcess]) -> MetaSubProcess:
        if process is CreateLonLat:
            return self.process_obj_dict['LonLat']
        elif process is PsFiles:
            return self.process_obj_dict['PsFiles']
        elif process is PsEstGamma:
            return self.process_obj_dict['PsEstGamma']
        elif process is PsSelect:
            return self.process_obj_dict['PsSelect']
        elif process is PsWeed:
            return self.process_obj_dict['PsWeed']
        elif process is PhaseCorrection:
            return self.process_obj_dict['PhaseCorrection']

    def __init_process(self, process: Type[MetaSubProcess]) -> MetaSubProcess:
        if process is CreateLonLat:
            return process(self.__path, self.__geo_file_path)
        elif process is PsFiles:
            return process(self.__path, self.process_obj_dict['LonLat'])
        elif process is PsEstGamma:
            return process(self.process_obj_dict['PsFiles'], self.__rand_dist_cached)
        elif process is PsSelect:
            return process(self.process_obj_dict['PsFiles'], self.process_obj_dict['PsEstGamma'])
        elif process is PsWeed:
            return process(self.__path, self.process_obj_dict['PsFiles'],
                           self.process_obj_dict['PsEstGamma'], self.process_obj_dict['PsSelect'])
        elif process is PhaseCorrection:
            return process(self.process_obj_dict['PsFiles'], self.process_obj_dict['PsWeed'])