from abc import ABCMeta, abstractmethod


class MetaSubProcess(metaclass=ABCMeta):
    @abstractmethod
    def start_process(self): pass

    @abstractmethod
    def save_results(self, save_path: str): pass

    @abstractmethod
    def load_results(self, load_path): pass
