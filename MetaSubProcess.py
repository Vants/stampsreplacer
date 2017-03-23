from abc import ABCMeta, abstractmethod


class MetaSubProcess(metaclass=ABCMeta):
    @abstractmethod
    def start_process(self, save_process=False): pass
