from abc import ABCMeta, abstractmethod

class Extractor(metaclass=ABCMeta):
    '''Interface that enables text content extraction'''

    @classmethod
    def __subclasshook__(cls, subclass):
        return(hasattr(subclass,"read_page") and callable(subclass.read_page) or NotImplementedError)

    @abstractmethod
    def read_page(self):
        pass