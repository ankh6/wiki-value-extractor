from abc import ABCMeta, abstractmethod

class Extractor(metaclass=ABCMeta):
    '''Interface that enables text content extraction'''

    @classmethod
    def __subclasshook__(cls, subclass):
        return(hasattr(subclass,"extract_content") and callable(subclass.extract_content) or NotImplementedError)

    @abstractmethod
    def extract_content(self):
        pass