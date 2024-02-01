from abc import abstractmethod, ABC


class BaseLogger(ABC):

    @abstractmethod
    def to_log(self, **kwargs):
        raise NotImplementedError
