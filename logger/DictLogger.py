from logger.Base import BaseLogger


class DictLogger(BaseLogger):

    def __init__(self, names_log_list):
        self._names_log_list = names_log_list
        self._log_state = None

    def to_log(self, **kwargs):
        self._log_state = {name: kwargs.get(name, None) for name in self._names_log_list}

    def get_log(self):
        return self._log_state
