from logger.Base import BaseLogger


class StdOutputLogger(BaseLogger):

    def __init__(self, names_log_list):
        self._names_log_list = names_log_list

    def to_log(self, **kwargs):
        print("----------------- iteration -----------------")
        for name in self._names_log_list:
            print(name, " ", kwargs.get(name, "- No log -"))
