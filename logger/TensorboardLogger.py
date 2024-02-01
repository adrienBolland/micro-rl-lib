import os

from torch.utils.tensorboard import SummaryWriter

from logger.Base import BaseLogger


class TensorboardLogger(BaseLogger):
    """ Logger writing in a tensorboard the results """

    def __init__(self, names_log_list, log_path, log_period=1):
        self.writer = SummaryWriter(log_path + f"/version-{get_version(log_path)}")
        self.step_it = 0
        self.log_period = log_period
        self._names_log_list = names_log_list

    def to_log(self, **kwargs):
        if not self.step_it % self.log_period:
            if self._names_log_list:
                for name in self._names_log_list:
                    self.writer.add_scalar(name, kwargs.get(name, None), self.step_it)
            else:
                for name, value in kwargs.items():
                    self.writer.add_scalar(name, value, self.step_it)

        self.step_it += 1

    def __del__(self):
        # flush and close the writer before destructing the object
        self.writer.flush()
        self.writer.close()


def get_version(log_path, width=3):
    """returns str repr of the version"""
    os.makedirs(log_path, exist_ok=True)
    files = list(sorted([f for f in os.listdir(log_path) if "version" in f]))
    if len(files) < 1:
        version = '1'.rjust(width, '0')
    else:
        last_version = int(files[-1][-width:])
        version = str(last_version + 1).rjust(width, '0')
    return version
