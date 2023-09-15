import math
from datetime import datetime


class ProgressBar:
    """
    Utility class for the management of progress bars showing training progress in the form
    "[<date>] Epoch <epoch_number>.<step_number> │<progress_bar>│ <completion_percentage>"
    """

    def __init__(self, max_step: int, max_epoch: int, current_epoch: int = 0, test: bool = False) -> None:
        self.max_step = max_step
        self.max_epoch = max_epoch
        self.curr_epoch = current_epoch
        self.step = 0
        self.mode = "Test" if test else "Train"
        self.len_bar = 20
        self.e = math.ceil(math.log10(self.max_epoch))
        self.s = math.ceil(math.log10(self.max_step + 1))

    def inc(self) -> None:
        """
        Increase the progress bar value by one unit
        """
        self.step += 1
        if self.step == self.max_step:
            self.step = 0
            self.curr_epoch += 1

    @property
    def progress(self) -> float:
        return (self.step + 1) / self.max_step

    @property
    def progress_bar(self) -> str:
        value = int(round(self.progress * self.len_bar))
        progress_bar = ('█' * value + ('┈' * (self.len_bar - value)))
        return f'│{progress_bar}│ {self.progress:.2%}'

    def __str__(self) -> str:
        date = datetime.now().strftime("[%b-%d@%H:%M]").lower()
        bar = f'\r{date} {self.mode:>5} {self.curr_epoch:0{self.e}d}.{self.step + 1:0{self.s}d} {self.progress_bar}'
        return bar
