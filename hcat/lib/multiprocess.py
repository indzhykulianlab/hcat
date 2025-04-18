import sys
import traceback
from typing import *

from PySide6.QtCore import *


# from hcat.backend.backend import init_model, eval_crop, model_from_path


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    begin = Signal(object)


class Worker(QRunnable):
    begin = Signal(object)

    def __init__(
        self, fn, begin_passthrough: Dict[str, object] | None = None, **kwargs
    ):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.kwargs = kwargs
        self.begin_passthrough: Dict[str, object] | None = begin_passthrough
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.signals.begin.emit(self.begin_passthrough)
            result = self.fn(**self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
