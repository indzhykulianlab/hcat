from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import *
from scipy import signal

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from copy import copy

from hcat.lib.types import *


class ABRExperiment:
    def __init__(self, filepath: str, frequency: float, sample_rate: float, metadata: Dict[str, str] | None = None):
        self.waveforms: List[ABRWaveform] = []
        self.frequency = frequency
        self.filepath = filepath
        self.file_data: FileData | None = None
        self.sample_rate = sample_rate  # in us
        self.thr_prediction_method: str | None = None
        self._filter_params: FilterParams = {}
        self.metadata = metadata
        self.search_fields: Fields | None = None

        # sometimes a trace has no response anywhere, or a response everwhere...
        self.all_above_threshold = False
        self.all_below_threshold = False


    def set_file_data(self, file_data: FileData):
        self.file_data = file_data

    def get_file_data(self) -> FileData | None:
        return self.file_data

    def clear_file_data(self):
        self.file_data = None

    def set_filter_params(self, f0: float, f1: float, order: int):
        for w in self.get_waveforms():
            w.set_filter_params(f0, f1, order)

        self._filter_params: FilterParams = {"f0": f0, "f1": f1, "order": order}

    def get_filter_params(self) -> FilterParams:
        return self._filter_params

    def clear_filter_params(self):
        self._filter_params = {}
        for w in self.get_waveforms():
            w.clear_filter_params()

    def get_peaks(self, *, inverted: bool = False) -> ABRPeaks:
        """Dict[Level: Dict[N_wave, Dict["x": float, "y": float]]]"""
        guess: ABRPeaks = {}
        for w in self.get_waveforms():
            peaks: List[Tuple[float, float]] = w.get_peaks(inverted=inverted)
            guess[w.get_level()] = {
                i + 1: {"x": p[0], "y": p[1]} for i, p in enumerate(peaks)
            }
        return guess

    def get_notches(self, *, inverted: bool = False) -> ABRNotches:
        """Dict[Level: Dict[N_wave, Dict["x": float, "y": float]]]"""
        guess: ABRNotches = {}
        for w in self.get_waveforms():
            notches: List[Tuple[float, float]] = w.get_notches(inverted=inverted)
            guess[w.get_level()] = {
                i + 1: {"x": p[0], "y": p[1]} for i, p in enumerate(notches)
            }
        return guess

    def set_peaks(self, guesses: ABRPeaks):
        for w in self.get_waveforms():
            peaks = guesses[w.get_level()]
            w.set_peaks(peaks)
        return self

    def set_notches(self, guesses: ABRNotches):
        for w in self.get_waveforms():
            notches = guesses[w.get_level()]
            w.set_notches(notches)
        return self

    def clear_peaks(self):
        for w in self.get_waveforms():
            w.clear_peaks()

    def clear_notches(self):
        for w in self.get_waveforms():
            w.clear_notches()

    def set_all_above_thr(self):
        self.all_above_threshold = True
        self.all_below_threshold = False

        for w in self.get_waveforms():
            w.is_thr = False

    def set_all_below_thr(self):
        self.all_below_threshold = True
        self.all_above_threshold = False

        for w in self.get_waveforms():
            w.is_thr = False

    def set_threshold_prediction_method(self, method: str):
        self.thr_prediction_method = method

    def get_threshold_prediction_method(self):
        return self.thr_prediction_method

    def get_frequency(self):
        return self.frequency

    def get_filepath(self) -> str:
        return self.filepath

    def add_waveform(self, waveform: ABRWaveform):
        self.waveforms.append(waveform)
        self.waveforms.sort(key=lambda x: x.level, reverse=True)  # sort by level from

    def get_waveforms(self) -> List[ABRWaveform]:
        return self.waveforms

    def get_levels(self) -> List[int]:
        _levels = [w.get_level() for w in self.waveforms]
        return _levels

    def get_sample_rate(self):
        return self.sample_rate

    def get_threshold(self) -> int | str | None:
        if self.all_above_threshold:
            return "ALL_ABOVE"
        elif self.all_below_threshold:
            return "ALL_BELOW"

        for w in self.get_waveforms():
            if w.is_thr:
                return w.level

    def get_normalization_factor(self) -> float:
        _max = -float("inf")
        for w in self.get_waveforms():
            _max = max(_max, w.data.max())
        return _max

    def get_average_waveform_length(self) -> float | None:
        lengths = [len(w) for w in self.get_waveforms()]
        if lengths:
            lengths = np.array(lengths).mean()
            return np.mean(lengths)
        else:
            return None

    def __len__(self):
        return len(self.waveforms)

    def items(self):
        levels = self.get_levels()
        return zip(levels, self.waveforms)

    def plot(self):
        fig = plt.figure()

        for i, w in enumerate(self.get_waveforms()):
            _x = np.linspace(0, 1, len(w))
            data = w.get_data()
            plt.plot(_x, data + (1 * i))
        plt.yticks(
            ticks=[i for i, _ in enumerate(self.get_levels())], labels=self.get_levels()
        )
        plt.show()

    def clear_threshold(self):
        for w in self.waveforms:
            w.clear_threshold()


class ABRWaveform:
    def __init__(
        self,
        level: int,
        data: List[float] | np.ndarray,
        dt: float,
        parent: ABRExperiment,
    ):
        """
        ABR Waveform Dataclass

        :param level:
        :param data:
        :param dt:
        """
        self.level: int = level
        self.d_time: int = dt
        self.offset: float = 0.0
        self.x_scale: float = 1.0
        self.y_scale: float = 0.3
        self.thr_probability = None

        self._x = None  # set when drawing. Allows for drawing...
        self._y = None

        self.filter = None

        self.y_pos = None

        self.data: np.ndarray = (
            data if isinstance(data, np.ndarray) else np.ndarray(data)
        )
        self.parent: ABRExperiment = parent

        self.peaks: List[
            Tuple[float, float]
        ] = []  # [P1, P2, P3, ... ] (time, value (mV))
        self.notches: List[Tuple[float, float]] = []

        self.is_thr = False
        self.thr_set_manually = False

    def __len__(self):
        return len(self.data)

    def set_thr_probability(self, probability: float):
        self.thr_probability = probability

    def get_thr_probability(self):
        return self.thr_probability

    def get_time(self, as_array: bool = False) -> np.ndarray | List[float]:
        """gets the time in SECONDS"""

        fs = 1e6 / self.parent.get_sample_rate()
        t = np.arange(self.data.shape[-1]) / fs * 1e3

        # time = [i * 1e6 self.parent.get_sample_rate()for i in range(self.data.shape[0])]
        if as_array:
            return t
        else:
            raise RuntimeError

    def set_filter_params(self, f0: float, f1: float, order: int):
        if f0 >= f1:
            raise RuntimeError("F0 must be lower than F1")
        self.filter = {"lower": f0, "upper": f1, "order": order}

    def clear_filter_params(self):
        self.filter = None

    def set_peaks(self, peaks: WaveformPeaks | List[Tuple[float, float]]):
        """
        Dict keys are the wave number, then "x" or "y"

        :param peaks:
        :return:  None
        """
        if isinstance(peaks, dict):
            keys = sorted([k for k in peaks.keys()])
            self.peaks: List[Tuple[float, float]] = []
            for k in keys:
                self.peaks.append((peaks[k]["x"], peaks[k]["y"]))
        elif isinstance(peaks, list):
            self.peaks = peaks

    def get_peaks(self, *, inverted: bool = False) -> List[Tuple[float, float]]:
        peaks = [(x, y * (-1 if inverted else 1)) for (x, y) in copy(self.peaks)]
        return peaks

    def get_notches(self, *, inverted: bool = False) -> List[Tuple[float, float]]:
        return [(x, y * (-1 if inverted else 1)) for (x, y) in copy(self.notches)]

    def clear_peaks(self):
        self.peaks = []

    def clear_notches(self):
        self.notches = []

    def set_notches(self, notches: WaveformPeaks | List[Tuple[float, float]]):
        """
        Dict keys are the wave number, then "x" or "y"

        :param notches:
        :return:  None
        """
        if isinstance(notches, dict):
            keys = sorted([k for k in notches.keys()])
            self.notches = []
            for k in keys:
                self.notches.append((notches[k]["x"], notches[k]["y"]))
        elif isinstance(notches, list):
            self.notches = notches

    def set_as_threshold(self):
        self.is_thr = True

    def clear_threshold(self):
        self.is_thr = False

    def get_level(self) -> float:
        return self.level

    def get_dt(self) -> float:
        return self.d_time

    def get_scale(self) -> float:
        return self.scale

    def get_data(self, *, inverted: bool = False):
        if self.filter is not None:
            Wn = (self.filter["lower"], self.filter["upper"])
            b, a = signal.iirfilter(
                self.filter["order"],
                Wn,
                fs=1e6 / self.parent.get_sample_rate(),
                output="ba",
            )
            data = signal.filtfilt(b, a, self.data, axis=-1)

        else:
            data = self.data

        return -1 * data.copy() if inverted else data.copy()

    def adjust_offset(self, d_offset):
        self.offset += d_offset

    def get_parent(self):
        return self.parent

    def get_offset(self) -> float:
        return self.offset

    def reset_offset(self):
        self.offset = 0.0

    def set_x_scale(self, x_scale):
        self.x_scale = x_scale

    def reset_x_scale(self):
        self.x_scale = 1.0

    def set_y_scale(self, y_scale):
        self.y_scale = max(y_scale - 0.7, 0)  # default has to be 0.3

    def reset_y_scale(self):
        self.y_scale = 1.0

    def thr_set_by_user(self):
        self.thr_set_manually = True

    def __len__(self):
        return len(self.data)

    def waveform_y_pos(self):
        return self.y_pos

    def draw(
        self, p: QPainter, x: float, y: float, normalization_factor: float | None = None
    ):
        """
        Paints the waveform with a painter...

        :param p: QPainter of a QWidget
        :param x: starting x point of the waveform
        :param y: starting y point of the waveform
        :param normalization_factor: factor by which to scale the waveform
        :return: None
        """
        width = p.device().width()
        height = p.device().height()

        self._x = x
        self._y = y

        normalization_factor: float = (
            normalization_factor if normalization_factor else self.data.max()
        )

        pad = width * 0.05
        time = np.linspace(0, width, len(self)).tolist()
        mV = self.get_data(inverted=True).copy()
        mV = mV / normalization_factor
        mV = mV.tolist()

        self.y_pos = (mV[0] * self.y_scale) + y + self.offset

        path = QPainterPath()
        path.moveTo(
            QPointF(
                (time[0] * self.x_scale) + x + pad,
                (mV[0] * self.y_scale) + y + self.offset,
            )
        )

        data_to_show = [
            (_x, _y)
            for _x, _y in zip(time, mV)
            if _x * self.x_scale + x + pad < (width * 0.98)
        ]

        for _x, _y in data_to_show:
            path.lineTo(
                QPointF(
                    (_x * self.x_scale) + x + pad,
                    (_y * self.y_scale * height / len(self.parent)) + y + self.offset,
                )
            )

        if self.is_thr and not self.thr_set_manually:
            pen = QPen()
            pen.setColor(QColor(255, 0, 0))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidth(4)
            p.setPen(pen)
            p.drawPath(path)

        elif self.is_thr and self.thr_set_manually:
            pen = QPen()
            pen.setColor(QColor(0, 255, 0))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidth(4)
            p.setPen(pen)
            p.drawPath(path)

        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setCapStyle(Qt.RoundCap)
        # pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawPath(path)

        visible_y = [y for x, y in data_to_show]
        absolute_y = [(y * self.y_scale) + y + self.offset for x, y in data_to_show]
        # (_y * self.y_scale * height / len(self.parent))
        axes_width = width * 0.98
        x_text = min(axes_width, (time[-1] * self.x_scale) + x + pad)
        y_text = (
            (min(visible_y[-30:-1:1]) * self.y_scale * height / len(self.parent))
            + y
            + self.offset
            - 20
        )
        # print(visible_y[-10::], y_text, [(y * self.y_scale) + y + self.offset - 30 for x, y in data_to_show][-10::])

        p.drawText(
            QRectF(x_text - 85, y_text, 80, 20),
            f"{self.level} dB",
            Qt.AlignRight | Qt.AlignBottom,
        )
