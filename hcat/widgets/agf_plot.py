from __future__ import annotations
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from typing import *
from hcat.lib.types import *
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
import numpy as np


class AGFplot(QWidget):
    def __init__(self, parent: QWidget | None):
        super(AGFplot, self).__init__()

        self.experiment: ABRExperiment | None = None
        self.peaks: ABRPeaks | None = None
        self.notches: ABRNotches | None = None
        self.amplitudes: ABRAmplitudes | None = None
        self.parent = parent


        self.checks = [QCheckBox('1'), QCheckBox('2'), QCheckBox('3'), QCheckBox('4'), QCheckBox('5')]

        self.xmin = 0
        self.xmax = 100
        self.ymin = 0
        self.ymax = 100

        self.painter = QPainter()

        self.colors: Tuple[str, str, str, str, str] = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        )
        self.plot_margin = (20,10,20,50) # left top right bottom

        self.setMinimumSize(QSize(20,20))

        self.prepare_legend()
        self.create_layout()

    def minimumSizeHint(self) -> QSize:
        return QSize(425, 175)

    def create_layout(self):
        check_layout = QHBoxLayout()
        for c in self.checks:
            check_layout.addWidget(c)
        check_layout.addStretch(1)
        check_layout.setContentsMargins(self.plot_margin[0]+10, self.plot_margin[1] + 8, 5, 5)

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addLayout(check_layout)
        layout.addStretch(1)
        self.setLayout(layout)


    def prepare_legend(self):
        style = lambda color: f"""
                QCheckBox{{
                    font: 12px; 
                    margin: 0px {8 if QSysInfo.productType() == 'macos' else -6}px 0px 0px;
                }}
                QCheckBox::indicator{{
                    width: 10px;
                    height: 10px;
                    background-color: rgba(180,180,180,255); 
                    border: 1px solid black;
                    border-radius: 0px;
                    margin: 0 0 0 0;
                }}
                QCheckBox::indicator::checked{{
                    width: 10px;
                    height: 10px;
                    background-color: {color}; 
                    border: 1px solid black;
                    border-radius: 0px;
                    margin: 0 0 0 0;
                    }}
                """

        for color, check in zip(self.colors, self.checks):
            check.setObjectName('LegendCheckBox')
            check.setContentsMargins(0,0,0,0)
            check.setChecked(True)
            check.setStyleSheet(style(color))
            check.clicked.connect(self.update)

    def update(self):
        if self.parent:
            self.set_experiment(self.parent.get_experiment())
        super(AGFplot, self).update()

    def set_colors(self, colors: Tuple[str, str, str, str, str]) -> AGFplot:
        self.colors = colors
        return self

    def reset_colors(self) -> AGFplot:
        self.colors: Tuple[str, str, str, str, str] = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        )
        return self

    def set_experiment(self, experiment: ABRExperiment) -> AGFplot:
        self.experiment = experiment
        self.set_peaks(self.experiment.get_peaks())
        self.set_notches(self.experiment.get_notches())
        self._calculate_amplitude()
        self._calculate_xrange()
        self._calculate_yrange()
        return self

    def clear_experiment(self) -> AGFplot:
        self.experiment = None
        return self

    def set_peaks(self, peaks: ABRPeaks) -> AGFplot:
        self.peaks: ABRPeaks = peaks
        return self

    def clear_peaks(self) -> AGFplot:
        self.peaks = None
        return self

    def set_notches(self, notches: ABRNotches) -> AGFplot:
        self.notches: ABRNotches = notches
        return self

    def clear_notches(self) -> AGFplot:
        self.notches = None
        return self

    def _calculate_amplitude(self) -> AGFplot:
        if self.peaks is None or self.notches is None:
            self.amplitudes = None
            return self

        for k0, k1 in zip(self.peaks, self.notches):
            assert k0 == k1, "Key Mismatch in peaks and notches"

        self.amplitudes: ABRAmplitudes = {}
        for peak_level, notch_level in zip(self.peaks, self.notches):
            waveform_peaks: WaveformPeaks = self.peaks[peak_level]
            waveform_notches: WaveformNotches = self.notches[notch_level]

            waveform_amplitudes: WaveformAmplitudes = {}
            for wave_number in waveform_peaks.keys():
                peak: Point = waveform_peaks[wave_number]
                notch: Point = waveform_notches[wave_number]

                waveform_amplitudes[wave_number] = abs(peak['y'] - notch['y'])

            self.amplitudes[peak_level] = waveform_amplitudes

        self._calculate_xrange()
        self._calculate_yrange()

        return self

    def _calculate_xrange(self):
        if self.amplitudes is None:
            return
        keys = self.amplitudes.keys()
        self.xmin = min(keys) - 5
        self.xmax = max(keys) + 5

        # self.xmin = 5 * round(self.xmin/5)
        # self.xmax = 5 * round(self.xmax/5)

    def _calculate_yrange(self):
        if self.amplitudes is None:
            return
        self.ymin = float('inf')
        self.ymax = float('-inf')
        for level, waveform_amplitudes in self.amplitudes.items():
            for amp in waveform_amplitudes.values():
                self.ymin = min(self.ymin, amp)
                self.ymax = max(self.ymax, amp)

        range = abs(self.ymax - self.ymin)
        self.ymin = self.ymin - range*0.05
        self.ymax = self.ymax + range * 0.05

    def transpose(self, x: float, y: float) -> QPointF:
        """ transpose from relative to absolute position """

        left, top, right, bottom = self.plot_margin

        x_total_dist = self.xmax - self.xmin
        y_total_dist = self.ymax - self.ymin


        percent_x = (x_total_dist - (self.xmax - x)) / x_total_dist
        percent_y = (y_total_dist - (y - self.ymin)) / y_total_dist

        width = self.width() - (left + right)
        height = self.height() - (top + bottom)

        x = width * percent_x + left
        y = height * percent_y + top
        return QPointF(x, y)

    def paintEvent(self, event):
        self.painter.begin(self)
        self.painter.setRenderHint(QPainter.Antialiasing, True)
        self.painter.setRenderHint(QPainter.TextAntialiasing, True)
        self.painter.setRenderHint(QPainter.VerticalSubpixelPositioning, True)

        self._paint_background()
        self._paint_axes()
        self._paint_agf()
        self._paint_ticks()

        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, self.painter, self)

        self.painter.end()

    def _paint_background(self):
        left, top, right, bottom = self.plot_margin
        width, height = self.width(), self.height()
        pen = QPen()
        pen.setWidthF(0)
        pen.setColor('white')
        self.painter.setPen(pen)

        self.painter.fillRect(
            left, top,
            width-(right + left), height-(top+bottom), 'white'
        )




    def _paint_axes(self):
        pen = QPen()
        pen.setColor('black')
        pen.setWidth(2)
        self.painter.setPen(pen)

        # All Constants
        left, top, right, bottom = self.plot_margin
        width, height = self.width(), self.height()

        # Axes without ticks
        y_axes = QLineF(QPointF(left, top), QPointF(left, height - bottom))
        x_axes = QLineF(
            QPointF(left, height - bottom), QPointF(width - right, height - bottom)
        )
        self.painter.drawLine(y_axes)
        self.painter.drawLine(x_axes)


        # X Ticks
        N_xticks = len(self.amplitudes) if self.amplitudes else 8
        for x in np.linspace(left, width - right, N_xticks):
            line = QLineF(
                QPointF(x, height - bottom),
                QPointF(x, height - bottom + 3)
            )
            self.painter.drawLine(line)

        # Y Ticks
        N_xticks = 5
        for y in np.linspace(top, height - bottom, N_xticks):
            line = QLineF(
                QPointF(left, y),
                QPointF(left - 3, y)
            )
            self.painter.drawLine(line)

    def _paint_ticks(self):
        left, top, right, bottom = self.plot_margin
        width, height = self.width(), self.height()
        pen = QPen()
        pen.setColor('black')
        self.painter.setPen(pen)

        # X Ticks
        N_xticks = len(self.amplitudes) if self.amplitudes else 8
        for x, label in zip(np.linspace(left, width - right, N_xticks),
            np.linspace(self.xmin, self.xmax, N_xticks)
                            ):
            rect = QRectF(x - 20, height - bottom + 4, 40, 20)
            self.painter.drawText(rect, Qt.AlignCenter, f'{label:0.0f}')

        # # Y Ticks
        # N_xticks = 5
        # for y in np.linspace(top, height - bottom, N_xticks):
        #     self.painter.drawText(QRectF(left - 70, y-20, 65, 40), Qt.AlignVCenter | Qt.AlignRight, 'test')
        #

        self.painter.drawText(QRectF(left, height - bottom + 22, width - (left + right), 20), Qt.AlignCenter | Qt.AlignVCenter, 'Level (dbSPL)')


    def _paint_agf(self):
        if self.amplitudes is None:
            return

        pen = QPen()
        pen.setWidthF(2)
        self.painter.setPen(pen)

        array = [[],[],[],[],[]]  # Wave 1, 2, 3, 4, 5

        for level, waveform_amplitudes in self.amplitudes.items():
            for n, peak in waveform_amplitudes.items():
                point: QPointF = self.transpose(level, peak)
                array[n-1].append(point)

        for color, points, checkbox in zip(self.colors, array, self.checks):
            if not checkbox.isChecked():
                continue

            pen.setColor(color)
            pen.setWidthF(2)
            self.painter.setPen(pen)

            path = QPainterPath()
            path.moveTo(points[0])
            for p in points[1::]:
                path.lineTo(p)

            pen.setCapStyle(Qt.RoundCap)
            pen.setWidthF(5)
            self.painter.drawPath(path)

            self.painter.setPen(pen)
            self.painter.drawPoints(points)



if __name__ == "__main__":
    from hcat.cabr.abr_store import ABRStore

    store = ABRStore()
    key = store.keys()
    for key in key:
        break
    experiment: ExperimentDict = store[key]

    peaks: ABRPeaks = experiment['peaks']
    notches: ABRNotches = experiment['notches']


    app = QApplication()
    w = AGFplot(None)
    w.set_peaks(peaks)
    w.set_notches(notches)
    w._calculate_amplitude()
    w.show()
    app.exec()


