from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from typing import *

from hcat.widgets.file_picker import WFilePickerWidget
from hcat.cabr.EPL_parser import parse_abr_file
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from hcat.lib.utils import qcolor_to_hex, hex_to_qcolor
from hcat.style.cabr_macos_style import MACOS_STYLE

import hcat.utils.colors


class PointWidget(QWidget):
    valueChanged = Signal()

    def __init__(
        self,
        waveform: ABRWaveform,
        point_number: int,
        parent=None,
        color: QColor | None = None,
        type: str = "peak",
    ):
        super(PointWidget, self).__init__(parent=parent)
        self.setFixedSize(15, 15)
        self.color: QColor | None = color
        self.point_number = point_number
        self.waveform = waveform

        if type == "peak":
            x, y = self.waveform.get_peaks()[self.point_number]
        else:
            x, y = self.waveform.get_notches()[self.point_number]
        self.label = QLabel(f"")
        self.y_offset = 0.0
        self.color = color

        self.painter = QPainter()

        if type not in ["peak", "notch"]:
            raise RuntimeError("Type must be peak or notch")

        self.type = type
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    def _waveform_index_from_global_position(self, global_x: float):
        global_x += self.width() / 2
        width = self.parent().width()
        pad = width * 0.05
        dx = width / (len(self.waveform) - 1) * self.waveform.x_scale
        index = round((global_x - pad) / dx)
        return min(len(self.waveform) - 1, max(0, index))

    def set_pos(self, x, y):
        self.move(int(x), int(y))
        self.startPos = self.pos()
        self.update()

    def move_to_init_position(self):
        # time, mV = self.waveform.get_peaks()[self.peak_number]
        if self.type == "peak":
            time, mV = self.waveform.get_peaks(inverted=True)[self.point_number]
        elif self.type == "notch":
            time, mV = self.waveform.get_notches(inverted=True)[self.point_number]

        width, height = self.parent().width(), self.parent().height()
        norm: float = self.waveform.parent.get_normalization_factor()

        fs = 1e6 / self.waveform.parent.get_sample_rate()
        x_index = round((time / 1e3) * fs)
        pad = width * 0.05  # x scale should be 2.02
        dx = width / (len(self.waveform) - 1) * self.waveform.x_scale
        x_pos = x_index * dx + pad - self.width() / 2

        new_y = (
            (
                mV
                / norm
                * self.waveform.y_scale
                * self.parent().height()
                / len(self.waveform.parent)
            )
            + self.waveform._y
            + self.waveform.offset
            - self.height() / 2
        )

        # print(index, self.waveform.get_data()[index], new)
        # if self.waveform is self.waveform.parent.get_waveforms()[0]:
        #     print(f'{self.point_number} -> ({x_pos:0.2f}, {new_y:0.2f})')
        self.move(x_pos, new_y)

    def snap_to_waveform(self):
        width = self.parent().width()
        pad = width * 0.05
        new_x = self.x()
        new_x = max(pad, min(width - pad, new_x))

        index = self._waveform_index_from_global_position(new_x)

        norm = self.waveform.parent.get_normalization_factor()

        data = self.waveform.get_data(inverted=True)
        time = self.waveform.get_time(as_array=True)

        new_y = (
            (
                data[index]
                / norm
                * self.waveform.y_scale
                * self.parent().height()
                / len(self.waveform.parent)
            )
            + self.waveform._y
            + self.waveform.offset
            - self.height() / 2
        )
        # print(index, self.waveform.get_data()[index], new)
        self.move(new_x, new_y)
        self.valueChanged.emit()

        if self.type == "peak":
            peaks = self.waveform.get_peaks()
            peaks[self.point_number]: List[Tuple[float, float]] = (
                time[index],
                -1 * data[index],
            )
            self.waveform.set_peaks(peaks)
        elif self.type == "notch":
            notches = self.waveform.get_notches()
            notches[self.point_number]: List[Tuple[float, float]] = (
                time[index],
                -1 * data[index],
            )
            self.waveform.set_notches(notches)
        else:
            raise RuntimeError("MUST BE EITHER PEAK OR NOTCH!")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.startPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.globalPosition().toPoint() - self.startPos
            width = self.parent().width()
            pad = width * 0.05

            # self.deltaPosition.emit(delta.y())

            new_x = self.x() + delta.x()
            new_x = max(pad, min(width - pad, new_x))

            index = self._waveform_index_from_global_position(new_x)

            norm = self.waveform.parent.get_normalization_factor()

            data = self.waveform.get_data(inverted=True)
            time = self.waveform.get_time(as_array=True)

            new_y = (
                (
                    data[index]
                    / norm
                    * self.waveform.y_scale
                    * self.parent().height()
                    / len(self.waveform.parent)
                )
                + self.waveform._y
                + self.waveform.offset
                - self.height() / 2
            )
            # print(index, self.waveform.get_data()[index], new)
            self.move(new_x, new_y)
            self.startPos = event.globalPosition().toPoint()
            self.valueChanged.emit()

            if self.type == "peak":
                peaks = self.waveform.get_peaks()
                peaks[self.point_number]: List[Tuple[float, float]] = (
                    time[index],
                    -1 * data[index],
                )
                self.waveform.set_peaks(peaks)
            elif self.type == "notch":
                notches = self.waveform.get_notches()
                notches[self.point_number]: List[Tuple[float, float]] = (
                    time[index],
                    -1 * data[index],
                )
                self.waveform.set_notches(notches)
            else:
                raise RuntimeError("MUST BE EITHER PEAK OR NOTCH!")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def paintEvent(self, event: QPaintEvent) -> None:
        p = self.painter
        p.begin(self)
        self.painter.setRenderHint(QPainter.Antialiasing, True)
        self.painter.setRenderHint(QPainter.TextAntialiasing, True)
        self.painter.setRenderHint(QPainter.VerticalSubpixelPositioning, True)

        x, y = self.width() / 2, self.height() / 2

        pad = 2
        width = self.width() - pad
        height = self.height() - pad

        if self.type == "peak":
            path = QPainterPath()
            path.moveTo(QPointF(pad, height))  # bottom left
            path.lineTo(QPointF(width, height))  # bottom right
            path.lineTo(QPointF(self.width() / 2, pad))  # Top middle
            path.closeSubpath()
        elif self.type == "notch":
            path = QPainterPath()
            path.moveTo(QPointF(pad, pad))
            path.lineTo(QPointF(width, pad))
            path.lineTo(QPointF(self.width() / 2, height))
            path.closeSubpath()
        else:
            raise RuntimeError

        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(2)
        p.setPen(pen)
        p.fillPath(path, self.color)
        p.drawPath(path)

        p.end()


class DraggableWidget(QWidget):
    deltaPosition = Signal(float)

    def __init__(self, waveform: ABRWaveform | None, parent=None):
        super(DraggableWidget, self).__init__(parent=parent)
        self.waveform = waveform
        self.setWindowTitle("Draggable Widget")
        self.setFixedSize(30, 80)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Set up a QLabel as the content of the widget
        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(10, 40)

        pallete = QPalette()
        pallete.setColor(QPalette.Window, "gray")
        self.label.setAutoFillBackground(True)
        self.label.setPalette(pallete)

        self.label.setStyleSheet(
            """
        QLabel {
            border-style: inset;
            border-width: 0.1em;
            border-color: Black;
            background-clip: border;
            }
        """
        )

        self.setToolTip("Click and drag")

        self.layout.addWidget(self.label)

        self.dragging = False
        self.startPos = QPoint()

    def set_pos(self, x, y):
        self.move(int(x), int(y))
        self.startPos = self.pos()
        self.update()

    def set_y_pos_from_waveform(self):
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.startPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.globalPosition().toPoint() - self.startPos
            if (self.y() + delta.y()) > 0 and self.y() + delta.y() < (
                self.parent().height() - 40
            ):
                self.deltaPosition.emit(delta.y())
                self.move(self.x(), self.y() + delta.y())
                self.startPos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def set_color(self, hex_color: str):
        pallete = QPalette()
        pallete.setColor(QPalette.Window, hex_color)
        self.label.setAutoFillBackground(True)
        self.label.setPalette(pallete)
        self.update()


class ABRViewerWidget(QWidget):
    pointChanged = Signal()

    def __init__(self):
        super(ABRViewerWidget, self).__init__()

        self.painter = QPainter(self)

        self.drag_widgets = []
        self._max_data_length = 250
        self.experiment: ABRExperiment | None = None
        self.is_blinded = True

        # self.setMinimumWidth(500)
        # self.setMinimumHeight(500)

        self.peak_widgets: List[PointWidget] = []

        # self.setStyleSheet("ABRViewerWidget { background-color: white; }")
        self.setStyleSheet(MACOS_STYLE)

    def sizeHint(self) -> QSize:
        return QSize(500, 500)

    def minimumSizeHint(self) -> QSize:
        return QSize(50, 50)

    def set_experiment(self, experiment: ABRExperiment):
        self.experiment = experiment
        self.reset_drag_widget_positions()
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.update_peak_widgets_on_resize()
        self.reset_drag_widget_positions()

    def set_y_scale(self, scale):
        if self.experiment is not None:
            for w in self.experiment.waveforms:
                w.reset_y_scale()
                w.set_y_scale(scale)

    def set_x_scale(self, scale):
        if self.experiment is not None:
            for w in self.experiment.waveforms:
                w.reset_x_scale()
                w.set_x_scale(scale)

    def update_peak_widgets_on_resize(self):
        for w in self.peak_widgets:
            w.move_to_init_position()

    def clear_peak_widgets(self):
        while self.peak_widgets:
            to_delete = self.peak_widgets.pop(-1)
            to_delete.deleteLater()
            to_delete.setVisible(False)
            del to_delete

        del self.peak_widgets
        self.peak_widgets = []

    def reset_peak_widgets(self):
        self.clear_peak_widgets()

        default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for w in self.experiment.get_waveforms():
            N = min(len(w.get_peaks()), len(w.get_notches()))
            for i in range(N):
                peak_widget = PointWidget(
                    waveform=w,
                    point_number=i,
                    parent=self,
                    color=hex_to_qcolor(default_colors[i]),
                    type="peak",
                )
                notch_widget = PointWidget(
                    waveform=w,
                    point_number=i,
                    parent=self,
                    color=hex_to_qcolor(default_colors[i]),
                    type="notch",
                )
                peak_widget.move_to_init_position()
                notch_widget.move_to_init_position()

                if peak_widget.x() > self.width() * 0.96:
                    peak_widget.hide()
                else:
                    peak_widget.show()

                if notch_widget.x() > self.width() * 0.96:
                    notch_widget.hide()
                else:
                    notch_widget.show()

                peak_widget.valueChanged.connect(self.emit_point_changed)
                notch_widget.valueChanged.connect(self.emit_point_changed)

                self.peak_widgets.append(peak_widget)
                self.peak_widgets.append(notch_widget)

        # This cose esentially snaps every widget to the waveform by clicking...
        # for w in self.peak_widgets:
        #     w.snap_to_waveform()
        # self.emit_point_changed()

    def emit_point_changed(self):
        self.pointChanged.emit()

    def color_drag_widgets(self):
        for w in self.drag_widgets:
            prob = w.waveform.get_thr_probability()
            if prob is not None:
                index = round(prob * 255)
                w.set_color(hcat.utils.colors.RED_GREEN[index])
            else:
                w.set_color("grey")
        self.update()

    def reset_drag_widget_positions(self):
        if not isinstance(self.experiment, ABRExperiment):
            return

        while self.drag_widgets:
            w: DraggableWidget = self.drag_widgets.pop(-1)
            w.deleteLater()
            w.setVisible(False)
            del w
        del self.drag_widgets
        self.drag_widgets = []

        for w, y_pos in zip(
            self.experiment.get_waveforms(), self.get_waveforms_y_positions()
        ):
            drag = DraggableWidget(w, parent=self)
            drag.deltaPosition.connect(w.adjust_offset)
            drag.deltaPosition.connect(self.update_peak_widgets_on_resize)
            drag.deltaPosition.connect(self.update)
            drag.set_pos(9, 3)
            drag.show()

            self.drag_widgets.append(drag)

        # for d in self.drag_widgets:
        #     d.set_y_pos_from_waveform()
        for drag, wave, y_pos in zip(
            self.drag_widgets,
            self.experiment.get_waveforms(),
            self.get_waveforms_y_positions(),
        ):
            wave.reset_offset()
            wave._y = y_pos  # fucked hack
            drag.set_pos(
                5,
                y_pos
                + wave.get_offset()
                + (wave.data.mean() * wave.y_scale)
                - drag.height() / 2,
            )

        self.color_drag_widgets()

    def get_waveforms_y_positions(self) -> List[float]:
        if not isinstance(self.experiment, ABRExperiment):
            return

        vertical_space = self.height() / len(self.experiment) * 0.80
        positions = []
        for i, _ in enumerate(self.experiment.get_waveforms()):
            positions.append(vertical_space * i + (vertical_space * 1.2))
        return positions

    def paintEvent(self, event: QPaintEvent) -> None:
        self.painter.begin(self)
        self.painter.setRenderHint(QPainter.Antialiasing, True)
        self.painter.setRenderHint(QPainter.TextAntialiasing, True)
        self.painter.setRenderHint(QPainter.VerticalSubpixelPositioning, True)

        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, self.painter, self)

        if isinstance(self.experiment, ABRExperiment):
            self._paint_waveforms()
            self._paint_axes()
            self._paint_freq()
            self._paint_response_thr_hint()

        self.painter.end()

    def _paint_waveforms(self):
        if not isinstance(self.experiment, ABRExperiment):
            return

        positions = self.get_waveforms_y_positions()
        normalization_factor: float = self.experiment.get_normalization_factor()
        for w, y_pos in zip(self.experiment.get_waveforms(), positions):
            w.draw(self.painter, 0, y_pos, normalization_factor)

    def _paint_axes(self):
        p = self.painter
        width, height = self.width(), self.height()

        # x_start = width * 0.05

        base_line = QLineF(
            QPointF(width * 0.05, height - 20), QPointF(width * 0.98, height - 20)
        )

        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setCapStyle(Qt.RoundCap)
        # pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidthF(1.5)
        p.setPen(pen)
        p.drawLine(base_line)

        # Draw the axes
        axes_width = (width * 0.98) - (width * 0.05)
        for percent in [i * 0.1 for i in range(11)]:
            tick_x = axes_width * percent + (width * 0.05)
            line = QLineF(QPointF(tick_x, height - 20), QPointF(tick_x, height - 15))
            p.drawLine(line)

        # Draw the total time...
        sample_rate = self.experiment.get_sample_rate()  # Samples per second...
        total_time = (
            self.experiment.get_average_waveform_length() * sample_rate
        )  # in uSec
        x_scale: float = self.experiment.waveforms[0].x_scale  # kinda fucked...
        total_time /= 1e3  # msec

        total_time /= x_scale

        p.drawText(
            QRectF(tick_x - 82, height - 40, 80, 20),
            f"{total_time:0.2f} mSec",
            Qt.AlignRight | Qt.AlignTop,
        )

        pen = QPen()
        pen.setColor(QColor(0, 0, 0, 50))
        pen.setCapStyle(Qt.RoundCap)
        # pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidthF(0.5)
        p.setPen(pen)
        p.drawLine(base_line)
        for percent in [i * 0.1 for i in range(11)]:
            tick_x = axes_width * percent + (width * 0.05)
            line = QLineF(QPointF(tick_x, height * 0.05), QPointF(tick_x, height - 20))
            p.drawLine(line)

    def _paint_freq(self):
        p = self.painter
        pen = QPen()
        alpha = 50 if self.is_blinded else 255
        pen.setColor(QColor(0, 0, 0, alpha))
        pen.setCapStyle(Qt.RoundCap)
        # pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidthF(0.5)
        p.setPen(pen)
        loc = 10
        text = (
            f"{self.experiment.get_frequency():3.2f} kHz"
            if not self.is_blinded
            else "Frequency Blinded"
        )
        p.drawText(QRectF(loc, loc, 300, 100), text, Qt.AlignTop | Qt.AlignLeft)

    def _paint_response_thr_hint(self):
        p = self.painter
        pen = QPen()
        alpha = 255
        pen.setColor(QColor(0, 0, 0, alpha))
        pen.setCapStyle(Qt.RoundCap)
        # pen.setJoinStyle(Qt.RoundJoin)
        pen.setWidthF(0.5)
        loc = self.width() - 10
        if self.experiment.all_above_threshold:
            text = "ALL ABOVE THRESHOLD"
            pen.setColor(QColor(28, 130, 16))
        elif self.experiment.all_below_threshold:
            text = "NO RESPONSE"
            pen.setColor(QColor(255, 0, 0))
        else:
            text = ""

        p.setPen(pen)
        p.drawText(
            QRectF(self.width() - 310, 10, 300, 100), text, Qt.AlignTop | Qt.AlignRight
        )


if __name__ == "__main__":
    app = QApplication()
    w = ABRViewerWidget()
    w.show()
    w.reset_drag_widget_positions()
    app.exec()
