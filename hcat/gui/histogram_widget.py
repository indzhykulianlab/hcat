from typing import List, Tuple

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal, QRectF, QLineF, QSize, QSysInfo
from PySide6.QtGui import *
from PySide6.QtWidgets import *

"""
TODO:

Add render hints allowing user to change color and transparency of EVERYTHING
Add render hints allowing selective rendering of certain objects
Add ability for these render hints to be controlled by additional applicaiton widget


"""

CYAN = QColor(0, 255, 255)
MAGENTA = QColor(255, 0, 255)
GREY = QColor(200, 200, 200)

BLACK = "rgb(0,0,0)"
RED = "rgb(228, 26, 27)"
red = QColor(228, 26, 27)
BLUE = "rgb(55, 126, 184)"
blue = QColor(55, 126, 184)
ORANGE = "rgb(246, 122, 0)"
GREEN = "rgb(78, 175, 74)"
green = QColor(78, 175, 74)

class HistogramWidget(QWidget):
    """
    Canvas! This draws everything
    Every time the state is changed, we emit a signal to main
    """
    oneColorVisible = Signal(int)

    def __init__(self, values: List[np.array], positions: List[np.array]):
        super(HistogramWidget, self).__init__()

        palette = QPalette()
        palette.setColor(QPalette.Window, 'white')
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.values: List[np.array] = values
        self.log_values = []
        self.positions: List[np.array] = positions
        self._histogram_pixmap = None

        # Where we draw the image
        self.label = QLabel()
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # Constants about where we are looking...
        self.center = QPointF(0.0, 0.0)
        self.painter = QPainter()
        self.image = None

        self.hist_pixmap = None

        # log scale the hist
        self.log_scale = False

        self.checkbox = [QCheckBox() for _ in range(3)]

        self.alpha = 200

        layout = QHBoxLayout()
        # layout.addWidget(self.red_checkbox, alignment=Qt.AlignTop | Qt.AlignLeft)
        # layout.addWidget(self.green_checkbox, alignment=Qt.AlignTop | Qt.AlignLeft)
        # layout.addWidget(self.blue_checkbox, alignment=Qt.AlignTop | Qt.AlignLeft)
        for w in self.checkbox:
            layout.addWidget(w, alignment=Qt.AlignTop | Qt.AlignLeft)

        layout.addStretch(10000)
        layout.setContentsMargins(6,6,0,0)
        layout.setSpacing(0)
        self.setLayout(layout)

        BLACK = "rgb(0,0,0)"
        RED = "rgb(228, 26, 27)"
        BLUE = "rgb(55, 126, 184)"
        ORANGE = "rgb(246, 122, 0)"
        GREEN = "rgb(78, 175, 74)"

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
        for w, c in zip(self.checkbox, [RED, GREEN, BLUE]):
            w.setStyleSheet(style(c))
            w.click() # set checkbox to on!

        # adjust_zone

        self.left = [0,0,0]
        self.right = [256,256,256]

        for w in self.checkbox:
            w.clicked.connect(self.update)
            w.clicked.connect(self.redrawPixmap)
            w.clicked.connect(self.handle_one_color_chosen)

    def handle_one_color_chosen(self):
        # for c, val, pos, checkbox in zip(colors, _values, self.positions, self.checkbox):
        #     if checkbox.checkState() != Qt.CheckState.Checked:
        #         continue

        checked = [c.checkState() == Qt.CheckState.Checked for c in self.checkbox]
        if sum(checked) == 1:
            self.oneColorVisible.emit([i for i, v in enumerate(checked) if v][0])

    def resetColorSelectors(self):
        for c in self.checkbox:
            c.setChecked(True)


    def set_log_scale(self,val):
        self.log_scale = val
        self.hist_pixmap=[]
        self.update()


    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        return QSize(250, 80)

    def _calculate_center_ratio(self):
        """ sets _current_center_ratio """
        if self.pixmap is not None:
            screen_size: QPointF = self._get_viewport_size()
            image_size: QPointF = self._get_pixmap_size()
            center = screen_size / 2

            center_ratio_x: float = (center.x() - self._pixmap_offset.x()) / (image_size.x())
            center_ratio_y: float = (center.y() - self._pixmap_offset.y()) / (image_size.y())

            self._current_center_ratio: Tuple[float, float] = (center_ratio_x, center_ratio_y)

    def _get_viewport_size(self):
        return QPointF(self.width() / self.scale, self.height() / self.scale)

    def set_min_max(self, l: int, r: int, c=None):
        if c is None:
            self.left = [l,l,l]
            self.right = [r,r,r]
        else:
            # assert c < 4, f'{c=} !< 4'
            self.left[c] = l
            self.right[c] = r

        self.update()
        self.repaint()

    def set_hist_to_none(self):
        self.values = None
        self.log_values = None
        self.redrawPixmap()
        self.update()

    def set_hist_vals_positions(self, vals:List[np.array], pos:List[np.array]):
        self.values=vals
        self.log_values = []
        for v in self.values:
            if v is not None:
                self.log_values.append(np.log10(v + 1))
            else:
                self.log_values.append(v)

        self.positions=pos
        self.redrawPixmap()
        self.update()
        self.repaint()

    def redrawPixmap(self):
        self.hist_pixmap = []

    def paintEvent(self, event):
        """ main render loop. Everything drawn on the screen goes here """
        self.painter.begin(self)

        if self.values and self.positions:
            self._paint_histogram()  # draws the image

        self._paint_adjustment_zone()
        self._paint_warning()
        self._paint_axes()

        # self._paint_center_cross_overlay()  # draws a green dot at the center and a pair of crosses
        self.painter.end()

    def resizeEvent(self, event):
        self.hist_pixmap = []

    def _paint_histogram(self):
        p = self.painter
        if not self.hist_pixmap:
            pixmap = QPixmap(self.width(), self.height())
            pixmap.fill(QColor(255,255,255))
            pixmap_painter = QPainter(pixmap)

            colors = [red, green, blue]  # r, g, b
            _values = self.log_values if self.log_scale else self.values
            pen = QPen()
            pen.setWidthF(0)
            pen.setColor(QColor(0, 0, 0, 0))
            pixmap_painter.setPen(pen)
            for c, val, pos, checkbox in zip(colors, _values, self.positions, self.checkbox):
                if checkbox.checkState() != Qt.CheckState.Checked:
                    continue
                c.setAlpha(self.alpha)

                _max_v = max(val[1::]) + 1e-10

                w = self.width() / len(val)
                h = self.height()

                rects = [QRectF(i * w, (1 - v / _max_v) * h + (h * 0.02), w, v / _max_v * h) for i, (p, v) in
                         enumerate(zip(pos, val))]

                path = QPainterPath()
                for r in rects:
                    path.addRect(r)
                pixmap_painter.fillPath(path, c)
                pixmap_painter.drawPath(path)

            self.hist_pixmap = pixmap

        p.drawPixmap(QPointF(0., 0.), self.hist_pixmap)

    def _paint_adjustment_zone(self):
        p = self.painter
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen()
        pen.setWidthF(1)
        pen.setColor(QColor(0, 0, 0))
        p.setPen(pen)

        w, h = self.width(), self.height()
        colors = [red, green, blue]

        for c, l, r, checkbox in zip(reversed(colors), reversed(self.left), reversed(self.right), reversed(self.checkbox)):
            if checkbox.checkState() != Qt.CheckState.Checked:
                continue
            c.setAlpha(255)
            pen.setColor(c)
            p.setPen(pen)

            right_line = QLineF(r / 255 * w, h, r / 255 * w, h * 0.8)
            slope = QLineF(l / 255 * w, h, r / 255 * w, 0)

            p.drawLine(right_line)
            p.drawLine(slope)

    def _paint_axes(self):
        p = self.painter
        pen = QPen()
        pen.setWidthF(2)
        pen.setColor(QColor(0, 0, 0))
        p.setPen(pen)

        outline_rect = QRectF(0, 0, self.width(), self.height())
        p.drawRect(outline_rect)

    def _paint_warning(self):
        p = self.painter
        if all([w.checkState() == Qt.CheckState.Unchecked for w in self.checkbox]):

            primary_color =  QColor(255, 0, 0, 160)
            secondary_color = QColor(255,255,255,160)
            primary_pen = QPen()
            primary_pen.setColor(primary_color)
            primary_pen.setWidth(15)

            secondary_pen = QPen()
            secondary_pen.setColor(secondary_color)
            secondary_pen.setWidth(15)

            _x = -500
            for i in range(100):
                p.setPen(primary_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))
                _x += 20

                p.setPen(secondary_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))

                _x += 20



if __name__ == '__main__':
    import torch
    import numpy as np

    a = torch.randint(0, 255, (100, 100, 3)).to(torch.uint8).numpy()
    vals, pos = [], []
    for i in range(3):
        _vals, _pos = np.histogram(a[..., i], bins=100)
        vals.append(_vals)
        pos.append(_pos)

    app = QApplication()
    w = HistogramWidget(vals, pos)
    w.show()
    app.exec()
