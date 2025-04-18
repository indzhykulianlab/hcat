from typing import Dict

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class RenderHintWidget(QWidget):
    renderHintChanged = Signal(dict)

    def __init__(self):
        super(RenderHintWidget, self).__init__()

        # Make the checkboxes
        self.box_hint = self._click_and_link(QCheckBox('BOX'))
        self.diag_hint = QCheckBox('DIAGNOSTIC')
        self.diag_hint.clicked.connect(self.emit_render_hint_dict)

        self.eval_region_hint = self._click_and_link(QCheckBox('EVAL REGION'))
        self.freq_region_hint = self._click_and_link(QCheckBox('FREQ REGION'))
        self.synapse_hint = self._click_and_link(QCheckBox('SYNAPSE'))

        # Prep the layout
        layout = QGridLayout()
        layout.addWidget(self.box_hint, 0, 0)
        layout.addWidget(self.synapse_hint, 0, 2)
        layout.addWidget(self.eval_region_hint, 1, 0)
        layout.addWidget(self.freq_region_hint, 1, 2)
        layout.addWidget(self.diag_hint, 2, 0)

        layout.setContentsMargins(10, 10, 10, 10)
        # layout.setSpacing(5)

        # for rendering
        self._warning = False
        self._error = False
        self._alpha = 255
        self.setLayout(layout)

        self.setFixedHeight(50 if QSysInfo.productType() == 'macos' else 65)

        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

        self.setStyleSheet(self._get_checkbox_style('rgba(114, 254, 0, 255)'))
        self.emit_render_hint_dict()

    def _click_and_link(self, w: QCheckBox) -> QCheckBox:
        """ all checkboxes should be on by default """
        w.click()
        w.clicked.connect(self.emit_render_hint_dict)
        return w

    def _get_checkbox_style(self, color):
        margin = 3
        style = f"""
        QCheckBox{{
            font: 10px; 
        }}
        QCheckBox::indicator{{
            width: 8px;
            height: 8px;
            background-color: rgba(255,255,255,0); 
            border: 1px solid black;
            border-radius: 0px;
            margin: {margin}ex, {margin}ex, 0ex, 0ex;
        }}
        QCheckBox::indicator::checked{{
            width: 8px;
            height: 8px;
            background-color: {color}; 
            border: 1px solid black;
            border-radius: 0px;
            margin: {margin}ex, {margin}ex, 0ex, 0ex;
            }}
        """
        return style

    def _construct_render_hint_dict(self) -> Dict[str, bool]:
        """returns a dict of each checkbox and if it's checked"""
        widgets = [
            self.box_hint,
            self.diag_hint,
            self.eval_region_hint,
            self.freq_region_hint,
            self.synapse_hint
        ]
        keys = ['box', 'diagnostic', 'eval_region', 'freq_region', 'synapse']
        render_hint_dict = {}
        for k, w in zip(keys, widgets):
            render_hint_dict[k] = w.checkState() == Qt.CheckState.Checked

        values = list(render_hint_dict.values())

        _good = (values[0] or values[1]) and all(values[2::])
        # _warn = (values[0] or values[1]) and any(values[2::])
        _error = not (values[0] and values[1]) and not any(values[2::])

        if _good:
            self._warning = False
            self._error = False
        elif not _good and not _error:
            self._warning = True
            self._error = False
        elif _error:
            self._warning = False
            self._error = True

        return render_hint_dict

    def emit_render_hint_dict(self):
        self.renderHintChanged.emit(self._construct_render_hint_dict())
        self.update()

    def paintEvent(self, event):
        self._paint_warning()
        super(RenderHintWidget, self).paintEvent(event)

    def _paint_warning(self):
        p = QPainter(self)
        if self._warning or self._error:
            primary_color = QColor(247, 83, 20, self._alpha) if self._warning else QColor(255, 0, 0, 60)
            secondary_color = QColor(200, 200, 200, self._alpha) if self._warning else (QColor(255, 255, 255, 60))
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

            if not self._error:
                palette: QPalette = self.palette()
                color = palette.color(self.backgroundRole())
                # red, green, blue = color.red(), color.green(), color.blue()
                # color = QColor(int(red*255), int(green*255), int(blue*255), 255)
                white_pen = QPen()
                white_pen.setColor(color)
                white_pen.setWidth(0)
                p.setPen(white_pen)

                w, h = self.width(), self.height()
                rect = QRectF(3, 3, w - 6, h - 6)
                path = QPainterPath()
                path.addRect(rect)
                p.fillPath(path, color)
            # p.drawPath(path)


if __name__ == '__main__':
    app = QApplication()
    w = RenderHintWidget()
    w.show()
    app.exec()
