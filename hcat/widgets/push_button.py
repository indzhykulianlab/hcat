from PySide6.QtCore import *
from PySide6.QtGui import QPen, QColor, QPainter, QPainterPath
from PySide6.QtWidgets import *


class WPushButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super(WPushButton, self).__init__(*args, **kwargs)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.warning_button = False
        self.danger_button = False

        self.popup_message = 'Warning! This action cannot be undone.'
        self.warn_on = 'danger'

        self._default_background_color = QColor(255, 255, 255)
        self.background_color = QColor(255, 255, 255)

        self.border_width = '1px'
        self.border_color = 'black'

        self._set_style()

    def _set_style(self):
        alpha = 0 if (self.warning_button or self.danger_button) else 255

        r, g, b = self.background_color.red(), self.background_color.green(), self.background_color.blue()
        font_color = 'black' if self.is_black_text_visible(r, g, b) else 'white'
        color_str = f'rgba({r}, {g}, {b}, {alpha})'
        pressed_str = f'rgba(200,200,200,{alpha}'


        self.setStyleSheet(f"""
                        WPushButton {{
                            background-color: {color_str};
                            border-style: outset;
                            border-width: {self.border_width};
                            border-color: {self.border_color};
                            font: bold;
                            color: {font_color if self.isEnabled() else 'rgba(0,0,0,100)'};
                            margin: 0px;
                            }}
                        WPushButton:pressed {{
                            background-color: rgba(200,200,200,{alpha});
                            border-style: outset;
                            border-width: 2px;
                            border-color: black;
                            margin: 0px;
                            padding: 2px;
                            }}
                        """)
        self.update()

    def setDisabled(self, arg__1: bool) -> None:
        self._set_style()
        super(WPushButton, self).setDisabled(arg__1)

    @staticmethod
    def is_black_text_visible(r: int, g: int, b: int):
        # Calculate the relative luminance of the color using the formula from the WCAG 2.0
        # accessibility guidelines
        return (r*0.299 + g*0.587 + b*0.114) > 186

    def getBackgroundColor(self):
        return self.background_color

    def resetBackgroundColor(self):
        self.background_color = self._default_background_color
        self._set_style()

    def paintEvent(self, event):
        self._paint_warning()
        self._paint_danger()
        super(WPushButton, self).paintEvent(event)

    def _paint_warning(self):
        p = QPainter(self)
        if self.warning_button:
            yellow_pen = QPen()
            yellow_pen.setColor(QColor(240, 200, 43, 255))
            yellow_pen.setWidth(15)

            black_pen = QPen()
            black_pen.setColor(QColor(0, 0, 0, 255))
            black_pen.setWidth(15)

            _x = -500
            for i in range(100):
                p.setPen(yellow_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))
                _x += 20

                p.setPen(black_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))

                _x += 20

            white_pen = QPen()
            white_pen.setColor(QColor(255,255,255,255))
            white_pen.setWidth(0)
            p.setPen(white_pen)
            w, h = self.width(), self.height()
            rect = QRectF(3, 3, w-6, h-6)
            path = QPainterPath()
            path.addRect(rect)
            p.fillPath(path, QColor(255,255,255))
            p.drawPath(path)

    def _paint_danger(self):
        p = QPainter(self)
        if self.danger_button:
            yellow_pen = QPen()
            yellow_pen.setColor(QColor(255, 0, 0, 255))
            yellow_pen.setWidth(15)

            black_pen = QPen()
            black_pen.setColor(QColor(255, 255, 255, 255))
            black_pen.setWidth(15)

            _x = -500
            for i in range(100):
                p.setPen(yellow_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))
                _x += 20

                p.setPen(black_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))

                _x += 20

            white_pen = QPen()
            white_pen.setColor(QColor(255,255,255,255))
            white_pen.setWidth(0)
            p.setPen(white_pen)
            w, h = self.width(), self.height()
            rect = QRectF(3, 3, w-6, h-6)
            path = QPainterPath()
            path.addRect(rect)
            p.fillPath(path, QColor(255,255,255))
            p.drawPath(path)

    def setDangerButton(self, val: bool):
        self.warning_button = False
        self.danger_button = val
        self._set_style()
        return self

    def setWarningButton(self, val: bool):
        self.danger_button = False
        self.warning_button = val
        self._set_style()
        return self

    def setBackgroundColor(self, color: QColor):
        self.background_color = color
        self._set_style()

    def setBorder(self, width, color):
        self.border_color = color
        self.border_width = width
        self._set_style()


class Test(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.layout = QGridLayout(self.mainWidget)

        self.button = WPushButton("TEST")
        self.button.setDangerButton(True)
        self.layout.addWidget(self.button)


if __name__ == '__main__':
    app = QApplication()
    w = Test()
    w.show()
    app.exec()
