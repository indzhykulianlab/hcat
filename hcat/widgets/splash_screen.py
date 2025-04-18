from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class WSplashScreen(QSplashScreen):
    def __init__(self, *args, **kwargs):
        super(WSplashScreen, self).__init__(*args, **kwargs)

    def drawContents(self, p: QPainter):
        super(WSplashScreen, self).drawContents(p)
        _font = p.font()
        _pen = p.pen()

        w = self.width()
        h = self.height()

        logo_font = QFont('Inconsolata')
        logo_font.setWeight(QFont.Black)
        logo_font.setPointSizeF(60)

        copyright_font = QFont('Inconsolata')
        copyright_font.setWeight(QFont.Thin)
        copyright_font.setPointSizeF(8)

        color = QColor(255,255,255)
        pen = QPen()
        pen.setColor(color)

        # logo_font.setWeight(900)
        p.setFont(logo_font)
        p.setPen(pen)
        p.drawText(QPointF(10, 53), 'hcat')

        p.setFont(copyright_font)
        p.drawText(QPointF(10, h-10), '© 2024 Christopher Buswinka')

        p.drawRect(QRectF(0, 0, w, h))

        p.setFont(_font)
        p.setPen(_pen)

        super(WSplashScreen, self).drawContents(p)

