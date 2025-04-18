from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

class VerticalPushButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super(VerticalPushButton, self).__init__(*args, **kwargs)

        self.painter = QStylePainter()

    def sizeHint(self) -> QSize:
        size = super(VerticalPushButton, self).sizeHint()
        print(size, end='->')
        size.transpose()
        print(size)
        return size


    def paintEvent(self, event: QPaintEvent) -> None:
        # self.painter.begin(self)
        painter = QStylePainter(self)
        painter.rotate(90)
        painter.translate(0, self.width())

        opt = QStyleOption()
        opt.initFrom(self)
        opt.rect = opt.rect.transposed()
        painter.drawControl(QStyle.CE_PushButton, opt)
        # self.painter.end()
        painter.end()

if __name__ == "__main__":
    app = QApplication()
    w = VerticalPushButton("test1234")
    w.show()
    app.exec()