from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class KeyEnterWidget(QWidget):
    keySubmitted = Signal(str)
    canceledSignal = Signal()

    def __init__(self):
        super(KeyEnterWidget, self).__init__()

        self.label = QLabel('KEY: ')
        self.input = QLineEdit()
        self.input.setFixedWidth(250)
        self.ok_button = QPushButton('OK')
        self.has_submitted = False

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.input)
        hlayout.addWidget(self.ok_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Please enter your product key'))
        layout.addLayout(hlayout)

        self.setLayout(layout)
        self.notReadyToSubmit()
        self.link_slots_and_signal()

    def link_slots_and_signal(self):
        self.input.textChanged.connect(self.checkTextInputState)
        self.ok_button.clicked.connect(self.submitKey)

    def checkTextInputState(self):
        text = self.input.text()
        if text is not None:
            self.readyToSubmit()
        else:
            self.notReadyToSubmit()

    def notReadyToSubmit(self):
        self.ok_button.setDisabled(True)
        self.ok_button.setEnabled(False)
        self.ok_button.blockSignals(True)

    def readyToSubmit(self):
        self.ok_button.setDisabled(False)
        self.ok_button.setEnabled(True)
        self.ok_button.blockSignals(False)

    def submitKey(self):
        key: str = self.input.text()
        self.keySubmitted.emit(key)
        self.has_submitted = True
        self.close()

    def closeEvent(self, event):
        if not self.has_submitted:
            self.canceledSignal.emit()
        super(KeyEnterWidget, self).closeEvent(event)

if __name__ == '__main__':
    app = QApplication()
    w = KeyEnterWidget()
    w.show()

    app.exec()


