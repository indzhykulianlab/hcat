import sys

from PySide6.QtWidgets import QDialog, QApplication, QLabel, QVBoxLayout


class WPopupDialog(QDialog):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle("Popup Dialog")
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setModal(True)

def show_popup(message):
    app = QApplication(sys.argv)
    popup = WPopupDialog(message)
    popup.show()
    app.processEvents()
    return popup

def my_function():
    popup = show_popup("This is a pop-up dialog")
    # Call your function here
    # ...
    popup.accept() # close the dialog