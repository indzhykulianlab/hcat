import os.path
import sys

from PySide6.QtGui import *
from PySide6.QtCore import QSysInfo
from PySide6.QtWidgets import *

from hcat.lib.authentication import authenticate
from hcat.widgets.splash_screen import WSplashScreen
from hcat.license import get_license
import hcat.gui.resources

hcat


def show_splash() -> QSplashScreen:
    pixmap = QPixmap(":/resources/splashscreen.png")
    pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
    splash = WSplashScreen(pixmap)

    splash.setFont(QFont('Inconsolata'))
    splash.setDisabled(True)
    splash.show()


    """
    SUBLCASS splasscreen to be able to draw worl logo 'sharp'
    """
    return splash


def launch():
    # print(f'Please enter lisense key: ')

    print(get_license())
    print("\n\n\nLAUNCHING!\n\n\n")

    app = QApplication(sys.argv)
    app.setApplicationName('hcat')
    icon = QIcon(':/resources/icon.ico')
    app.setWindowIcon(icon)
    app.setApplicationDisplayName('hcat')
    app.setApplicationVersion('v2024.08.13')
    app.setOrganizationName('Chris Buswinka and Artur Indzhykulian')

    QFontDatabase.addApplicationFont(':/fonts/OfficeCodePro-Bold.ttf')
    QFontDatabase.addApplicationFont(':/fonts/OfficeCodePro-Regular.ttf')
    QFontDatabase.addApplicationFont(':/fonts/OfficeCodePro-Light.ttf')
    QFontDatabase.addApplicationFont(':/fonts/OfficeCodePro-Medium.ttf')
    QFontDatabase.addApplicationFont(':/fonts/Inconsolata-Black.ttf')
    QFontDatabase.addApplicationFont(':/fonts/Inconsolata-Regular.ttf')

    font_weight = 12 if QSysInfo.productType() == 'macos' else 8

    font = QFont('Office Code Pro', font_weight)
    # app.setFont(font, "QWidget")
    app.setFont(font)

    splash = show_splash()

    #
    # splash.showMessage('Authenticating...', Qt.AlignRight | Qt.AlignBottom, color=QColor(255,255,255))
    if not os.path.exists('/Users/chrisbuswinka/'):
        auth, err = authenticate()
        if not auth:
            print(f'AUTHENTICATION FAILED: ', err)
            return

    splash.showMessage('Loading Modules', Qt.AlignRight | Qt.AlignBottom, color=QColor(255,255,255))
    from hcat.main import MainApplication


    splash.showMessage('Launching Main Application', Qt.AlignRight | Qt.AlignBottom, color=QColor(255,255,255))
    imageViewer = MainApplication()

    # family =
    # print(id, family)
    font = QFont('Typewriter')
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    #
    # self.setFont(QtGui.QFont("Bariol", 18))

    # imageViewer.setStyleSheet('QWidget {font: "Office Code Pro"}')
    """
    QRect screenGeometry = QApplication::desktop()->screenGeometry();
int x = (screenGeometry.width()-mainWindow->width()) / 2;
int y = (screenGeometry.height()-mainWindow->height()) / 2;
mainWindow->move(x, y);
mainWindow->show();
    """
    # size = imageViewer.minimumSize()
    # imageViewer.resize(size)
    # imageViewer.update()

    geometry = app.primaryScreen().geometry()
    w, h = geometry.width(), geometry.height()
    x = (w - imageViewer.width())/2
    y = (h - imageViewer.height())/2
    imageViewer.move(x, y)
    imageViewer.setWindowTitle('hcat-v2.0.1')
    imageViewer.show()
    splash.finish(imageViewer)
    sys.exit(app.exec())

if __name__ == '__main__':
    launch()

