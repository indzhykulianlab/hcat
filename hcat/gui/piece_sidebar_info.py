from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.state import Cochlea, Piece, Synapse, Cell
from hcat.widgets.push_button import WPushButton


class TonotopyGraphic(QWidget):
    def __init__(self, *args, **kwargs):
        super(TonotopyGraphic, self).__init__(*args, **kwargs)
        self.painter = QPainter()

    def paintEvent(self, event) -> None:
        self.painter.begin(self)
        p = self.painter
        w, h = self.width(), self.height()
        vcenter = h/2

        blue_pen = QPen()
        blue_pen.setColor(QColor(0,0,255))
        blue_pen.setWidth(10)
        blue_pen.setCapStyle(Qt.RoundCap)

        red_pen = QPen()
        red_pen.setColor(QColor(255,0,0))
        red_pen.setWidth(10)
        red_pen.setCapStyle(Qt.RoundCap)

        line_pen = QPen()
        line_pen.setWidth(2)

        p.setPen(line_pen)
        p.drawLine(QLineF(QPointF(0.1*w, vcenter), QPointF(0.9*w, vcenter)))

        p.setPen(blue_pen)
        p.drawPoint(QPointF(0.1*w, vcenter))

        p.setPen(red_pen)
        p.drawPoint(QPointF(0.9*w, vcenter))

        self.painter.end()



class PieceSidebarInfoWidget(QWidget):
    """
    Piece info needs showing

    basal_freq
    apical_freq
    ID:
    #IHC
    #OHC
    #SYN
    IMG.shape ->
    IMG memory usage
    IMG device: CPU / CUDA / MPS
    Eval Region: SET/NOT SET

    staining???
    """
    pieceChanged = Signal()
    def __init__(self, state: Cochlea):
        super(PieceSidebarInfoWidget, self).__init__()
        self.state: Cochlea = state

        #
        # self.painter = QPainter()
        #
        # self.color_text = lambda text, color: f'<font color={color}>{text}</font>'
        #
        # self.SET = self.color_text('SET', '"Green"')
        # self.NOT_SET = self.color_text('NOT SET', '"RED"')#f'<font color="Red">[NOT SET]</font>'
        # self.CPU= self.color_text('CPU', '"SlateGrey"')
        # self.CUDA = self.color_text('CUDA', '#76b900')
        #
        #
        # self.textbox = QTextEdit()
        # self.textbox.setReadOnly(True)
        # self.textbox.setAlignment(Qt.AlignCenter)
        #
        #
        # layout = QHBoxLayout()
        # layout.addWidget(self.textbox)
        # layout.setContentsMargins(0,0,0,0)
        #
        #
        # policy = self.sizePolicy()
        # policy.setVerticalPolicy(QSizePolicy.Fixed)
        # self.setSizePolicy(policy)


        # self.setStyleSheet("border-width: 1px; border-color: black; border-style: inset;")

        self.piece_name_label = QLabel('NONE')
        self.img_memory_label = QLabel('0.00B')
        self.device_button = WPushButton('NONE')

        self.ohc_label = QLabel('OHC: 0000')
        self.ihc_label = QLabel('IHC: 0000')
        self.syn_label = QLabel('SYN: 0000')

        # borders
        self.ohc_label.setStyleSheet(f'border: 0.5px solid gray; margin-top: -1px')
        self.ihc_label.setStyleSheet(f'border: 0.5px solid gray; margin-top: -1px')
        self.syn_label.setStyleSheet(f'border: 0.5px solid gray; margin-top: -1px; margin-bottom: -1px')

        self.basal_freq_label = QLabel('NOTSET')
        self.apical_freq_label = QLabel('NOTSET')
        self.tonotopy_graphic = TonotopyGraphic()

        label_style = """
            QLabel {
                font: 10px;
            }
            """

        self.setStyleSheet(label_style)

        self._create_layout()

        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

        self.setFixedHeight(45)

    def setState(self, state: Cochlea):
        self.state = state
        self.update()

    @staticmethod
    def format_bytes(num, suffix="B"):
        for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def populate_from_piece(self):
        piece: Piece | None = self.state.get_selected_child()
        if isinstance(piece, Piece):
            ohc = 0
            ihc = 0
            syn = 0
            for c in piece.children.values():
                if isinstance(c, Cell) and c.type == 'IHC':
                    ihc += 1
                if isinstance(c, Cell) and c.type == 'OHC':
                    ohc += 1
                if isinstance(c, Synapse):
                    syn += 1

            nbytes = self.format_bytes(piece.image.nbytes)
            basal_freq = piece.basal_frequency
            apical_freq = piece.apical_frequency

            basal_freq = f'{basal_freq:02.2f}' if basal_freq else 'NOTSET'
            apical_freq = f'{apical_freq:02.2f}' if apical_freq else 'NOTSET'

            device: str = piece.device
            name = piece.tree_item.text(0)

            self.piece_name_label.setText(name)
            self.img_memory_label.setText(nbytes)
            self.device_button.setText(device.upper())
            self.device_button.setBackgroundColor(QColor(255,255,255))
            self.ohc_label.setText(f'OHC: {ohc:04}')
            self.ihc_label.setText(f'IHC: {ihc:04}')
            self.syn_label.setText(f'SYN: {syn:04}')
            self.basal_freq_label.setText(basal_freq)
            self.apical_freq_label.setText(apical_freq)

        else:
            self.piece_name_label.setText('NONE')
            self.img_memory_label.setText('0.00B')
            self.device_button.setText('NONE')
            self.device_button.setBackgroundColor(QColor(255,255,255))
            self.ohc_label.setText('OHC: 0000')
            self.ihc_label.setText('IHC: 0000')
            self.syn_label.setText('SYN: 0000')
            self.basal_freq_label.setText('NOTSET')
            self.apical_freq_label.setText('NOTSET')


    def update(self) -> None:
        self.populate_from_piece()
        super(PieceSidebarInfoWidget, self).update()

    def _create_layout(self):

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(1,1,1,1)
        top_layout.addWidget(self.piece_name_label)

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(self.img_memory_label)
        middle_layout.addWidget(QLabel('Device:'))
        middle_layout.addWidget(self.device_button)
        middle_layout.addStretch(1)
        middle_layout.setSpacing(5)
        middle_layout.setContentsMargins(1,1,1,1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.basal_freq_label, Qt.AlignLeft)
        bottom_layout.addWidget(self.tonotopy_graphic, Qt.AlignCenter)
        bottom_layout.addWidget(self.apical_freq_label, Qt.AlignRight)
        bottom_layout.setSpacing(0)
        bottom_layout.setContentsMargins(1,1,1,1)

        left_layout = QVBoxLayout()
        left_layout.addLayout(top_layout)
        left_layout.addLayout(middle_layout)
        left_layout.addLayout(bottom_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.ihc_label)
        right_layout.addWidget(self.ohc_label)
        right_layout.addWidget(self.syn_label)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0,0,0,0)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        layout.setSpacing(3)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)



if __name__=='__main__':
    import numpy as np

    c = Piece(id=0, filename='tity', image=np.random.randint(0, 255, (100, 100, 3)), parent=None)
    cochlea = Cochlea(0,None)
    cochlea.add_child(c)

    app = QApplication()
    w = PieceSidebarInfoWidget(cochlea)
    w.show()
    app.exec()
