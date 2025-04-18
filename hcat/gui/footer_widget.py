from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.state import Piece, Synapse, Cell


class FooterWidget(QWidget):
    def __init__(self, piece: Piece = None):
        super(FooterWidget, self).__init__()

        self._piece = piece

        fixedFont = QFont('Office Code Pro')

        self.layout = QHBoxLayout()
        self.current_mode = QLabel('MOVE')
        self.current_mode.setFont(fixedFont)

        self.key_buffer = QLabel('')
        self.key_buffer.setFont(fixedFont)

        self._image_str = ''
        self._count_str = ''

        self.mouse_position = QLabel(f'POSITION: X/Y (0000, 0000)')
        # self.mouse_position.setFixedWidth(220)
        self.mouse_position.setFont(fixedFont)

        self.image_info = QLabel(f'')
        self.image_info.setFont(fixedFont)

        self.mode_label = QLabel(f'MOVE MODE')
        self.mode_label.setStyleSheet('color: red; font: bold')
        self.mode_label.setFont(fixedFont)
        # self.mode_label.setFixedWidth(150)

        # self.piece_counts = QLabel(f"| OHC: {0} IHC: {0} Synapse: {0}")
        # self.piece_counts.setFont(fixedFont)
        self.layout.addWidget(self.mode_label, alignment=Qt.AlignLeft)
        self.layout.addWidget(self.image_info, alignment=Qt.AlignLeft)
        self.layout.addStretch(1)
        self.layout.addWidget(self.mouse_position, alignment=Qt.AlignRight)
        self.layout.setContentsMargins(10,2,10,2)
        self.layout.setSpacing(10)
        self.setLayout(self.layout)

        policy = self.sizePolicy()
        # policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

        self.populate_info_from_piece()
        self.update_cell_counts()

    def setModeLabel(self, text: str):
        self.mode_label.setText(text)
        self.update()

    def setActivePiece(self, piece: Piece):
        """
        sets the active piece of the image viewer.
        is supposed to be called outside this widget!
        """
        self._piece = piece
        self.populate_info_from_piece()


    def populate_info_from_piece(self):
        if isinstance(self._piece, Piece) and self._piece.image is not None:
            shape = self._piece.image.shape
            x,y,c = shape
            shape = f'{(x, y, c)}'

            dtype = self._piece.image.dtype
            _min = self._piece.image.min()
            _max = self._piece.image.max()

            px_size = self._piece.pixel_size_xy

            if px_size:
                w = x * px_size / 100
                h = y * px_size / 100
            else:
                w = x
                h = y
                px_size = 1

            ihc = 0
            ohc = 0
            syn = 0
            for k,v in self._piece.children.items():
                if isinstance(v, Cell):
                    if v.type == 'OHC':
                        ohc += 1
                    else:
                        ihc += 1
                if isinstance(v, Synapse):
                    syn += 1

        else:
            shape = (0,0,0)
            dtype=None
            _min=-1
            _max=-1
            px_size=None
            w=None
            h=None
            ohc=0
            ihc=0
            syn=0

        self._image_str = f'SHAPE: {shape} px | SIZE: {(w,h)} um | MIN: {_min} | MAX: {_max} | dTYPE: {dtype} | px: ({px_size} nm)'
        self._update_labels()
        # self.image_info.setText(label)


    def _update_labels(self):
        self.image_info.setText(self._image_str + self._count_str)


    @Slot(list)
    def populate_mouse_position(self, pos):
        x, y = pos
        x, y = int(x), int(y)
        self.mouse_position.setText(f'POSITION: X/Y ({x:04}, {y:04})')

    def maximumHeight(self):
        return 20

    def sizeHint(self):
        return QSize(100, 20)

    def update_cell_counts(self):
        " | OHC: {ohc} IHC: {ihc} Synapse: {syn} "
        if isinstance(self._piece, Piece):
            ihc = 0
            ohc = 0
            syn = 0
            for k, v in self._piece.children.items():
                if isinstance(v, Cell):
                    if v.type == 'OHC':
                        ohc += 1
                    else:
                        ihc += 1
                if isinstance(v, Synapse):
                    syn += 1
        else:
            ihc, ohc, syn = 0,0,0

        self._count_str = f"| OHC: {ohc} IHC: {ihc} SYNAPSE: {syn}"
        self._update_labels()



