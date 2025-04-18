from typing import List

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import hcat.lib.utils
from hcat.state import Piece, Cell
from hcat.widgets.push_button import WPushButton


class CellSidebarInfoWidget(QWidget):
    cellChanged = Signal()

    def __init__(self, active_piece: Piece = None):
        super(CellSidebarInfoWidget, self).__init__()

        self.painter = QPainter()

        self.active_piece: Piece = active_piece  # allow multiple active cells
        self.active_cell = active_piece.get_selected_children() if self.active_piece else []
        # colors = {'None': GREY, 'OHC': QColor(255, 255, 0, 128), 'IHC': QColor(255, 0, 255, 128)}

        self.ohc_color = QColor(255,255,0,128) # 'rgba(255, 255, 0, 128)'
        self.ihc_color = QColor(255,0, 255,128) # 'rgba(255, 0, 255, 128)'
        # self.setFrameStyle(QFrame.Box | QFrame.Plain)
        # self.setLineWidth(1)

        label_style = """
            QLabel {
                font: 10px;
            }
            """
        self.top_label = QLabel('NO CELL SELECTED')
        self.bottom_label = QLabel('')
        self.top_label.setStyleSheet(label_style)
        self.bottom_label.setStyleSheet(label_style)

        button_style = """
        QPushButton {
            background-color: white;
            border-style: outset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 1px;
            font-size: 10pt;
            font: bold;
            }
        QPushButton:pressed {
            background-color: grey;
            border-style: outset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 2px;
            }
        """
        self.toggle_type = QPushButton('NONE')
        self.toggle_type.setVisible(False)
        self.toggle_type.setStyleSheet(button_style)
        self.toggle_type.setFixedWidth(30)

        self.set_all_ihc = WPushButton('SET ALL IHC')
        self.set_all_ihc.setVisible(False)
        self.set_all_ihc.setWarningButton(True)
        self.set_all_ihc.setStyleSheet(button_style)
        self.set_all_ihc.setFixedWidth(75)

        self.set_all_ohc = WPushButton('SET ALL OHC')
        self.set_all_ohc.setWarningButton(True)
        self.set_all_ohc.setVisible(False)
        self.set_all_ohc.setStyleSheet(button_style)
        self.set_all_ohc.setFixedWidth(75)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.top_label, alignment=Qt.AlignLeft | Qt.AlignTop)
        top_layout.addWidget(self.toggle_type, alignment=Qt.AlignLeft | Qt.AlignTop)
        # top_layout.addWidget(self.set_all_ihc, alignment=Qt.AlignLeft)
        top_layout.addStretch(10)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.bottom_label, alignment=Qt.AlignLeft | Qt.AlignTop)
        bottom_layout.addWidget(self.set_all_ihc, alignment=Qt.AlignLeft | Qt.AlignTop)
        bottom_layout.addWidget(self.set_all_ohc, alignment=Qt.AlignLeft | Qt.AlignTop)
        bottom_layout.addStretch(10)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(2)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

        self.setFixedHeight(30)

        self.setLayout(layout)

        self.toggle_type.clicked.connect(self.update_type_button_text)
        self.toggle_type.clicked.connect(self.update_button_style)
        self.toggle_type.clicked.connect(self.update_cell_type)

        self.set_all_ohc.clicked.connect(self.change_all_cells_type_to_ohc)
        self.set_all_ihc.clicked.connect(self.change_all_cells_type_to_ihc)

        # self.setStyleSheet("border-width: 1px; border-color: black; border-style: inset;")
        self.setMinimumWidth(275)

    @Slot(dict)
    def setRenderParams(self, style_dict: dict):
        # r, g, b, a = hcat.lib.utils.hex_to_qcolor(style_dict['ihc_color']).getRgb()
        self.ihc_color: QColor = hcat.lib.utils.hex_to_qcolor(style_dict['ihc_color'])#f'rgba({r},{g},{b},{a})'

        # r, g, b, a = hcat.lib.utils.hex_to_qcolor(style_dict['ohc_color']).getRgb()
        self.ohc_color: QColor =hcat.lib.utils.hex_to_qcolor(style_dict['ohc_color']) #f'rgba({r},{g},{b},{a})'
        self.update_button_style()

    def update_cell_type(self):
        if len(self.active_cell) == 1:
            self.active_cell[0].type = self.toggle_type.text()
            self.cellChanged.emit()

    def change_all_cells_type_to_ihc(self):
        if len(self.active_cell) > 0:
            for i, c in enumerate(self.active_cell):
                c.type = 'IHC'
            self.cellChanged.emit()

    def change_all_cells_type_to_ohc(self):
        if len(self.active_cell) > 0:
            for i, c in enumerate(self.active_cell):
                c.type = 'OHC'
            self.cellChanged.emit()

    def set_active_piece(self, piece: Piece | None):
        self.active_piece: Piece = piece
        if self.active_piece is not None:
            self.set_active_cell(self.active_piece.get_selected_children())

    def set_active_cell(self, cell: Cell | List[Cell] = []):
        cell = cell if isinstance(cell, list) else [cell]
        cell = [c for c in cell if isinstance(c, Cell)]
        self.active_cell = cell
        self.update_type_button_text()
        self.populate_info()
        self.update_button_style()
        self.update()

    def populate_info(self):
        if len(self.active_cell) == 0:
            top = 'NO CELL SELECTED'
            bottom = ''
            self.toggle_type.setText('NONE')
            self.toggle_type.setVisible(False)
            self.set_all_ihc.setVisible(False)
            self.set_all_ohc.setVisible(False)
            self.bottom_label.setVisible(False)
        elif len(self.active_cell) > 1:
            top = 'MULTIPLE CELLS SELECTED'
            bottom = 'SET→'
            self.toggle_type.setText('True')
            self.bottom_label.setVisible(False)
            self.toggle_type.setVisible(False)
            self.set_all_ihc.setVisible(True)
            self.set_all_ohc.setVisible(True)
        else:
            c = self.active_cell[0]
            x0, y0, x1, y1 = c.bbox

            freq = f'{c.frequency:02.2f}' if c.frequency is not None else 'NOTSET'

            top = f'ID: {c.id:04} | SCORE: {c.score:01.2f} | FREQ: {freq} | TYPE:'
            bottom = f'BOX [x0, y0, x1, y1]: [{int(x0):04}, {int(y0):04}, {int(x1):04}, {int(y1):04}]'
            self.toggle_type.setText(c.type.upper())
            self.bottom_label.setVisible(True)
            self.toggle_type.setVisible(True)
            self.set_all_ihc.setVisible(False)
            self.set_all_ohc.setVisible(False)

        self.top_label.setText(top)
        self.bottom_label.setText(bottom)
        self.update()

    def update_type_button_text(self):
        button_text = self.toggle_type.text()
        if button_text == 'OHC':
            self.toggle_type.setText('IHC')
        elif button_text == 'IHC':
            self.toggle_type.setText('OHC')
        elif len(self.active_cell) == 0:
            self.toggle_type.setText('NONE')
        elif len(self.active_cell) > 1:
            self.toggle_type.setText('MULT')

    def update_button_style(self):

        def button_style(r,g,b,a):
            color = f"rgba({r},{g},{b},{a})"
            text_color = 'black' if (r*0.299 + g*0.587 + b*0.114) > 186 else 'white'
            return f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            border-style: outset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 1px;
            font-size: 10pt;
            font: bold;
            }}
        QPushButton:pressed {{
            background-color: grey;
            border-style: outset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 2px;
            }}
        """
        # qcolor_to_rgba = lambda r, g, b, a: f"rgba({r},{g},{b},{a})"
        button_text = self.toggle_type.text()
        if button_text == 'OHC':
            self.toggle_type.setStyleSheet(button_style(*self.ohc_color.getRgb()))
        elif button_text == 'IHC':
            self.toggle_type.setStyleSheet(button_style(*self.ihc_color.getRgb()))
        else:
            self.toggle_type.setStyleSheet(button_style(255,255,255,255))

        self.set_all_ohc.setStyleSheet(button_style(*self.ohc_color.getRgb()))
        self.set_all_ihc.setStyleSheet(button_style(*self.ihc_color.getRgb()))

    # def paintEvent(self, event):
    #     painter = self.painter
    #     painter.begin(self)
    #
    #     pen = QPen()
    #     pen.setColor(QColor(0,0,31, 128))
    #     pen.setWidth(1)
    #     painter.setPen(pen)
    #
    #     painter.drawLine(QLine(-10, self.height()-1, 1000, self.height()-1))
    #     # painter.drawLine(QLine(0, 1, 0, 1))
    #
    #     super(PieceSidebarInfo, self).paintEvent(event)
    #     painter.end()


if __name__ == '__main__':
    c = Cell(id=0, parent=None, score=0.1093984701293874, type='OHC', frequency=2342.23, bbox=[0.0, 0.0, 100.0, 1000.0])

    app = QApplication()
    w = CellSidebarInfoWidget()
    w.set_active_cell([c])
    w.show()
    app.exec()
