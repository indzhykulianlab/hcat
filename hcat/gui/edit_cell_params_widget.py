# from PySide6.QtCore import Qt, QSize, QDir, QPointF, Slot, Signal, QLineF, QRectF
# from PySide6.QtWidgets import *
# from PySide6.QtGui import *
#
# from hcat.state import Piece, Cell, Synapse, StateItem
# import hcat.lib.utils
#
# from typing import List, Tuple
# from math import sqrt
#
#
# class EditCellParamsWidget(QWidget):
#     def __init__(self, cell: Cell):
#         super(EditCellParamsWidget, self).__init__()
#
#
#         self.cell = cell
#
#         layout = QVBoxLayout()
#
#         # choose a celltype
#         _layout = QVBoxLayout()
#         self.celltype = QComboBox(self)
#         self.celltype.addItem('None')
#         self.celltype.addItem('IHC')
#         self.celltype.addItem('OHC')
#         self.type_index_map = {'None': 0, 'IHC': 1, 'OHC': 2}
#         self.celltype.setCurrentIndex(self.type_index_map[str(self.cell.type)])
#         self.celltype_label = QLabel(self)
#         self.celltype_label.setText('Cell Type: ')
#         _layout.addWidget(self.celltype_label)
#         _layout.addWidget(self.celltype)
#         layout.addLayout(_layout)

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLabel,
    QVBoxLayout, QComboBox
)


class EditCellParamsWidget(QDialog):
    type_changed = Signal(str)
    update_canvas = Signal()

    def __init__(self, cell, parent=None):
        super().__init__(parent)
        self.cell = cell

        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(QLabel('Box Coordinates'))
        bbox_layout.addWidget(QLabel('---------------'))
        for i, k in enumerate(['X0', 'Y0', 'X1', 'Y1']):
            _label = QLabel(f'{k}: {cell.bbox[i]:.2f}')
            bbox_layout.addWidget(_label)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["OHC", "IHC"])
        self.type_combo.setCurrentText(self.cell.type)

        form_layout = QFormLayout()
        form_layout.addRow(bbox_layout)
        form_layout.addRow("type:", self.type_combo)

        form_layout.setAlignment(Qt.AlignTop)

        self.type_combo.currentTextChanged.connect(self.update_cell)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel(f'Cell ID: {cell.id}'))
        main_layout.addLayout(form_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("Cell Editor")

    def update_cell(self):
        self.type_changed.emit(self.type_combo.currentText())
        self.update_canvas.emit()




