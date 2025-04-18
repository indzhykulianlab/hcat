from __future__ import annotations
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from typing import *
from hcat.lib.types import *
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
import numpy as np


class TableWithCopy(QTableWidget):
    """
    this class extends QTableWidget
    * supports copying multiple cell's text onto the clipboard
    * formatted specifically to work with multiple-cell paste into programs
      like google sheets, excel, or numbers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        copied_cells = sorted(self.selectedIndexes())

        copy_text = ''
        max_column = copied_cells[-1].column()
        for c in copied_cells:
            copy_text += self.item(c.row(), c.column()).text()
            if c.column() == max_column:
                copy_text += '\n'
            else:
                copy_text += '\t'

        QApplication.clipboard().setText(copy_text)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_C and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.copy()


class ABRBatchTableViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None):
        super(ABRBatchTableViewerWidget, self).__init__()

        self.parent = parent

        self.table = TableWithCopy()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.save_button = QPushButton('Save')
        self.clear_button = QPushButton('Clear')

        self._init_table()
        self.create_layout()
        self.link_slots_and_signals()

    def _init_table(self):
        self.table.setColumnCount(2)  # filename, Threshold
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem('Filename'))
        self.table.setHorizontalHeaderItem(1, QTableWidgetItem('Threshold (dB)'))

    def add_column(self, field: Fields):
        n_columns = self.table.horizontalHeader().count()
        self.table.setColumnCount(n_columns + 1)
        self.table.setHorizontalHeaderItem(n_columns, QTableWidgetItem(field['attribute']))

    def remove_headder(self, header: str):
        raise NotImplementedError

    def add_row(self, filedata: FileData):
        n_rows = self.table.verticalHeader().count() + 1
        n_columns = self.table.horizontalHeader().count()
        self.table.setRowCount(n_rows)
        self.table.setItem(n_rows-1, 0, QTableWidgetItem(filedata['filename']))

        headers = [self.table.horizontalHeaderItem(i).text() for i in range(n_columns)]
        for field in filedata['fields']:
            if field['attribute'] not in headers:
                headers.append(field['attribute'])
                n_columns+=1
                self.add_column(field)

            for i, h in enumerate(headers):
                if h == field['attribute']:
                    self.table.setItem(n_rows-1, i, QTableWidgetItem(field['value']))

    def update_row(self, filedata: FileData):
        n_rows = self.table.verticalHeader().count()
        n_columns = self.table.horizontalHeader().count()
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(n_columns)]

        for i in range(n_rows):
            item = self.table.item(i, 0)
            if item.text() != filedata['filename']:
                continue

            for field in filedata['fields']:
                for j, h in enumerate(headers):
                    if h == field['attribute']:
                        self.table.setItem(n_rows - 1, j, QTableWidgetItem(field['value']))
            return

        # If weve got here, then we need to add a row...
        self.add_row(filedata)

    def update(self):

        if self.parent is not None:
            experiment: ABRExperiment = self.parent.get_experiment()
            file_data = experiment.get_file_data()
            if file_data is not None:
                self.update_row(file_data)

        super(ABRBatchTableViewerWidget, self).update()

    def remove_column(self, field: Fields):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError

    def link_slots_and_signals(self):
        pass

    def create_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.table)

        savelayout = QHBoxLayout()
        savelayout.addWidget(self.save_button)
        savelayout.addWidget(self.clear_button)
        savelayout.addStretch(1)
        layout.addLayout(savelayout)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication()
    w = ABRBatchTableViewerWidget(None)

    field0: Fields = {
        'attribute': 'test0',
        'regex': 'regex_test',
        'value': 'TEST0'
    }

    field1: Fields = {
        'attribute': 'test1',
        'regex': 'regex_test',
        'value': 'TEST1'
    }

    file_data: FileData = {
        'filename': 'test.png',
        'fields': [field0, field1]
    }

    w.update_row(file_data)

    field0: Fields = {
        'attribute': 'test0',
        'regex': 'regex_test',
        'value': 'TEST0'
    }

    field1: Fields = {
        'attribute': 'test1',
        'regex': 'regex_test',
        'value': 'UPDATED'
    }

    file_data: FileData = {
        'filename': 'test.png',
        'fields': [field0, field1]
    }

    w.add_row(file_data)
    w.add_row(file_data)
    w.add_row(file_data)
    w.add_row(file_data)


    w.show()
    app.exec()
