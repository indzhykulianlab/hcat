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

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_C and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
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

class ABRPeakInfoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super(ABRPeakInfoWidget, self).__init__()
        self.parent = parent

        self.waveform_selecter_combobox = QComboBox()
        self.waveform_selecter_combobox.addItems(["1", "2", "3", "4", "5"])
        self.waveform_selecter_combobox.setCurrentIndex(0)

        self.table = TableWithCopy()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnCount(5)
        self.table.setRowCount(5)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # self._prepare_line_edits()
        self.make_headers()
        self.create_layout()
        self.link_slots_and_signals()

    def link_slots_and_signals(self):
        self.waveform_selecter_combobox.currentIndexChanged.connect(self.update)
        self.waveform_selecter_combobox.currentIndexChanged.connect(self.alert)

    def alert(self):
        print('ALERT!')

    def make_headers(self):
        nul = QTableWidgetItem(" ")
        header = (
            QTableWidgetItem("Amplitude"),
            QTableWidgetItem("P Value"),
            QTableWidgetItem("P Latency"),
            QTableWidgetItem("N Value"),
            QTableWidgetItem("N Latency"),
        )

        for i, w in enumerate(header):
            self.table.setHorizontalHeaderItem(i, w)

        for i in range(self.table.verticalHeader().count()):
            self.table.setVerticalHeaderItem(i, QTableWidgetItem(''))

    def create_layout(self):
        layout = QVBoxLayout()
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave Number: "))
        wave_layout.addWidget(self.waveform_selecter_combobox)
        layout.addWidget(self.table)
        layout.addLayout(wave_layout)

        layout.addStretch(1)
        self.setLayout(layout)

    def update(self):
        if self.parent is not None:
            experiment: ABRExperiment = self.parent.get_experiment()

            peaks: ABRPeaks = experiment.get_peaks()
            notches: ABRNotches = experiment.get_notches()

            # Get the wave number
            # for wave_number in peaks.keys():
                # if wave_number == self.waveform_selecter_combobox.currentText():
                #     peaks = peaks[wave_number]
                #     notches = notches[wave_number]
            self.table.setRowCount(max(len(peaks), len(notches)))

            peak_y = []
            notch_y = []

            for i, v in enumerate(peaks.keys()):
                self.table.setVerticalHeaderItem(i, QTableWidgetItem(f'{v:0.0f}'))

            for i, k in enumerate(peaks.keys()):
                for wave_number in peaks[k].keys():
                    if str(wave_number) == self.waveform_selecter_combobox.currentText():
                        v: Point = peaks[k][wave_number]
                        x = v['x']
                        y = v['y']
                        peak_y.append(y)
                        self.table.setItem(i, 1, QTableWidgetItem(f'{x:0.2e}'))
                        self.table.setItem(i, 2, QTableWidgetItem(f'{y:0.2e}'))

            for i, k in enumerate(notches.keys()):
                for wave_number in notches[k].keys():
                    if str(wave_number) == self.waveform_selecter_combobox.currentText():
                        v: Point = notches[k][wave_number]
                        x = v['x']
                        y = v['y']
                        notch_y.append(y)
                        self.table.setItem(i, 3, QTableWidgetItem(f'{x:0.2e}'))
                        self.table.setItem(i, 4, QTableWidgetItem(f'{y:0.2e}'))

            for i, (p, n) in enumerate(zip(peak_y, notch_y)):
                self.table.setItem(i, 0, QTableWidgetItem(f'{p-n:0.2e}'))

        super(ABRPeakInfoWidget, self).update()


if __name__ == "__main__":
    app = QApplication()
    w = ABRPeakInfoWidget()
    w.show()
    app.exec()
