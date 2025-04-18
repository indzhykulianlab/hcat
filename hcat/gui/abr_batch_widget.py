import re

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from typing import *
from torch import Tensor

from hcat.widgets.file_picker import WFilePickerWidget
from hcat.cabr.EPL_parser import parse_abr_file
from hcat.cabr.abr_store import ABRStore, TempStore
from hcat.cabr.kirupa_threshold import calculate_threshold as kirupa_thr_function
from hcat.cabr.abr_evaluator import calculate_threshold as deepABR_thr_function
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from hcat.gui.abr_viewer import ABRViewerWidget
from hcat.cabr.peak_finder import guess_abr_peaks, guess_abr_notches
from hcat.lib.types import *
import hcat.utils.colors
from hcat.widgets.push_button import WPushButton
from hcat.widgets.vertical_button import VerticalPushButton
from hcat.style.cabr_macos_style import MACOS_STYLE
import hcat.gui.resources

import glob
import os.path
from random import shuffle
from hcat.lib.types import *


class DeleteButton(QPushButton):
    pressed = Signal(int)

    def __init__(self, row_num: int, *args, **kwargs):
        super(DeleteButton, self).__init__(*args, **kwargs)
        self.row_num = row_num
        self.clicked.connect(self.emit_pressed)
        self.setObjectName("field_delete_button")

    def get_row_num(self):
        return self.row_num

    def set_row_num(self, num: int):
        self.row_num = num

    def emit_pressed(self):
        self.pressed.emit(self.row_num)


class RegexLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super(RegexLineEdit, self).__init__(*args, **kwargs)

        self.textChanged.connect(self.compile_regex)

    def compile_regex(self, text: str) -> bool:
        try:
            re.compile(text)
            self.setStyleSheet("background-color: white")
            return True
        except re.error:
            self.setStyleSheet("background-color: rgba(255,0,0,50)")


class AttrRegexListWidget(QWidget):
    fieldsChanged = Signal()

    def __init__(self):
        super(AttrRegexListWidget, self).__init__()

        self.attribute_inputs: List[QLineEdit] = []
        self.regex_inputs: List[RegexLineEdit] = []
        self.delete_buttons: List[DeleteButton] = []
        self._label: List[QLabel] = []

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.row_layout_list: List[QHBoxLayout] = []

        self.painter = QPainter()

    def reset(self):
        """call when resetting the form widget..."""
        raise NotImplementedError

    def new_row(self):
        att_input = QLineEdit()
        att_input.setPlaceholderText("Attribute")
        self.attribute_inputs.append(att_input)

        regex_input = RegexLineEdit()
        regex_input.setPlaceholderText("Regular Expression")
        self.regex_inputs.append(regex_input)
        self._label.append(QLabel("⟶"))

        # Create delete button which is linked to the row...
        del_button = DeleteButton(text="-", row_num=len(self.delete_buttons))
        del_button.pressed.connect(self.delete_row)
        self.delete_buttons.append(del_button)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(10)
        row_layout.addWidget(self.attribute_inputs[-1])
        row_layout.addWidget(self._label[-1])
        row_layout.addWidget(self.regex_inputs[-1])
        row_layout.addWidget(self.delete_buttons[-1])
        self.row_layout_list.append(row_layout)

        self.refresh_layout()

        self.setLayout(self.layout)
        return self

    def delete_row(self, row: int):
        for widget_list in [
            self.attribute_inputs,
            self.regex_inputs,
            self.delete_buttons,
            self._label,
        ]:
            w = widget_list.pop(row)
            w.deleteLater()

        self.refresh_layout()

        # now update button row...
        self._update_delete_button_row_number()

        if len(self.attribute_inputs) == 0:
            self.new_row()

    def clear(self):
        for widget_list in [
            self.attribute_inputs,
            self.regex_inputs,
            self.delete_buttons,
            self._label,
        ]:
            for widget in widget_list:
                widget.deleteLater()

        self.attribute_inputs = []
        self.regex_inputs = []
        self.delete_buttons = []
        self._label = []
        self.refresh_layout()
        self.new_row()

    def refresh_layout(self):
        while self.layout.count() > 0:
            self.layout.takeAt(self.layout.count() - 1)

        for row in self.row_layout_list:
            self.layout.addLayout(row)

        self.layout.addStretch(1)
        self.setLayout(self.layout)

    def _update_delete_button_row_number(self):
        for i, b in enumerate(self.delete_buttons):
            b.set_row_num(i)

    def get_fields(self) -> List[Fields]:
        fields = []
        for att, reg in zip(self.attribute_inputs, self.regex_inputs):
            field = {"attribute": att.text(), "regex": reg.text()}

    def paintEvent(self, event: QPaintEvent) -> None:
        self.painter.begin(self)
        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, self.painter, self)
        self.painter.end()


class ABRFileViewer(QWidget):
    def __init__(self):
        super(ABRFileViewer, self).__init__()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Filename", "Attribute", "Value"])

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        self.setLayout(layout)

    def populate_from_data(self, data: List[FileData]):
        self.tree.clear()
        items = []
        for file_data in data:
            filename: str = file_data["filename"]
            field_results: Dict[str, str] = file_data["data"]

            item = QTreeWidgetItem([filename])
            for key, value in field_results.items():
                child = QTreeWidgetItem([None, key, value])
                item.addChild(child)

            items.append(item)

        self.tree.insertTopLevelItems(0, items)


class ABRBatchWidget(QWidget):
    def __init__(self):
        super(ABRBatchWidget, self).__init__()

        # Custon Widgets
        self.attr_regex_widget = AttrRegexListWidget().new_row()
        # self.metadata_tree_widget = ABRFileViewer()

        # Buttons
        # self.choose_folder_button = QPushButton("Choose Folder")
        self.add_new_attribute_button = QPushButton("Add Attribute")
        self.clear_all_attributes_button = QPushButton("Clear All")
        # self.add_folder_button = QPushButton("ADD")  # For adding batch fils
        # self.remove_file_button = QPushButton("Remove")  # For removing files
        # self.clear_all_files_button = QPushButton("Clear")  # For removing files
        # self.run_batch_button = QPushButton("RUN")  # For running the full analysis

        # # Checkbox
        # self.filter_checkbox = QCheckBox("")
        # self.filter_checkbox.setChecked(True)
        #
        # # Spinbox for filters
        # self.f0_spinbox = QSpinBox()
        # self.f0_spinbox.setRange(1, 12500)
        # self.f0_spinbox.setValue(30)
        #
        # self.f1_spinbox = QSpinBox()
        # self.f1_spinbox.setRange(1, 12500)
        # self.f1_spinbox.setValue(3000)
        #
        # self.order_spinbox = QSpinBox()
        # self.order_spinbox.setRange(1, 4)

        # # Combobox
        # self.choose_parser = QComboBox()
        # self.choose_parser.addItems(["EPL"])
        #
        # self.threshold_prediction = QComboBox()
        # self.threshold_prediction.addItems(["Suthakar-2019", "DeepABR"])

        self.create_layout()
        self.link_slots_and_signals()

    def minimumHeight(self) -> int:
        return 150

    @staticmethod
    def new_frame():
        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")
        return frame

    def link_slots_and_signals(self):
        self.add_new_attribute_button.clicked.connect(self.attr_regex_widget.new_row)
        self.clear_all_attributes_button.clicked.connect(self.attr_regex_widget.clear)

    def create_layout(self):


        self.scroll_area = QScrollArea()
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setViewportMargins(10, 10, 10, 10)
        self.scroll_area.setWidget(self.attr_regex_widget)


        attribute_layout = QHBoxLayout()
        attribute_layout.addWidget(self.add_new_attribute_button)
        attribute_layout.addWidget(self.clear_all_attributes_button)
        attribute_layout.setContentsMargins(0, 0, 0, 0)

        # filter_layout = QFormLayout()
        # filter_layout.setContentsMargins(0, 0, 0, 0)
        # filter_layout.setSpacing(2)
        # filter_layout.addRow(QLabel("Apply"), self.filter_checkbox)
        # filter_layout.addRow(QLabel("Lower"), self.f0_spinbox)
        # filter_layout.addRow(QLabel("Upper"), self.f1_spinbox)
        # filter_layout.addRow(QLabel("Order"), self.order_spinbox)

        # filter_group = QGroupBox("Filter")
        # filter_group.setLayout(filter_layout)
        # margins: QMargins = filter_group.contentsMargins()
        # filter_group.setContentsMargins(2, margins.top(), 2, margins.bottom())

        # run_layout = QHBoxLayout()
        # run_layout.addWidget(self.run_batch_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # layout.addWidget(filter_group)
        # layout.addWidget(self.new_frame())
        layout.addWidget(self.scroll_area)
        layout.addLayout(attribute_layout)
        # layout.addWidget(self.new_frame())
        # layout.addWidget(self.metadata_tree_widget)
        # layout.addWidget(self.choose_folder_button)
        # layout.addWidget(self.new_frame())
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication()
    w = ABRBatchWidget()
    print(w.sizeHint())
    w.show()
    app.exec()
