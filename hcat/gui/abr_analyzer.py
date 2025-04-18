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
from hcat.gui.abr_batch_table_viewer_widget import ABRBatchTableViewerWidget
from hcat.cabr.peak_finder import guess_abr_peaks, guess_abr_notches
from hcat.widgets.agf_plot import AGFplot
from hcat.widgets.abr_info_view import ABRPeakInfoWidget
from hcat.lib.types import *
import hcat.utils.colors
from hcat.widgets.push_button import WPushButton
from hcat.widgets.vertical_button import VerticalPushButton
from hcat.style.cabr_macos_style import MACOS_STYLE
from hcat.gui.abr_batch_widget import ABRBatchWidget
import hcat.gui.resources

import glob
import os.path
from random import shuffle

"""
TODO:
- Add button to re-guess the predictions
- Add the probabilities to the drag widgets when DEEP Abr is available...
- Add a little training widget 
- style the whole thing
- bug fixes...
- save file exporting...
- validate DeepABR Widget



show amplitude in excel like spreadsheed.
Show latencies and other statistics ...
draw lines between peaks...

"""


class ABRToolbar(QWidget):
    def __init__(self):
        super(ABRToolbar, self).__init__()

        self.analyzer_tab_button = QPushButton("Analyze")
        self.analyzer_tab_button.setObjectName("tab_button")
        self.trainer_tab_button = QPushButton("Train")
        self.trainer_tab_button.setObjectName("tab_button")
        self.batch_process_button = QPushButton("Batch")
        self.batch_process_button.setObjectName("tab_button")

        self.create_layouts()

    def create_layouts(self):
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.analyzer_tab_button)
        layout.addWidget(self.trainer_tab_button)
        layout.addWidget(self.batch_process_button)
        layout.addStretch(1)
        self.setLayout(layout)


class ABRControls(QWidget):
    ySliderChanged = Signal(float)
    xSliderChanged = Signal(float)
    filterParamsChanged = Signal()

    def __init__(self):
        super(ABRControls, self).__init__()

        self.next_file_button = QPushButton(QIcon(":/resources/right_arrow.svg"), "")
        self.next_file_button.setIconSize(QSize(30, 10))
        self.next_file_button.setObjectName("next_button")
        self.previous_file_button = QPushButton(QIcon(":/resources/left_arrow.svg"), "")
        self.previous_file_button.setIconSize(QSize(30, 10))
        self.previous_file_button.setObjectName("previous_button")
        self.open_folder_button = QPushButton("OPEN FOLDER")

        self.reset_button = QPushButton("RESET")
        self.export_as_csv_button = QPushButton("EXPORT CSV")
        self.recursive_search_checkbox = QCheckBox("Recursive Search")
        self.recursive_search_checkbox.setChecked(True)

        self.blinder_checkbox = QCheckBox("Blind Info")
        self.blinder_checkbox.setChecked(True)

        self.autosave_on_next_checkbox = QCheckBox("Autosave")
        self.autosave_on_next_checkbox.setChecked(True)

        self.increase_thr_button = QPushButton(QIcon(":/resources/up_arrow.svg"), "")
        self.increase_thr_button.setIconSize(QSize(10, 10))

        self.decrease_thr_button = QPushButton(QIcon(":/resources/down_arrow.svg"), "")
        self.decrease_thr_button.setIconSize(QSize(10, 10))

        self.set_all_above_thr_button = QPushButton("ALL ABOVE THR")
        self.set_all_above_thr_button.setObjectName("all_above_button")
        self.set_all_below_thr_button = QPushButton("NO RESPONSE")
        self.set_all_below_thr_button.setObjectName("all_below_button")

        self.normalize_waveforms_button = QPushButton("NORMALIZE")

        self.y_scale_slider = self.prepare_slider(
            QSlider(Qt.Horizontal), -20, 50, 70 // 15
        )
        self.x_scale_slider = self.prepare_slider(
            QSlider(Qt.Horizontal), -20, 100, 120 // 15
        )

        self.y_scale_slider_init_val = 0
        self.x_scale_slider_init_val = 51

        self.f0_spinbox = QSpinBox()
        self.f0_spinbox.setRange(1, 12500)
        self.f0_spinbox.setValue(30)

        self.f1_spinbox = QSpinBox()
        self.f1_spinbox.setRange(1, 12500)
        self.f1_spinbox.setValue(3000)

        self.filter_checkbox = QCheckBox("")
        self.filter_checkbox.setChecked(True)

        self.order_spinbox = QSpinBox()
        self.order_spinbox.setRange(1, 4)

        self.choose_parser = QComboBox()
        self.choose_parser.addItems(["EPL"])

        self.threshold_prediction = QComboBox()
        self.threshold_prediction.addItems(["Suthakar-2019"])
        self.threshold_prediction.addItems(["DeepABR"])

        self.painter = QPainter()
        self.create_layout()
        self.create_tooltips()
        self.link_slots_and_signals()
        self.reset_sliders()

    # def minimumSizeHint(self) -> QSize:
    #     return QSize(153, 663)

    def create_layout(self):
        # scale_layout = QFormLayout()
        # scale_layout.setContentsMargins(1, 1, 1, 1)
        # scale_layout.setSpacing(2)
        # scale_layout.addRow(QLabel("X"), self.x_scale_slider)
        # scale_layout.addRow(QLabel("Y"), self.y_scale_slider)
        # scale_layout.setAlignment(Qt.AlignLeft)

        xlayout = QHBoxLayout()
        xlayout.setSpacing(2)
        xlayout.setContentsMargins(0, 0, 2, 0)
        xlayout.addWidget(QLabel("X"))
        xlayout.addWidget(self.x_scale_slider)

        ylayout = QHBoxLayout()
        ylayout.setSpacing(2)
        ylayout.setContentsMargins(0, 0, 2, 0)
        ylayout.addWidget(QLabel("Y"))
        ylayout.addWidget(self.y_scale_slider)

        scale_layout = QVBoxLayout()
        scale_layout.setSpacing(0)
        scale_layout.setContentsMargins(2, 0, 2, 0)
        scale_layout.addLayout(xlayout)
        scale_layout.addLayout(ylayout)

        scale_group = QGroupBox("Scale")
        # scale_group.setContentsMargins(1,10,1,1)
        scale_group.setLayout(scale_layout)
        margins: QMargins = scale_group.contentsMargins()
        scale_group.setContentsMargins(2, margins.top(), 2, margins.bottom())

        next_previous_layout = QHBoxLayout()
        next_previous_layout.setContentsMargins(0, 0, 0, 0)
        next_previous_layout.setSpacing(0)
        next_previous_layout.addWidget(self.previous_file_button)
        next_previous_layout.addWidget(self.next_file_button)
        next_previous_group = QGroupBox("File Selector")
        next_previous_group.setLayout(next_previous_layout)

        threshold_buttons_layout = QVBoxLayout()
        threshold_buttons_layout.setContentsMargins(2, 2, 2, 2)
        threshold_buttons_layout.setSpacing(0)
        threshold_buttons_layout.addWidget(self.increase_thr_button)
        threshold_buttons_layout.addWidget(self.decrease_thr_button)
        threshold_buttons_layout.addWidget(self.set_all_above_thr_button)
        threshold_buttons_layout.addWidget(self.set_all_below_thr_button)
        threshold_buttons_group = QGroupBox("Threshold")
        # margins: QMargins = threshold_buttons_group.contentsMargins()
        threshold_buttons_group.setLayout(threshold_buttons_layout)
        # threshold_buttons_group.setContentsMargins(2, margins.top(), 2, margins.bottom()+2)

        filter_layout = QFormLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(2)
        filter_layout.addRow(QLabel("Apply"), self.filter_checkbox)
        filter_layout.addRow(QLabel("Lower"), self.f0_spinbox)
        filter_layout.addRow(QLabel("Upper"), self.f1_spinbox)
        filter_layout.addRow(QLabel("Order"), self.order_spinbox)

        filter_group = QGroupBox("Filter")
        filter_group.setLayout(filter_layout)
        margins: QMargins = filter_group.contentsMargins()
        filter_group.setContentsMargins(2, margins.top(), 2, margins.bottom())
        #
        blind_layout = QVBoxLayout()
        blind_layout.addWidget(self.blinder_checkbox)
        blind_layout.setContentsMargins(5, 3, 3, 3)

        search_layout = QVBoxLayout()
        search_layout.addWidget(self.recursive_search_checkbox)
        search_layout.setContentsMargins(5, 3, 3, 3)

        autosave_layout = QVBoxLayout()
        autosave_layout.addWidget(self.autosave_on_next_checkbox)
        autosave_layout.setContentsMargins(5, 3, 3, 3)
        #
        layout = QVBoxLayout()
        layout.addSpacing(5)
        layout.setObjectName("control_layout")
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(next_previous_group)
        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")
        layout.addWidget(frame, Qt.AlignLeft)
        layout.addWidget(threshold_buttons_group)
        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")
        layout.addWidget(frame, Qt.AlignLeft)
        layout.addWidget(scale_group)
        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")
        layout.addWidget(frame, Qt.AlignLeft)
        layout.addWidget(filter_group)
        # layout.addSpacing(10)
        #
        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")
        layout.addWidget(frame, Qt.AlignLeft)

        dropdown_layout = QVBoxLayout()
        dropdown_layout.addWidget(QLabel("File Parser"))
        dropdown_layout.addWidget(self.choose_parser)
        dropdown_layout.addSpacing(10)
        dropdown_layout.addWidget(QLabel("Threshold Predictor"))
        dropdown_layout.addWidget(self.threshold_prediction)
        dropdown_layout.setContentsMargins(4, 2, 4, 2)
        dropdown_layout.setSpacing(2)

        bottom_button_layout = QVBoxLayout()
        bottom_button_layout.setContentsMargins(2, 2, 2, 2)
        bottom_button_layout.setSpacing(0)
        bottom_button_layout.addWidget(self.open_folder_button)
        bottom_button_layout.addWidget(self.export_as_csv_button)
        bottom_button_layout.addWidget(self.reset_button)

        frame = QFrame()
        frame.setFrameStyle(QFrame.HLine)
        frame.setObjectName("frame")

        layout.addLayout(dropdown_layout)
        layout.addWidget(frame)
        layout.addStretch(1)
        layout.addLayout(blind_layout)
        layout.addLayout(autosave_layout)
        layout.addLayout(search_layout)
        layout.addLayout(bottom_button_layout)

        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint() + QSize(1, 1)

    def create_tooltips(self):
        self.next_file_button.setToolTip("Display the next ABR Waveform")
        self.previous_file_button.setToolTip("Display the previous ABR Waveform")
        self.increase_thr_button.setToolTip("Increase Threshold")
        self.decrease_thr_button.setToolTip("Decrease Threshold")
        self.open_folder_button.setToolTip("Open a folder of ABR Files")
        self.choose_parser.setToolTip("Sets expected format of ABR files")
        self.threshold_prediction.setToolTip("Sets the default ABR threshold generator")
        self.x_scale_slider.setToolTip("change X scaling of waveform")
        self.y_scale_slider.setToolTip("change Y scaling of waveform")
        self.recursive_search_checkbox.setToolTip("Search sub-folders when opening")
        self.blinder_checkbox.setToolTip("Blind identifying ABR data")

    def prepare_slider(
        self,
        slider_widget: QSlider,
        minval: int,
        maxval: int,
        tick: float,
        init_val: None | float = None,
    ):
        slider_widget.setTickPosition(QSlider.TicksBelow)
        init_val = init_val if init_val else round((maxval - minval) / 2 + minval)
        slider_widget.setMinimum(minval)
        slider_widget.setMaximum(maxval)
        slider_widget.setTickInterval(tick)
        slider_widget.setValue(int(init_val))
        return slider_widget

    def link_slots_and_signals(self):
        self.x_scale_slider.valueChanged.connect(self.emit_x_slider_changed)
        self.y_scale_slider.valueChanged.connect(self.emit_y_slider_changed)
        self.reset_button.clicked.connect(self.reset_sliders)

        self.f0_spinbox.valueChanged.connect(self.update_spinbox_ranges)
        self.f1_spinbox.valueChanged.connect(self.update_spinbox_ranges)

        self.f0_spinbox.valueChanged.connect(self.emit_filter_params_changed)
        self.f1_spinbox.valueChanged.connect(self.emit_filter_params_changed)
        self.order_spinbox.valueChanged.connect(self.emit_filter_params_changed)
        self.filter_checkbox.clicked.connect(self.emit_filter_params_changed)

    def emit_filter_params_changed(self):
        self.filterParamsChanged.emit()

    def update_spinbox_ranges(self):
        self.f0_spinbox.setMaximum(self.f1_spinbox.value() - 1)
        self.f1_spinbox.setMinimum(self.f0_spinbox.value() + 1)

    def reset_sliders(self):
        self.x_scale_slider.setValue(self.x_scale_slider_init_val)
        self.y_scale_slider.setValue(self.y_scale_slider_init_val)

        self.emit_x_slider_changed()
        self.emit_y_slider_changed()

    def block_signals(self):
        self.x_scale_slider.blockSignals(True)
        self.y_scale_slider.blockSignals(True)

    def enable_signals(self):
        self.x_scale_slider.blockSignals(False)
        self.y_scale_slider.blockSignals(False)

    def emit_y_slider_changed(self):
        scale = self.y_scale_slider.value() / 50
        self.ySliderChanged.emit(1 + scale)

    def emit_x_slider_changed(self):
        scale = self.x_scale_slider.value() / 50
        self.xSliderChanged.emit(1 + scale)

    def paintEvent(self, event):
        self.painter.begin(self)
        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, self.painter, self)
        self.painter.end()


class ABRFooter(QWidget):
    def __init__(self):
        super(ABRFooter, self).__init__()

        self.file_label = QLabel("")
        layout = QHBoxLayout()
        layout.setContentsMargins(3, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self.file_label)
        self.setLayout(layout)
        self.setMinimumHeight(20)
        self.painter = QPainter()
        self.setStyleSheet(MACOS_STYLE)

    def paintEvent(self, event):
        self.painter.begin(self)
        opt = QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, self.painter, self)
        self.painter.end()

    def resetText(self):
        self.file_label.setText("")

    def setText(self, text):
        self.file_label.setText(text)
        self.update()


class ABRAnalyzer(QMainWindow):
    abrFileChanged = Signal(str)

    def __init__(self):
        super(ABRAnalyzer, self).__init__()

        self.setFocusPolicy(Qt.StrongFocus)

        self.viewer = ABRViewerWidget()
        self.agf_plot_widget = AGFplot(parent=self)
        self.agf_info_widget = ABRPeakInfoWidget(parent=self)
        self.abr_batch_info_table = ABRBatchTableViewerWidget(parent=self)
        self.control_widget = ABRControls()
        self.control_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.footer_widget = ABRFooter()
        self.left_toolbar_widget = ABRToolbar()

        self.abr_batch_widget = ABRBatchWidget()
        # self.abr_batch_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.store = ABRStore()
        self.temp_store: TempStore | None = None

        self.abr_files: List[str] = []
        self.abr_cache: List[ABRExperiment] = []
        self.abr_file_index: int = -1
        self.abr_cache_index: int = -1

        self.manual_thresholds = {}

        self.experiment: ABRExperiment | None = None

        # Layouts and organization
        self.setCentralWidget(self.viewer)
        self.create_layout()
        self.link_slots_and_signals()
        self.create_actions()
        self.create_menus()

        self.setStyleSheet(MACOS_STYLE)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)
        self.setDockOptions(QMainWindow.VerticalTabs)
        self.setDocumentMode(True)

        self.update()

    def get_experiment(self):
        return self.experiment

    def open_folder(self):
        folderpath = QFileDialog.getExistingDirectory(self, "Open", QDir.homePath())
        self.find_abr_files_from_filepath(folderpath)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        # if QGuiApplication.queryKeyboardModifiers() not in Qt.ShiftModifier:
        if Qt.ShiftModifier not in QGuiApplication.queryKeyboardModifiers():
            if key == Qt.Key_Down:
                self.decrease_thr()
            elif key == Qt.Key_Up:
                self.increase_thr()
            elif key == Qt.Key_Left:
                self.previous_abr()
            elif key == Qt.Key_Right:
                self.next_abr()

    def emit_abr_file_changed(self):
        if self.abr_files:
            text = self.abr_files[self.abr_file_index]
            if self.control_widget.blinder_checkbox.isChecked():
                self.abrFileChanged.emit("  ABR Filepath Blinded")

            else:
                self.abrFileChanged.emit("  " + text)

    def update_filter(self):
        if self.control_widget.filter_checkbox.isChecked():
            f0 = self.control_widget.f0_spinbox.value()
            f1 = self.control_widget.f1_spinbox.value()
            order = self.control_widget.order_spinbox.value()
            if isinstance(self.experiment, ABRExperiment):
                self.experiment.set_filter_params(f0, f1, order)
        else:
            self.experiment.clear_filter_params()
        self.update()

    def add_experiment_to_temp_store(self):
        if self.temp_store is not None and isinstance(self.experiment, ABRExperiment):
            self.temp_store.add_experiment(self.experiment)

    def find_abr_files_from_filepath(self, folderpath: str):
        if self.control_widget.recursive_search_checkbox.isChecked():
            search_path = os.path.join(os.path.join(folderpath, "**"), "ABR*")
        else:
            search_path = os.path.join(folderpath, "ABR*")

        # find all candidates for ABR Files. May not be abr files
        _candidates: List[str] = glob.glob(search_path, recursive=True)

        # shuffle them for blinding...
        if self.control_widget.blinder_checkbox.isChecked():
            shuffle(_candidates)

        # assert each valid file is indeed an ABR file...
        self.abr_files: List[str] = []
        for f in _candidates:
            with open(f, "rb") as file:
                if file.read(4).decode("utf-8") == ":RUN":
                    self.abr_files.append(f)

        if self.abr_files:
            parser: Callable[[str], ABRExperiment] = self.get_parser()
            self.abr_cache = parser(self.abr_files[0])
            self.abr_file_index = 0
            self.abr_cache_index = 0
            self.set_current_experiment(self.abr_cache[self.abr_cache_index])
            self.temp_store = TempStore()
            self.emit_abr_file_changed()
            self.control_widget.reset_sliders()

            self.update_filter()
            self.guess_peaks()
            self.agf_plot_widget.update()
            self.update()
        else:
            raise NotImplementedError("POPUP - no abr files found!")

    def guess_peaks(self):
        """
        guess the peaks of an ABR waveform!

        :return:
        :rtype:
        """
        # if isinstance(self.experiment, ABRExperiment):
        #     if self.experiment in self.temp_store:
        #         experiment_dict: ExperimentDict = self.temp_store[self.experiment]
        #         peaks: ABRPeaks = experiment_dict["peaks"]
        #         notches: ABRNotches = experiment_dict["notches"]
        #         self.experiment.set_peaks(peaks)
        #         self.experiment.set_notches(notches)
        #         self.viewer.reset_peak_widgets()
        #
        #     else:
        try:
            peaks = guess_abr_peaks(self.experiment)
            notches = guess_abr_notches(self.experiment, peaks)
            self.experiment.set_peaks(peaks)
            self.experiment.set_notches(notches)
            self.viewer.reset_peak_widgets()
            self.add_experiment_to_temp_store()
        except ValueError:
            self.experiment.clear_peaks()
            self.experiment.clear_notches()
            self.viewer.clear_peak_widgets()

        self.update()

    def update_experiment_from_store(self) -> bool:
        """
        sets the threshold of self.waveform if it exists in the store currently

        :return:
        """
        # Is the experiment in the store already
        if self.experiment in self.store:
            store_dict: ExperimentDict = self.store[self.experiment]
            thr: str = store_dict["threshold"]
            self.experiment.clear_threshold()
            if thr == "ALL_ABOVE":
                self.experiment.set_all_above_thr()
            elif thr == "ALL_BELOW":
                self.experiment.set_all_below_thr()
            elif thr is None:
                thr = None
            else:
                thr = int(thr)

            # Set peaks and notches...
            peaks: ABRPeaks = store_dict["peaks"]
            notches: ABRNotches = store_dict["notches"]
            self.experiment.set_peaks(peaks)
            self.experiment.set_notches(notches)

            for w in self.experiment.get_waveforms():
                w.clear_threshold()
                w.set_thr_probability(None)

            for w in self.experiment.get_waveforms():
                if w.level == thr:
                    w.set_as_threshold()
                    w.thr_set_by_user()

            return True

        # else not in store...
        return False

    def update_experiment_from_temp_store(self) -> bool:
        """
        sets the threshold of self.waveform if it exists in the store currently

        :return:
        """
        if not self.temp_store:
            return

        # Is the experiment in the store already
        if self.experiment in self.temp_store:
            store_dict: ExperimentDict = self.temp_store[self.experiment]
            thr: str = store_dict["threshold"]
            self.experiment.clear_threshold()
            if thr == "ALL_ABOVE":
                self.experiment.set_all_above_thr()
            elif thr == "ALL_BELOW":
                self.experiment.set_all_below_thr()
            elif thr is None:
                thr = None
            else:
                thr = int(thr)

            # Set peaks and notches...
            peaks: ABRPeaks = store_dict["peaks"]
            notches: ABRNotches = store_dict["notches"]
            self.experiment.set_peaks(peaks)
            self.experiment.set_notches(notches)

            for w in self.experiment.get_waveforms():
                w.clear_threshold()
                w.set_thr_probability(None)

            for w in self.experiment.get_waveforms():
                if w.level == thr:
                    w.set_as_threshold()
                    w.thr_set_by_user()

            return True

        # else not in store...
        return False

    def update_scale_from_sliders(self):
        self.control_widget.emit_x_slider_changed()
        self.control_widget.emit_y_slider_changed()

    def increase_y_slider_by_constant(self):
        val = self.control_widget.y_scale_slider.value()
        val += 2
        self.control_widget.y_scale_slider.setValue(val)

    def increase_x_slider_by_constant(self):
        val = self.control_widget.x_scale_slider.value()
        val += 2
        self.control_widget.x_scale_slider.setValue(val)

    def decrease_y_slider_by_constant(self):
        val = self.control_widget.y_scale_slider.value()
        val -= 2
        self.control_widget.y_scale_slider.setValue(val)

    def decrease_x_slider_by_constant(self):
        val = self.control_widget.x_scale_slider.value()
        val -= 2
        self.control_widget.x_scale_slider.setValue(val)

    def next_abr(self):
        """
        Goes to the next abr in the queue, if the queue is at the end, then we
        go to the next file.

        :return:
        """

        if self.experiment.get_threshold() is not None:
            self.store.add_experiment(self.experiment)

        self.abr_cache_index = min(self.abr_cache_index + 1, len(self.abr_cache) - 1)

        # If we're at the end of the cache, go to next file
        if self.abr_cache_index == len(self.abr_cache) - 1:
            self.abr_cache_index = 0

            if self.abr_file_index + 1 == len(self.abr_files):
                popup = QMessageBox()
                popup.setText("All ABR Files Analyzed")
                popup.setInformativeText(
                    "All ABR files found in the current folder have been inspected and saved."
                )
                popup.exec()

            self.abr_file_index = min(self.abr_file_index + 1, len(self.abr_files) - 1)

            parser: Callable[[str], ABRExperiment] = self.get_parser()
            self.abr_cache = parser(self.abr_files[self.abr_file_index])

        # new temp store on new experiment...
        del self.temp_store
        self.temp_store = TempStore()

        # Set the current experiment
        self.set_current_experiment(self.abr_cache[self.abr_cache_index])
        self.emit_abr_file_changed()

    def previous_abr(self):
        # If we're at the beginning of the cache, go to the previous file
        self.abr_cache_index = max(self.abr_cache_index - 1, 0)
        if self.abr_cache_index == 0:
            self.abr_file_index = max(self.abr_file_index - 1, 0)
            parser: Callable[[str], ABRExperiment] = self.get_parser()
            self.abr_cache = parser(self.abr_files[self.abr_file_index])
            self.abr_cache_index = len(self.abr_cache) - 1

        # new temp store on new experiment...
        del self.temp_store
        self.temp_store = TempStore()

        self.set_current_experiment(self.abr_cache[self.abr_cache_index])
        self.emit_abr_file_changed()

    def on_abr_change(self):
        self.update_scale_from_sliders()
        self.update_filter()

        # If in temp store, we do this first...
        if self.experiment in self.temp_store:
            was_in_temp = self.update_experiment_from_temp_store()

        # next check the actual store...
        elif self.experiment in self.store:
            self.update_experiment_from_store()

        else:
            self.guess_peaks()
            self.guess_threshold()

        self.viewer.reset_peak_widgets()
        self.add_experiment_to_temp_store()
        self.agf_plot_widget.update()
        self.agf_info_widget.update()
        self.abr_batch_info_table.update()
        self.update()

        # was_in_gt = self.update_experiment_from_store()
        # was_in_temp = self.update_experiment_from_temp_store()
        #
        # self.update_scale_from_sliders()
        # self.update_filter()
        # if not was_in_gt or not was_in_temp:
        #     self.guess_peaks()
        #     self.guess_threshold()
        #     self.add_experiment_to_temp_store()
        #
        # else:
        #
        # self.update()

    def get_threshold_predictor(self):
        if self.control_widget.threshold_prediction.currentText() == "Suthakar-2019":
            return kirupa_thr_function

        elif self.control_widget.threshold_prediction.currentText() == "DeepABR":
            return deepABR_thr_function()

    def get_parser(self):
        if self.control_widget.choose_parser.currentText() == "EPL":
            return parse_abr_file

    def increase_thr(self):
        """when up key is pressed"""
        if not isinstance(self.experiment, ABRExperiment):
            return

        if self.experiment.all_above_threshold or self.experiment.all_below_threshold:
            self.experiment.all_below_threshold = False
            self.experiment.all_above_threshold = False

        has_thr = any(w.is_thr for w in self.experiment.waveforms)
        if not has_thr:  # no threshold set
            self.experiment.waveforms[-1].set_as_threshold()
            self.experiment.waveforms[-1].thr_set_by_user()

        else:
            for w in self.experiment.get_waveforms():
                w.set_thr_probability(None)

            for i, w in enumerate(self.experiment.get_waveforms()):
                if w.is_thr:
                    w.clear_threshold()
                    self.experiment.waveforms[max(i - 1, 0)].set_as_threshold()
                    self.experiment.waveforms[max(i - 1, 0)].thr_set_by_user()
                    break

        self.viewer.color_drag_widgets()
        self.update()

    def decrease_thr(self):
        """when down key is pressed"""
        if not isinstance(self.experiment, ABRExperiment):
            return

        if self.experiment.all_above_threshold or self.experiment.all_below_threshold:
            self.experiment.all_below_threshold = False
            self.experiment.all_above_threshold = False

        has_thr = any(w.is_thr for w in self.experiment.waveforms)
        if not has_thr:  # no threshold set
            self.experiment.waveforms[0].set_as_threshold()
            self.experiment.waveforms[0].thr_set_by_user()

        else:
            for w in self.experiment.get_waveforms():
                w.set_thr_probability(None)

            for i, w in enumerate(self.experiment.waveforms):
                if w.is_thr:
                    w.clear_threshold()
                    self.experiment.waveforms[
                        min(i + 1, len(self.experiment) - 1)
                    ].set_as_threshold()
                    self.experiment.waveforms[
                        min(i + 1, len(self.experiment) - 1)
                    ].thr_set_by_user()
                    break

        self.viewer.color_drag_widgets()
        self.update()

    def set_all_waveforms_above_threshold(self):
        if isinstance(self.experiment, ABRExperiment):
            self.experiment.set_all_above_thr()
        self.update()

    def set_all_waveforms_below_threshold(self):
        if isinstance(self.experiment, ABRExperiment):
            self.experiment.set_all_below_thr()
        self.update()

    def update_blinded_state(self):
        self.viewer.is_blinded = self.control_widget.blinder_checkbox.isChecked()
        if self.abr_files:
            text = self.abr_files[self.abr_file_index]
            if self.control_widget.blinder_checkbox.isChecked():
                self.footer_widget.setText("  ABR Filepath Blinded")
            else:
                self.footer_widget.setText("  " + text)
        self.viewer.update()

    def set_current_experiment(self, experiment: ABRExperiment):
        """sets the current experiment. Likely from a parser!"""
        self.experiment = experiment
        self.viewer.set_experiment(self.experiment)
        self.viewer.reset_drag_widget_positions()

        self.update()

    def guess_threshold(self):
        if not isinstance(self.experiment, ABRExperiment):
            return

        threshold_fn: Callable[
            [ABRExperiment], Tuple[float, str, str]
        ] = self.get_threshold_predictor()
        thr, prob, _ = threshold_fn(self.experiment)

        for w in self.experiment.get_waveforms():
            w.set_thr_probability(None)
            w.clear_threshold()

        if thr is not None:
            thr = round(thr / 10) * 10

            for i, w in enumerate(self.experiment.get_waveforms()):
                if w.level == thr:
                    w.set_as_threshold()

                if isinstance(prob, List):  # kirupa method returns a str...
                    w.set_thr_probability(prob[i])

                else:
                    w.set_thr_probability(
                        None
                    )  # if a probaiblity array wasnt give, the prob should be none

        self.experiment.set_threshold_prediction_method(
            self.control_widget.threshold_prediction.currentText()
        )

        if isinstance(prob, List):  # kirupa method returns a str...
            self.viewer.color_drag_widgets()
        self.update()

    def export_as_csv(self):
        raise NotImplementedError

    @Slot(QDockWidget)
    def updateDockSize(self, dock: QDockWidget):
        for dock_widget in self.tabifiedDockWidgets(dock):
            if dock_widget is not dock:
                dock_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        dock.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        width = dock.widget().sizeHint().width()
        self.resizeDocks([dock], [width], Qt.Orientation.Horizontal)

    def create_layout(self):
        """creates all widget layouts"""
        self.setDockNestingEnabled(True)


        self.left_dock = QDockWidget("Analyze", self)
        self.left_dock.setObjectName("left_dock")
        self.left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.left_dock.setTitleBarWidget(QWidget(None))
        self.left_dock.setWidget(self.control_widget)
        self.left_dock.setObjectName("ControlDock")
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        self.batch_info = QDockWidget("AGF Info", self)
        self.batch_info.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.batch_info.setTitleBarWidget(QWidget(None))
        # self.info_dock.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.batch_info.setContentsMargins(0, 0, 0, 0)
        self.batch_info.setWidget(self.abr_batch_info_table)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.batch_info)



        self.plot_dock = QDockWidget("AGF Plot", self)
        self.plot_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        # self.plot_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.plot_dock.setTitleBarWidget(QWidget(None))
        self.plot_dock.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.plot_dock.setContentsMargins(0, 0, 0, 0)
        self.plot_dock.setWidget(self.agf_plot_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.plot_dock)

        self.left_toolbar_dock = QDockWidget("Batch", self)
        self.left_toolbar_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.left_toolbar_dock.setTitleBarWidget(QWidget(None))
        self.left_toolbar_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.left_toolbar_dock.setContentsMargins(0, 0, 0, 0)
        self.left_toolbar_dock.setMinimumSize(QSize(1,1))
        self.left_toolbar_dock.setWidget(self.abr_batch_widget)
        self.left_toolbar_dock.setObjectName("BatchDock")
        self.addDockWidget(Qt.RightDockWidgetArea, self.left_toolbar_dock)

        self.info_dock= QDockWidget("AGF Info", self)
        self.info_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.info_dock.setTitleBarWidget(QWidget(None))
        # self.info_dock.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.info_dock.setContentsMargins(0, 0, 0, 0)
        self.info_dock.setWidget(self.agf_info_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)

        toolbar = QToolBar()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setFixedHeight(2)
        toolbar.setStyleSheet("border-bottom: 2px solid black;")
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        # self.tabifyDockWidget(self.left_dock, self.left_toolbar_dock)

        self.footer_toolbar = QToolBar("Dockable", self)
        # self.footer_toolbar.setFeatures(QDockWidget.NoDockWidgetFeatures)
        # self.footer_toolbar.setTitleBarWidget(QWidget(None))
        # self.footer_toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.footer_toolbar.setContentsMargins(0, 0, 0, 0)
        # self.footer_toolbar.setWidget(self.footer_widget)
        self.footer_toolbar.addWidget(self.footer_widget)
        self.addToolBar(Qt.BottomToolBarArea, self.footer_toolbar)


        # self.updateDockSize(self.left_toolbar_dock)

        self.left_dock.raise_()  # sets the tab activated...

    def link_slots_and_signals(self):
        self.tabifiedDockWidgetActivated.connect(self.updateDockSize)

        """links all widget slots and associated signals... Effectivley the control flow"""
        self.control_widget.increase_thr_button.clicked.connect(self.increase_thr)
        self.control_widget.decrease_thr_button.clicked.connect(self.decrease_thr)
        self.control_widget.next_file_button.clicked.connect(self.next_abr)
        self.control_widget.previous_file_button.clicked.connect(self.previous_abr)
        self.control_widget.open_folder_button.clicked.connect(self.open_folder)
        self.control_widget.set_all_below_thr_button.clicked.connect(
            self.set_all_waveforms_below_threshold
        )
        self.control_widget.set_all_above_thr_button.clicked.connect(
            self.set_all_waveforms_above_threshold
        )
        self.control_widget.export_as_csv_button.clicked.connect(self.export_as_csv)

        self.control_widget.reset_button.clicked.connect(
            self.viewer.reset_drag_widget_positions
        )
        self.control_widget.reset_button.clicked.connect(self.update)

        # self.control_widget.blinder_checkbox.clicked.connect(self.emit_abr_file_changed)
        self.control_widget.blinder_checkbox.clicked.connect(self.update_blinded_state)
        self.control_widget.blinder_checkbox.clicked.connect(self.viewer.update)

        self.control_widget.filterParamsChanged.connect(self.on_abr_change)

        self.abrFileChanged.connect(self.footer_widget.setText)
        self.abrFileChanged.connect(self.on_abr_change)

        self.control_widget.xSliderChanged.connect(self.viewer.set_x_scale)
        self.control_widget.xSliderChanged.connect(self.viewer.reset_peak_widgets)
        self.control_widget.xSliderChanged.connect(self.update)
        self.control_widget.ySliderChanged.connect(self.viewer.set_y_scale)
        self.control_widget.ySliderChanged.connect(self.viewer.reset_peak_widgets)
        self.control_widget.ySliderChanged.connect(self.update)

        self.viewer.pointChanged.connect(self.add_experiment_to_temp_store)
        self.viewer.pointChanged.connect(self.agf_plot_widget.update)
        self.viewer.pointChanged.connect(self.agf_info_widget.update)

    def test_hotkey(self):
        print("triggered!")
        raise NotImplementedError

    def create_actions(self):
        self.increaseYscale = QAction(
            "Increase Y Scale",
            self,
            shortcut="Ctrl+Shift+Up",
            enabled=True,
            triggered=self.increase_y_slider_by_constant,
        )

        self.decreaseYscale = QAction(
            "Decrease Y Scale",
            self,
            shortcut="Ctrl+Shift+Down",
            enabled=True,
            triggered=self.decrease_y_slider_by_constant,
        )

        self.decreaseXscale = QAction(
            "Decrease X Scale",
            self,
            shortcut="Ctrl+Shift+Left",
            enabled=True,
            triggered=self.decrease_x_slider_by_constant,
        )

        self.increaseXscale = QAction(
            "Increase X Scale",
            self,
            shortcut="Ctrl+Shift+Right",
            enabled=True,
            triggered=self.increase_x_slider_by_constant,
        )

        self.reset_action = QAction(
            "Reset Viewer",
            self,
            shortcut="Ctrl+R",
            triggered=self.control_widget.reset_button.click,
        )

        self.open_folder_action = QAction(
            "Open Folder",
            self,
            shortcut="Ctrl+O",
            triggered=self.control_widget.open_folder_button.click,
        )

        self.save_action = QAction(
            "Save", self, shortcut="Ctrl+S", triggered=self.test_hotkey
        )

        self.save_as_action = QAction(
            "Save As", self, shortcut="Ctrl+Shift+S", triggered=self.test_hotkey
        )

        self.export_csv_action = QAction(
            "Export as CSV", self, shortcut="Ctrl+E", triggered=self.export_as_csv
        )

        self.train_deep_abr_predictor_action = QAction(
            "Train DeepABR Model", self, triggered=self.test_hotkey
        )

        self.eval_abr_threshold_precitor_action = QAction(
            "Predict ABR Threshold", self, triggered=self.guess_threshold
        )

        self.launch_train_widget_action = QAction(
            "Launch Training Wizard", self, triggered=self.test_hotkey
        )

        self.import_train_data_action = QAction(
            "Import Training Data", self, triggered=self.test_hotkey
        )

        self.launch_batch_widget_action = QAction(
            "Launch Batch Eval Wizard", self, triggered=self.test_hotkey
        )

        self.choose_deep_abr_model_file_action = QAction(
            "Choose DeepABR Model", self, triggered=self.test_hotkey
        )

        self.predict_peaks_action = QAction(
            "Predict ABR Peaks", self, triggered=self.guess_peaks
        )

        self.show_peaks_action = QAction(
            "Show Peaks", self, checkable=True, triggered=self.test_hotkey
        )
        self.show_peaks_action.setChecked(True)

    def create_menus(self):
        self.file_menu = QMenu("&File", self)
        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addAction(self.export_csv_action)

        self.view_menu = QMenu("&View", self)
        self.view_menu.addAction(self.increaseYscale)
        self.view_menu.addAction(self.decreaseYscale)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.increaseXscale)
        self.view_menu.addAction(self.decreaseXscale)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.show_peaks_action)
        self.view_menu.addSeparator()
        self.view_menu.addAction(self.reset_action)

        self.predict_menu = QMenu("&Predict", self)
        self.predict_menu.addAction(self.eval_abr_threshold_precitor_action)
        self.predict_menu.addAction(self.predict_peaks_action)
        self.predict_menu.addSeparator()
        self.predict_menu.addAction(self.launch_train_widget_action)
        self.predict_menu.addAction(self.launch_batch_widget_action)
        self.predict_menu.addSeparator()
        self.predict_menu.addAction(self.train_deep_abr_predictor_action)
        self.predict_menu.addAction(self.choose_deep_abr_model_file_action)
        self.predict_menu.addSeparator()

        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.view_menu)
        self.menuBar().addMenu(self.predict_menu)


if __name__ == "__main__":
    app = QApplication()
    app.setApplicationName("hcat")
    icon = QIcon(":/resources/icon.ico")
    app.setWindowIcon(icon)
    app.setApplicationDisplayName("hcat")
    app.setApplicationVersion("v2023.06.02")
    app.setOrganizationName("Buswinka CG LCC")

    QFontDatabase.addApplicationFont(":/fonts/OfficeCodePro-Bold.ttf")
    QFontDatabase.addApplicationFont(":/fonts/OfficeCodePro-Regular.ttf")
    QFontDatabase.addApplicationFont(":/fonts/OfficeCodePro-Light.ttf")
    QFontDatabase.addApplicationFont(":/fonts/OfficeCodePro-Medium.ttf")
    QFontDatabase.addApplicationFont(":/fonts/Inconsolata-Black.ttf")
    QFontDatabase.addApplicationFont(":/fonts/Inconsolata-Regular.ttf")

    font_weight = 12 if QSysInfo.productType() == "macos" else 8

    font = QFont("Office Code Pro", font_weight)

    w = ABRAnalyzer()
    w.show()
    w.find_abr_files_from_filepath(
        "/Users/chrisbuswinka/Documents/Projects/hcat/test_abr_data"
    )
    app.exec()
