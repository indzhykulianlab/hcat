import os.path
import warnings

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.gui.colors import RED, BLUE, GREEN
from hcat.widgets.push_button import WPushButton


class NamedSliderSpinBox(QWidget):
    valueChanged = Signal(float)
    def __init__(self, name, min_val, max_val, step, init_val):
        super(NamedSliderSpinBox, self).__init__()

        spin_style = """
            QDoubleSpinBox {
                 padding-right: 3px; /* make room for the arrows */
                 border-width: 0px;
                 background-color: rgba(0,0,0,0);
                 margin: 0px 0px 0px 0px;
             }

        """

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(init_val)
        self.spinbox.setMinimumWidth(50)
        self.spinbox.setStyleSheet(spin_style)

        self.slider =QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val*100)
        self.slider.setMaximum(max_val*100)
        self.slider.setValue(init_val * 100)
        self.slider.setTickInterval(step*100)
        # self.slider.setTickPosition(QSlider.TicksBelow)

        #signals

        self.slider.valueChanged.connect(self.update_spinbox_from_slider)
        self.slider.valueChanged.connect(self.emit_val_changed)
        self.spinbox.valueChanged.connect(self.update_slider_from_spinbox)
        self.slider.valueChanged.connect(self.emit_val_changed)

        group_layout = QHBoxLayout()
        group_layout.addWidget(self.slider, alignment=Qt.AlignTop)
        group_layout.addWidget(self.spinbox, alignment=Qt.AlignCenter)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)

        group = QGroupBox(name)
        group.setContentsMargins(0, 0, 0, 0)
        group.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.setLayout(layout)
        # self.setStyleSheet(style)

    def update_slider_from_spinbox(self):
        """ changes slider from value of the spinbox """
        self.diable_signals()
        value = self.spinbox.value()
        self.slider.setValue(value*100)
        self.enalbe_signals()

    def update_spinbox_from_slider(self):
        """ changes spinbox from value of the slider """
        self.diable_signals()
        value = self.slider.value()
        self.spinbox.setValue(value/100)
        self.enalbe_signals()

    def diable_signals(self):
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)

    def enalbe_signals(self):
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)

    def emit_val_changed(self):
        self.valueChanged.emit(self.slider.value)

    def value(self):
        return self.spinbox.value()

    def setValue(self, value):
        self.spinbox.setValue(value / 100)
        self.slider.setValue(value)

class TextEdit(QTextEdit):
    def __init__(self, parent = None):
        super(TextEdit, self).__init__(parent)
        self.text_set = False

    def paintEvent(self, event):
        if not self.text_set:
            painter = QPainter(self.viewport())

            red_pen = QPen()
            red_pen.setColor(QColor(255, 0, 0, 60))
            red_pen.setWidth(15)

            white_pen = QPen()
            white_pen.setColor(QColor(255, 255, 255, 255))
            white_pen.setWidth(15)

            _x = -50
            for i in range(20):
                painter.setPen(red_pen)
                painter.drawLine(QLineF(_x, 0, 200+_x, 200))
                _x += 20

                painter.setPen(white_pen)
                painter.drawLine(QLineF(_x, 0, 200+_x, 200))

                _x += 20

        super(TextEdit, self).paintEvent(event)

class ModelFileSelector(QWidget):
    modelFileSelected = Signal(str)
    def __init__(self):
        super().__init__()
        self.file_path = None

        style = f"""
        QPushButton {{
            background-color: white;
            margin: 0px;
            padding: 0px;
            font-size: 12pt;
            font: bold;
            border-style: inset;
            border-width: 1px 1px 1px 1px; 
            border-color: black black black black; 
            background-color: white;
            }}
        QPushButton:pressed {{
        background-color: grey;
            border-style: inset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 2px;
            }}
        """
        self.setStyleSheet(style)

        # Create the button and label widgets
        self.button = QPushButton('SELECT MODEL', clicked=self.select_file)
        self.label = TextEdit('NOT SET')
        self.label.setReadOnly(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.label.setFixedHeight(30)
        self.label.setStyleSheet("""
        QTextEdit {
            margin: -1px 0px 0px 0px; 
            background-color: rgba(0,0,0,0);
            font: bold 16px;
            border: 1px solid black;
        }
        """)

        # Create a layout to contain the button and label
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        layout.setSpacing(0)
        layout.setContentsMargins(0,10,0,0)
        layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Set the fixed size of the widget
        # self.setFixedSize(QSize(250, 70))

    def set_file(self, file_path: str):
        _justfile = os.path.split(file_path)[-1]
        self.label.setText(_justfile)
        self.label.setFixedHeight(60)
        self.label.text_set = True
        self.label.setAlignment(Qt.AlignLeft)
        self.label.setStyleSheet("""
        QTextEdit {
            background-color: rgba(0,0,0,0);
            font: 12px;
        }
        """)
        self.label.update()
        self.modelFileSelected.emit(str(self.file_path))

    def remove_file(self):
        self.label.setText('NOT SET')
        self.label.text_set = False
        self.label.setReadOnly(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.label.setFixedHeight(30)
        self.label.setStyleSheet("""
        QTextEdit {
            margin: -1px 0px 0px 0px; 
            background-color: rgba(0,0,0,0);
            font: bold 16px;
            border: 1px solid black;
        }
        """)

    def select_file(self):
        # Open a file dialog to choose a file
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select File')

        if file_path:
            # Update the label with the selected file path
            if not self.validate_model_file(self.file_path):
                self.file_path = None
                return

            self.file_path = file_path
            self.set_file(file_path)
        else:
            # No file selected, update the label to display 'None'
            self.file_path = None
            self.label.text_set = False
            self.label.setText('NOT SET')
            self.label.setFixedHeight(30)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setStyleSheet("""
            QTextEdit {
                margin: -1px 0px 0px 0px; 
                background-color: rgba(0,0,0,0);
                font: bold 16px;
                border: 1px solid black;
            }
            """)
            self.label.update()

    def validate_model_file(self, file):
        warnings.warn('Validation of model files not implemented')
        return True

class EvalWidget(QWidget):
    thresholdSliderChanged = Signal(tuple)
    modelFileSelected = Signal(str)
    defaultModelSelected = Signal()
    modelDeletedSignal = Signal()
    liveUpdate = Signal(bool)
    runSnapshot = Signal()
    runFullAnalysis = Signal()
    setEvalChannels = Signal(list)

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Create widgets for the CELL_THRESHOLD and NMS_THRESHOLD options
        self.cell_threshold_spinbox = NamedSliderSpinBox('CELL THRESHOLD', 0, 1, 0.01, self.config.EVAL.CELL_THRESHOLD)
        self.nms_threshold_spinbox = NamedSliderSpinBox('NMS THRESHOLD', 0, 1, 0.01, self.config.EVAL.NMS_THRESHOLD)

        # Create checkboxes for the USE_RED, USE_GREEN, and USE_BLUE options
        self.channel_group = QGroupBox('Analysis Channels')

        self.use_red_checkbox = QCheckBox("RED")
        self.use_red_checkbox.setChecked(self.config.EVAL.USE_RED)
        self.use_red_checkbox.setStyleSheet(self._get_checkbox_style(RED))

        self.use_green_checkbox = QCheckBox("GREEN")
        self.use_green_checkbox.setChecked(self.config.EVAL.USE_GREEN)
        self.use_green_checkbox.setStyleSheet(self._get_checkbox_style(GREEN))

        self.use_blue_checkbox = QCheckBox("BLUE")
        self.use_blue_checkbox.setChecked(self.config.EVAL.USE_BLUE)
        self.use_blue_checkbox.setStyleSheet(self._get_checkbox_style(BLUE))

        # Create a checkbox for the LIVE_UPDATE option
        self.live_update_checkbox = QCheckBox("Live Update")
        self.live_update_checkbox.setChecked(self.config.EVAL.LIVE_UPDATE)

        # Create a checkbox for the QUICK_EVAL option
        self.quick_eval_checkbox = QCheckBox("Quick Eval")
        self.quick_eval_checkbox.setChecked(self.config.EVAL.QUICK_EVAL)

        # Create a widget to select a model
        self.model_select = ModelFileSelector()
        self.model_select.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.model_select.modelFileSelected.connect(self.emitModelFileSelected)

        self.clear_model_button = WPushButton('DEL MDL')
        self.clear_model_button.setWarningButton(True)
        self.choose_defualt_model_button = WPushButton('DEFUALT')


        self.quick_eval = WPushButton('SNAPSHOT')
        self.run_detection = WPushButton('RUN ANALYSIS')
        self.assign_freq_button = WPushButton('ASSIGN FREQ')
        self.set_gt = WPushButton('SET ALL GT')
        self.set_gt.setWarningButton(True)

        self._create_layouts()
        self._link_signals()

    def emitModelFileSelected(self, x):
        self.modelFileSelected.emit(str(x))

    def _link_signals(self):

        self.clear_model_button.clicked.connect(self.emitModelDeletedSignal)
        self.quick_eval.clicked.connect(self.emitRunSnapshot)
        self.run_detection.clicked.connect(self.emitRunFullAnalysis)
        self.choose_defualt_model_button.clicked.connect(self.emitDefaultModelSelected)
        self.live_update_checkbox.clicked.connect(self.emit_liveUpdate)
        self.cell_threshold_spinbox.valueChanged.connect(self.emit_thresholdSliderChanged)
        self.nms_threshold_spinbox.valueChanged.connect(self.emit_thresholdSliderChanged)

    def _create_layouts(self):
        color_checkbox_layout = QVBoxLayout()
        color_checkbox_layout.addWidget(self.use_red_checkbox)
        color_checkbox_layout.addWidget(self.use_green_checkbox)
        color_checkbox_layout.addWidget(self.use_blue_checkbox)
        color_checkbox_layout.setContentsMargins(0,0,0,0)
        color_checkbox_layout.setSpacing(0)

        self.channel_group.setLayout(color_checkbox_layout)
        # channel_group.setContentsMargins(0,0,0,0)

        run_layout = QVBoxLayout()
        run_layout.addWidget(self.quick_eval)
        run_layout.addWidget(self.assign_freq_button)
        run_layout.addWidget(self.set_gt)
        run_layout.addWidget(self.run_detection)
        run_layout.setSpacing(5)
        run_layout.setContentsMargins(0,0,0,0)


        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(self.choose_defualt_model_button)
        model_button_layout.addWidget(self.clear_model_button)
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setSpacing(2)

        # Create layouts for the spinboxes and checkboxes
        threshold_layout = QVBoxLayout()
        threshold_layout.addWidget(self.cell_threshold_spinbox)
        threshold_layout.addWidget(self.nms_threshold_spinbox)

        # Create a layout for the widget
        layout = QVBoxLayout()
        layout.addWidget(self.model_select)
        layout.addLayout(model_button_layout)
        layout.addLayout(threshold_layout)
        layout.addWidget(self.channel_group)
        # layout.addWidget(self.live_update_checkbox)
        layout.addStretch(10000)
        layout.addLayout(run_layout)
        layout.setAlignment(Qt.AlignTop)

        self.setLayout(layout)

    def emitDefaultModelSelected(self):
        print('emitted default model selected')
        self.defaultModelSelected.emit()

    def emitRunSnapshot(self):
        print('emitted run selected ')
        self.runSnapshot.emit()

    def emitRunFullAnalysis(self):
        print('emitted run full analysis')
        self.runFullAnalysis.emit()

    def emitModelDeletedSignal(self):
        print('emitted model deleted signal ')
        self.modelDeletedSignal.emit()

    def get_model_filename(self):
        return self.model_select.file_path


    def emit_liveUpdate(self):
        self.liveUpdate.emit(self.live_update_checkbox.checkState() == Qt.CheckState.Checked)

    def get_use_channel_state(self):
        _use_channel = []
        for v in [
            self.use_red_checkbox,
            self.use_green_checkbox,
            self.use_blue_checkbox
        ]:
            _use_channel.append(not v.checkState() == Qt.CheckState.Checked)
        return _use_channel

    def setModel(self, file_path: str):
        print('SET MODEL TO: ', file_path)
        self.model_select.blockSignals(True)
        self.model_select.set_file(file_path)
        self.model_select.blockSignals(False)

    def deleteModel(self):
        self.model_select.remove_file()

    def emit_thresholdSliderChanged(self):
        self.thresholdSliderChanged.emit((
            self.nms_threshold_spinbox.value(), self.cell_threshold_spinbox.value()
        ))

    def _get_checkbox_style(self, color: QColor):
        red, green, blue = color.red(), color.green(), color.blue()
        color_str = f'rgb({red}, {green}, {blue})'
        style = f"""
        QCheckBox{{
            font: 12px; 
        }}
        QCheckBox::indicator{{
            width: 10px;
            height: 10px;
            background-color: rgba(0,0,0,0); 
            border: 1px solid black;
            border-radius: 0px;
            margin: 3px, 3px, 0px, 0px;
        }}
        QCheckBox::indicator::checked{{
            width: 10px;
            height: 10px;
            background-color: {color_str}; 
            border: 1px solid black;
            border-radius: 0px;
            margin: 3px, 3px, 0px, 0px;
            }}
        """
        return style

    def _get_button_style(self, color):
        style = f"""
        QPushButton {{
            background-color: white;
            margin: 1px;
            padding: 0px;
            font-size: 12pt;
            font: bold;
            border-style: inset;
            border-width: 1px 1px 1px 1px; 
            border-color: black black black black; 
            background-color: white;
            }}
        QPushButton:pressed {{
        background-color: grey;
            border-style: inset;
            border-width: 1px;
            border-color: black;
            margin: 1px;
            padding: 2px;
            }}
        """
        return style

    def setThresholdValues(self, cell: float, nms: float):

        self.cell_threshold_spinbox.setValue(cell)
        self.nms_threshold_spinbox.setValue(nms)
        self.update()

if __name__ == '__main__':
    from hcat.config.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    app = QApplication()
    w = EvalWidget(cfg)
    # w = ModelFileSelector()
    w.show()
    app.exec()
