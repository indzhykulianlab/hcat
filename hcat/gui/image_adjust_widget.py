from typing import Tuple, List, Dict

import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.gui.histogram_widget import HistogramWidget
from hcat.state import *
from hcat.widgets.push_button import WPushButton
import hcat.gui.resources


class SliderWithLabel(QWidget):
    valueChanged = Signal(int)
    def __init__(self, name, min_val, max_val, init_val, tick_interval):
        super(SliderWithLabel, self).__init__()

        font = QFont(':/fonts/OfficeCodePro-Bold.ttf')
        fixedFont = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        self.setFont(fixedFont)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        self.slider.setTickInterval(tick_interval)
        self.slider.valueChanged.connect(self.emit_val_changed)
        self.slider.setTickPosition(QSlider.TicksBelow)
        # self.slider.setMinimumHeight(20)

        policy = self.slider.sizePolicy()
        policy.setHorizontalStretch(1)
        self.slider.setSizePolicy(policy)

        group_layout = QVBoxLayout()
        group_layout.addWidget(self.slider)
        group_layout.setContentsMargins(0,0,0,0)
        group_layout.setSpacing(0)

        group = QGroupBox(name)
        group.setContentsMargins(0,0,0,50)
        # group.setMinimumHeight(20)
        group.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(group)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        self.setStyleSheet("""
        QGroupBox {
            background-color: none;
            border: 0px solid black;
            margin-top: 10px; /* leave space at the top for the title */
            margin-bottom: 0px;
            padding-bottom:-10px;
            font-size: 10px;
            border-radius: 5px;
            }

        QGroupBox::title {
            subcontrol-origin: margin;
            padding: 0 3px;
        }

        """)
        layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.setLayout(layout)

    def value(self):
        return self.slider.value()

    def set_slider_value(self, val):
        self.slider.setValue(val)
        self.update()

    def emit_val_changed(self):
        self.valueChanged.emit(self.slider.value)


class ImageAdjustmentWidget(QWidget):
    adjustment_changed = Signal(list)
    def __init__(self):
        super().__init__()
        """
        Create a simplistic widget for controling sliders for image adjustment...
        
        """

        self.red = {'channel': 0, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
        self.green = {'channel': 1, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
        self.blue = {'channel': 2, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
        """
        QPushButton {border-width: 1px 3px 1px 1px; border-color: black rgb(246, 122, 0) black black;}
        """
        button_style = lambda color: f"""
                QPushButton {{
                    border-style: outset;
                    border-width: 1px;
                    border-color: black;
                    font: bold;
                    }}
                QPushButton:pressed {{
                    border-style: outset;
                    border-width: 2px;
                    border-color: black;
                    }}
                """

        self.modified_color_channels = []

        self.min_slider = SliderWithLabel('MIN', 0, 255, 0, 10) # Note min/max is different for max slider
        self.min_slider.valueChanged.connect(self._validate_max_slider)
        self.min_slider.valueChanged.connect(self.adjust_min_max)
        self.min_slider.valueChanged.connect(self.update_lines)

        self.max_slider = SliderWithLabel('MAX', 1, 256, 255, 10)
        self.max_slider.valueChanged.connect(self._validate_min_slider)
        self.max_slider.valueChanged.connect(self.adjust_min_max)
        self.max_slider.valueChanged.connect(self.update_lines)

        self.brightness_slider = SliderWithLabel('BRIGHTNESS', -255, 255, 0, 512//(255/10))
        self.brightness_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.brightness_slider.valueChanged.connect(self.update_lines)

        self.contrast_slider = SliderWithLabel('CONTRAST', -200, 200, 0,  512//(255/10))
        self.contrast_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self.update_lines)


        self.channel_label = QLabel('Channel')
        # self.channel_label.setMinimumSize(QSize(100, 20))
        # self.channel_label.setAlignment(Qt.AlignCenter)
        self.channel_picker = QComboBox()
        self.channel_picker.addItems(['Red', 'Green', 'Blue'])
        self.channel_picker.setCurrentIndex(0)
        self.channel_picker.currentIndexChanged.connect(self.update_slider_and_label)


        self.reset = WPushButton('RESET')
        self.reset.clicked.connect(self.reset_channel)

        self.live_update = QCheckBox('INSTANT APPLY')
        self.live_update.clicked.connect(self.apply_adjustment)
        self.log_scale_checkbox = QCheckBox('LOG SCALE')
        self.log_scale_checkbox.clicked.connect(self.update_histogram_scaling)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.live_update)
        checkbox_layout.addWidget(self.log_scale_checkbox)
        checkbox_layout.addStretch(1)
        checkbox_layout.setContentsMargins(0,0,5,0)
        checkbox_layout.setSpacing(10)


        self.apply_adjust_button = WPushButton('APPLY')
        self.apply_adjust_button.clicked.connect(self.apply_adjustment)

        self.reset_all_button = WPushButton('RESET ALL')
        self.reset_all_button.setWarningButton(True)
        self.reset_all_button.clicked.connect(self.reset_all)

        # Create layout for sliders and labels
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.min_slider, alignment=Qt.AlignTop)
        slider_layout.addWidget(self.max_slider, alignment=Qt.AlignTop)
        slider_layout.addWidget(self.brightness_slider, alignment=Qt.AlignTop)
        slider_layout.addWidget(self.contrast_slider, alignment=Qt.AlignTop)
        slider_layout.setSpacing(0)
        slider_layout.setContentsMargins(0,0,0,0)

        choose_channel_layout = QHBoxLayout()
        choose_channel_layout.addWidget(self.channel_label, alignment=Qt.AlignLeft)
        choose_channel_layout.addWidget(self.channel_picker, alignment=Qt.AlignLeft)
        choose_channel_layout.addStretch(1)
        choose_channel_layout.setSpacing(2)
        choose_channel_layout.setContentsMargins(1,1,1,1)

        # hcat histogram
        self.histogram_widget = HistogramWidget(None, None)
        self.histogram_widget.oneColorVisible.connect(self.set_color_spinbox)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_adjust_button)
        button_layout.addWidget(self.reset)
        button_layout.addWidget(self.reset_all_button)
        button_layout.setContentsMargins(0,0,0,0)
        button_layout.setSpacing(2)

        layout = QVBoxLayout()
        layout.addWidget(self.histogram_widget)
        layout.addLayout(choose_channel_layout)
        layout.addLayout(slider_layout)
        layout.addStretch(10)
        # layout.addWidget(self.live_update)
        layout.addLayout(checkbox_layout)
        layout.addLayout(button_layout)
        layout.setContentsMargins(4,10,4,5)
        layout.setSpacing(3)

        # Create group box and set layout
        group_box = QGroupBox('IMAGE ADJUSTMENTS')
        group_box.setLayout(layout)
        group_box.setStyleSheet("""
        QGroupBox {
            background-color: none;
            border: 0px solid black; 
            /* margin-top: 3ex;  leave space at the top for the title */
            margin: 1ex 0 0 0;
            font-size: 10px;
            border-radius: 0px; 
            }

        QGroupBox::title {
            subcontrol-origin: margin;
            padding: 0 5px;
        }
        
        """)

        # Create grid layout and add group box
        grid_layout = QGridLayout()
        grid_layout.addWidget(group_box)
        grid_layout.setContentsMargins(1,5,1,1)

        self.setLayout(grid_layout)
        self.setStyleSheet(self._get_checkbox_style('gray'))

    def resetHistogramAndSliders(self):
        self.reset_all()
        self.histogram_widget.set_hist_to_none()  # sets to none
        self.histogram_widget.resetColorSelectors()
        self.update()


    @Slot(int)
    def set_color_spinbox(self, index):
        self.channel_picker.setCurrentIndex(index)
        self.update()

    def update_histogram_scaling(self):
        self.histogram_widget.set_log_scale(self.log_scale_checkbox.checkState() == Qt.CheckState.Checked)

    def generate_histograms(self, histograms: List[Dict[str, np.array]]):
        """
        Redraws the histogram for the histogram widget

        :param histograms:
        :type histograms:
        :return:
        :rtype:
        """

        vals = [h['values'] for h in histograms]
        pos = [h['positions'] for h in histograms]

        self.histogram_widget.set_hist_vals_positions(vals, pos)

    def _min_max(self) -> Tuple[float, float]:
        """
        Given a contrast an brightness, calculate a minimum and maximum
        value which maps to zero and max brightness

        used for drawing vertical lines in the histogram widget.

        contrast = 10 ** (contrast / 100) # slider goes between -20 and 200
        # brightness = brightness + 128
        midpoint = (128.0 - brightness) / contrast
        max_val = midpoint + brightness + 128
        min_val = -midpoint + brightness + 128

        """
        contrast = 10 ** (self.contrast_slider.value() / 200) # slider goes between -200 and 200
        brightness = self.brightness_slider.value()
        midpoint = (128.0 - brightness)
        max_val = midpoint + (128 / contrast)
        min_val = midpoint - (128 / contrast)
        return min_val, max_val

    def _brightnes_contrast(self):
        min_val = self.min_slider.value()
        max_val = self.max_slider.value()

        b = 128 - (max_val + min_val)/ 2
        c = np.log10(256 / (max_val - min_val)) * 200  # Reversing the calculation of 'c'
        return b, c

    def update_lines(self):
        """ updates the min/max vertical lines """
        current_val = self.channel_picker.currentText().lower()
        index_map = {'red': 0, 'green': 1, 'blue': 2}

        min_val, max_val = self._min_max()
        self.histogram_widget.set_min_max(min_val, max_val, c=index_map[current_val])
        self.histogram_widget.update()


    def _validate_min_slider(self):
        """ called by max_slider! """
        self._disable_slider_signals(True)
        _min = self.min_slider.value()
        _max = self.max_slider.value()
        if _min >= _max:
            _min = _max - 1
            self.min_slider.set_slider_value(_min)
            self.min_slider.update()
            self.min_slider.repaint()
        self._disable_slider_signals(False)

    def _validate_max_slider(self):
        """ called by min_slider! """
        self._disable_slider_signals(True)
        _min = self.min_slider.value()
        _max = self.max_slider.value()
        if _max <= _min:
            _max = _min + 1
            self.max_slider.set_slider_value(_max)
            self.max_slider.update()
            self.max_slider.repaint()
        self._disable_slider_signals(False)

    def adjust_min_max(self):
        self._disable_slider_signals(True)

        brightness, contrast = self._brightnes_contrast()

        self.brightness_slider.set_slider_value(brightness)
        self.contrast_slider.set_slider_value(contrast)

        self._adjust_bright_contrast(brightness, contrast)
        self._disable_slider_signals(False)

    def adjust_brightness_contrast(self):
        """
        this runs when a slider is changed.
        Updates the rgb values and populates the QLabel with correct text.
        When live update is checked, will emit the signal: self.adjustment_changed.emit([self.red, self.green, self.blue])
        """

        # Get current slider values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        min_val, max_val = self._min_max()

        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.set_slider_value(min_val)
        self.max_slider.set_slider_value(max_val)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)

        self._adjust_bright_contrast(brightness, contrast)

    def _adjust_bright_contrast(self, brightness, contrast):
        LIVE = self.live_update.checkState() == Qt.CheckState.Checked

        if self.channel_picker.currentText()=='Red':
            self.red['brightness'] = int(brightness)
            self.red['contrast'] = contrast
            if 0 not in self.modified_color_channels:
                self.modified_color_channels.append(0)

            if LIVE:
                self.adjustment_changed.emit([self.red])

        elif self.channel_picker.currentText()=='Green':
            self.green['brightness'] = int(brightness)
            self.green['contrast'] = contrast
            if 1 not in self.modified_color_channels:
                self.modified_color_channels.append(1)

            if LIVE:
                self.adjustment_changed.emit([self.green])

        elif self.channel_picker.currentText()=='Blue':
            self.blue['brightness'] = int(brightness)
            self.blue['contrast'] = contrast
            if 2 not in self.modified_color_channels:
                self.modified_color_channels.append(2)

            if LIVE:
                self.adjustment_changed.emit([self.blue])

        if LIVE:
            self.modified_color_channels = []

    def apply_adjustment(self):
        """ emits a signal to apply adjustment """
        to_emit = []
        if 0 in self.modified_color_channels:
            to_emit.append(self.red)
        if 1 in self.modified_color_channels:
            to_emit.append(self.green)
        if 2 in self.modified_color_channels:
            to_emit.append(self.blue)
        self.adjustment_changed.emit(to_emit)
        self.modified_color_channels = []

    def get_current_adjustments(self):
        return [self.red, self.green, self.blue]

    def update_slider_and_label(self):
        """ update the slides and labels when the user chooses a different channel """

        if self.channel_picker.currentText() == 'Red':
            brightness, contrast = [self.red[v] for v in ['brightness', 'contrast']]
        elif self.channel_picker.currentText() == 'Green':
            brightness, contrast = [self.green[v] for v in ['brightness', 'contrast']]
        elif self.channel_picker.currentText() == 'Blue':
            brightness, contrast = [self.blue[v] for v in ['brightness', 'contrast']]

        self._disable_slider_signals(True)
        self.brightness_slider.set_slider_value(brightness)
        self.contrast_slider.set_slider_value(contrast)
        _min, _max = self._min_max()
        self.max_slider.set_slider_value(_max)
        self.min_slider.set_slider_value(_min)
        self._disable_slider_signals(False)

        self.update_lines()

    def _disable_slider_signals(self, val):
        self.brightness_slider.blockSignals(val)
        self.contrast_slider.blockSignals(val)
        self.min_slider.blockSignals(val)
        self.max_slider.blockSignals(val)

    def reset_channel(self):
        """ resets current channel to default """
        if self.channel_picker.currentText() == 'Red':
            self.red = {'channel': 0, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
            self.modified_color_channels.append(0)
        elif self.channel_picker.currentText() == 'Green':
            self.green = {'channel': 1, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
            self.modified_color_channels.append(1)
        elif self.channel_picker.currentText() == 'Blue':
            self.blue = {'channel': 2, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
            self.modified_color_channels.append(2)

        if self.live_update.checkState() == Qt.CheckState.Checked:
            self.apply_adjustment()

        self.update_slider_and_label()
        self.update_lines()

    def reset_all(self):
        """ resets all channels to default """
        self.red = {'channel': 0, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
        self.green = {'channel': 1, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}
        self.blue = {'channel': 2, 'brightness': 0, 'contrast': 0.0, 'saturation': 0}

        self.update_slider_and_label()

        index_map = {'red': 0, 'green': 1, 'blue': 2}

        for c in range(3):
            self.histogram_widget.set_min_max(0, 255, c=c)
        self.modified_color_channels = [0,1,2]
        self.apply_adjustment()
        self.histogram_widget.update()

    def set_sliders_from_piece(self, piece: Piece | None):
        if piece is None:
            return

        rgb = piece.adjustments

        self.red = rgb[0]
        self.green = rgb[1]
        self.blue = rgb[2]

        self.update_slider_and_label()
        self.update_lines()
        self.update()

    def _get_checkbox_style(self, color):
        style = f"""
        QCheckBox{{
            font: 10px; 
            margin-right: 15px;
        }}
        QCheckBox::indicator{{
            width: 10px;
            height: 10px;
            background-color: rgba(255,255,255,0); 
            border: 1px solid black;
            border-radius: 0px;
        }}
        QCheckBox::indicator::checked{{
            width: 10px;
            height: 10px;
            background-color: {color}; 
            border: 1px solid black;
            border-radius: 0px;
            }}
        """
        return style


if __name__ == '__main__':
    app = QApplication()
    w = ImageAdjustmentWidget()
    w.show()
    app.exec()
