from copy import copy

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import hcat.lib.utils
from hcat.lib.utils import hex_to_qcolor
from hcat.widgets.push_button import WPushButton


class RenderStyleWidget(QWidget):
    renderStyleChanged = Signal(dict)

    def __init__(self):
        super(RenderStyleWidget, self).__init__()

        layout = QGridLayout()
        style = """
            QSpinBox {
                 padding-right: 3px; /* make room for the arrows */
                 border-width: 0px;
                 background-color: rgba(0,0,0,0);
                 margin: 0px 0px 0px 0px;
             }

        """
        self._default_style = {'ihc_color': '#ff00ff',
                               'ohc_color': '#ffff00',
                               'cell_transparency': 50,
                               'eval_region_color': '#c8c8c8',
                               'eval_region_stroke': 1,
                               'eval_region_transparency': 100,
                               'ihc_stroke': 1,
                               'ohc_stroke': 1,
                               'ihc_transparency': 50,
                               'ohc_transparency': 50,
                               'basal_point_color': hcat.lib.utils.qcolor_to_hex(QColor(55, 126, 184)),
                               'apex_point_color': hcat.lib.utils.qcolor_to_hex(QColor(228, 26, 27)),
                               'freq_line_stroke': 2,
                               'freq_line_color': hcat.lib.utils.qcolor_to_hex(QColor(255, 255, 255, 255)),
                               }

        self.style_dict = copy(self._default_style)
        self._button_side = 15

        # IHC Line
        self.ihc_color = self._prepare_button(WPushButton("", self), QColor(255,0,255,128))
        self.ihc_transparency = self._prepare_spinbox(QSpinBox(), 0, 100, 50, style)
        self.ihc_stroke = self._prepare_spinbox(QSpinBox(), 0, 10, 1, style)

        # OHC Line
        self.ohc_color = self._prepare_button(WPushButton("", self), QColor(255,255,0,128))
        self.ohc_transparency = self._prepare_spinbox(QSpinBox(), 0, 100, 50, style)
        self.ohc_stroke = self._prepare_spinbox(QSpinBox(), 0, 10, 1, style)

        # add layout for first row
        layout.addWidget(QLabel('IHC→ '), 0, 0)
        layout.addWidget(self.ihc_color, 0, 1)
        layout.addWidget(QLabel('ALPHA→ '), 0, 3)
        layout.addWidget(self.ihc_transparency, 0, 4)
        layout.addWidget(QLabel('STROKE→ '), 0, 6)
        layout.addWidget(self.ihc_stroke, 0, 7)


        # add layout for second row
        layout.addWidget(QLabel('OHC→ '), 1, 0)
        layout.addWidget(self.ohc_color, 1, 1)
        layout.addWidget(QLabel('ALPHA→ '), 1, 3)
        layout.addWidget(self.ohc_transparency, 1, 4)
        layout.addWidget(QLabel('STROKE→ '), 1, 6)
        layout.addWidget(self.ohc_stroke, 1, 7)
        layout.setColumnStretch(8, 100)

        # set some spacing
        layout.setColumnMinimumWidth(2, 5)
        layout.setColumnMinimumWidth(5, 5)
        layout.setHorizontalSpacing(1)
        layout.setVerticalSpacing(10)
        layout.setContentsMargins(0,1,0,1)

        self._warning = False
        self._error = False
        self._alpha = 255
        self.setLayout(layout)
        self.setContentsMargins(5,5,5,5)

        # Set the size policy
        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

        self.setStyleSheet(self._get_checkbox_style('rgba(114, 254, 0, 255)'))

        # slots
        self.ohc_color.clicked.connect(self.show_ohc_color_dialog)
        self.ihc_color.clicked.connect(self.show_ihc_color_dialog)

    def show_ohc_color_dialog(self):
        self.show_color_dialog(self.ohc_color)

    def show_ihc_color_dialog(self):
        self.show_color_dialog(self.ihc_color)

    def _prepare_button(self, w: WPushButton, color: QColor) -> WPushButton:
        """ called before each button initialization """
        w.setBackgroundColor(color)
        w.setFixedWidth(self._button_side)
        w.setFixedHeight(self._button_side)
        return w

    def _prepare_spinbox(self, s: QSpinBox, minimum, maxmimum, init_val, style):
        """ called before each spinbox initialization """
        s.setMaximum(maxmimum)
        s.setMinimum(minimum)
        s.setValue(init_val)
        s.setStyleSheet(style)
        s.valueChanged.connect(self.update_style_dict)
        return s

    def show_color_dialog(self, w: WPushButton):
        color = QColorDialog.getColor(initial=w.getBackgroundColor())
        w.setBackgroundColor(color)
        self.update_style_dict()

    def update_style_dict(self):
        self.style_dict = {
            'ihc_color': hcat.lib.utils.qcolor_to_hex(self.ihc_color.getBackgroundColor()),
            'ohc_color': hcat.lib.utils.qcolor_to_hex(self.ohc_color.getBackgroundColor()),
            'ohc_stroke': self.ohc_stroke.value(),
            'ihc_stroke': self.ihc_stroke.value(),
            'ihc_transparency': round(self.ihc_transparency.value() / 100 * 255),
            'ohc_transparency': round(self.ohc_transparency.value() / 100 * 255),
            'eval_region_color': '#c8c8c8',
            'eval_region_stroke': 1,
            'eval_region_transparency': 100,
            'basal_point_color': hcat.lib.utils.qcolor_to_hex(QColor(55, 126, 184)),
            'apex_point_color': hcat.lib.utils.qcolor_to_hex(QColor(228, 26, 27)),
            'freq_line_color': hcat.lib.utils.qcolor_to_hex(QColor(255, 255, 255, 255)),
            'freq_line_stroke': 2
        }

        self.renderStyleChanged.emit(self.style_dict)

    def _get_checkbox_style(self, color):
        style = f"""
        QLabel {{ font: 10px;}}
        QCheckBox{{
            font: bold 8px; 
        }}
        QCheckBox::indicator{{
            width: 8px;
            height: 8px;
            background-color: rgba(255,255,255,0); 
            border: 1px solid black;
            border-radius: 0px;
            margin: 3px, 3px, 0px, 0px;
        }}
        QCheckBox::indicator::checked{{
            width: 8px;
            height: 8px;
            background-color: {color}; 
            border: 1px solid black;
            border-radius: 0px;
            margin: 3px, 3px, 0px, 0px;
            }}
        """
        return style

    def paintEvent(self, event):
        self._paint_warning()
        super(RenderStyleWidget, self).paintEvent(event)

    def _paint_warning(self):
        p = QPainter(self)
        if self._warning or self._error:
            primary_color = QColor(247, 83, 20, self._alpha) if self._warning else QColor(255, 0, 0, 60)
            secondary_color = QColor(200, 200, 200, self._alpha) if self._warning else (QColor(255, 255, 255, 60))
            primary_pen = QPen()
            primary_pen.setColor(primary_color)
            primary_pen.setWidth(15)

            secondary_pen = QPen()
            secondary_pen.setColor(secondary_color)
            secondary_pen.setWidth(15)

            _x = -500
            for i in range(100):
                p.setPen(primary_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))
                _x += 20

                p.setPen(secondary_pen)
                p.drawLine(QLineF(_x, 0, 1000 + _x, 1000))

                _x += 20

            if not self._error:
                palette: QPalette = self.palette()
                color = palette.color(self.backgroundRole())
                # red, green, blue = color.red(), color.green(), color.blue()
                # color = QColor(int(red*255), int(green*255), int(blue*255), 255)
                white_pen = QPen()
                white_pen.setColor(color)
                white_pen.setWidth(0)
                p.setPen(white_pen)

                w, h = self.width(), self.height()
                rect = QRectF(3, 3, w - 6, h - 6)
                path = QPainterPath()
                path.addRect(rect)
                p.fillPath(path, color)
            # p.drawPath(path)

    def get_style_dict(self):
        return self.style_dict

    def set_style_dict(self, style_dict):
        self.style_dict = style_dict
        self.update_fields_from_style_dict()

    def update_fields_from_style_dict(self):

        # self.style_dict = {
        #     'ihc_color': hcat.lib.utils.qcolor_to_hex(self.ihc_color.getBackgroundColor()),
        #     'ohc_color': hcat.lib.utils.qcolor_to_hex(self.ohc_color.getBackgroundColor()),
        #     'ohc_stroke': self.ohc_stroke.value(),
        #     'ihc_stroke': self.ihc_stroke.value(),
        #     'ihc_transparency': round(self.ihc_transparency.value() / 100 * 255),
        #     'ohc_transparency': round(self.ohc_transparency.value() / 100 * 255),
        #     'eval_region_color': '#c8c8c8',
        #     'eval_region_stroke': 1,
        #     'eval_region_transparency': 100,
        #     'basal_point_color': hcat.lib.utils.qcolor_to_hex(QColor(55, 126, 184)),
        #     'apex_point_color': hcat.lib.utils.qcolor_to_hex(QColor(228, 26, 27)),
        #     'freq_line_color': hcat.lib.utils.qcolor_to_hex(QColor(255, 255, 255, 255)),
        #     'freq_line_stroke': 2
        # }
        sd = self.style_dict
        self.ihc_color.setBackgroundColor(hex_to_qcolor(sd['ihc_color']))
        self.ohc_color.setBackgroundColor(hex_to_qcolor(sd['ohc_color']))
        self.ihc_stroke.setValue(sd['ihc_stroke'])
        self.ohc_stroke.setValue(sd['ohc_stroke'])
        self.ihc_transparency.setValue(sd['ihc_transparency'])
        self.ohc_transparency.setValue(sd['ohc_transparency'])
        self.update()



if __name__ == '__main__':
    app = QApplication()
    w = RenderStyleWidget()
    w.show()
    app.exec()
