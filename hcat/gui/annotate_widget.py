from PySide6.QtCore import Qt, QSize, Slot, QSysInfo
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.gui.colors import BLACK, RED, BLUE, ORANGE, GREEN, GRAY
from hcat.widgets.push_button import WPushButton


class AnnotateWidget(QWidget):
    def __init__(self):
        super(AnnotateWidget, self).__init__()
        BUTTON_SIZE = QSize(75, 18)

        layout = QVBoxLayout()

        self._selected_group: str = None

        # Move Grouping!
        move_group_layout = QVBoxLayout()
        move_group_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.set_move_mode_button = WPushButton('MOVE')
        self.set_move_mode_button.setMinimumSize(BUTTON_SIZE)
        self.set_move_mode_button.clicked.connect(self.highlight_move_mode_button)

        self.zoom_in_button = WPushButton('ZOOM +')
        self.zoom_in_button.setFixedSize(BUTTON_SIZE)
        self.zoom_out_button = WPushButton('ZOOM -')
        self.zoom_out_button.setFixedSize(BUTTON_SIZE)
        self.reset_view_button = WPushButton('RESET')
        self.reset_view_button.setFixedSize(BUTTON_SIZE)

        move_group_layout.addWidget(self.set_move_mode_button)
        move_group_layout.addWidget(self.zoom_in_button)
        move_group_layout.addWidget(self.zoom_out_button)
        move_group_layout.addWidget(self.reset_view_button)

        move_group_layout.setContentsMargins(1, 1, 1, 1)
        move_group_layout.setSpacing(1)

        self.move_group = QGroupBox()
        move_style = "QPushButton {border-width: 1px 3px 1px 1px; border-color: black black black black;}"

        # self.move_group.setStyleSheet(move_style)
        self.move_group.setTitle('Navigate')
        self.move_group.setLayout(move_group_layout)

        # Cell Grouping ---------------------------------------
        cell_group_layout = QVBoxLayout()
        cell_group_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        cell_style = "QPushButton {border-width: 1px 3px 1px 1px; border-color: black rgb(228, 26, 27) black black;}"
        self.select_cell = WPushButton('SELECT')
        self.select_cell.clicked.connect(self.highlight_select_cell)
        self.select_cell.setMinimumSize(BUTTON_SIZE)
        self.edit_cell = WPushButton('EDIT')
        self.edit_cell.clicked.connect(self.highlight_edit_cell)
        self.edit_cell.setMinimumSize(BUTTON_SIZE)
        self.draw_cell = WPushButton('DRAW')
        self.draw_cell.clicked.connect(self.highlight_draw_cell)
        self.draw_cell.setMinimumSize(BUTTON_SIZE)

        self.celltype_combobox = QComboBox()
        self.celltype_combobox.addItem('IHC')
        self.celltype_combobox.addItem('OHC')

        cell_group_layout.addWidget(self.select_cell)
        cell_group_layout.addWidget(self.draw_cell)
        cell_group_layout.addWidget(self.edit_cell)
        cell_group_layout.addWidget(self.celltype_combobox)

        cell_group_layout.setContentsMargins(1, 1, 1, 1)
        cell_group_layout.setSpacing(2)
        cell_group_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.cell_group = QGroupBox()
        self.cell_group.setTitle('Cell')
        self.cell_group.setLayout(cell_group_layout)

        # ---- eval region
        self.eval_region_group = QGroupBox('Eval Region')
        self.draw_eval_region = WPushButton('DRAW')
        self.draw_eval_region.setMinimumSize(BUTTON_SIZE)
        self.draw_eval_region.clicked.connect(self.highlight_draw_eval_region)

        self.edit_eval_region = WPushButton('EDIT')
        self.edit_eval_region.setMinimumSize(BUTTON_SIZE)
        self.edit_eval_region.clicked.connect(self.highlight_edit_eval_region)

        self.clear_eval_region = WPushButton('CLEAR')
        self.clear_eval_region.setMinimumSize(BUTTON_SIZE)
        self.clear_eval_region.setWarningButton(True)

        eval_region_layout = QVBoxLayout()
        eval_region_layout.addWidget(self.draw_eval_region)
        eval_region_layout.addWidget(self.edit_eval_region)
        eval_region_layout.addWidget(self.clear_eval_region)
        eval_region_layout.setContentsMargins(1, 1, 1, 1)
        eval_region_layout.setSpacing(2)
        eval_region_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.eval_region_group.setLayout(eval_region_layout)

        # Synapse
        self.synapse_region_group = QGroupBox('Synapse')
        self.select_synapse = WPushButton('SELECT')
        self.select_synapse.setMinimumSize(BUTTON_SIZE)
        self.select_synapse.clicked.connect(self.highlight_select_synapse)
        self.draw_synapse = WPushButton('DRAW')
        self.draw_synapse.setMinimumSize(BUTTON_SIZE)
        self.draw_synapse.clicked.connect(self.highlight_draw_synapse)
        self.edit_synapse = WPushButton('EDIT')
        # self.edit_synapse.setStyleSheet(synapse_style)
        self.edit_synapse.setMinimumSize(BUTTON_SIZE)
        self.edit_synapse.clicked.connect(self.highlight_edit_synapse)

        synapse_region_layout = QVBoxLayout()
        synapse_region_layout.addWidget(self.select_synapse)
        synapse_region_layout.addWidget(self.draw_synapse)
        synapse_region_layout.addWidget(self.edit_synapse)
        synapse_region_layout.setContentsMargins(1, 1, 1, 1)
        synapse_region_layout.setSpacing(2)
        synapse_region_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.synapse_region_group.setLayout(synapse_region_layout)

        # Frequency Annotation --------------
        self.freq_region_group = QGroupBox('Frequency')
        freq_width = "1px 3px 1px 1px"
        freq_color = "black rgb(78, 175, 74) black black"
        self.draw_freq_region = WPushButton('DRAW')
        self.draw_freq_region.setMinimumSize(BUTTON_SIZE)
        self.draw_freq_region.clicked.connect(self.highlight_draw_freq_region)
        self.edit_freq_region = WPushButton('EDIT')
        self.edit_freq_region.setMinimumSize(BUTTON_SIZE)
        self.edit_freq_region.clicked.connect(self.highlight_draw_freq_region)
        self.switch_freq_region = WPushButton('SWITCH')
        self.switch_freq_region.setMinimumSize(BUTTON_SIZE)
        self.clear_freq_region = WPushButton('CLEAR')
        self.clear_freq_region.setWarningButton(True)
        self.clear_freq_region.setMinimumSize(BUTTON_SIZE)

        freq_region_layout = QVBoxLayout()
        freq_region_layout.addWidget(self.draw_freq_region)
        freq_region_layout.addWidget(self.edit_freq_region)
        freq_region_layout.addWidget(self.switch_freq_region)
        freq_region_layout.addWidget(self.clear_freq_region)
        freq_region_layout.setContentsMargins(1, 1, 1, 1)
        freq_region_layout.setSpacing(2)
        freq_region_layout.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.freq_region_group.setLayout(freq_region_layout)

        #  Eval buttons

        run_style = "QPushButton {border-width: 2px 2px 2px 2px; border-color: black;}"
        # self.run_detection_button = QPushButton('RUN')
        # self.run_detection_button.setStyleSheet(run_style)

        SPACER = 1
        layout.addSpacing(10)
        layout.addWidget(self.move_group)
        # layout.addSpacing(SPACER)
        layout.addWidget(self.cell_group)
        # layout.addSpacing(SPACER)
        layout.addWidget(self.synapse_region_group)
        # layout.addSpacing(SPACER)
        layout.addWidget(self.eval_region_group)
        # layout.addSpacing(SPACER)
        layout.addWidget(self.freq_region_group)
        layout.addStretch(10)

        # layout.addWidget(self.run_detection_button, alignment=Qt.AlignBottom)

        layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        margin = 12 if QSysInfo.productType() == 'macos' else 15
        layout.setContentsMargins(margin, 10, margin, 1)

        self.setLayout(layout)

        self._set_all_button_borders()
        self.reset_subdomain_style()
        self.reset_all_button_styles()

        # we start in move mode, so that button should be highlghted
        self.highlight_self(self.set_move_mode_button, GRAY)

        policy = self.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Fixed)
        self.setSizePolicy(policy)

    def highlight_move_mode_button(self):
        self.highlight_self(self.set_move_mode_button, GRAY)

    def highlight_select_cell(self):
        self.highlight_self(self.select_cell, RED)

    def highlight_edit_cell(self):
        self.highlight_self(self.edit_cell, RED)

    def highlight_draw_cell(self):
        self.highlight_self(self.draw_cell, RED)

    def highlight_draw_eval_region(self):
        self.highlight_self(self.draw_eval_region, ORANGE)

    def highlight_edit_eval_region(self):
        self.highlight_self(self.edit_eval_region, ORANGE)

    def highlight_select_synapse(self):
        self.highlight_self(self.select_synapse, BLUE)

    def highlight_draw_synapse(self):
        self.highlight_self(self.draw_synapse, BLUE)

    def highlight_edit_synapse(self):
        self.highlight_self(self.edit_synapse, BLUE)

    def highlight_draw_freq_region(self):
        self.highlight_self(self.draw_freq_region, GREEN)

    def highlight_edit_freq_region(self):
        self.highlight_self(self.edit_freq_region, GREEN)


    def _set_all_button_borders(self):
        width, colors = self._get_button_borders(RED)
        self.select_cell.setBorder(width, colors)
        self.draw_cell.setBorder(width, colors)
        self.edit_cell.setBorder(width, colors)

        width, colors = self._get_button_borders(ORANGE)
        self.draw_eval_region.setBorder(width, colors)
        self.edit_eval_region.setBorder(width, colors)
        self.clear_eval_region.setBorder(width, colors)

        width, colors = self._get_button_borders(BLUE)
        self.select_synapse.setBorder(width, colors)
        self.draw_synapse.setBorder(width, colors)
        self.edit_synapse.setBorder(width, colors)

        width, colors= self._get_button_borders(GREEN)
        self.draw_freq_region.setBorder(width, colors)
        self.edit_freq_region.setBorder(width, colors)
        self.switch_freq_region.setBorder(width, colors)
        self.clear_freq_region.setBorder(width, colors)

        width, colors = self._get_button_borders(BLACK)
        self.set_move_mode_button.setBorder(width, colors)

    def highlight_subdomain(self, key):
        if not self.isEnabled():
            return

        groups = [self.cell_group, self.eval_region_group, self.synapse_region_group, self.freq_region_group,
                  self.move_group]
        highlight_style = f"""
        QGroupBox {{
            background-color: darkgray;
            border: 0px solid black; 
            margin-top: {3 if QSysInfo.productType() == 'macos' else 6}ex; /* leave space at the top for the title */
            font-size: 10px;
            border-radius: 0px; 
            }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            padding: 0 3px;
        }}
        """

        for k, w in zip(['Cell', 'Eval', 'Synapse', 'Frequency', 'Navigate'], groups):
            if key == k:
                w.setStyleSheet(highlight_style)
                w.update()
                w.repaint()
                self._selected_group = k

        self.repaint()

    @Slot()
    def reset_subdomain_style(self):
        """ resets group styling """
        groups = [self.move_group, self.cell_group, self.synapse_region_group,
                  self.eval_region_group, self.freq_region_group, self.move_group]
        base_group_style = f"""
        QGroupBox {{
            background-color: None;
            border: 0px solid black; 
            margin-top: {3 if QSysInfo.productType() == 'macos' else 6}ex; /* leave space at the top for the title */
            font-size: 10px;
            border-radius: 0px; 
            }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            padding: 0 3px;
        }}
        """
        for w in groups:
            w.setStyleSheet(base_group_style)
            w.update()
            w.repaint()

        self._selected_group = None
        self.repaint()

    def highlight_button(self, key):
        """
        select, [cell, synapse]
        draw,[cell, synapse, eval, frequency]
        edit, [cell, synapse, eval, frequency]
        switch, Frequency
        clear [Eval, Frequency]

        """
        if self._selected_group is None:  # do nothing if nothing selected
            return

        self.reset_all_button_styles()

        if self._selected_group == 'Cell':
            width, colors = self._get_button_borders(RED)
            if key == 'S':
                self.select_cell.setBackgroundColor(RED)
            elif key == 'D':
                self.draw_cell.setBackgroundColor(RED)
            elif key == 'E':
                self.edit_cell.setBackgroundColor(RED)

        elif self._selected_group == 'Eval':
            width, colors = self._get_button_borders(ORANGE)
            if key == 'D':
                self.draw_eval_region.setBackgroundColor(ORANGE)
            if key == 'E':
                self.edit_eval_region.setBackgroundColor(ORANGE)

        elif self._selected_group == 'Synapse':
            width, colors = self._get_button_borders(BLUE)
            if key == 'S':
                self.select_synapse.setBackgroundColor(BLUE)
            elif key == 'D':
                self.draw_synapse.setBackgroundColor(BLUE)
            elif key == 'E':
                self.edit_synapse.setBackgroundColor(BLUE)

        elif self._selected_group == 'Frequency':
            width, colors= self._get_button_borders(GREEN)
            if key == 'D':
                self.draw_freq_region.setBackgroundColor(GREEN)
            if key == 'E':
                self.edit_freq_region.setBackgroundColor(GREEN)

        elif self._selected_group == 'Navigate':
            if key == 'M':
                self.set_move_mode_button.setBackgroundColor(GRAY)

    def reset_all_button_styles(self):
        """ resets all button styles """

        # black = QColor(0, 0, 0)
        # red = QColor(228, 26, 27)  # "rgb(228, 26, 27)"
        # blue = QColor(55, 126, 184)  # "rgb(55, 126, 184)"
        # orange = QColor(246, 122, 0)  # "rgb(246, 122, 0)"
        # green = QColor(78, 175, 74)  # "rgb(78, 175, 74)"

        # self.set_move_mode_button.setStyleSheet(style)
        self.set_move_mode_button.resetBackgroundColor()

        self.select_cell.resetBackgroundColor()
        self.draw_cell.resetBackgroundColor()
        self.edit_cell.resetBackgroundColor()

        self.draw_eval_region.resetBackgroundColor()
        self.edit_eval_region.resetBackgroundColor()
        self.clear_freq_region.resetBackgroundColor()

        self.select_synapse.resetBackgroundColor()
        self.draw_synapse.resetBackgroundColor()
        self.edit_synapse.resetBackgroundColor()

        self.draw_freq_region.resetBackgroundColor()
        self.edit_freq_region.resetBackgroundColor()
        self.clear_freq_region.resetBackgroundColor()
        self.switch_freq_region.resetBackgroundColor()

    def _get_button_borders(self, color: QColor):
        r, g, b= color.red(), color.green(), color.blue()
        width_str = '1px 3px 1px 1px'
        color_str = f'black rgb({r}, {g}, {b}) black black '
        return width_str, color_str

    def highlight_self(self, button: WPushButton, color: QColor):
        self.reset_all_button_styles()
        if isinstance(button, QPushButton):
            button.setBackgroundColor(color)
        elif isinstance(button, WPushButton):

            button.setBackgroundColor(color)

        button.update()
        button.repaint()
