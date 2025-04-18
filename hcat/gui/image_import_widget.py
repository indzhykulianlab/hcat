import os.path
from copy import copy
from typing import List, Dict

from PySide6.QtCore import QSize, QDir, Slot, Signal, QItemSelectionModel
from PySide6.QtWidgets import *

import hcat.lib.io
from hcat.state import Cochlea, Piece
from hcat.widgets.push_button import WPushButton

import os


# TODO add a final import button which returns each to the main app.
# TODO add a cancel button which just aborts the whole thing...

class Listener(QWidget):
    valueChanged = Signal(bool)
    def __init__(self, value):
        super(Listener, self).__init__()
        self.value = value

    def __setattr__(self, name, value):
        if name in self.__dict__ and self.__dict__[name] != value:
            self.valueChanged.emit(value)

        self.__dict__[name] = value



class LabeledInputWidget(QWidget):
    textChanges = Signal(int)

    def __init__(self, label: str, size: QSize):
        super(LabeledInputWidget, self).__init__()

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setFixedSize(size)

        self.input = QLineEdit()
        self.input.setInputMask('#######')
        self.input.setFixedSize(size)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input)
        self.setLayout(self.layout)

        self.input.editingFinished.connect(self.userEditedText)

    def setDefaultText(self, s: str):
        self.input.setText(s)

    def getValue(self):
        return self.input.text()

    def setText(self, s: str):
        self.input.setText(s)

    def enable(self):
        self.input.setEnabled(True)

    def disable(self):
        self.input.setDisabled(True)

    @Slot(int)
    def userEditedText(self):
        print('EDITED!')
        self.textChanges.emit(int(self.input.text()))


class ImportImageWidget(QWidget):
    updatedState = Signal()
    pieceRemoved = Signal(Piece)
    piecesAdded = Signal(list)

    def __init__(self, state: Cochlea):
        super(ImportImageWidget, self).__init__()
        self.candidate_pieces: List[Dict[str, QTreeWidgetItem | int | str | None | Dict[str, str | None]]] \
            = []
        self._removed_candidates = []
        self.default_candidate = {
            'filename': '',
            'px_size': 288,
            'animal': 'Mouse',
            'piece_order': None,
            'stains': {'red': 'Unspecified', 'green': 'Unspecified', 'blue': 'Unspecified'},
            'microscope': 'Unspecified',
            'zoom': 'Unspecified',
            'list_item': None,
            'from_piece': False
        }
        # Keys ----
        #   filename: str
        #   px_size: int | float
        #   animal: str
        #   piece_order: int
        #   stains = {'red: str, 'green': str, 'blue': str}
        #   microscope: str
        #   zoom: int | str
        #   list_item: QListWidgetItem
        #   from_piece: bool

        self.was_changed: Listener = Listener(False)
        self.new_candidate_added = Listener(False)

        self.state = state

        self.pieces_label = QLabel('Pieces')

        ## ---------- widgets
        self.piece_name_list = QListWidget()
        self.add_piece_button = WPushButton('ADD')
        self.remove_piece_button = WPushButton('REMOVE')
        self.remove_all_button = WPushButton('CLEAR')
        self.remove_all_button.setDangerButton(True)
        self.import_button = WPushButton('IMPORT')
        self.import_button.setToolTip('Imports new images')

        self.cancel_button = WPushButton('CANCEL').setWarningButton(True)
        self.cancel_button.setToolTip('Aborts imports or update')

        self.update_button = WPushButton('UPDATE')
        self.update_button.setToolTip('Saves updates to piece metadata')

        self.timeseries_checkbox = QCheckBox('Timeseries Import')
        self.timeseries_checkbox.setChecked(False)

        if not os.environ.get('ENABLE_HCAT_DEV', False):
            self.timeseries_checkbox.hide()


        # Required Metadata
        self.px_size_input = QSpinBox()
        self.px_size_input.setMaximum(9999)
        self.piece_ordering_input = QSpinBox()
        self.piece_ordering_input.setMaximum(0)
        self.animal_selector = QComboBox()
        self.animal_selector.addItems(['Mouse', 'Cat', 'Chinchilla', 'Rhesys Monkey', 'Guinea Pig'])

        # Optional Metadata
        self.red_channel_label = QLineEdit()
        self.green_channel_label = QLineEdit()
        self.blue_channel_label = QLineEdit()
        self.microscopy_selector = QComboBox()
        self.microscopy_selector.addItems(['Unspecified','Confocal', 'Widefield', 'Brightfield', 'SEM', 'Epifluorescence'])
        self.microscopy_selector.setCurrentIndex(0)
        self.zoom_selector = QLineEdit()
        self.apply_metadata_to_all_button = WPushButton('APPLY TO ALL')
        self.apply_metadata_to_all_button.setFixedWidth(120)
        self.apply_metadata_to_all_button.setWarningButton(True)

        self.create_layouts()
        self.slots_and_signals()
        self.enableUpdateButton(False)
        self.enableImportButton(False)
        # self.set_button_heights(16)

        # DEBUG
        # self._debug_create_candidate('/Users/chrisbuswinka/Pictures/1 - tJK66LE.jpg')
        # self._debug_create_candidate('/Users/chrisbuswinka/Pictures/2 - ceBv8nb.jpg')

        # set the defualt
        self._populate_fields_from_candidate(self._get_default_candidate())

    def set_tooltips(self):
        self.add_piece_button.setToolTip("Adds a piece for import")
        self.remove_piece_button.setToolTip("Removes a piece from import. "
                                            "Will delte a piece that's already been imported.")
        self.remove_all_button.setToolTip("Removes all pieces and associated data.")
        self.import_button.setToolTip("Imports all pieces and loads them into memory.")
        self.cancel_button.setToolTip("Cancels the import")
        self.update_button.setToolTip("Saves any changed piece metadata.")

    def set_button_heights(self, height):
        self.add_piece_button.setFixedHeight(height)
        self.remove_piece_button.setFixedHeight(height)
        self.cancel_button.setFixedHeight(height)
        self.update_button.setFixedHeight(height)
        self.import_button.setFixedHeight(height)

    def create_layouts(self):
        # Show the list view (LEFT SIDE)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.pieces_label)
        left_layout.addWidget(self.piece_name_list)

        # add and remove a piece
        left_button_layout = QHBoxLayout()
        left_button_layout.addWidget(self.add_piece_button)
        left_button_layout.addWidget(self.remove_piece_button)
        left_button_layout.addWidget(self.remove_all_button)
        left_layout.addLayout(left_button_layout)

        right_button_layout = QHBoxLayout()
        right_button_layout.addWidget(self.import_button)
        right_button_layout.addWidget(self.update_button)
        right_button_layout.addWidget(self.cancel_button)

        right_layout = QVBoxLayout()
        required_group = QGroupBox('Required Metadata')
        optional_group = QGroupBox('Optional Metadata')

        required_metadata_layout = QFormLayout()
        optional_metadata_layout = QFormLayout()

        required_metadata_layout.addRow(QLabel('Pixel Size (nm)'), self.px_size_input)
        required_metadata_layout.addRow(QLabel('Animal'), self.animal_selector)
        required_metadata_layout.addRow(QLabel('Piece Order'), self.piece_ordering_input)


        optional_metadata_layout.addRow(QLabel('Red Stain'), self.red_channel_label)
        optional_metadata_layout.addRow(QLabel('Green Stain'), self.green_channel_label)
        optional_metadata_layout.addRow(QLabel('Blue Stain'), self.blue_channel_label)
        optional_metadata_layout.addRow(None, None)
        optional_metadata_layout.addRow(QLabel('Microscope'), self.microscopy_selector)
        optional_metadata_layout.addRow(QLabel('Zoom'), self.zoom_selector)

        apply_to_all_layout = QHBoxLayout()
        apply_to_all_layout.addStretch(1)
        apply_to_all_layout.addWidget(self.apply_metadata_to_all_button)
        apply_to_all_layout.setContentsMargins(0,0,0,0)

        super_layout = QVBoxLayout()
        # super_layout.setContentsMargins(0,0,0,0)
        # super_layout.setSpacing(0)
        super_layout.addLayout(optional_metadata_layout)
        super_layout.addLayout(apply_to_all_layout)



        required_group.setLayout(required_metadata_layout)
        optional_group.setLayout(super_layout)

        right_layout.addWidget(required_group)
        right_layout.addWidget(optional_group)
        right_layout.addWidget(self.timeseries_checkbox)
        right_layout.addLayout(right_button_layout)
        right_layout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        self.setLayout(layout)

    def slots_and_signals(self):
        self.was_changed.valueChanged.connect(self.enableUpdateButton)
        self.new_candidate_added.valueChanged.connect(self.enableImportButton)

        self.add_piece_button.clicked.connect(self.create_candidate)
        self.remove_piece_button.clicked.connect(self.delete_candidate)
        self.remove_all_button.clicked.connect(self.clear_all)
        self.import_button.clicked.connect(self.update_state_from_candidates)
        self.cancel_button.clicked.connect(self.cancelImport)
        self.update_button.clicked.connect(self.update_pieces_from_candidates)
        self.apply_metadata_to_all_button.clicked.connect(self.assignOptionalMetadataToAll)

        self.piece_name_list.currentRowChanged.connect(self.onItemChange)

        self.piece_ordering_input.valueChanged.connect(self.update_piece_order)

        # Required Metadata
        self.px_size_input.valueChanged.connect(self.onFieldChange)
        self.animal_selector.currentIndexChanged.connect(self.onItemChange)

        # Optional Metadata
        self.red_channel_label.editingFinished.connect(self.onFieldChange)
        self.green_channel_label.editingFinished.connect(self.onFieldChange)
        self.blue_channel_label.editingFinished.connect(self.onFieldChange)
        self.microscopy_selector.currentIndexChanged.connect(self.onFieldChange)
        self.zoom_selector.editingFinished.connect(self.onFieldChange)

    def disable_slots(self):
        self.piece_name_list.blockSignals(True)
        self.add_piece_button.blockSignals(True)
        self.remove_all_button.blockSignals(False)
        self.remove_piece_button.blockSignals(True)
        self.import_button.blockSignals(True)
        self.cancel_button.blockSignals(True)
        self.update_button.blockSignals(True)

        # Required Metadata
        self.px_size_input.blockSignals(True)
        self.piece_ordering_input.blockSignals(True)
        self.animal_selector.blockSignals(True)

        # Optional Metadata
        self.red_channel_label.blockSignals(True)
        self.green_channel_label.blockSignals(True)
        self.blue_channel_label.blockSignals(True)
        self.microscopy_selector.blockSignals(True)
        self.zoom_selector.blockSignals(True)

    def enable_slots(self):
        self.piece_name_list.blockSignals(False)
        self.add_piece_button.blockSignals(False)
        self.remove_all_button.blockSignals(False)
        self.remove_piece_button.blockSignals(False)
        self.import_button.blockSignals(False)
        self.cancel_button.blockSignals(False)
        self.update_button.blockSignals(False)

        # Required Metadata
        self.px_size_input.blockSignals(False)
        self.piece_ordering_input.blockSignals(False)
        self.animal_selector.blockSignals(False)

        # Optional Metadata
        self.red_channel_label.blockSignals(False)
        self.green_channel_label.blockSignals(False)
        self.blue_channel_label.blockSignals(False)
        self.microscopy_selector.blockSignals(False)
        self.zoom_selector.blockSignals(False)

    def show(self):
        # set current row if a row available
        # do this to make the button go to err if piece
        # kind of a hack
        if self.piece_name_list.count() > 0:
            self.piece_name_list.setCurrentRow(0, QItemSelectionModel.ClearAndSelect)
        super(ImportImageWidget, self).show()

    @Slot()
    def enableUpdateButton(self, enabled):
        self.update_button.setEnabled(enabled)
        self.update_button.setDisabled(not enabled)
        self.update_button.update()

    @Slot()
    def enableImportButton(self, enabled):
        self.import_button.setEnabled(enabled)
        self.import_button.setDisabled(not enabled)
        self.import_button.update()

    @Slot()
    def removedElsewhere(self, piece: Piece):
        index = [c['filename'] == piece.filename for c in self.candidate_pieces]
        if not index: # nothing in the list
            return
        index = [i for i, v in enumerate(index) if v]
        index = index[0] # should only have one...
        self.piece_name_list.takeItem(index)
        removed_candidate = self.candidate_pieces.pop(index)  # I dont think we put it in self._removed_pieces...
        for i, candidate in enumerate(self.candidate_pieces):
            candidate['relative_order'] = i
        self.refresh_list()
        self.piece_ordering_input.setMaximum(len(self.candidate_pieces))

    def assignOptionalMetadataToAll(self):
        if len(self.candidate_pieces) <= 0:
            return
        stains = {
            'red': self.red_channel_label.text(),
            'green': self.green_channel_label.text(),
            'blue': self.blue_channel_label.text(),
        }
        for index, candidate in enumerate(self.candidate_pieces):
            candidate['stains'] = stains
            candidate['microscope'] = self.microscopy_selector.currentText()
            candidate['zoom'] = self.zoom_selector.text()

        self.update()

    @Slot(int)
    def onItemChange(self, selected: int):
        self.remove_piece_button.setDangerButton(False)
        if 0 <= selected < len(self.candidate_pieces):
            candidate = self.candidate_pieces[selected]
            if candidate['from_piece']:  # warn removing if candidate item has a piece
                self.remove_piece_button.setDangerButton(True)
            self._populate_fields_from_candidate(candidate)

    @Slot()
    def onFieldChange(self):
        self.was_changed.value = True
        if self.candidate_pieces:
            candidate = self._candidate_from_fields()
            i = self.piece_name_list.currentRow()
            from_piece = self.candidate_pieces[i]['from_piece']
            candidate['from_piece'] = from_piece
            print(from_piece)
            self.candidate_pieces[i] = candidate

        else: # change the default...
            self.default_candidate = self._candidate_from_fields()

    def cancelImport(self):
        """ called by pressing the cancel button... """
        for candidate in self.candidate_pieces:
            if not candidate['from_piece']:
                item = candidate['list_item']
                index: int = self.piece_name_list.indexFromItem(item).row()
                self.piece_name_list.takeItem(index)
        self.candidate_pieces = [c for c in self.candidate_pieces if c['from_piece']]
        self.close()

    def _update_piece_order(self, piece: Piece, old_position: int, new_position: int):
        """
        Create an order of piece
        :return:
        :rtype:
        """
        pass

    def _get_default_candidate(self):
        return copy(self.default_candidate)

    @staticmethod
    def _candidate_from_piece(piece: Piece) -> Dict:
        candidate = {
            'filename': piece.filename,
            'px_size': piece.pixel_size_xy,
            'animal': piece.animal,
            'piece_order': piece.relative_order,
            'stains': piece.stain_label,
            'microscope': piece.microscopy_type,
            'zoom': piece.microscopy_zoom,
            'list_item': QListWidgetItem(piece.filename),
            'from_piece': True
        }
        return candidate

    def _piece_from_candidate(self, candidate: Dict[str, int | str | None]):
        if candidate['from_piece']:
            return
        image = hcat.lib.io.imread(candidate['filename'])

        piece = Piece(
            id=hash(candidate['filename']),
            filename=candidate['filename'],
            image=image,
            parent=self.state,
        )
        piece.tree_item.setText(0, str(os.path.split(candidate['filename'])[-1]))
        piece.set_xy_pixel_size(float(candidate['px_size']))
        piece.set_animal(candidate['animal'])
        piece.set_relative_order(candidate['piece_order'])
        piece.set_stain_labels(candidate['stains'])
        piece.set_microscopy_type(candidate['microscope'])
        piece.set_microscopy_zoom(candidate['zoom'])

        index: int = self.piece_name_list.currentRow()
        candidate['from_piece'] = True
        return piece

    def _populate_fields_from_candidate(self, candidate):
        self.disable_slots()
        self.px_size_input.setValue(candidate['px_size'])

        # set the current combox selector # animal
        all_items: List[str] = [self.animal_selector.itemText(i) for i in range(self.animal_selector.count())]
        index: int = all_items.index(candidate['animal']) if candidate['animal'] in all_items else 0
        self.animal_selector.setCurrentIndex(index)

        piece_order = candidate['piece_order'] if candidate['piece_order'] else -1
        self.piece_ordering_input.setValue(piece_order)

        # set the current combox selector microscope
        all_items: List[str] = [self.microscopy_selector.itemText(i) for i in range(self.microscopy_selector.count())]
        index: int = all_items.index(candidate['microscope']) if candidate['microscope'] in all_items else 0
        self.microscopy_selector.setCurrentIndex(index)

        self.zoom_selector.setText(str(candidate['zoom']))

        stains = candidate['stains']
        self.red_channel_label.setText(str(stains['red']))
        self.green_channel_label.setText(str(stains['green']))
        self.blue_channel_label.setText(str(stains['blue']))

        self.enable_slots()

    def _candidate_from_fields(self) -> Dict:
        stains = {
            'red': self.red_channel_label.text(),
            'green': self.green_channel_label.text(),
            'blue': self.blue_channel_label.text(),
        }

        index: int = self.piece_name_list.currentRow()
        filename = self.candidate_pieces[index]['filename'] if self.candidate_pieces else None
        list_item = self.candidate_pieces[index]['list_item'] if self.candidate_pieces else QListWidgetItem()

        candidate = {
            'filename': filename,  # need a special way of doing this...
            'px_size': self.px_size_input.value(),
            'animal': self.animal_selector.currentText(),
            'piece_order': self.piece_ordering_input.value(),
            'stains': stains,
            'microscope': self.microscopy_selector.currentText(),
            'zoom': self.zoom_selector.text(),
            'list_item': list_item,
            'from_piece': False
        }

        return candidate

    def refresh_list(self):
        self.disable_slots()

        index: int = self.piece_name_list.currentRow()
        index = max(index, 0)

        # destroy all piece names
        while self.piece_name_list.count() > 0:
            self.piece_name_list.takeItem(0)

        for i, c in enumerate(self.candidate_pieces):

            if i == 0:
                prefex = '(Base)'
            elif i == len(self.candidate_pieces) - 1:
                prefex = '(Apex)'
            elif i == len(self.candidate_pieces) - 2:
                prefex = '  ↓   '
            else:
                prefex = '  │   '

            list_item = c['list_item']
            # if list_item is None:
            #     print(f'{list_item=}, {c=}, ')
            list_item.setText(prefex + ' ' + os.path.split(c['filename'])[-1])
            self.piece_name_list.addItem(c['list_item'])
            self.piece_name_list.setCurrentRow(index, QItemSelectionModel.ClearAndSelect)
        self.enable_slots()

    def update_piece_order(self):
        if len(self.candidate_pieces) <= 0:  # no op if nothing is there...
            return

        index = max(self.piece_name_list.currentRow(), 0)
        current = self.candidate_pieces[index]['piece_order']
        new = int(self.piece_ordering_input.text())

        if new > current:
            self.increase_piece_order()

        elif new < current:
            self.decrease_piece_order()

        for i, c in enumerate(self.candidate_pieces):
            c['piece_order'] = i

        # print(f'{index=}, {current=}, {new=}')
        self._populate_fields_from_candidate(self.candidate_pieces[new])
        self.refresh_list()
        self.piece_name_list.setCurrentRow(new, QItemSelectionModel.ClearAndSelect)

        # at this point, all of it should be fine

    def decrease_piece_order(self):
        """ decreases list order """
        if len(self.candidate_pieces) >= 2:
            index = self.piece_name_list.currentRow()
            if index > 0:
                self.candidate_pieces = self._swap(self.candidate_pieces, index, index-1)
                # self.candidate_pieces[index-1], self.candidate_pieces[index] = \
                #     self.candidate_pieces[index], self.candidate_pieces[index-1]

    def increase_piece_order(self):
        """ increases list order """
        if len(self.candidate_pieces) >= 2:
            index = self.piece_name_list.currentRow()
            if index < self.piece_name_list.count()-1:
                self.candidate_pieces = self._swap(self.candidate_pieces, index, index+1)
                # self.candidate_pieces[index], self.candidate_pieces[index+1] = \
                #     self.candidate_pieces[index+1], self.candidate_pieces[index]

    def _swap(self, array: list, old: int, new: int):
        array[old], array[new] = array[new], array[old]
        return array

    def update_pieces_from_candidates(self):
        for key, piece in self.state.children.items():
            if not isinstance(piece, Piece):
                continue
            for candidate in self.candidate_pieces:
                if candidate['filename'] == piece.filename:
                    piece.set_xy_pixel_size(float(candidate['px_size']))
                    piece.set_animal(candidate['animal'])
                    piece.set_relative_order(candidate['piece_order'])
                    piece.set_stain_labels(candidate['stains'])
                    piece.set_microscopy_type(candidate['microscope'])
                    piece.set_microscopy_zoom(candidate['zoom'])
                    self.updatedState.emit()

    def create_candidate(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open File",
                                                   QDir.currentPath(), 'Image Files (*.jpg *.tif *.png *.jpeg)')

        for file_path in file_paths:
            self.was_changed.value = True
            self.new_candidate_added.value = True

            previously_imported_filenames = [p['filename'] for p in self.candidate_pieces]
            if os.path.exists(file_path) and file_path not in previously_imported_filenames:
                candidate = self._get_default_candidate()  # base
                candidate['filename'] = file_path
                candidate['list_item'] = QListWidgetItem(os.path.split(file_path)[-1])  # just the filename
                candidate['piece_order'] = len(self.candidate_pieces)

                self.piece_name_list.addItem(candidate['list_item'])

                self._populate_fields_from_candidate(candidate)

                self.candidate_pieces.append(candidate)  # add to list of all candidates
                self.piece_ordering_input.setMaximum(len(self.candidate_pieces) - 1)
                self.piece_name_list.setCurrentRow(len(self.candidate_pieces) -1, QItemSelectionModel.ClearAndSelect)

                self.refresh_list()

    def _debug_create_candidate(self, file_path: str):
        previously_imported_filenames = [p['filename'] for p in self.candidate_pieces]
        if os.path.exists(file_path) and file_path not in previously_imported_filenames:
            candidate = self._get_default_candidate()  # base
            candidate['filename'] = file_path
            candidate['list_item'] = QListWidgetItem(os.path.split(file_path)[-1])  # just the filename
            candidate['piece_order'] = len(self.candidate_pieces)

            self.piece_name_list.addItem(candidate['list_item'])

            self._populate_fields_from_candidate(candidate)

            self.candidate_pieces.append(candidate)  # add to list of all candidates
            self.piece_ordering_input.setMaximum(len(self.candidate_pieces) - 1)

            self.refresh_list()

    def delete_candidate(self):
        index = self.piece_name_list.currentRow()
        if index >= 0:

            has_piece = self.candidate_pieces[index]['from_piece']
            if has_piece:
                msg = QMessageBox()
                msg.setText('Are you sure you want to delete?')
                msg.setInformativeText("You are attempting to delete a piece, image data, and all associated annotations. "
                                       "This action cannot be undone.")
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg.setDefaultButton(QMessageBox.Cancel)
                ret = msg.exec()
                if ret != QMessageBox.Ok:
                    return

                for key, piece in self.state.children.items():
                    print(f"{self.candidate_pieces[index]['filename'] == piece.filename}, " 
                          f"{self.candidate_pieces[index]['filename']}, {piece.filename}")
                    if not isinstance(piece, Piece):
                        continue
                    if self.candidate_pieces[index]['filename'] == piece.filename:
                        print('emitted a signal that the piece was removed...')
                        self.pieceRemoved.emit(piece)
                        break

            # delete and go to next closest candidate...
            candidate = self.candidate_pieces.pop(index)
            self.piece_name_list.takeItem(index)

            if len(self.candidate_pieces) > 0:
                self.refresh_list()
            # self._removed_candidates.append(candidate)
            # del candidate

        self.piece_ordering_input.setMaximum(len(self.candidate_pieces))

        # reset the order if deleted in the middle
        for i, c in enumerate(self.candidate_pieces):
            c['piece_order'] = i

        # if we deleted all pieces
        if self.piece_name_list.count() < 0 and index < 0:
            self._populate_fields_from_candidate(self._get_default_candidate())
            self.new_candidate_added.value = False  # no candidates...

        # if there is still a piece, we activate the next closest
        elif len(self.candidate_pieces) > 0:
            index = max(0, index - 1)
            self._populate_fields_from_candidate(self.candidate_pieces[index])
            # self.piece_name_list.setCurrentRow(-1, QItemSelectionModel.)
            self.piece_name_list.setCurrentRow(index, QItemSelectionModel.ClearAndSelect)

    def clear_all(self):

        if len(self.candidate_pieces) > 0:

            msg = QMessageBox()
            msg.setText('Are you sure you want to clear?')
            msg.setInformativeText(
                "You are attempting to clear all imports, image data, and all associated annotations. "
                "This action cannot be undone.")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Cancel)
            ret = msg.exec()
            if ret != QMessageBox.Ok:
                return

            # _candidates = list(enumerate(self.candidate_pieces))
            # _children = list(self.state.children)
            remove: List[Piece] = []
            for index, candidate_piece in enumerate(self.candidate_pieces):
                for key, piece in self.state.children.items():
                    if not isinstance(piece, Piece):
                        continue
                    if candidate_piece['filename'] == piece.filename:
                        remove.append(piece)

            for p in remove:
                self.pieceRemoved.emit(p)

            # delte all
            while self.piece_name_list.count() > 0:
                self.piece_name_list.takeItem(0)

        self.candidate_pieces = []
        self.piece_ordering_input.setMaximum(len(self.candidate_pieces))

        # we deleted all pieces so populate default
        if self.piece_name_list.count() <= 0:
            self._populate_fields_from_candidate(self._get_default_candidate())

    def update_state_from_candidates(self):
        """ both adds and removes """
        self.update_pieces_from_candidates()
        new_pieces = []
        if not self.timeseries_checkbox.isChecked():
            for candidate in self.candidate_pieces:
                if not candidate['from_piece']:
                    new_pieces.append(self._piece_from_candidate(candidate))
        else: # treat everything as a timeseries
            raise NotImplementedError('timeseries not done')



        removed_pieces = []
        for key, piece in self.state.children.items():
            if not isinstance(piece, Piece):
                continue
            for candidate in self._removed_candidates:
                if candidate['filename'] == piece.filename:
                    removed_pieces.append(piece)

        # self.piecesRemoved.emit(removed_pieces)
        self.piecesAdded.emit(new_pieces)
        self.close()

    def create_candidates_from_cochlea(self, cochlea: Cochlea):

        pieces = [p for p in cochlea.children.values()]
        key = lambda p: p.relative_order
        pieces.sort(key=key)
        _candidate = None
        for p in pieces:
            if not isinstance(p, Piece):
                continue
            _candidate = self._candidate_from_piece(p)
            self.candidate_pieces.append(_candidate)
            self.piece_name_list.addItem(_candidate['list_item'])

        if _candidate is not None:
            self._populate_fields_from_candidate(_candidate)

        self.piece_ordering_input.setMaximum(len(self.candidate_pieces) - 1)
        self.refresh_list()

    def set_state(self, state: Cochlea):
        self.state = state
        self.candidate_pieces = []
        self.create_candidates_from_cochlea(state)


if __name__ == '__main__':
    app = QApplication()
    cochlea = Cochlea(0, None)
    w = ImportImageWidget(state=cochlea)
    w.show()

    app.exec()
