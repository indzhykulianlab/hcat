import json
import os.path
from typing import *
from typing import List

import numpy as np
import skimage.io as io
import torch
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import hcat.lib.io
import hcat.state.load
import hcat.state.save
import hcat.lib.roi

# from hcat.backend.backend import init_model, eval_crop, model_from_path
from hcat.backends import backend
from hcat.gui.annotate_widget import AnnotateWidget
from hcat.gui.canvas import PieceViewerWidget
from hcat.gui.cell_sidebar_info import CellSidebarInfoWidget
from hcat.gui.eval_widget import EvalWidget
from hcat.gui.footer_widget import FooterWidget
from hcat.gui.image_adjust_widget import ImageAdjustmentWidget
from hcat.gui.image_import_widget import ImportImageWidget
from hcat.gui.piece_sidebar_info import PieceSidebarInfoWidget
from hcat.gui.piece_tree_viewer import PieceTreeViewerWidget
from hcat.gui.render_hint_widget import RenderHintWidget
from hcat.gui.render_style_widget import RenderStyleWidget
from hcat.gui.train_config_widget import TrainWidget
from hcat.lib.adjust import _adjust
from hcat.lib.analyze import assert_validity, assign_frequency, rescale_to_optimal_size
from hcat.lib.export import piece_to_xml
from hcat.lib.importfromxml import cells_from_xml
from hcat.lib.multiprocess import Worker
from hcat.state import Piece, Cochlea, Cell
import hcat.gui.resources

VALID_FILE_IMPORTS = [".png", ".jpg", ".tif"]

from hcat.config.config import get_cfg_defaults

cfg = get_cfg_defaults()

__model_url__ = (
    "https://www.dropbox.com/s/opf43jwcbgz02vm/detection_trained_model.trch?dl=1"
)


class MainApplication(QMainWindow):
    def __init__(self):
        super(MainApplication, self).__init__()

        # Cache for undo
        self._deleted_cache = []
        self._key_buffer = []
        self.is_unsaved = True

        self.setFocusPolicy(Qt.StrongFocus)

        # model info
        self.model = backend.init_model()
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.worker_queue: List[List[QRunnable]] = []

        # All widgets here!
        self.image_viewer = PieceViewerWidget()
        self.setCentralWidget(self.image_viewer)
        self.annotate_sidebar = AnnotateWidget()

        self.cochlea = Cochlea(id=0, parent=None)  # initalize empty cochlea
        self.import_widget = ImportImageWidget(state=self.cochlea)

        self.createActions()
        self.createMenus()

        self.image_viewer.enableMoveMode()
        self.resize(1200, 800)

        self.current_state_filename = None
        self.train_widget = TrainWidget(cfg)
        self.train_widget.setEnabled(False)
        self.eval_widget = EvalWidget(cfg)

        # thresholding info...
        self.cell_thr = 0.5
        self.nms_thr = 0.5

        # live update buffer
        self.live_update_tensor: torch.Tensor | None = None

        # Layouts -------------

        # Set the image adjust dock
        self.image_adjust_dock = QDockWidget("Docakble", self)
        self.image_adjust_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_adjust_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.image_adjust_dock.setTitleBarWidget(QWidget(None))
        self.image_adjust_widget = ImageAdjustmentWidget()
        self.image_adjust_dock.setWidget(self.image_adjust_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_adjust_dock)

        # Piece info Dock
        self.piece_sidebar = PieceSidebarInfoWidget(self.cochlea)
        self.piece_info_dock = QDockWidget("Dockable", self)
        self.piece_info_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.piece_info_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.piece_info_dock.setTitleBarWidget(QWidget(None))
        self.piece_info_dock.setWidget(self.piece_sidebar)
        self.addDockWidget(Qt.RightDockWidgetArea, self.piece_info_dock)

        # Cell infor widget
        self.cell_info = CellSidebarInfoWidget()
        self.cell_info_dock = QDockWidget("Dockable", self)
        self.cell_info_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.cell_info_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.cell_info_dock.setTitleBarWidget(QWidget(None))
        self.cell_info_dock.setWidget(self.cell_info)
        self.addDockWidget(Qt.RightDockWidgetArea, self.cell_info_dock)

        # Tree Piece viewer widget
        self.right_dock = QDockWidget("Docakble", self)
        self.right_dock.setMinimumSize(QSize(150, 100))
        self.right_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.right_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.right_dock.setTitleBarWidget(QWidget(None))
        self.tree = PieceTreeViewerWidget(self.cochlea)
        self.right_dock.setWidget(self.tree)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        # render Style Dock
        self.render_style_dock = QDockWidget("Dockable", self)
        self.render_style_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.render_style_dock.setFeatures(
            QDockWidget.NoDockWidgetFeatures | QDockWidget.DockWidgetVerticalTitleBar
        )
        self.render_style_dock.setTitleBarWidget(
            QWidget(None)
        )  # QLabel('RENDER HINT'))
        self.render_style_widget = RenderStyleWidget()
        self.render_style_dock.setWidget(self.render_style_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.render_style_dock)

        # render hint dock
        self.render_hint_dock = QDockWidget("Dockable", self)
        self.render_hint_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.render_hint_dock.setFeatures(
            QDockWidget.NoDockWidgetFeatures | QDockWidget.DockWidgetVerticalTitleBar
        )
        self.render_hint_dock.setTitleBarWidget(QWidget(None))  # QLabel('RENDER HINT'))
        self.render_hint_widget = RenderHintWidget()
        self.render_hint_dock.setWidget(self.render_hint_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.render_hint_dock)

        # Create the left side dock.
        self.left_dock = QDockWidget("Dockable", self)

        # self.left_dock.setMinimumSize(QSize(150, 100))
        # self.left_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.left_dock.setTitleBarWidget(QWidget(None))
        self.left_dock.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        # self.left_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # Left side tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.addTab(self.annotate_sidebar, "Annotate")
        self.tab_widget.addTab(self.train_widget, "Train")
        self.tab_widget.addTab(self.eval_widget, "Eval")
        self.left_dock.setWidget(self.tab_widget)
        self.left_dock.setContentsMargins(0,0,0,0)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.tab_widget.setElideMode(Qt.ElideNone)
        self.tab_widget.setStyleSheet(
            """
        QTabWidget::pane {
          border: 2px solid black;
          top:-2px; 
          left: 2px;
          right: 2px;
          background: rgb(245, 245, 245);
        } 

        QTabBar::tab {
          background: rgb(236, 236, 236); 
          border: 1px solid darkgray; 
          padding-top: 1px;
          padding-bottom: 1px;
          padding-right: 3px;
          padding-left: 3px;
          margin-right: -1px;
          font: 8px;
        } 

        QTabBar::tab:selected { 
          margin-bottom: -1px; 
        }
        QTabBar::tab:!selected {
            color: darkgray
        }
        """
        )

        # footer widget
        self.bottom_dock = QDockWidget("Dockable", self)
        self.bottom_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.bottom_dock.setTitleBarWidget(QWidget(None))
        self.bottom_dock.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.footer = FooterWidget(self.cochlea)
        self.bottom_dock.setWidget(self.footer)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

        # self.left_dock.setDisabled(True)
        # self.right_dock.setDisabled(True)
        # self.image_adjust_dock.setDisabled(True)

        self.updateTabSizes(0)
        self.link_slots_and_signals()
        self.setApplicationDisabled()
        # self._debugImportImage('/Users/chrisbuswinka/Desktop/Composite2-1.tif')

    def link_slots_and_signals(self):
        """for each widget of the main window, link a slot to a signal"""
        # slots and signals -------------------
        self.image_viewer.updated_state.connect(self.updateChildren)
        self.image_viewer.children_removed.connect(self.addItemToCache)
        self.image_viewer.children_removed.connect(self.tree.removeItems)
        self.image_viewer.children_removed.connect(self.updateChildren)
        self.image_viewer.mouse_position_signal.connect(
            self.footer.populate_mouse_position
        )
        self.image_viewer.liveUpdateRegionChanged.connect(self.update_live_mode_area)
        self.image_viewer.modeChanged.connect(self.footer.setModeLabel)
        self.image_viewer.tonotopyPathEdited.connect(
            self.assignFrequenciesForActivePiece
        )

        # Image Import
        self.import_widget.piecesAdded.connect(self.addPiecesFromImport)
        self.import_widget.pieceRemoved.connect(self._deletePiece)
        self.import_widget.updatedState.connect(self.tree.updateItemOrder)

        # Image Adjust
        self.image_adjust_widget.adjustment_changed.connect(
            self.image_viewer.adjustImage
        )
        self.image_adjust_widget.adjustment_changed.connect(self.run_live_model)

        # Tree Viewer
        self.tree.selected_changed.connect(self.updateAllActivePieces)
        self.tree.active_piece_changed.connect(self.updateOnPieceChange)
        self.tree.cell_changed.connect(self.updateChildren)
        self.tree.cell_changed.connect(self.footer.update_cell_counts)
        self.tree.cell_changed.connect(self.cell_info.set_active_cell)
        self.tree.add_piece_button.clicked.connect(self.import_image_via_widget)
        self.tree.delete_piece_button.clicked.connect(self.deletePiece)

        # Annotate sidebar
        self.annotate_sidebar.zoom_in_button.clicked.connect(self.image_viewer.zoomIn)
        self.annotate_sidebar.zoom_out_button.clicked.connect(self.image_viewer.zoomOut)
        self.annotate_sidebar.reset_view_button.clicked.connect(
            self.image_viewer.resetImageToViewport
        )
        self.annotate_sidebar.set_move_mode_button.clicked.connect(
            self.image_viewer.enableMoveMode
        )
        self.annotate_sidebar.draw_eval_region.clicked.connect(
            self.image_viewer.enableDrawEvalRegionMode
        )
        self.annotate_sidebar.edit_eval_region.clicked.connect(
            self.image_viewer.enableEditEvalRegionMode
        )
        self.annotate_sidebar.clear_eval_region.clicked.connect(
            self.image_viewer.clearROI
        )
        self.annotate_sidebar.select_cell.clicked.connect(
            self.image_viewer.enableSelectBoxMode
        )
        self.annotate_sidebar.draw_cell.clicked.connect(
            self.image_viewer.enableAnnotateCellBoxMode
        )
        self.annotate_sidebar.celltype_combobox.currentTextChanged.connect(
            self.image_viewer.setAnotatingCellType
        )
        self.annotate_sidebar.edit_cell.clicked.connect(
            self.image_viewer.enableEditCellBoxMode
        )

        self.annotate_sidebar.select_synapse.clicked.connect(
            self.image_viewer.enableSelectSynapseMode
        )
        self.annotate_sidebar.edit_synapse.clicked.connect(
            self.image_viewer.enableEditSynapseMode
        )
        self.annotate_sidebar.draw_synapse.clicked.connect(
            self.image_viewer.enableAnnotateSynapseMode
        )
        self.annotate_sidebar.draw_freq_region.clicked.connect(
            self.image_viewer.enableAnnotateFreqMode
        )
        self.annotate_sidebar.edit_freq_region.clicked.connect(
            self.image_viewer.enableEditFreqMode
        )
        self.annotate_sidebar.clear_freq_region.clicked.connect(
            self.image_viewer.clearFreqPath
        )
        self.annotate_sidebar.switch_freq_region.clicked.connect(
            self.image_viewer.reverseFreqPath
        )

        # Cell Info
        self.cell_info.cellChanged.connect(self.updateChildren)

        # Tab group (Sets size)
        self.tab_widget.currentChanged.connect(self.updateTabSizes)

        # Render Hint
        self.render_hint_widget.renderHintChanged.connect(
            self.image_viewer.setRenderHint
        )

        # rnder Style
        self.render_style_widget.renderStyleChanged.connect(
            self.image_viewer.setRenderParams
        )
        self.render_style_widget.renderStyleChanged.connect(
            self.cell_info.setRenderParams
        )

        # eval widget
        self.eval_widget.thresholdSliderChanged.connect(self.update_active_piece_thr)
        self.eval_widget.thresholdSliderChanged.connect(self.updateChildren)
        self.eval_widget.thresholdSliderChanged.connect(self.footer.update_cell_counts)
        self.eval_widget.modelFileSelected.connect(self.loadModelFile)
        self.eval_widget.liveUpdate.connect(self.run_live_model)
        self.eval_widget.liveUpdate.connect(self.liveUpdateModel)
        self.eval_widget.runSnapshot.connect(self.quickEval)
        self.eval_widget.runFullAnalysis.connect(self.runFullAnalysis)
        self.eval_widget.defaultModelSelected.connect(self.loadDefaultModel)
        self.eval_widget.modelDeletedSignal.connect(self.deleteModel)
        self.eval_widget.set_gt.clicked.connect(self.setAllVisibleChildrenAsGroundTruth)
        self.eval_widget.assign_freq_button.clicked.connect(
            self.checkedFrequencyAssignment
        )


    def _deletePiece(self, piece: Piece):
        """actually deltes the"""

        # first remove the child...
        self.cochlea.remove_child(piece.id)
        self.tree.removeTopLevelItem(piece)

        # try to re-draw another piece on the image_viewer...
        # len of > 1 means there are at least 2 pieces
        if len(self.cochlea.children) > 0 and len(self.cochlea.selected):
            self.image_viewer.setActivePiece(self.cochlea.get_selected_child())
            self.footer.setActivePiece(self.cochlea.get_selected_child())

        elif (
            len(self.cochlea.children) > 0 and not self.cochlea.selected
        ):  # if we deleted the selected piece, but there are still some
            for id, p in self.cochlea.children.items():
                break
            self.cochlea.set_selected_children(id)
            self.image_viewer.setActivePiece(self.cochlea.get_selected_child())
            self.footer.setActivePiece(self.cochlea.get_selected_child())

        # if that doesnt work, we just re-set it to nothing...
        else:
            self.image_viewer.resetImageViewer()
            self.footer.setActivePiece(None)
            # self.image_adjust_widget.blockSignals(True)  # need to block signals else live mode callsed
            self.image_adjust_widget.resetHistogramAndSliders()
            # self.image_adjust_widget.blockSignals(False)

        if self.cochlea.get_selected_child() is None:
            self.setApplicationDisabled()

        self.updateChildren()
        self.update()

        del piece  # finally remove...

    def deletePiece(self, pieces: Piece | List[Piece] | None = None):
        if len(self.cochlea.children) == 0:
            return  # does nothing

        # launch 'are you sure?' window
        msg = QMessageBox()
        msg.setText("Are you sure you want to delete?")
        msg.setInformativeText(
            "You are attempting to delete a piece, image data, and all associated annotations. "
            "This action cannot be undone."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        ret = msg.exec()
        if ret != QMessageBox.Ok:
            return

        pieces = (
            [pieces]
            if isinstance(pieces, Piece)
            else [self.cochlea.get_selected_child()]
        )

        for piece in pieces:
            self._deletePiece(piece)
            self.import_widget.removedElsewhere(piece)

    def import_image_via_widget(self):
        self.import_widget.show()

    @Slot(list)
    def addPiecesFromImport(self, pieces: List[Piece]):
        if not pieces:
            return

        for piece in pieces:
            self.cochlea.add_child(piece)
            self.cochlea.set_selected_children(piece.id)

            tree_item = self.cochlea.getLatestTreeItem()
            self.tree.addTopLevelItem(tree_item)

            # cochlea should only ever have ONE selected child...
            self.image_adjust_widget.generate_histograms(
                self.cochlea.get_selected_child().histogram
            )

            self.setActivePiece(piece.id)

            self.setApplicationEnabled()

    @Slot(list)
    def removePiecesFromImport(self, pieces: List[Piece]):
        raise NotImplementedError

    @Slot(bool)
    def assignFrequenciesForActivePiece(self, *args):
        is_valid, msg = assert_validity(self.cochlea)
        piece = self.cochlea.get_selected_child()
        if not isinstance(piece, Piece):
            return

        if is_valid:
            worker = Worker(fn=assign_frequency, state_item=piece)
            worker.signals.finished.connect(self.updateChildren)
            self.add_tasks_to_queue([worker])
            self.run_next_task()

    def update_active_piece_thr(self, thr):
        _piece: Piece | None = self.cochlea.get_selected_child()
        if _piece is None:
            return

        _piece.set_thresholds(thr)

    def addItemToCache(self, item):
        self._deleted_cache.append(item)

    def keyPressEvent(self, event):
        """runs when the user presses a key"""
        print(QGuiApplication.queryKeyboardModifiers())
        if not self.annotate_sidebar.isEnabled():
            return

        key = event.key()

        if key == Qt.Key_Backspace or key == Qt.Key_Delete:
            self.image_viewer.deleteActiveChildren()

        if key == Qt.Key_Escape:
            self.image_viewer.resetAllAnnotationState()
            self._key_buffer = []
            self.parse_key_buffer()
            self.cochlea.get_selected_child().set_selected_to_none()
            self.updateChildren()
            # self.cell_info.set_active_cell([])

        else:
            self._key_buffer.append(key)
            self.parse_key_buffer()

    def parse_key_buffer(self):
        """
        parses a key buffer from keyPressEvent. Basically lets people press a key combo to activate some functionalit
        of the software.

        """
        key_map = {
            Qt.Key_C: "Cell",
            Qt.Key_E: "Eval",
            Qt.Key_F: "Frequency",
            Qt.Key_S: "Synapse",
            Qt.Key_N: "Navigate",
            Qt.Key_D: "Delete",
            Qt.Key_T: "Toggle",
        }
        selector_map = {
            Qt.Key_C: "C",
            Qt.Key_E: "E",
            Qt.Key_D: "D",
            Qt.Key_S: "S",  #
            Qt.Key_M: "M",
            Qt.Key_T: "T",
        }

        if self._key_buffer:
            if len(self._key_buffer) > 2:  # dont let mutlple keys go into buffer
                self.annotate_sidebar.reset_subdomain_style()
                self._key_buffer = []

            if len(self._key_buffer) == 1:  # subdomain
                if self._key_buffer[0] not in key_map:  # pressed nonsense
                    self._key_buffer = []
                    self.annotate_sidebar.reset_subdomain_style()
                else:
                    self.annotate_sidebar.highlight_subdomain(
                        key_map[self._key_buffer[0]]
                    )
                    self.annotate_sidebar.update()

            if len(self._key_buffer) == 2:  # set the mode in each subdomain
                if self._key_buffer[1] not in selector_map.keys():
                    self._key_buffer = []
                    self.annotate_sidebar.reset_subdomain_style()
                else:
                    self.annotate_sidebar.highlight_button(
                        selector_map[self._key_buffer[1]]
                    )

                    group = key_map[self._key_buffer[0]]
                    selector = selector_map[self._key_buffer[1]]

                    if group == "Cell":
                        if selector == "D":
                            self.image_viewer.enableAnnotateCellBoxMode()
                        elif selector == "E":
                            self.image_viewer.enableEditCellBoxMode()
                        elif selector == "S":
                            self.image_viewer.enableSelectBoxMode()

                    if group == "Eval":
                        if selector == "D":
                            self.image_viewer.enableDrawEvalRegionMode()
                        elif selector == "E":
                            self.image_viewer.enableEditEvalRegionMode()

                    if group == "Navigate":
                        if selector == "M":
                            self.image_viewer.enableMoveMode()

                    if group == "Frequency":
                        if selector == "D":
                            self.image_viewer.enableAnnotateFreqMode()
                        elif selector == "E":
                            self.image_viewer.enableEditFreqMode()
                        elif selector == "S":
                            self.image_viewer.reverseFreqPath()

                    if group == "Synapse":
                        if selector == "S":
                            self.image_viewer.enableSelectSynapseMode()
                        elif selector == "D":
                            self.image_viewer.enableAnnotateSynapseMode()
                        elif selector == "E":
                            self.image_viewer.enableEditSynapseMode()

                    if group == "Delete":
                        if selector == "D":
                            self.image_viewer.deleteActiveChildren()

                    if group == "Toggle":
                        if selector == "T":
                            self.toggleSelectedType()
                        elif selector == "D":
                            index = (
                                self.annotate_sidebar.celltype_combobox.currentIndex()
                            )
                            self.annotate_sidebar.celltype_combobox.setCurrentIndex(
                                0 if index else 1
                            )
                            self.image_viewer.update()

                    self.annotate_sidebar.reset_subdomain_style()
                    self._key_buffer = []

    def _debugImportImage(self, file_name):
        # check if its an image
        if os.path.splitext(file_name)[-1] in VALID_FILE_IMPORTS:
            image = io.imread(file_name)
            existing_id_values = self.cochlea.children.keys()
            id = 0
            while id in existing_id_values:
                id += 1

            piece = Piece(image=image, filename=file_name, id=id, parent=self.cochlea)

            self.cochlea.add_child(piece)
            self.cochlea.set_selected_children(piece.id)

            tree_item = self.cochlea.getLatestTreeItem()
            self.tree.addTopLevelItem(tree_item)

            # cochlea should only ever have ONE selected child...
            self.image_adjust_widget.generate_histograms(
                self.cochlea.get_selected_child().histogram
            )

            self.setActivePiece(piece.id)

            self.left_dock.setEnabled(True)
            self.right_dock.setEnabled(True)
            self.image_adjust_dock.setEnabled(True)
            self.updateChildren()

    def importImage(self):
        """
        Brings up a widget letting you import a piece.
        Things you must do:
            - popup to select a file
            - give hcat some info on the piece
                - base/apex
                - piece number
                - channel info???
            - import the image and re-set the channel ordering
            - create a cochlea if none has been created yet
            - create a piece for the image and add it to cochlea
        :return:
        """
        self.import_widget.show()
        self.import_widget.activateWindow()
        self.import_widget.raise_()

    @Slot()
    def updateScrollBar(self, factor):
        value = self.scroll_area.horizontalScrollBar().value()
        page_step = self.scroll_area.horizontalScrollBar().pageStep()
        self.scroll_area.horizontalScrollBar().setValue(
            int(factor * value + ((factor - 1) * page_step / 2))
        )

        value = self.scroll_area.verticalScrollBar().value()
        page_step = self.scroll_area.verticalScrollBar().pageStep()
        self.scroll_area.verticalScrollBar().setValue(
            int(factor * value + ((factor - 1) * page_step / 2))
        )

    def setActivePiece(self, index: int):
        self.cochlea.set_selected_children(index)
        self.image_viewer.setActivePiece(self.cochlea.get_selected_child())
        self.footer.setActivePiece(self.cochlea.get_selected_child())
        self.updateChildren()

    @Slot(list)
    def updateAllActivePieces(self, index: List[int]):
        self.cochlea.set_selected_to_none()
        self.cochlea.set_selected_for_all_children(index)
        if self.cochlea.get_selected_child() is None:
            return

        piece = self.cochlea.get_selected_child()

        self.image_adjust_widget.set_sliders_from_piece(piece)

        self.image_viewer.setActivePiece(piece, len(index) > 1)
        self.cell_info.set_active_piece(piece)

        self.footer.setActivePiece(piece)
        self.footer.update_cell_counts()

        self.piece_sidebar.update()

        self.eval_widget.setThresholdValues(piece.cell_thr * 100, piece.nms_thr * 100)

        self.cell_info.set_active_cell(piece.get_selected_children())
        self.image_adjust_widget.generate_histograms(piece.histogram)
        self.updateChildren()

    def updateOnPieceChange(self):
        """updates all widgets which only need to be changed when the active piece changes"""
        self.image_adjust_widget.generate_histograms(
            self.image_viewer.active_piece.get_histogram()
        )
        self.image_adjust_widget.update()

    @Slot()
    def updateChildren(self):
        self.image_viewer.update()
        self.footer.update()
        self.tree.update()
        self.cell_info.set_active_piece(self.cochlea.get_selected_child())
        self.cell_info.update()
        self.footer.update_cell_counts()
        self.footer.update()
        self.piece_sidebar.update()

    @Slot()
    def loadWhorlStateFile(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open", QDir.homePath(), "Whorl Cochlea File (*.hcat *.json)"
        )
        if not filename:
            return

        self.cochlea, style_dict, _ = hcat.state.load.load_hcat_file(filename)
        self.current_state_filename = filename

        if style_dict is not None:
            self.render_style_widget.set_style_dict(style_dict)

        self.import_widget.set_state(self.cochlea)

        self.tree.setNewSate(self.cochlea)
        for tree_item in self.cochlea.top_level_tree_items:
            self.tree.addTopLevelItem(tree_item)

        # cochlea should only ever have ONE selected child...
        self.image_viewer.setActivePiece(self.cochlea.get_selected_child())
        self.image_adjust_widget.generate_histograms(
            self.cochlea.get_selected_child().histogram
        )

        self.cell_info.set_active_piece(self.cochlea.get_selected_child())
        self.footer.setActivePiece(self.cochlea.get_selected_child())
        self.footer.update_cell_counts()
        self.cell_info.set_active_cell(
            self.cochlea.get_selected_child().get_selected_children()
        )
        self.piece_sidebar.setState(self.cochlea)
        self.setApplicationEnabled()

        self.updateChildren()
        self.update()

    @Slot()
    def saveWhorlStateFile(self):
        if (
            self.cochlea.get_selected_child() is None
        ):  # dont do shit if nothign to save...
            return
        #
        # if self.freq_is_stale:
        #     # launch 'are you sure?' window
        #     msg = QMessageBox()
        #     msg.setText("Cell frequency assignments may be invalid!")
        #     msg.setInformativeText(
        #     """
        #     You have updated the tonotopy path of one or more pieces without re-assigning cell frequency. Some cell
        #     frequencies may be invalid. Press "Yes" to re-assign frequencies and continue saving, "No" to save without
        #     assignment, and "Cancel" to abort saving.
        #     """
        #     )
        #     msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        #     msg.setDefaultButton(QMessageBox.Yes)
        #     ret = msg.exec()
        #     if ret == QMessageBox.Yes:
        #         assign_frequency(self.cochlea)
        #     elif ret == QMessageBox.Cancel:
        #         return

        if not self.current_state_filename:
            self.saveAsWhorlStateFile()
        else:
            json_dict = hcat.state.save.cochlea_to_json(
                self.cochlea, style_hint=self.render_style_widget.style_dict
            )
            with open(f"{self.current_state_filename}", "w") as f:
                json.dump(json_dict, f, ensure_ascii=False)

    @Slot()
    def saveAsWhorlStateFile(self):
        filename, type = QFileDialog.getSaveFileName(
            self, "Save", QDir.currentPath(), "Whorl Cochlea File (*.hcat)"
        )
        if not filename:
            return

        self.current_state_filename = filename

        # if self.freq_is_stale:
        #     # launch 'are you sure?' window
        #     msg = QMessageBox()
        #     msg.setText("Cell frequency assignments may be invalid!")
        #     msg.setInformativeText(
        #     "You have updated the tonotopy path of one or more pieces without re-assigning cell frequency. Some cell frequencies may be invalid. Press 'Yes' to re-assign frequencies and continue saving, 'No' to save without assignment, and 'Cancel' to abort saving."
        #     )
        #     msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        #     msg.setDefaultButton(QMessageBox.Yes)
        #     ret = msg.exec()
        #     if ret == QMessageBox.Yes:
        #         assign_frequency(self.cochlea)
        #     elif ret == QMessageBox.Cancel:
        #         return

        json_dict = hcat.state.save.cochlea_to_json(
            self.cochlea, style_hint=self.render_style_widget.style_dict
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False)

    @Slot()
    def exportCSV(self):
        filename, type = QFileDialog.getSaveFileName(
            self, "Save", QDir.currentPath(), "csv (*.csv)"
        )
        if not filename:
            return

        hcat.state.save.export_to_csv(self.cochlea, filename)

    def _run_model(
        self, piece: Piece, coords: Tuple[float, float, float, float]
    ) -> Worker:
        x, y, w, h = coords
        excluded_colors: List[bool] = self.eval_widget.get_use_channel_state()
        image = piece.image  # X Y C
        crop = image[x : x + w, y : y + h, :].copy()

        for j, excluded in enumerate(excluded_colors):
            if excluded:
                crop[..., j] = crop[..., j] * 0.0

            for c in piece.get_adjustments():
                brightness = float(c["brightness"])  # , dtype=np.float64)
                contrast = float(c["contrast"])  # , dtype=np.float64)
                _img = crop[..., c["channel"]].astype(np.float64)  # needs input float64
                crop[..., c["channel"]] = _adjust(
                    _img, brightness, contrast
                )  # output uint8

            for j, excluded in enumerate(excluded_colors):
                if excluded:
                    crop[..., j] = crop[..., j] * 0.0

        crop: torch.Tensor = rescale_to_optimal_size(
            torch.from_numpy(crop), px_size=piece.pixel_size_xy
        )

        worker = Worker(
            fn=backend.eval_crop,
            crop=crop,
            model=self.model,
            device="cpu",
            parent=piece,
            coords=[x, y],
            return_window=True,
            metadata={"creator": self.eval_widget.get_model_filename()},
            begin_passthrough={"coords": (x, y, w, h)},
        )

        worker.signals.begin.connect(
            self.set_eval_region_crop_box
        )
        worker.signals.result.connect(self.add_children_from_worker)
        return worker

    @Slot()
    def quickEval(self):
        active_piece: Piece = self.cochlea.get_selected_child()
        image = active_piece.image  # X Y C
        regions = active_piece.eval_region_boxes

        if not regions:
            verts = [
                (0, 0),
                (0, image.shape[0]),
                (image.shape[0], image.shape[1]),
                (image.shape[1], 0),
            ]
            regions = hcat.lib.roi.get_squares(
                verts, round(255 * (289 / active_piece.pixel_size_xy)), 0.1
            )

        tasks = []
        for i, (y, x, w, h) in enumerate(regions):
            w = min(image.shape[0] - 1, w)
            h = min(image.shape[1] - 1, h)
            x = min(max(0, x), image.shape[0] - w)
            y = min(max(0, y), image.shape[1] - h)

            x, y, w, h = int(x), int(y), int(w), int(h)
            worker = self._run_model(active_piece, (x, y, w, h))
            if i + 1 == len(regions):
                worker.signals.finished.connect(self.image_viewer.clearEvalWindow)
                worker.signals.finished.connect(self.run_next_task)
            tasks.append(worker)
        self.add_tasks_to_queue(tasks)
        self.run_next_task()

    def batch_quick_eval(self):
        pieces = [p for p in self.cochlea.children.values() if isinstance(p, Piece)]

        for piece in pieces:
            if not piece.eval_region:
                verts = [
                    (0, 0),
                    (0, piece.image.shape[0]),
                    (piece.image.shape[0], piece.image.shape[1]),
                    (piece.image.shape[1], 0),
                ]
                regions = hcat.lib.roi.get_squares(
                    verts, round(255 * (289 / piece.pixel_size_xy)), 0.1
                )
                piece.set_eval_region_boxes(regions)


        for piece in pieces:
            worker = Worker(fn=lambda p: p, p=piece)
            worker.signals.result.connect(
                lambda p: self.cochlea.set_selected_children(p.id)
            )
            worker.signals.result.connect(
                lambda p: self.image_viewer.setActivePiece(p)
                )

            worker.signals.finished.connect(self.image_viewer.resetImageToViewport)
            worker.signals.finished.connect(self.run_next_task)
            self.add_tasks_to_queue([worker])

            image = piece.image  # X Y C
            regions = piece.eval_region_boxes

            piece_tasks = []
            for i, (y, x, w, h) in enumerate(regions):
                w = min(image.shape[0] - 1, w)
                h = min(image.shape[1] - 1, h)
                x = min(max(0, x), image.shape[0] - w)
                y = min(max(0, y), image.shape[1] - h)

                x, y, w, h = int(x), int(y), int(w), int(h)
                worker = self._run_model(piece, (x, y, w, h))
                if i + 1 == len(regions):
                    worker.signals.finished.connect(self.image_viewer.clearEvalWindow)
                    worker.signals.finished.connect(self.run_next_task)

                piece_tasks.append(worker)
            self.add_tasks_to_queue(piece_tasks)

        worker = Worker(fn=self.cochlea.set_selected_children, id=pieces[0].id)
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        worker = Worker(fn=self._return_input, arg__0=pieces[0])
        worker.signals.finished.connect(
            self.image_viewer.setActivePiece
        )
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        worker = Worker(fn=assign_frequency, state_item=self.cochlea)
        worker.signals.finished.connect(self.enable_for_eval)
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        self.run_next_task()


    def runFullAnalysis(self):
        """
        evaluates model on everything. We need a way to abort...
        :return:
        """
        valid, error_msg = assert_validity(self.cochlea)
        if not valid:
            msg = QMessageBox()
            msg.setText("Error")
            msg.setInformativeText(error_msg)
            msg.setDefaultButton(QMessageBox.Ok)
            ret = msg.exec()
            return

        # if we've gotten here, we know that each piece has a freq path, and each
        # piece has an eval path...
        # we are safe to say fuck it, lets go

        # STEP 1: Disable everything...

        self.annotate_sidebar.reset_all_button_styles()
        self.image_viewer.enableMoveMode()

        # highlight annotate sidebar...
        self.annotate_sidebar.highlight_subdomain("Navigate")
        self.annotate_sidebar.highlight_button("M")
        self.annotate_sidebar.reset_subdomain_style()

        self.disable_for_eval()

        # STEP 2: Loop over all pieces...
        pieces = [p for p in self.cochlea.children.values() if isinstance(p, Piece)]
        pieces.sort(key=lambda p: p.relative_order)

        # ALL THIS ASSUMES THE THREADS ARE FIRST IN, FIRST OUT
        for piece in pieces:
            if piece.eval_region is None:  # we ignore pieces without regions...
                print('no eval region')
                continue

            # Make the current cochlea selected...
            worker = Worker(fn=lambda p: p, p=piece)
            worker.signals.result.connect(
                lambda p: self.cochlea.set_selected_children(p.id)
            )
            worker.signals.result.connect(self.image_viewer.setActivePiece)
            worker.signals.finished.connect(self.image_viewer.resetImageToViewport)
            worker.signals.finished.connect(self.run_next_task)
            self.add_tasks_to_queue([worker])

            image = piece.image  # X Y C
            regions = piece.eval_region_boxes
            scale = piece.pixel_size_xy / 289 if piece.pixel_size_xy is not None else 1

            excluded_colors: List[bool] = self.eval_widget.get_use_channel_state()
            piece_tasks = []
            for i, (y, x, w, h) in enumerate(regions):
                w = min(image.shape[0] - 1, w)
                h = min(image.shape[1] - 1, h)
                x = min(max(0, x), image.shape[0] - w)
                y = min(max(0, y), image.shape[1] - h)

                x, y, w, h = int(x), int(y), int(w), int(h)
                worker = self._run_model(piece, (x, y, w, h))
                if i + 1 == len(regions):
                    worker.signals.finished.connect(self.image_viewer.clearEvalWindow)
                    worker.signals.finished.connect(self.run_next_task)
                piece_tasks.append(worker)
            self.add_tasks_to_queue(piece_tasks)

        worker = Worker(fn=self.cochlea.set_selected_children, id=pieces[0].id)
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        worker = Worker(fn=self._return_input, arg__0=pieces[0])
        worker.signals.finished.connect(
            self.image_viewer.setActivePiece
        )
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        worker = Worker(fn=assign_frequency, state_item=self.cochlea)
        worker.signals.finished.connect(self.enable_for_eval)
        worker.signals.finished.connect(self.run_next_task)
        self.add_tasks_to_queue([worker])

        self.run_next_task()


    def add_tasks_to_queue(self, tasks: List[QRunnable] | QRunnable):
        if isinstance(tasks, QRunnable):
            tasks = [tasks]
        self.worker_queue.append(tasks)

    def run_next_task(self):
        self.threadpool.waitForDone(-1)
        if len(self.worker_queue) <= 0:
            return
        tasks = self.worker_queue.pop(0)
        for worker in tasks:
            self.threadpool.start(worker)

    def _return_input(self, arg__0: object):
        """ helper class to get rid of lambdas """
        return arg__0

    def checkedFrequencyAssignment(self):
        valid, error_msg = assert_validity(self.cochlea)
        if not valid:
            msg = QMessageBox()
            msg.setText("Error")
            msg.setInformativeText(error_msg)
            msg.setDefaultButton(QMessageBox.Ok)
            ret = msg.exec()
            return
        self.assign_freq_from_analyzed_cochlea()
        self.updateChildren()
        self.update()

    def assign_freq_from_analyzed_cochlea(self):
        self.threadpool.waitForDone(-1)
        assign_frequency(self.cochlea)

    def disable_for_eval(self):
        self.annotate_sidebar.setDisabled(True)
        self.image_adjust_widget.setDisabled(True)
        self.import_widget.setDisabled(True)
        self.tree.setDisabled(True)

    def enable_for_eval(self):
        self.annotate_sidebar.setDisabled(False)
        self.image_adjust_widget.setDisabled(False)
        self.import_widget.setDisabled(False)
        self.tree.setDisabled(False)

    def add_children_from_worker(self, *args):
        args = args[0]  # passed as a list...

        children = args[0]

        if len(children) > 0:
            parent: Piece = children[0].parent  # they should all be the same
            parent.add_children(children)

        self.updateChildren()
        self.update()

    def set_eval_region_crop_box(self, coords: Tuple[float, float, float, float]):
        if isinstance(coords, dict) and 'coords' in coords:
            coords = coords['coords'] # bad way of handling kwarg

        x, y, w, h = coords
        self.image_viewer.setEvalWindow([x, y, w, h])
        self.update()

    @Slot()
    def chooseModelFile(self):
        file_path, type = QFileDialog.getOpenFileName(
            self, "Load Model File", QDir.currentPath(), "Detection Model (*.trch)"
        )
        print(file_path)
        # self.model = backend.model_from_path(file_path, 'cpu')
        if file_path:
            self.loadModelFile(file_path)
            self.eval_widget.setModel(file_path)
            self.eval_widget.update()
        else:
            print(file_path)
            raise RuntimeError(f"shouldnt happen -> {file_path}")

    @Slot()
    def loadDefaultModel(self):
        file_path = hcat.lib.io.defualt_path_from_url(
            "https://www.dropbox.com/s/opf43jwcbgz02vm/detection_trained_model.trch?dl=1"
        )
        self.loadModelFile(file_path)
        self.eval_widget.setModel(file_path)
        self.eval_widget.update()

    @Slot()
    def deleteModel(self):
        print('DELETED THE MODEL')
        self.model = backend.init_model()
        self.eval_widget.deleteModel()
        self.eval_widget.update()

    @Slot(bool)
    def liveUpdateModel(self, live: bool):
        if live and self.cochlea.get_selected_child() is not None:
            self.image_viewer.enableLiveEvalmode()
            piece: Piece = self.cochlea.get_selected_child()
            if not isinstance(piece, Piece):
                return

            scale = piece.pixel_size_xy / 289 if piece.pixel_size_xy is not None else 1

            piece: Piece = self.cochlea.get_selected_child()
            piece.enable_live_mode([0, 0, round(255 * scale), round(255 * scale)])
            self.image_viewer.live_eval_square = [
                0,
                0,
                round(255 * scale),
                round(255 * scale),
            ]
            self.update_live_mode_area(
                [0, 0, round(255 * scale), round(255 * scale)]
            )  # have to do this...
        else:
            piece: Piece = self.cochlea.get_selected_child()
            if not isinstance(piece, Piece):
                return

            scale = piece.pixel_size_xy / 289 if piece.pixel_size_xy is not None else 1

            self.image_viewer.resetPixmapFromActivePiece()
            self.image_viewer.enableMoveMode()
            self.image_viewer.live_eval_square = [
                0,
                0,
                round(255 * scale),
                round(255 * scale),
            ]
            piece: Piece = self.cochlea.get_selected_child()
            piece.disable_live_mode()
            self.image_viewer.update()

        self.annotate_sidebar.setEnabled(not live)

    @Slot(list)
    def update_live_mode_area(self, area):
        if area and self.cochlea.get_selected_child() is not None:
            piece: Piece = self.cochlea.get_selected_child()

            if not piece.live_mode:  # do nothing if not live mode
                return

            piece.enable_live_mode(area)
            # this is to re-apply image adjustments when the region is moved
            self.image_adjust_widget.modified_color_channels.extend([0, 1, 2])

            # I am an awful programer
            # self.image_adjust_widget.blockSignals(True)
            # self.image_adjust_widget.apply_adjustment()
            self.image_viewer.adjustImage(
                self.image_adjust_widget.get_current_adjustments()
            )
            # self.image_adjust_widget.blockSignals(False)

            y0, x0, y1, x1 = piece.live_area
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            self.live_update_tensor = torch.from_numpy(
                piece._display_image_buffer[x0:x1, y0:y1, :]
            )
            self.run_live_model()

    def run_live_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.cochlea.get_selected_child() is not None:
            piece: Piece = self.cochlea.get_selected_child()
            if not isinstance(piece, Piece):
                return

            if piece is not None and not piece.live_mode:  # do nothing if not live mode
                return

            image = piece.image  # X Y C
            excluded_colors: List[bool] = self.eval_widget.get_use_channel_state()

            y0, x0, y1, x1 = piece.live_area
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            if self.live_update_tensor is None:
                crop = image[x0:x1, y0:y1, :]
                for c in piece.get_adjustments():
                    brightness = float(c["brightness"])  # , dtype=np.float64)
                    contrast = float(c["contrast"])  # , dtype=np.float64)
                    _img = crop[..., c["channel"]].astype(
                        np.float64
                    )  # needs input float64
                    crop[..., c["channel"]] = _adjust(
                        _img, brightness, contrast
                    )  # output uint8

                for j, excluded in enumerate(excluded_colors):
                    if excluded:
                        crop[..., j] = crop[..., j] * 0.0

                crop: torch.Tensor = rescale_to_optimal_size(
                    torch.from_numpy(crop), px_size=piece.pixel_size_xy
                )

                self.live_update_tensor = crop

            out: List[Cell] = backend.eval_crop(
                crop=self.live_update_tensor.to(device),
                model=self.model.to(device),
                device=device,
                parent=piece,
                coords=[x0, y0],
                return_window=False,
                metadata={},
            )

            """
                    excluded_colors: List[bool] = self.eval_widget.get_use_channel_state()
        image = piece.image  # X Y C
        crop = image[x : x + w, y : y + h, :].copy()

        for j, excluded in enumerate(excluded_colors):
            if excluded:
                crop[..., j] = crop[..., j] * 0.0

            for c in piece.get_adjustments():
                brightness = float(c["brightness"])  # , dtype=np.float64)
                contrast = float(c["contrast"])  # , dtype=np.float64)
                _img = crop[..., c["channel"]].astype(
                    np.float64
                )  # needs input float64
                crop[..., c["channel"]] = _adjust(
                    _img, brightness, contrast
                )  # output uint8

            for j, excluded in enumerate(excluded_colors):
                if excluded:
                    crop[..., j] = crop[..., j] * 0.0

        crop: torch.Tensor = rescale_to_optimal_size(
            torch.from_numpy(crop), px_size=piece.pixel_size_xy
        )


        worker = Worker(
            fn=backend.eval_crop,
            crop=crop,
            model=self.model,
            device="cpu",
            parent=piece,
            coords=[x, y],
            return_window=True,
            metadata={"creator": self.eval_widget.get_model_filename()},
            begin_passthrough={"coords": (x, y, w, h)},
        )
            
            
            """

            piece.set_live_children(out)
            self.image_viewer.update()
            self.update()

    @Slot(str)
    def loadModelFile(self, file_path):
        assert os.path.exists(file_path), f"{file_path} should exist???"
        self.model = backend.model_from_path(file_path, "cpu")
        if self.image_viewer.live_eval_mode:
            self.run_live_model()

    @Slot(int)
    def updateTabSizes(self, index: int):
        """
        Updates the tab Sizes
        """
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).setSizePolicy(
                QSizePolicy.Ignored, QSizePolicy.Ignored
            )
        widget = self.tab_widget.currentWidget()
        widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        const_width = self.tab_widget.widget(index).minimumSizeHint().width()
        widget.setFixedWidth(const_width)
        self.tab_widget.setMinimumWidth(const_width)
        self.resizeDocks([self.left_dock], [const_width], Qt.Orientation.Horizontal)

    def exportXML(self):
        "fuck me"
        piece: Piece = self.cochlea.get_selected_child()
        piece_to_xml(piece)
        print("Saved to xml!")

    def importXML(self):
        piece: Piece = self.cochlea.get_selected_child()
        if piece is not None:
            filename = piece.filename
            noext = os.path.splitext(filename)[0]

            if os.path.exists(noext + ".xml"):
                cells = cells_from_xml(noext + ".xml", piece)
                piece.add_children(cells)
                self.updateChildren()
                self.update()

    def setSelectedChildrenAsGroundTruth(self):
        print("called")
        piece: Piece = self.cochlea.get_selected_child()
        if piece is not None:
            for c in piece.get_selected_children():
                if not isinstance(c, Cell):
                    continue
                c.setGroundTruth(True)
                c.score = 1.0

            self.updateChildren()
            self.update()

    def setAllVisibleChildrenAsGroundTruth(self):
        piece: Piece = self.cochlea.get_selected_child()
        # launch 'are you sure?' window
        msg = QMessageBox()
        msg.setText("Label each visible cell as ground truth?")
        msg.setInformativeText(
            "By Pressing 'Ok' you are confirming that each visible cell has been"
            " verified and properly annotated."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        ret = msg.exec()
        if ret != QMessageBox.Ok:
            return

        for id, c in piece.children.items():
            if isinstance(c, Cell):
                c.setGroundTruth(True)
                c.score = 1.0
        piece._candidate_children = {}

    def unsetSelectedChildrenAsGroundTruth(self):
        piece: Piece = self.cochlea.get_selected_child()
        if piece is not None:
            for c in piece.get_selected_children():
                if not isinstance(c, Cell):
                    continue
                c.setGroundTruth(False)
            self.updateChildren()
            self.update()

    def toggleSelectedType(self):
        piece: Piece = self.cochlea.get_selected_child()
        for c in piece.get_selected_children():
            if not isinstance(c, Cell):
                continue
            if c.type == "OHC":
                c.type = "IHC"
            else:
                c.type = "OHC"
        self.updateChildren()
        self.update()

    def _setApplicationDisabledState(self, disabled):
        self.image_viewer.setDisabled(disabled)
        self.footer.setDisabled(disabled)
        self.cell_info.setDisabled(disabled)

        self.cell_info_dock.setDisabled(disabled)
        self.piece_info_dock.setDisabled(disabled)
        self.render_style_widget.setDisabled(disabled)
        self.render_hint_widget.setDisabled(disabled)

        self.saveFileAction.setDisabled(disabled)
        self.saveAsFileAction.setDisabled(disabled)
        self.exportCSVAction.setDisabled(disabled)

        # self.tree.setDisabled(disabled)
        self.tree.delete_piece_button.setDisabled(disabled)
        # self.tree.add_piece_button.setDisabled(disabled)
        # self.tree.delete_piece_button.setDisabled(disabled)

        self.eval_widget.setDisabled(disabled)

        self.image_adjust_widget.setDisabled(disabled)
        self.image_adjust_widget.apply_adjust_button.setDisabled(disabled)
        self.image_adjust_widget.reset_all_button.setDisabled(disabled)
        self.image_adjust_widget.reset.setDisabled(disabled)

        self.evalMenu.setDisabled(disabled)
        self.viewMenu.setDisabled(disabled)
        self.editMenu.setDisabled(disabled)

        self.left_dock.setDisabled(disabled)
        self.annotate_sidebar.setDisabled(disabled)
        self.annotate_sidebar.set_move_mode_button.setDisabled(disabled)
        self.annotate_sidebar.zoom_in_button.setDisabled(disabled)
        self.annotate_sidebar.zoom_out_button.setDisabled(disabled)
        self.annotate_sidebar.reset_view_button.setDisabled(disabled)
        self.annotate_sidebar.select_cell.setDisabled(disabled)
        self.annotate_sidebar.edit_cell.setDisabled(disabled)
        self.annotate_sidebar.draw_cell.setDisabled(disabled)
        self.annotate_sidebar.edit_eval_region.setDisabled(disabled)
        self.annotate_sidebar.draw_eval_region.setDisabled(disabled)
        self.annotate_sidebar.clear_eval_region.setDisabled(disabled)
        self.annotate_sidebar.select_synapse.setDisabled(disabled)
        self.annotate_sidebar.draw_synapse.setDisabled(disabled)
        self.annotate_sidebar.edit_synapse.setDisabled(disabled)
        self.annotate_sidebar.draw_freq_region.setDisabled(disabled)
        self.annotate_sidebar.edit_freq_region.setDisabled(disabled)
        self.annotate_sidebar.switch_freq_region.setDisabled(disabled)
        self.annotate_sidebar.clear_freq_region.setDisabled(disabled)

        self.update()

    def _setApplicationEnabledState(self, enabled):
        self.image_viewer.setEnabled(enabled)
        self.footer.setEnabled(enabled)
        self.cell_info.setEnabled(enabled)

        self.cell_info_dock.setEnabled(enabled)
        self.piece_info_dock.setEnabled(enabled)
        self.render_style_widget.setEnabled(enabled)
        self.render_hint_widget.setEnabled(enabled)

        # self.tree.setDisabled(disabled)
        self.tree.delete_piece_button.setEnabled(enabled)
        # self.tree.add_piece_button.setDisabled(disabled)
        # self.tree.delete_piece_button.setDisabled(disabled)

        self.eval_widget.setEnabled(enabled)

        self.saveFileAction.setEnabled(enabled)
        self.saveAsFileAction.setEnabled(enabled)
        self.exportCSVAction.setEnabled(enabled)

        self.image_adjust_widget.setEnabled(enabled)
        self.image_adjust_widget.apply_adjust_button.setEnabled(enabled)
        self.image_adjust_widget.reset_all_button.setEnabled(enabled)
        self.image_adjust_widget.reset.setEnabled(enabled)

        self.evalMenu.setEnabled(enabled)
        self.viewMenu.setEnabled(enabled)
        self.editMenu.setEnabled(enabled)

        self.left_dock.setEnabled(enabled)
        self.annotate_sidebar.setEnabled(enabled)
        self.annotate_sidebar.set_move_mode_button.setEnabled(enabled)
        self.annotate_sidebar.zoom_in_button.setEnabled(enabled)
        self.annotate_sidebar.zoom_out_button.setEnabled(enabled)
        self.annotate_sidebar.reset_view_button.setEnabled(enabled)
        self.annotate_sidebar.select_cell.setEnabled(enabled)
        self.annotate_sidebar.edit_cell.setEnabled(enabled)
        self.annotate_sidebar.draw_cell.setEnabled(enabled)
        self.annotate_sidebar.edit_eval_region.setEnabled(enabled)
        self.annotate_sidebar.draw_eval_region.setEnabled(enabled)
        self.annotate_sidebar.clear_eval_region.setEnabled(enabled)
        self.annotate_sidebar.select_synapse.setEnabled(enabled)
        self.annotate_sidebar.draw_synapse.setEnabled(enabled)
        self.annotate_sidebar.edit_synapse.setEnabled(enabled)
        self.annotate_sidebar.draw_freq_region.setEnabled(enabled)
        self.annotate_sidebar.edit_freq_region.setEnabled(enabled)
        self.annotate_sidebar.switch_freq_region.setEnabled(enabled)
        self.annotate_sidebar.clear_freq_region.setEnabled(enabled)

        self.update()

    def setApplicationDisabled(self):
        self._setApplicationEnabledState(False)
        self._setApplicationDisabledState(True)

    def setApplicationEnabled(self):
        self._setApplicationEnabledState(True)
        self._setApplicationDisabledState(False)

    def createActions(self):
        ### EDIT ###

        self.deleteLatestActiveChild = QAction(
            "&Delete",
            self,
            shortcut=Qt.Key_Delete,
            enabled=True,
            triggered=self.image_viewer.deleteActiveChildren,
        )

        ### VIEW ###
        # self.toggleShowCellBoxesModeAction = QAction(
        #     "Show Cell Boxes",
        #     self,
        #     checkable=True,
        #     checked=True,
        #     enabled=True,
        #     triggered=self.image_viewer.toggleShowCellBoxesMode,
        # )
        # self.toggleShowCellTypeModeAction = QAction(
        #     "Show Cell Types",
        #     self,
        #     checkable=True,
        #     checked=True,
        #     enabled=True,
        #     triggered=self.image_viewer.toggleShowCellTypeMode,
        # )

        self.zoomInAct = QAction(
            "Zoom &In (10%)",
            self,
            shortcut="Ctrl++",
            enabled=True,
            triggered=self.image_viewer.zoomIn,
        )

        self.zoomOutAct = QAction(
            "Zoom &Out (10%)",
            self,
            shortcut="Ctrl+-",
            enabled=True,
            triggered=self.image_viewer.zoomOut,
        )

        self.normalSizeAct = QAction(
            "&Normal Size",
            self,
            shortcut="Ctrl+R",
            enabled=True,
            triggered=self.image_viewer.resetImageToViewport,
        )
        # Open files
        self.loadFileAction = QAction(
            "Open",
            self,
            shortcut="Ctrl+O",
            checkable=False,
            enabled=True,
            triggered=self.loadWhorlStateFile,
        )

        ### FILE ###
        self.importAct = QAction(
            "&Import", self, shortcut="Ctrl+I", enabled=True, triggered=self.importImage
        )

        # Save files
        self.saveFileAction = QAction(
            "&Save",
            self,
            shortcut="Ctrl+S",
            enabled=True,
            triggered=self.saveWhorlStateFile,
        )
        self.saveAsFileAction = QAction(
            "&Save As",
            self,
            shortcut="Ctrl+Shift+S",
            enabled=True,
            triggered=self.saveAsWhorlStateFile,
        )

        self.exportCSVAction = QAction(
            "&Export as CSV",
            self,
            shortcut="Ctrl+E",
            enabled=True,
            triggered=self.exportCSV,
        )

        self.exportXMLAction = QAction(
            " &Export",
            self,
            shortcut="Ctrl+Shift+Alt+E",
            enabled=True,
            triggered=self.exportXML,
        )
        self.importXMLAction = QAction(
            " &XML Import",
            self,
            shortcut="Ctrl+Shift+Alt+I",
            enabled=True,
            triggered=self.importXML,
        )

        ### EVAL MODEL ###
        # Eval Action
        self.quickEvalAction = QAction(
            "&Eval Model",
            self,
            shortcut="Ctrl+Shift+R",
            enabled=True,
            triggered=self.quickEval,
        )
        self.loadModelFileAction = QAction(
            "Load Model File", self, enabled=True, triggered=self.chooseModelFile
        )

        self.setGTAction = QAction(
            "Set Cell as &Ground Truth",
            self,
            shortcut="Ctrl+G",
            enabled=True,
            triggered=self.setSelectedChildrenAsGroundTruth,
        )
        self.unsetGTAction = QAction(
            "Set Selected Cells as &Ground Truth",
            self,
            shortcut="Ctrl+Shift+G",
            enabled=True,
            triggered=self.unsetSelectedChildrenAsGroundTruth,
        )
        self.batch_quick_eval_action  = QAction(
            "Batch Quick &Eval",
            self,
            shortcut="Ctrl+Shift+E",
            enabled=True,
            triggered=self.batch_quick_eval,
        )

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.importAct)
        self.fileMenu.addAction(self.loadFileAction)
        self.fileMenu.addAction(self.saveFileAction)
        self.fileMenu.addAction(self.saveAsFileAction)
        self.fileMenu.addAction(self.exportCSVAction)
        self.fileMenu.addAction(self.exportXMLAction)
        self.fileMenu.addAction(self.importXMLAction)

        self.editMenu = QMenu("&Edit", self)
        self.editMenu.addAction(self.deleteLatestActiveChild)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()

        # Adjust Images
        # self.viewMenu.addAction(self.increaseBrightnessAct)
        # self.viewMenu.addAction(self.decreateBrightnessAct)
        # self.viewMenu.addAction(self.increaseContrastAct)
        # self.viewMenu.addAction(self.decreaseContrastAct)
        # self.viewMenu.addAction(self.resetImageAdjustmentsAct)

        # Set render views... one for each.
        # boxes
        # diagnostic
        # etc...

        # self.viewMenu.addAction(self.toggleShowCellBoxesModeAction)
        # self.viewMenu.addAction(self.toggleShowCellTypeModeAction)
        self.viewMenu.addSeparator()

        self.evalMenu = QMenu("Eval", self)
        self.evalMenu.addAction(self.quickEvalAction)
        self.evalMenu.addAction(self.loadModelFileAction)
        self.evalMenu.addAction(self.setGTAction)
        self.evalMenu.addAction(self.unsetGTAction)
        self.evalMenu.addAction(self.batch_quick_eval_action)

        # eval
        # assign freq
        # delete model
        # choose new model
        # increase cell thr
        # decrease cell thr
        # increase nms thr
        # decrease nms thr
        # run full analysis

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.editMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.evalMenu)

