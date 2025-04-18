from copy import copy
from typing import List, Tuple, Dict

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


import hcat.lib.roi
from hcat.lib.analyze import assign_frequency, assert_validity
from hcat.lib.utils import hex_to_qcolor
from hcat.state import Piece, Cell, Synapse, StateItem

"""
TODO:

Add render hints allowing user to change color and transparency of EVERYTHING
Add render hints allowing selective rendering of certain objects
Add ability for these render hints to be controlled by additional applicaiton widget

"""

CYAN = QColor(0, 255, 255)
MAGENTA = QColor(255, 0, 255)
GREY = QColor(200, 200, 200)


class PieceViewerWidget(QWidget):
    rescaled = Signal(float)  # when image is re-scaled
    updated_state = Signal()  # when a child is added to the piece
    children_removed = Signal(list)  # when a child(ren) is removed
    mouse_position_signal = Signal(list)  # emitted when mouse moves (for footer)
    selected_child_changed = Signal(list)  # selected child changed
    liveUpdateRegionChanged = Signal(list)  # live mode dragged around
    modeChanged = Signal(str)
    tonotopyPathEdited = Signal(bool)

    """
    Canvas! This draws everything
    Every time the state is changed, we emit a signal to main
    """

    def __init__(self, piece: Piece = None):
        super(PieceViewerWidget, self).__init__()
        palette = QPalette()
        palette.setColor(QPalette.Window, "gray")
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        # state
        self.active_piece: Piece = piece

        # Where we draw the image
        self.label = QLabel()
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap = None

        # Constants about where we are looking...
        self.center = QPointF(0.0, 0.0)
        self.painter = QPainter()
        self.image = None
        self._last_drag_pos = QPointF(0.0, 0.0)
        self._pixmap_offset = QPointF()
        self._dragging = [False, False]  # flags for it we are moving around..

        self._last_center_ratio = (0.0, 0.0)
        self._current_center_ratio = (0.0, 0.0)

        # Interactivity modes --------------
        self.move_mode = True

        # eval region lets users draw a polygon to define an eval region
        self.draw_eval_region_mode = False
        self.edit_eval_region_mode = False

        # Cell boxes
        self.select_cell_box_mode = False
        self.annotate_cell_box_mode = False
        self.edit_cell_box_mode = False
        self.move_cell_box_mode = False

        # Synapse Modes
        self.select_synapse_mode = False
        self.annotate_synapse_mode = False
        self.edit_synapse_mode = False

        # Display modes --------------------
        self.show_cell_outlines_mode = True
        self.show_cell_boxes_mode = True
        self.show_cell_type_mode = True

        # freq region mode
        self.annotating_freq_mode = False
        self.edit_freq_mode = False

        # live eval ----------------
        self.live_eval_mode = False
        self.live_eval_square = [0, 0, 255, 255]  # x0 y0 x1 y1
        self.past_live_eval_mouse_pos = None

        # Render Hints ---------------------
        self.render_hints = {
            "box": True,
            "diagnostic": False,
            "eval_region": True,
            "freq_region": True,
            "synapse": True,
        }
        self.ihc_color = QColor(255, 0, 255)
        self.ohc_color = QColor(255, 255, 0)
        self.eval_region_color = QColor(200, 200, 200)
        self.freq_region_color = QColor()
        self.synapse_color = [QColor(), QColor()]

        self.basal_point_color = QColor(55, 126, 184)
        self.apex_point_color = QColor(228, 26, 27)
        self.freq_line_color = QColor(255, 255, 255, 255)
        self.freq_line_stroke = 2

        self.ihc_stroke: int = 1
        self.ohc_stroke: int = 1
        self.ohc_transparency: int = 128
        self.ihc_transparency: int = 128
        self.eval_region_stroke: int = 1
        self.eval_region_transparency: int = 100

        # Cell Outline Annotation Point Storage
        self._cell_points = []

        # annotate cell type
        self._cell_type = "IHC"

        # Cell Box Annotation. We need explicit control over each corner I think
        self._cell_box_corner_0: QPointF = None
        self._cell_box_corner_1: QPointF = None

        # Cell box annotation, for adjusting the edges of a box
        self._cell_held_edge = None  # [left, right, top, bottom] [0, 1, 2, 3]
        self._cell_edge_previous = None

        # cell dragging around, need to know the start and the end
        self._cell_bbox_start = None

        # Cell box editing
        self._cell_to_edit = None
        self._corner_index_of_edit = None
        self._opposite_corner_of_editable_box = None

        # polygon for eval region
        self._is_drawing_eval_region = (
            False  # true if user is activley drawing the region
        )
        self._eval_mouse_position = None
        self._eval_region_edit_vertex_index = None

        # cordinates of selection box:
        self._select_region_corner_0: QPointF = None  # expect List[QPointF, ...]
        self._select_region_corner_1: QPointF = None

        # synapses
        self._synapse_pos_0 = []
        self._synapse_pos_1 = []

        # freq region
        self._freq_path_buffer = []
        self._freq_path = []
        self._freq_mouse_pos = []
        self._freq_path_edit_vertex_index = None

        # window of eval_region
        self._eval_window: List[int] | None = None

        # edit synapses
        self._synapse_to_edit = None
        self._keypoint_index_of_edit = None

        # image viewer state
        self.scale = 1.0
        self._temp_ratio = (0.0, 0.0)
        # self._undo_action_available()

        # where is the mouse?
        self.mouse_position = QPointF()
        self.setMouseTracking(True)

        # Points
        self.points = []

    def deleteActiveChildren(self):
        """only deletes children of the piece, not the piece itself"""
        if self.active_piece is not None and self.active_piece.get_selected_children():
            selected_children: List[
                StateItem
            ] = self.active_piece.get_selected_children()

            to_remove = []
            for child in selected_children:
                removed_item: StateItem = child.get_parent().remove_child(child.id)
                # index = removed_item.tree_item.index
                # _: QTreeWidgetItem = removed_item.get_parent().tree_item.takeChild(index)

                to_remove.append(removed_item)

            self.children_removed.emit(to_remove)

        # If there is nothing to delete, we may want to delete the piece...
        else:
            pass

    def resetImageViewer(self):
        """returns image viewer to basically none"""
        self.image = None
        self.active_piece: Piece = None
        self.pixmap = None

        # Constants about where we are looking...
        self.center = QPointF(0.0, 0.0)
        self.painter = QPainter()
        self.image = None
        self._last_drag_pos = None
        self._pixmap_offset = QPointF()

        self._last_center_ratio = (0.0, 0.0)
        self._current_center_ratio = (0.0, 0.0)
        self.resetAllAnnotationState()

    def resetAllAnnotationState(self):
        """
        Meant to be called by an escape key in the main application.
        forcefully intercepts everything.
        """
        self._cell_points = []
        self._cell_box_corner_0: QPointF = None
        self._cell_box_corner_1: QPointF = None

        # Cell box annotation, for adjusting the edges of a box
        self._cell_held_edge = None  # [left, right, top, bottom] [0, 1, 2, 3]
        self._cell_edge_previous = None

        # cell dragging around, need to know the start and the end
        self._cell_bbox_start = None

        # Cell box editing
        self._cell_to_edit = None
        self._corner_index_of_edit = None
        self._opposite_corner_of_editable_box = None

        # polygon for eval region
        if self._is_drawing_eval_region:
            self.active_piece.eval_region = []
            self.active_piece.eval_region_boxes = []
        self._eval_mouse_position = None
        self._eval_region_edit_vertex_index = None
        self._is_drawing_eval_region = False

        # cordinates of selection box:
        self._select_region_corner_0: QPointF = None  # expect List[QPointF, ...]
        self._select_region_corner_1: QPointF = None

        # synapse
        self._synapse_pos_0 = []
        self._synapse_pos_1 = []

        # freq region
        self._freq_path_buffer = []
        self._freq_mouse_pos = []

        self.update()

    def addGTCellToParent(
        self, verticies: List[QPointF] = None, bbox: List[int | float] = None
    ):
        """
        SHOULD THIS BE HERE?

        eventually, this will add a cell from a specified region
        with a dedicated process. Likely a dialog box which lets the user
        delinate a cell themselves. This includes a template cell.

        For now, we add one cell, and call it good
        We also need a way to select cells.

        bbox = [x0, y0, x1, y1]
        vertifi
        """
        _id = 0
        while _id in self.active_piece._candidate_children.keys():
            _id += 1

        new_cell = Cell(
            id=_id,
            mask_verticies=verticies,
            bbox=bbox,
            parent=self.active_piece,
            score=1.0,
        )
        new_cell.setGroundTruth(True)
        new_cell.set_type(self._cell_type)
        new_cell.set_creator("user")
        new_cell.set_image_adjustments_at_creation(self.active_piece.adjustments)


        self.active_piece.add_child(new_cell)
        self.active_piece.set_selected_children([new_cell.id])

        is_valid, msg = assert_validity(self.active_piece.parent)
        if is_valid:
            assign_frequency(new_cell)

        self.selected_child_changed.emit(self.active_piece.get_selected_children())
        self.updated_state.emit()

    def addSynapseToParent(self, x0, y0, x1, y1):
        """
        Adds a synapse to the active child in the active piece...
        Creates a new synapse object, and adds it to the active child.
        If the active child is a cell, then the synapse is added to the cell,
        if no cell has been created, we add to the synapse to the piece.
        Because it changes the state, we emit a signal telling the rest of the program to
        refresh.

        """
        _id = 0
        while _id in self.active_piece._candidate_children.keys():
            _id += 1

        new_synapse = Synapse(
            x0, y0, x1, y1, id=_id, mask=None, parent=self.active_piece, score=1.0
        )
        self.active_piece.add_child(new_synapse)
        self.active_piece.set_selected_children([new_synapse.id])
        self.selected_child_changed.emit(self.active_piece.get_selected_children())
        self.updated_state.emit()

    def disableAllInteractivityModes(self):
        # Interactivity modes --------------
        self.move_mode = False

        # Cell boxes
        self.select_cell_box_mode = False
        self.annotate_cell_box_mode = False
        self.edit_cell_box_mode = False
        self.move_cell_box_mode = False
        self.draw_eval_region_mode = False
        self.edit_eval_region_mode = False
        self.live_eval_mode = False
        self.select_synapse_mode = False
        self.annotate_synapse_mode = False
        self.edit_synapse_mode = False
        self.annotating_freq_mode = False
        self.edit_freq_mode = False


        # Clear all animation drawing...

        self._cell_box_corner_0: QPointF = None
        self._cell_box_corner_1: QPointF = None
        self._cell_held_edge = None  # [left, right, top, bottom] [0, 1, 2, 3]
        self._cell_edge_previous = None
        self._cell_bbox_start = None
        self._cell_to_edit = None
        self._corner_index_of_edit = None
        self._opposite_corner_of_editable_box = None
        # polygon for eval region
        if self._is_drawing_eval_region:
            # true if user is activley drawing the region
            self._is_drawing_eval_region = False
            self._eval_mouse_position = None
            self._eval_region_edit_vertex_index = None
            _verts = [(v.x(), v.y()) for v in self.active_piece.eval_region]

            scale = (
                289 / self.active_piece.pixel_size_xy
                if self.active_piece.pixel_size_xy is not None
                else 1
            )
            self.active_piece.eval_region_boxes = hcat.lib.roi.get_squares(
                _verts, round(256 * scale), 0.1
            )
            self.updated_state.emit()


        # cordinates of selection box:
        self._select_region_corner_0: QPointF = None  # expect List[QPointF, ...]
        self._select_region_corner_1: QPointF = None

        # synapses
        self._synapse_pos_0 = []
        self._synapse_pos_1 = []

        # freq region
        if self._freq_path_buffer:
            self._freq_path_buffer = []
            self.tonotopyPathEdited.emit(True)

        self._freq_path = []
        self._freq_mouse_pos = []
        self._freq_path_edit_vertex_index = None


    def emitCurrentMode(self):
        if self.move_mode:
            text = "MOVE MODE"
        elif self.annotate_cell_box_mode:
            text = "ANNOTATE CELL BBOX MODE"
        elif self.edit_cell_box_mode:
            text = "EDIT CELL BBOX MODE"
        elif self.draw_eval_region_mode:
            text = "DRAW EVAL REGION MODE"
        elif self.edit_eval_region_mode:
            text = "EDIT EVAL REGION MODE"
        elif self.select_cell_box_mode:
            text = "SELECT CELL MODE"
        elif self.select_synapse_mode:
            text = "SELECT SYNAPSE MODE"
        elif self.edit_synapse_mode:
            text = "EDIT SYNAPSE MODE"
        elif self.annotate_synapse_mode:
            text = "DRAW SYNAPSE MODE"
        elif self.live_eval_mode:
            text = "LIVE EVAL MODE"
        elif self.annotating_freq_mode:
            text = "DRAW FREQ PATH MODE"
        elif self.edit_freq_mode:
            text = "EDIT FREQ PATH MODE"
        else:
            raise RuntimeError("UNKOWN MODE")

        self.modeChanged.emit(text)

    def enableAnnotateFreqMode(self):
        self.disableAllInteractivityModes()
        self.annotating_freq_mode = True
        self.emitCurrentMode()
        self.update()

    def enableEditFreqMode(self):
        self.disableAllInteractivityModes()
        self.edit_freq_mode = True
        self.emitCurrentMode()
        self.update()

    def enableSelectSynapseMode(self):
        self.disableAllInteractivityModes()
        self.select_synapse_mode = True
        self.emitCurrentMode()
        self.update()

    def enableAnnotateSynapseMode(self):
        self.disableAllInteractivityModes()
        self.annotate_synapse_mode = True
        self.emitCurrentMode()
        self.update()

    def enableEditSynapseMode(self):
        self.disableAllInteractivityModes()
        self.edit_synapse_mode = True
        self.emitCurrentMode()
        self.update()

    def enableLiveEvalmode(self):
        self.disableAllInteractivityModes()
        self.live_eval_mode = True
        self.liveUpdateRegionChanged.emit(self.live_eval_square)
        self.emitCurrentMode()
        self.update()

    def enableDrawEvalRegionMode(self):
        self.disableAllInteractivityModes()
        self.draw_eval_region_mode = True
        self.emitCurrentMode()
        self.update()

    def enableEditEvalRegionMode(self):
        self.disableAllInteractivityModes()
        self.edit_eval_region_mode = True
        self.active_piece.deselect()
        self.emitCurrentMode()
        self.update()

    def enableMoveMode(self):
        self.disableAllInteractivityModes()
        self.move_mode = True
        self.emitCurrentMode()
        self.update()

    def enableAnnotateCellBoxMode(self):
        self.disableAllInteractivityModes()
        self.annotate_cell_box_mode = True
        self.emitCurrentMode()
        self.update()

    def enableEditCellBoxMode(self):
        self.disableAllInteractivityModes()
        self.edit_cell_box_mode = True
        self.emitCurrentMode()
        self.update()

    def enableSelectBoxMode(self):
        self.disableAllInteractivityModes()
        self.select_cell_box_mode = True
        self.emitCurrentMode()
        self.update()

    def toggleShowCellBoxesMode(self):
        self.show_cell_boxes_mode = not self.show_cell_boxes_mode
        self.emitCurrentMode()
        self.update()

    def toggleShowCellTypeMode(self):
        self.show_cell_type_mode = not self.show_cell_type_mode
        self.emitCurrentMode()
        self.update()

    @Slot(dict)
    def setRenderHint(self, hint):
        self.render_hints = hint
        self.update()

    def setActivePiece(self, piece: Piece = None, goto_cell: bool = False):
        """
        sets the active piece of the image viewer.
        is supposed to be called outside this widget!
        """
        if piece is None:
            return

        scale = copy(self.scale)
        self.active_piece = piece
        self.setImage(self.active_piece.get_image_buffer_as_qimage())

        if goto_cell:  # if a cell was clicked, we center the active cell
            self.resetImageToViewport()
            factor = scale / self.scale
            self.zoomBy(factor)
            """
            pos = event.position()
            _x = (pos.x() / self.scale) - self._pixmap_offset.x()
            _y = (pos.y() / self.scale) - self._pixmap_offset.y()
            
            po + x = (w/2) / self.scale
            """

            child: Cell = self.active_piece.get_selected_children()[0]
            x0, y0, x1, y1 = child.bbox
            cx, cy = (x1 - x0) / 2 + x0, (y1 - y0) / 2 + y0
            _pox = (self.width() / 2 / self.scale) - cx
            _poy = (self.height() / 2 / self.scale) - cy
            self._pixmap_offset = QPointF(_pox, _poy)
            self.update()

        else:
            self.resetImageToViewport()

    def adjustImage(self, c: Dict[str, int] | List[Dict[str, int]]):
        """sets adjustment val in active piece, redraws the pixmap"""
        if self.active_piece is not None:
            self.active_piece.apply_adjustment(c)
            self.pixmap = QPixmap.fromImage(
                self.active_piece.get_image_buffer_as_qimage()
            )
            self.update()

    def clearROI(self):
        if self.active_piece is not None:
            self.active_piece.remove_eval_roi()
            self.update()

    def reverseFreqPath(self):
        # can reverse when the user is not drawing a line...
        if self.active_piece is not None and len(self._freq_path_buffer) == 0:
            self.active_piece.reverse_freq_path()
            self._freq_path.reverse()
            self._freq_path_buffer.reverse()
            self.tonotopyPathEdited.emit(True)
        self.update()

    def clearFreqPath(self):
        if self.active_piece is not None:
            self.active_piece.clear_freq_path()
            self._freq_path = []
            self._freq_path_buffer = []
            self._freq_mouse_pos = []
            self.update()

    @Slot(str)
    def setAnotatingCellType(self, type: str):
        if type in ["OHC", "IHC"]:
            self._cell_type = "OHC" if type == "OHC" else "IHC"
        else:
            raise ValueError(
                f"Tried to set annotating cell type to somethign nonsensical: {type}"
            )

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(PieceViewerWidget, self).minimumSizeHint()

    def _calculate_center_ratio(self):
        """sets _current_center_ratio"""
        if self.pixmap is not None:
            screen_size: QPointF = self._get_viewport_size()
            image_size: QPointF = self._get_pixmap_size()
            center = screen_size / 2

            center_ratio_x: float = (center.x() - self._pixmap_offset.x()) / (
                image_size.x()
            )
            center_ratio_y: float = (center.y() - self._pixmap_offset.y()) / (
                image_size.y()
            )

            self._current_center_ratio: Tuple[float, float] = (
                center_ratio_x,
                center_ratio_y,
            )

    def zoomBy(self, scalefactor: float, offset: QPointF | None = None):
        """
        Zooms by a scale factor up or down. scale factor must be POSITIV

        :param scalefactor: scale factor. 1.1 zooms by in by 10% at 110% zoom.
        :return:
        """
        if scalefactor > 1.0 and self.scale >= 20:
            scalefactor = 1.0
        elif scalefactor < 1.0 and self.scale < 0.01:
            scalefactor = 1.0

        # we need viewport size before and after scaling to zoom to the center
        _ss0 = self._get_viewport_size()  # init viewport size
        self.scale = self.scale * scalefactor
        _ss1 = self._get_viewport_size()  # after viewport size

        # how much do we change the pixmap offset such that we zoom to the center...
        dx = 0.5 * (_ss0.x() - _ss1.x())
        dy = 0.5 * (_ss0.y() - _ss1.y())

        # set the new _pixmap_offset
        self._pixmap_offset = QPointF(
            self._pixmap_offset.x() - dx, self._pixmap_offset.y() - dy
        )

        self.rescaled.emit(scalefactor)
        self.update()

    def zoomIn(self):
        self.zoomBy(1.1)

    def zoomOut(self):
        self.zoomBy(1 / 1.1)

    def zoomReset(self):
        self.scale = 1.0

    def zoomToScreen(self):
        """
        zooms to fit the entire image in the screen.
        DOES NOT place the image in the center of the screen. For that, go to self.resetImageToViewport()
        """
        if self.pixmap is not None:
            # get height of the current view and of the pixmap (image)
            w0, h0 = self.pixmap.width(), self.pixmap.height()
            w1, h1 = self.width(), self.height()

            if w0 > w1 or h0 > h1:  # zooms out if the image is much smaller
                _scale = min(w1 / w0, h1 / h0)
            else:  # zooms in otherwise
                _scale = max(w1 / w0, h1 / h0)
        else:
            _scale = 1

        self.zoomReset()  # have to reset the zoom first.
        self.zoomBy(_scale)

    def _get_viewport_size(self):
        return QPointF(self.width() / self.scale, self.height() / self.scale)

    def _get_pixmap_size(self):
        return QPointF(self.pixmap.width(), self.pixmap.height())

    def point_outside_pixmap(self, x, y) -> bool:
        x0, y0 = self._pixmap_offset

    def setImage(self, image: QImage):
        """gets a pixmap from the image, assigns to self.pixmap, and calls a pixmap update"""
        self.image = image
        self.pixmap = QPixmap.fromImage(self.image)
        self.zoomToScreen()
        self.update()

    def setEvalWindow(self, window: list):
        self._eval_window = window

    def clearEvalWindow(self):
        self._eval_window = None
        self.update()

    def resetImageToViewport(self):
        """
        Set the current zoom to the perfectly fit the image into the viewport.
        Centers the image as best it can. Calls a view refresh.
        """
        self.zoomToScreen()

        # now we center the image...
        if self.pixmap is not None:
            w, h = self.pixmap.width(), self.pixmap.height()
            sw, sh = (
                self.width() / self.scale,
                self.height() / self.scale,
            )  # screenwidth, screenheight
            x = (sw - w) / 2
            y = (sh - h) / 2
            self._pixmap_offset = QPointF(x, y)
            self.zoomBy(
                1.0
            )  # we have to do this to update the _pixmap_offset... kinda dumb... im dumb..
            self.update()

    def resetPixmapFromActivePiece(self):
        # why do
        self.active_piece.reset_live_mode_adjustment()
        self.active_piece.clear_live_mode_children()
        self.pixmap = QPixmap.fromImage(self.active_piece.get_image_buffer_as_qimage())
        self.update()

    def wheelEvent(self, event):
        """zooms in and out if you scroll"""
        num_degrees = event.angleDelta().y() / 8
        num_steps = num_degrees / 15.0
        num_steps = min(num_steps, 0.1) if num_steps > 0 else max(num_steps, -0.1)
        self.zoomBy(pow(0.1, num_steps))

    def mousePressEvent(self, event):
        """
        Mouse Press Event happens when the user clicks the mouse. We have two systems
        1. Move mode: lets the user drag the screen around.
        2. Annotate Mode: lets the user add synapses.

        :param event:
        :type event:
        :return:
        :rtype:
        """

        pos = event.position()
        _x = (pos.x() / self.scale) - self._pixmap_offset.x()
        _y = (pos.y() / self.scale) - self._pixmap_offset.y()

        # Drag the screen around!
        if (
            event.buttons() == Qt.MiddleButton or event.buttons() == Qt.RightButton
        ) and not self._dragging[0]:
            self._dragging[1] = True
            self._last_drag_pos = event.position()

        # This is such that if someone tries to click while already dragging, it just fails..
        if event.buttons() == Qt.LeftButton|Qt.RightButton and self._dragging[1]:
            self._last_drag_pos = QPointF()
            self._dragging[1] = False
        elif (
            event.buttons() == Qt.MiddleButton|Qt.LeftButton or event.buttons() == Qt.LeftButton|Qt.RightButton
        ) and self._dragging[0]:
            self._last_drag_pos = QPointF()
            self._dragging[0] = False

        if event.buttons() != Qt.LeftButton:
            self.update()
            return

        if self.move_mode and not self._dragging[1]:
            self._dragging[0] = True
            self._last_drag_pos = event.position()

        # Draw a box
        if self.annotate_cell_box_mode:
            self.show_cell_boxes_mode = True
            self._cell_box_corner_0 = QPointF(_x, _y)

        # Click a corner or edge to edit a box...
        # only edit a box if we're displaying a box
        if self.edit_cell_box_mode and self.render_hints["box"]:
            _min_dist = 1e10
            _closest_coord_index = None
            _closest_cell = None

            if self.active_piece.children:  # check if we are touching an corner
                for id, c in self.active_piece.children.items():
                    if isinstance(c, Cell):
                        corner_distance, corner_ind = c.dist_to_closest_corner(_x, _y)

                        if corner_distance < (
                            50 / self.scale
                        ):  # grabable distance threshold
                            self._cell_to_edit = c
                            self._corner_index_of_edit = corner_ind
                            self._opposite_corner_of_editable_box = (
                                c.get_opposite_corner_by_index(corner_ind)
                            )

                            self.active_piece.deselect_all()
                            c.select()
                            self.updated_state.emit()
                            break

            if (
                self.active_piece.children and self._cell_to_edit is None
            ):  # if not corner, how about an edge?
                for id, c in self.active_piece.children.items():
                    if isinstance(c, Cell):
                        edge_distance, edge_ind = c.dist_to_closest_edge(
                            _x, _y
                        )  # are we touching an edge
                        if edge_distance < (50 / self.scale):
                            self._cell_to_edit = c
                            self._cell_held_edge = edge_ind
                            self._cell_edge_previous = c.bbox[edge_ind]

                            self.active_piece.deselect_all()
                            c.select()
                            self.updated_state.emit()
                            break

            # We should be able to move a cell by clicking its center
            selected = self.active_piece.get_selected_children()
            for c in selected:  # self.active_piece.children.items():
                if isinstance(c, Cell) and self._cell_to_edit is None:
                    center_distance = c.dist_to_bbox_center(_x, _y)
                    if center_distance < (50 / self.scale):
                        self._cell_bbox_start = c.bbox
                        self._cell_to_edit = c

        # draw a region for evaluation
        if self.draw_eval_region_mode:
            self._eval_mouse_position = QPointF(pos.x(), pos.y())
            # only do this when there hasnt been a eval region drawn yet
            # I.e. its the first point
            if not self._is_drawing_eval_region and not self.active_piece.eval_region:
                self._is_drawing_eval_region = True

            # this is if the user has already drawn a polygon
            elif not self._is_drawing_eval_region and self.active_piece.eval_region:
                self.active_piece.clear_eval_region()
                self._is_drawing_eval_region = True

            self.active_piece.eval_region.append(QPointF(_x, _y))

        # edit the polygon of the eval region
        if self.edit_eval_region_mode and self.active_piece.eval_region:
            verts = self.active_piece.eval_region
            dist = [(p.x() - _x) ** 2 + (p.y() - _y) ** 2 for p in verts]

            closest_index = 0
            closest_dist = 1e10
            for i, d in enumerate(dist):
                if d < closest_dist:
                    closest_dist = d
                    closest_index = i
            if closest_dist < (200 / self.scale):
                self._eval_region_edit_vertex_index = closest_index
                self.active_piece.eval_region_boxes = []

        # edit the polygon of the freq path
        if self.edit_freq_mode and self.active_piece.freq_path:
            verts = self.active_piece.freq_path
            dist = [(x - _x) ** 2 + (y - _y) ** 2 for x, y in verts]

            closest_index = 0
            closest_dist = 1e10
            for i, d in enumerate(dist):
                if d < closest_dist:
                    closest_dist = d
                    closest_index = i
            if closest_dist < (200 / self.scale):
                self._freq_path_edit_vertex_index = closest_index


        # draw a square for selecting boxes
        if self.select_cell_box_mode or self.select_synapse_mode:
            self._select_region_corner_0 = QPointF(_x, _y)

        # select a cell by pressing a cell
        if self.select_cell_box_mode:  # if not corner, how about an edge?
            for id, c in self.active_piece.children.items():
                if isinstance(c, Cell):
                    edge_distance, edge_ind = c.dist_to_closest_edge(
                        _x, _y
                    )  # are we touching an edge
                    if edge_distance < (200 / self.scale):
                        self.active_piece.deselect_all()
                        c.select()
                        self.updated_state.emit()
                        break

        # click and drag the live update box around
        if self.live_eval_mode:
            x0, y0, x1, y1 = self.live_eval_square
            X_IN = x0 <= _x <= x1
            Y_IN = y0 <= _y <= y1
            if X_IN and Y_IN:
                self.past_live_eval_mouse_pos = [_x, _y]
                self.resetPixmapFromActivePiece()

        # Draw Synapses
        if self.annotate_synapse_mode:
            self._synapse_pos_0 = [_x, _y]
            self._synapse_pos_1 = [_x, _y]

        # Edit Synapses
        # Click a corner or edge to edit a box...
        if self.edit_synapse_mode and self.render_hints["synapse"]:
            _min_dist = float("inf")
            _closest_coord_index = None
            _closest_cell = None

            if self.active_piece.children:  # check if we are touching an corner
                for id, s in self.active_piece.children.items():
                    if isinstance(s, Synapse):
                        keypoint_distance, keypoint_ind = s.dist_to_closest_keypoint(
                            _x, _y
                        )

                        if keypoint_distance < (
                            50 / self.scale
                        ):  # grabable distance threshold
                            self._synapse_to_edit = s
                            self._keypoint_index_of_edit = keypoint_ind

                            self.active_piece.deselect_all()
                            s.select()
                            self.updated_state.emit()
                            break

        # draw a frequency path...
        if self.annotating_freq_mode:
            # already have a freq path, but starting to draw a new one...
            if not self._freq_path_buffer and self.active_piece.freq_path:
                self.active_piece.freq_path = []
            self._freq_path_buffer.append((_x, _y))

        self.updated_state.emit()
        self.update()

    def mouseMoveEvent(self, event):
        """
        The mouse may move, and we wish to do various things when it does.

        :param event:
        :type event:
        :return:
        :rtype:
        """
        self.mouse_position = event.position()

        pos = event.position()
        _x = (pos.x() / self.scale) - self._pixmap_offset.x()
        _y = (pos.y() / self.scale) - self._pixmap_offset.y()

        self.mouse_position_signal.emit([_x, _y])

        # Moves the pixmap around...
        if (
            (event.buttons() == Qt.LeftButton and self.move_mode)
            or event.buttons() == Qt.MiddleButton
            or event.buttons() == Qt.RightButton
        ) and any(self._dragging):
            pos = event.position()
            if self._last_drag_pos is not None:
                self._pixmap_offset += (pos - self._last_drag_pos) / self.scale
                self._last_drag_pos = pos
            self.zoomBy(1.0)

        # Handles how we draw a box!
        if (event.buttons() == Qt.LeftButton) and self.annotate_cell_box_mode:
            self._cell_box_corner_1 = QPointF(_x, _y)

        if (
            (event.buttons() == Qt.LeftButton)
            and self.edit_cell_box_mode
            and self._opposite_corner_of_editable_box is not None
            and self._cell_to_edit is not None
        ):
            _x1, _y1 = self._opposite_corner_of_editable_box
            self._cell_to_edit.bbox = (_x, _y, _x1, _y1)
            self._cell_to_edit.score = 1.0  # if we edited a box, its a gt box now...
            self._cell_to_edit._correct_bbox_order()

        # Move a box edge
        if (
            (event.buttons() == Qt.LeftButton)
            and self.edit_cell_box_mode
            and self._cell_to_edit is not None
            and self._cell_held_edge is not None
        ):
            if self._cell_held_edge in [0, 2]:  # x
                self._cell_to_edit.bbox[self._cell_held_edge] = _x
            if self._cell_held_edge in [1, 3]:  # x
                self._cell_to_edit.bbox[self._cell_held_edge] = _y

        if (
            (event.buttons() == Qt.LeftButton)
            and self.edit_cell_box_mode
            and self._cell_to_edit is not None
            and self._cell_bbox_start is not None
        ):
            x0, y0, x1, y1 = self._cell_bbox_start
            cx = (abs(x1 - x0) / 2) + x0
            cy = (abs(y1 - y0) / 2) + y0

            dx = cx - _x
            dy = cy - _y

            self._cell_to_edit.bbox = [x0 - dx, y0 - dy, x1 - dx, y1 - dy]

        # are we drawing an eval region mode polygon?
        if (
            self.draw_eval_region_mode
            and self._is_drawing_eval_region
            and self.active_piece.eval_region
        ):
            pos = event.position()

            # this lets us draw a line from the last position to the mouse...
            # similar to leica lasx
            self._eval_mouse_position = QPointF(pos.x(), pos.y())

        # editing an eval region
        if (
            (event.buttons() == Qt.LeftButton)
            and self.edit_eval_region_mode
            and self.active_piece.eval_region
            and self._eval_region_edit_vertex_index is not None
        ):
            self.active_piece.eval_region[
                self._eval_region_edit_vertex_index
            ] = QPointF(_x, _y)

        # editing a freq region
        if (
            (event.buttons() == Qt.LeftButton)
            and self.edit_freq_mode
            and self.active_piece.freq_path
            and self._freq_path_edit_vertex_index is not None
        ):
            self.active_piece.freq_path[
                self._freq_path_edit_vertex_index
            ] = (_x, _y)

        if (
            (event.buttons() == Qt.LeftButton)
            and (self.select_cell_box_mode or self.select_synapse_mode)
            and self._select_region_corner_0
        ):
            self._select_region_corner_1 = QPointF(_x, _y)

        if (
            (event.buttons() == Qt.LeftButton)
            and self.live_eval_mode
            and self.past_live_eval_mouse_pos is not None
        ):
            px, py = self.past_live_eval_mouse_pos
            dx, dy = px - _x, py - _y
            x0, y0, x1, y1 = self.live_eval_square
            shape = self.active_piece.image.shape
            x0 -= dx
            y0 -= dy
            x1 -= dx
            y1 -= dy
            scale = 289 / self.active_piece.pixel_size_xy if self.active_piece.pixel_size_xy is not None else 1

            clamp = lambda num, min_value, max_value: max(
                min(num, max_value), min_value
            )

            x0 = clamp(x0, 0, shape[1] - round(255 * scale)+1)
            y0 = clamp(y0, 0, shape[0] - round(255 * scale)+1)

            x1 = clamp(x1, round(255 * scale)-1, shape[1] - 1)
            y1 = clamp(y1, round(255 * scale)-1, shape[0] - 1)

            self.live_eval_square = [x0, y0, x1, y1]
            self.past_live_eval_mouse_pos = [_x, _y]
            self.update()

        if self.annotate_synapse_mode and self._synapse_pos_0:
            self._synapse_pos_1 = [_x, _y]

        if self.edit_synapse_mode and self._synapse_to_edit is not None:
            if self._keypoint_index_of_edit == 0:
                self._synapse_to_edit.pos0 = [_x, _y]
            elif self._keypoint_index_of_edit == 1:
                self._synapse_to_edit.pos1 = [_x, _y]

        if self._freq_path_buffer and self.annotating_freq_mode:
            self._freq_mouse_pos = (_x, _y)

        self.update()

    def mouseReleaseEvent(self, event):
        """if the mouse is released, we redraw the pixmap and reset self._last_drag_pos"""

        pos = event.position()
        _x = (pos.x() / self.scale) - self._pixmap_offset.x()
        _y = (pos.y() / self.scale) - self._pixmap_offset.y()

        if (
            event.button() == Qt.MiddleButton or event.button() == Qt.RightButton
        ) and self._dragging[
            1
        ]:  # can always move by pressing the middle button down
            pos = event.position()
            if self._last_drag_pos is not None:
                self._pixmap_offset += pos - self._last_drag_pos
            self._last_drag_pos = QPointF()
            self._dragging[1] = False
            self.update()

        if (
            event.button() == Qt.LeftButton and self.move_mode and self._dragging[0]
        ):  # if youre in move mode then you can click and drag.
            pos = event.position()
            if self._last_drag_pos:  # need to do this to avoid double click bug..
                self._pixmap_offset += pos - self._last_drag_pos
            self._last_drag_pos = QPointF()
            self._dragging[0] = False
            self.update()

        # released the box draw annotation...
        if (
            event.button() == Qt.LeftButton
            and self.annotate_cell_box_mode
            and self._cell_box_corner_1
            and self._cell_box_corner_0
        ):
            pos = event.position()
            _x = (pos.x() / self.scale) - self._pixmap_offset.x()
            _y = (pos.y() / self.scale) - self._pixmap_offset.y()
            self._cell_box_corner_1 = QPointF(_x, _y)
            box = QRectF(self._cell_box_corner_0, self._cell_box_corner_1)
            bottom_left = box.bottomLeft()
            top_right = box.topRight()

            self.addGTCellToParent(
                verticies=None,
                bbox=[bottom_left.x(), bottom_left.y(), top_right.x(), top_right.y()],
            )

            self._cell_box_corner_0 = None
            self._cell_box_corner_1 = None

        # released a corner, center, or edge a box
        if (
            event.button() == Qt.LeftButton
            and self.edit_cell_box_mode
            and self._cell_to_edit is not None
        ):
            self._cell_box_corner_0: QPointF = None
            self._cell_box_corner_1: QPointF = None
            self._cell_held_edge = None  # [left, right, top, bottom] [0, 1, 2, 3]
            self._cell_edge_previous = None
            self._corner_index_of_edit = None
            self._opposite_corner_of_editable_box = None
            self._cell_to_edit = None
            self._cell_bbox_start = None

        # changing a vertex of the eval region
        if (
            event.button() == Qt.LeftButton
            and self.edit_eval_region_mode
            and self._eval_region_edit_vertex_index is not None
        ):
            self._eval_region_edit_vertex_index = None
            scale = (
                289 / self.active_piece.pixel_size_xy
                if self.active_piece.pixel_size_xy is not None
                else 1
            )
            _verts = [(v.x(), v.y()) for v in self.active_piece.eval_region]
            self.eval_region_boxes = hcat.lib.roi.get_squares(
                _verts, round(256 * scale), 0.1
            )

        # changing a vertex of the freq region
        if (
            event.button() == Qt.LeftButton
            and self.edit_freq_mode
            and self._freq_path_edit_vertex_index is not None
        ):
            self._freq_path_edit_vertex_index = None
            self.tonotopyPathEdited.emit(True)

        # Select a synapse or box!
        if (
            event.button() == Qt.LeftButton
            and (self.select_cell_box_mode or self.select_synapse_mode)
            and self._select_region_corner_0
        ):
            self._select_region_corner_1 = QPointF(_x, _y)
            box = QRectF(self._select_region_corner_0, self._select_region_corner_1)
            bottom_left = box.bottomLeft()
            top_right = box.topRight()

            selected_type = Cell if self.select_cell_box_mode else Synapse
            if abs(box.width() * box.height()) > (20 * self.scale):
                self.active_piece.set_selected_from_regionbox(
                    bottom_left, top_right, selected_type
                )
                self.selected_child_changed.emit(
                    self.active_piece.get_selected_children()
                )
                # send info to tree viewer

            self._select_region_corner_0 = None
            self._select_region_corner_1 = None

        # Draggning live update mode around
        if self.live_eval_mode and self.past_live_eval_mouse_pos is not None:
            self.past_live_eval_mouse_pos = None
            self.liveUpdateRegionChanged.emit(self.live_eval_square)

            # we need to regenerate pixmap because the live mode only adjusts
            # the small region for speed...
            self.pixmap = QPixmap.fromImage(
                self.active_piece.get_image_buffer_as_qimage()
            )
            self.update()

        # Synapse release
        if (
            event.button() == Qt.LeftButton
            and self.annotate_synapse_mode
            and self._synapse_pos_0
        ):
            pos = event.position()
            _x0 = self._synapse_pos_0[0]
            _y0 = self._synapse_pos_0[1]
            _x1 = (pos.x() / self.scale) - self._pixmap_offset.x()
            _y1 = (pos.y() / self.scale) - self._pixmap_offset.y()
            self.addSynapseToParent(_x0, _y0, _x1, _y1)
            self._synapse_pos_0 = []
            self._synapse_pos_1 = []

        # edit synapse release
        if event.button() == Qt.LeftButton and self._keypoint_index_of_edit is not None:
            self._synapse_to_edit = None
            self._keypoint_index_of_edit = None

        self.updated_state.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        # end the drawing of a eval region polygon
        if event.buttons() == Qt.LeftButton and self.draw_eval_region_mode:
            pos = event.position()
            _x = (pos.x() / self.scale) - self._pixmap_offset.x()
            _y = (pos.y() / self.scale) - self._pixmap_offset.y()

            # self.active_piece.eval_region.append(QPointF(_x, _y))
            self._is_drawing_eval_region = False
            self._eval_mouse_position = None
            self._eval_region_edit_vertex_index = None

            _verts = [(v.x(), v.y()) for v in self.active_piece.eval_region]

            scale = (
                289 / self.active_piece.pixel_size_xy
                if self.active_piece.pixel_size_xy is not None
                else 1
            )
            self.active_piece.eval_region_boxes = hcat.lib.roi.get_squares(
                _verts, round(256 * scale), 0.1
            )
            self.updated_state.emit()

        # end the drawing of a tonotopy path
        if event.buttons() == Qt.LeftButton and self._freq_path_buffer:
            self._freq_path = copy(self._freq_path_buffer)
            self.active_piece.clear_freq_path()
            self.active_piece.set_freq_path(copy(self._freq_path))
            self.tonotopyPathEdited.emit(True)

            self._freq_path = []
            self._freq_path_buffer = []
            self._freq_mouse_pos = []

    def setRenderParams(self, style_dict: Dict[str, str | int]):
        ihc_color = hex_to_qcolor(style_dict["ihc_color"])
        ohc_color = hex_to_qcolor(style_dict["ohc_color"])
        ihc_stroke = style_dict["ihc_stroke"]
        ihc_transparency = style_dict["ihc_transparency"]
        ohc_stroke = style_dict["ohc_stroke"]
        ohc_transparency = style_dict["ohc_transparency"]

        self.ihc_color = ihc_color
        self.ohc_color = ohc_color
        self.ihc_stroke = ihc_stroke
        self.ihc_transparency = ihc_transparency

        self.ohc_stroke = ohc_stroke
        self.ohc_transparency = ohc_transparency

        color = hex_to_qcolor(style_dict["eval_region_color"])

        self.eval_region_color = color
        self.eval_region_stroke = style_dict["eval_region_stroke"]
        self.eval_region_transparency = style_dict["eval_region_transparency"]

        self.basal_point_color = hex_to_qcolor(
            style_dict["basal_point_color"]
        )  # QColor(0, 255, 255, 255),
        self.apex_point_color = hex_to_qcolor(
            style_dict["apex_point_color"]
        )  # QColor(0, 255, 255, 255),
        self.freq_line_color = hex_to_qcolor(
            style_dict["freq_line_color"]
        )  # QColor(255, 255, 255, 255),
        self.freq_line_stroke = style_dict["freq_line_stroke"]  # 2

        self.update()

    def resetRenderParams(self):
        self.ihc_color = QColor(255, 0, 255)
        self.ohc_color = QColor(255, 255, 0)
        self.ohc_stroke: int = 128
        self.ihc_stroke: int = 128
        self.ohc_transparency: int = 128
        self.ihc_transparency: int = 128
        self.eval_region_color = QColor(200, 200, 200)
        self.eval_region_stroke: int = 1
        self.eval_region_transparency: int = 100
        self.basal_point_color = QColor(55, 126, 184)
        self.apex_point_color = QColor(228, 26, 27)
        self.freq_line_color = QColor(255, 255, 255, 255)
        self.freq_line_stroke = 2

    def getCurrentWindowCoords(self) -> List[float]:
        x0 = -self._pixmap_offset.x()
        y0 = -self._pixmap_offset.y()
        x1 = self.width() / self.scale + x0
        y1 = self.height() / self.scale + y0
        return [x0,y0,x1,y1]

    def paintEvent(self, event):
        """main render loop. Everything drawn on the screen goes here"""
        self.painter.begin(self)
        self.painter.scale(self.scale, self.scale)

        if self.active_piece is not None and not self.live_eval_mode:
            self._paint_pixmap()  # draws the image
            # self._paint_pixmap_corner_coords_overlay()  # draws red dots at all corners

            self._calculate_center_ratio()  # caluclates the ratio of center position, to the image...
            # self._paint_current_mode()  # Puts red text on bottom left corner...
            self._paint_select_box()
            self._paint_annotate_synapses()

            if self.render_hints["freq_region"] or self._freq_path_buffer:
                self._paint_freq_region()
                self._paint_cell_to_freq_path()

            # Cell Boxes
            if self.annotate_cell_box_mode or self.annotate_synapse_mode:
                self._paint_center_cross()
                self._paint_cell_annotation_boxes()
                self._paint_annotate_synapses()

            if self.render_hints["diagnostic"]:
                self._paint_cell_diagnostic()

            if self.render_hints["synapse"]:
                self._paint_synapses()

            if self.render_hints["box"]:
                self._paint_bboxes()

            if (
                self.render_hints["eval_region"]
                or self.draw_eval_region_mode
                or self.edit_eval_region_mode
            ):
                self._paint_eval_region()

        elif self.live_eval_mode:
            self._paint_pixmap()  # draws the image
            self._calculate_center_ratio()  # caluclates the ratio of center position, to the image...
            # self._paint_current_mode()  # Puts red text on bottom left corner...
            self._paint_live_bboxes()
            self._paint_live_region()

        # self._paint_center_cross_overlay()  # draws a green dot at the center and a pair of crosses
        self._paint_eval_window()
        self.painter.end()

    def _paint_select_box(self):
        if (
            self.active_piece is not None
            and self._select_region_corner_0
            and self._select_region_corner_1
        ):
            p = self.painter
            pen = QPen()
            pen.setWidthF(1 / self.scale)
            pen.setColor(QColor(255, 255, 255))
            p.setPen(pen)
            p.drawRect(
                QRectF(
                    self._select_region_corner_0 + self._pixmap_offset,
                    self._select_region_corner_1 + self._pixmap_offset,
                )
            )

    def _paint_bboxes(self):
        # colors = {'None': GREY, 'OHC': QColor(255, 255, 0, 128), 'IHC': QColor(255, 0, 255, 128)}
        if self.active_piece is not None and self.render_hints["box"]:
            cells = [c for c in self.active_piece if isinstance(c, Cell)]

            window = self.getCurrentWindowCoords()
            cells = [c for c in cells if c.is_visible(window)]

            p = self.painter
            font = QFont()
            font.setPointSizeF(5)
            p.setFont(font)

            ohc_pen = QPen()
            ohc_pen.setWidthF(self.ohc_stroke / self.scale)
            self.ohc_color.setAlpha(self.ohc_transparency)
            ohc_pen.setColor(self.ohc_color)

            ihc_pen = QPen()
            ihc_pen.setWidthF(self.ihc_stroke / self.scale)
            self.ihc_color.setAlpha(self.ihc_transparency)
            ihc_pen.setColor(self.ihc_color)

            for c in cells:  # draws all the cells
                bbox = [bb + self._pixmap_offset for bb in c.get_bbox_as_QPointF()]
                pen = ihc_pen if c.type == "IHC" else ohc_pen
                p.setPen(pen)

                p.drawRect(QRectF(*bbox))

                if c.groundTruth():
                    p.drawRect(QRectF(bbox[0], bbox[0] + QPointF(2, 2)))

            # Re-set the color and pen stroke and alpha
            self.ohc_color.setAlpha(255)
            ohc_pen.setColor(self.ohc_color)
            ohc_pen.setWidthF((self.ohc_stroke + 1) / self.scale)

            self.ihc_color.setAlpha(255)
            ihc_pen.setColor(self.ihc_color)
            ihc_pen.setWidthF((self.ihc_stroke + 1) / self.scale)

            x_pen = QPen()
            x_pen.setColor(QColor(255, 255, 255, 50))
            x_pen.setWidthF(2 / self.scale)

            selected_list: List = self.active_piece.get_selected_children()
            for selected in selected_list:
                if isinstance(selected, Cell):
                    pen = ihc_pen if selected.type == "IHC" else ohc_pen
                    width = (
                        self.ihc_stroke if selected.type == "IHC" else self.ohc_stroke
                    )

                    p.setPen(pen)
                    rect = QRectF(
                        *[
                            bb + self._pixmap_offset
                            for bb in selected.get_bbox_as_QPointF()
                        ]
                    )
                    p.drawRect(rect)

                    # Draw an X in the box
                    tl, bl, br, tr = selected._get_bbox_corners()
                    p.setPen(x_pen)
                    p.drawLine(
                        QPointF(*tl) + self._pixmap_offset,
                        QPointF(*br) + self._pixmap_offset,
                    )
                    p.drawLine(
                        QPointF(*bl) + self._pixmap_offset,
                        QPointF(*tr) + self._pixmap_offset,
                    )

                    # If we are editing, we draw a bunch of circles
                    if self.edit_cell_box_mode:
                        pen = QPen()
                        pen.setColor(QColor(255, 255, 255))
                        pen.setCapStyle(Qt.RoundCap)
                        pen.setWidthF((width + 2) * 2 / self.scale)
                        p.setPen(pen)

                        # corner
                        for point in selected._get_bbox_corners():
                            point = QPointF(*point) + self._pixmap_offset
                            p.drawPoint(point)
                        # center_edges
                        for point in selected._get_bbox_edge_centers():
                            point = QPointF(*point) + self._pixmap_offset
                            p.drawPoint(point)

                        pen.setColor(QColor(0, 0, 0))
                        pen.setCapStyle(Qt.RoundCap)
                        pen.setWidthF((width + 2) * 1.5 / self.scale)
                        p.setPen(pen)
                        for point in selected._get_bbox_corners():
                            point = QPointF(*point) + self._pixmap_offset
                            p.drawPoint(point)

                        for point in selected._get_bbox_edge_centers():
                            point = QPointF(*point) + self._pixmap_offset
                            p.drawPoint(point)

    def _paint_live_bboxes(self):
        # colors = {'None': GREY, 'OHC': QColor(255, 255, 0, 128), 'IHC': QColor(255, 0, 255, 128)}
        if self.active_piece is not None:
            p = self.painter
            cells = [
                c
                for id, c in self.active_piece.live_children.items()
                if isinstance(c, Cell)
            ]
            ohc_pen = QPen()
            ohc_pen.setWidthF(self.ohc_stroke / self.scale)
            self.ohc_color.setAlpha(self.ohc_transparency)
            ohc_pen.setColor(self.ohc_color)

            ihc_pen = QPen()
            ihc_pen.setWidthF(self.ihc_stroke / self.scale)
            self.ihc_color.setAlpha(self.ihc_transparency)
            ihc_pen.setColor(self.ihc_color)

            for c in cells:
                pen = ihc_pen if c.type == "IHC" else ohc_pen
                rect = QRectF(
                    *[bb + self._pixmap_offset for bb in c.get_bbox_as_QPointF()]
                )
                p.setPen(pen)
                p.drawRect(rect)

    def _paint_current_mode(self):
        """draws the text at the bottom"""
        p = self.painter

        screensize = QPointF(self.width(), self.height()) / self.scale

        x = screensize.x() * 0.02
        y = screensize.y() * 0.98

        _pen = p.pen()
        _font = p.font()

        pen = QPen()
        pen.setWidthF(10 / self.scale)
        pen.setCapStyle(Qt.RoundCap)
        pen.setColor(QColor(255, 0, 0))
        font = QFont()
        font.setPointSizeF(font.pointSize() / self.scale)

        p.setPen(pen)
        p.setFont(font)

        if self.move_mode:
            text = "MOVE"
        elif self.annotate_cell_box_mode:
            text = "ANNOTATE CELL BBOX"
        elif self.edit_cell_box_mode:
            text = "EDIT CELL BBOX"
        elif self.draw_eval_region_mode:
            text = "DRAW EVAL REGION"
        elif self.edit_eval_region_mode:
            text = "EDIT EVAL REGION"
        elif self.select_cell_box_mode:
            text = "SELECT CELL"
        elif self.select_synapse_mode:
            text = "SELECT SYNAPSE"
        elif self.edit_synapse_mode:
            text = "EDIT SYNAPSE"
        elif self.annotate_synapse_mode:
            text = "DRAW SYNAPSE"
        elif self.live_eval_mode:
            text = "LIVE EVAL"
        elif self.annotating_freq_mode:
            text = "DRAW FREQ PATH"
        elif self.edit_freq_mode:
            text = "EDIT FREQ PATH"
        else:
            raise RuntimeError("UNKOWN MODE")

        p.drawText(QPointF(x, y), text)

        p.setPen(_pen)
        p.setFont(_font)

    def _paint_pixmap_corner_coords_overlay(self):
        """for debug only"""
        if self.pixmap is not None:
            p = self.painter

            _pen = p.pen()
            _font = p.font()

            pen = QPen()
            pen.setWidthF(10 / self.scale)
            pen.setCapStyle(Qt.RoundCap)
            pen.setColor(QColor(255, 0, 0))
            font = QFont()
            font.setPointSizeF(font.pointSize() / self.scale)

            p.setPen(pen)
            p.setFont(font)

            # Top Left
            point = self._pixmap_offset
            p.drawPoints([point])
            p.drawText(point, f"  {point.x():.1f}, {point.y():.1f}")

            # Top Right
            point = self._pixmap_offset + QPointF(self.pixmap.width(), 0.0)
            p.drawPoints([point])
            p.drawText(point, f"  {point.x():.1f}, {point.y():.1f}")

            # Bottom Left
            point = self._pixmap_offset + QPointF(0.0, self.pixmap.height())
            p.drawPoints([point])
            p.drawText(point, f"  {point.x():.1f}, {point.y():.1f}")

            # Bottom Right
            point = self._pixmap_offset + QPointF(
                self.pixmap.width(), self.pixmap.height()
            )
            p.drawPoints([point])
            p.drawText(point, f"  {point.x():.1f}, {point.y():.1f}")

            p.setPen(_pen)
            p.setFont(_font)

    def _paint_cell_annotation_boxes(self):
        p = self.painter

        box_pen = QPen()
        box_pen.setColor(QColor(255, 0, 0))
        box_pen.setWidthF(2 / self.scale)
        p.setPen(box_pen)
        if (
            self._cell_box_corner_0 and self._cell_box_corner_1
        ):  # has all four coordinates
            box = QRectF(
                self._cell_box_corner_0 + self._pixmap_offset,
                self._cell_box_corner_1 + self._pixmap_offset,
            )  # should be a list of QPointFs
            p.drawRect(box)

    def _paint_center_cross_overlay(self):
        """
        paints the center cross and the mouse coordinates
        """

        p = self.painter

        _pen = p.pen()
        _font = p.font()

        # draws a cross
        screensize = QPointF(self.width(), self.height()) / self.scale
        _pen = self.painter.pen()
        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        pen.setWidthF(0.5 / self.scale)
        self.painter.setPen(pen)

        self.painter.drawLine(
            QPointF(screensize.x() / 2, 0.0),
            QPointF(screensize.x() / 2, screensize.y()),
        )
        self.painter.drawLine(
            QPointF(0.0, screensize.y() / 2),
            QPointF(screensize.x(), screensize.y() / 2),
        )

        # style the painter
        pen = QPen()
        pen.setWidthF(3 / self.scale)
        pen.setCapStyle(Qt.RoundCap)
        pen.setColor(QColor(255, 255, 255))
        font = QFont()
        font.setPointSizeF(font.pointSize() / self.scale)
        p.setPen(pen)
        p.setFont(font)

        # places a point at the dead center.
        point = (QPointF(self.width(), self.height()) / self.scale) / 2
        p.drawPoints([point])
        # p.drawText(point, f'  {point.x():.1f}, {point.y():.1f} \n ({self._current_center_ratio[0]:.2f}, {self._current_center_ratio[1]:.2f})')
        p.drawText(
            point, f" {self.mouse_position.x():.2f}, {self.mouse_position.y():.2f}"
        )

        p.setPen(_pen)
        p.setFont(_font)

    def _paint_live_region(self):
        if self.live_eval_mode and len(self.live_eval_square) == 4:
            screensize = QPointF(self.width(), self.height()) / self.scale
            width = screensize.x()
            height = screensize.y()

            p = self.painter
            pen = QPen()
            pen.setWidthF(4)
            pen.setColor(QColor(0, 0, 0, 255))
            p.setPen(pen)

            # verts = [v + self._pixmap_offset for v in self.live_eval_square]
            x0, y0, x1, y1 = self.live_eval_square
            bl = QPointF(x0, y0) + self._pixmap_offset
            tr = QPointF(x1, y1) + self._pixmap_offset
            rect = QRectF(bl, tr)

            x0, y0 = bl.x(), bl.y()
            x1, y1 = tr.x(), tr.y()

            p.drawRect(rect)  # draw the square...

            pen = QPen()
            pen.setWidthF(2)
            pen.setColor(QColor(255, 255, 255, 255))
            p.setPen(pen)

            p.drawRect(rect)

            pen = QPen()
            pen.setWidthF(0)
            pen.setColor(QColor(0, 0, 0, 0))
            p.setPen(pen)

            # gray out everything else...
            top = QRectF(0, 0, width, y0)
            bottom = QRectF(0, y1, width, height - y1)
            left = QRectF(0, y0, x0, y1 - y0)
            right = QRectF(x1, y0, width - x1, y1 - y0)

            path = QPainterPath()
            path.addRect(top)
            path.addRect(bottom)
            path.addRect(left)
            path.addRect(right)

            p.fillPath(path, QColor(255, 255, 255, 160))
            p.drawPath(path)

    def _paint_eval_window(self):
        if self._eval_window:
            p = self.painter
            x, y, w, h = self._eval_window

            rect = QRectF(
                y + self._pixmap_offset.x(), x + self._pixmap_offset.y(), h, w
            )
            pen = QPen()
            pen.setWidthF(4)
            pen.setColor(QColor(255, 255, 255, 255))
            p.setPen(pen)
            p.drawRect(rect)

    def _paint_eval_region(self):
        if self.active_piece.eval_region:
            p = self.painter
            pen = QPen()
            pen.setWidthF(self.eval_region_stroke + 2 / self.scale)
            color = self.eval_region_color
            color.setAlpha(self.eval_region_transparency)
            pen.setColor(color)

            verts = [v + self._pixmap_offset for v in self.active_piece.eval_region]
            if self._eval_mouse_position:
                verts += [self._eval_mouse_position / self.scale]

            p.setPen(pen)
            p.drawConvexPolygon(verts)
            shape = self.active_piece.image.shape

            # draw the boxes
            if self.active_piece.eval_region_boxes:
                pen.setWidthF(self.eval_region_stroke / self.scale)
                p.setPen(pen)

                for b in self.active_piece.eval_region_boxes:
                    x, y, w, h = b

                    w = min(shape[0] - 1, w)
                    h = min(shape[1] - 1, h)
                    x = min(max(0, x), shape[1] - w)
                    y = min(max(0, y), shape[0] - h)

                    p.drawRect(
                        QRectF(
                            x + self._pixmap_offset.x(),
                            y + self._pixmap_offset.y(),
                            w,
                            h,
                        )
                    )

            # draw edit bubbles
            if self.edit_eval_region_mode:
                pen.setColor(QColor(255, 255, 255))
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidthF(self.eval_region_stroke + 5 / self.scale)
                p.setPen(pen)
                for point in verts:
                    p.drawPoint(point)

                pen.setColor(QColor(0, 0, 0))
                pen.setCapStyle(Qt.RoundCap)
                pen.setWidthF(self.eval_region_stroke + 4 / self.scale)
                p.setPen(pen)
                for point in verts:
                    p.drawPoint(point)

    def _paint_synapses(self):
        if self.active_piece is not None:
            p = self.painter

            _pen = p.pen()

            green_pen = QPen()
            green_pen.setColor(QColor(0, 200, 0))
            green_pen.setWidthF(5 / self.scale)
            green_pen.setCapStyle(Qt.RoundCap)

            red_pen = QPen()
            red_pen.setColor(QColor(200, 0, 0))
            red_pen.setWidthF(5 / self.scale)
            red_pen.setCapStyle(Qt.RoundCap)

            line_pen = QPen()
            line_pen.setColor(QColor(100, 100, 100))
            line_pen.setWidthF((2 / self.scale))

            pos0, pos1 = self.active_piece.get_synapse_pos_as_QPoints()
            pos0 = [p + self._pixmap_offset for p in pos0]
            pos1 = [p + self._pixmap_offset for p in pos1]

            # DRAW LINE HERE!!!
            lines = [QLineF(p0, p1) for p0, p1 in zip(pos0, pos1)]
            p.setPen(line_pen)
            p.drawLines(lines)

            p.setPen(green_pen)
            p.drawPoints(pos0)

            p.setPen(red_pen)
            p.drawPoints(pos1)

        selected_list: List = self.active_piece.get_selected_children()
        if selected_list:
            pos0 = [
                QPointF(*s.pos0) + self._pixmap_offset
                for s in selected_list
                if isinstance(s, Synapse)
            ]
            pos1 = [
                QPointF(*s.pos1) + self._pixmap_offset
                for s in selected_list
                if isinstance(s, Synapse)
            ]

            green_pen = QPen()
            green_pen.setColor(QColor(0, 200, 0))
            green_pen.setWidthF(7 / self.scale)
            green_pen.setCapStyle(Qt.RoundCap)

            red_pen = QPen()
            red_pen.setColor(QColor(200, 0, 0))
            red_pen.setWidthF(7 / self.scale)
            red_pen.setCapStyle(Qt.RoundCap)

            line_pen = QPen()
            line_pen.setColor(QColor(100, 100, 100))
            line_pen.setWidthF((3 / self.scale))

            # DRAW LINE HERE!!!
            lines = [QLineF(p0, p1) for p0, p1 in zip(pos0, pos1)]
            p.setPen(line_pen)
            p.drawLines(lines)

            p.setPen(green_pen)
            p.drawPoints(pos0)

            p.setPen(red_pen)
            p.drawPoints(pos1)

            if self.edit_synapse_mode:  # draw edit bubbles
                pos = pos0 + pos1

                pen = QPen()
                pen.setColor(QColor(255, 255, 255, 255))
                pen.setWidthF(4 / self.scale)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.drawPoints(pos)  # draw black circle

                pen.setColor(QColor(0, 0, 0, 255))
                pen.setWidthF(2 / self.scale)
                p.setPen(pen)
                p.drawPoints(pos)  # draw white circle

    def _paint_center_cross(self):
        p = self.painter
        _x, _y = (
            self.mouse_position.x() / self.scale,
            self.mouse_position.y() / self.scale,
        )
        _w, _h = self.width() / self.scale, self.height() / self.scale

        _pen = self.painter.pen()
        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        pen.setWidthF(0.5 / self.scale)
        p.setPen(pen)

        # draw a cross at the cursor position
        p.drawLine(QPointF(0.0, _y), QPointF(_w, _y))  # horrizontal line
        p.drawLine(QPointF(_x, 0.0), QPointF(_x, _h))  # horrizontal line

        if self.annotate_cell_box_mode:
            pen = QPen()
            if self._cell_type == "IHC":
                pen.setColor(self.ihc_color)
            else:
                pen.setColor(self.ohc_color)

            pen.setWidthF(4 / self.scale)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)

            # lines = [
            #     QLineF(QPointF(_x-10, _y), QPointF(_x+10, _y)), # horizontal
            #     QLineF(QPointF(_x, _y-10), QPointF(_x, _y+10)),
            # ]

            # p.drawLines(lines)  # horrizontal line
            offset = 7 / self.scale
            points = [
                QPointF(0 + offset, _y),
                QPointF(_w - offset, _y),
                QPointF(_x, 0 + offset),
                QPointF(_x, _h - offset),
            ]
            p.drawPoints(points)

    def _paint_annotate_synapses(self):
        if self.active_piece is not None:
            p = self.painter

            green_pen = QPen()
            green_pen.setColor(QColor(0, 200, 0))
            green_pen.setWidthF(7 / self.scale)
            green_pen.setCapStyle(Qt.RoundCap)

            red_pen = QPen()
            red_pen.setColor(QColor(200, 0, 0))
            red_pen.setWidthF(7 / self.scale)
            red_pen.setCapStyle(Qt.RoundCap)

            line_pen = QPen()
            line_pen.setColor(QColor(100, 100, 100))
            line_pen.setWidthF((2 / self.scale))

            if self._synapse_pos_0 and self._synapse_pos_1:
                pos0 = (
                    QPointF(*self._synapse_pos_0) + self._pixmap_offset
                )  # [p + self._pixmap_offset for p in pos0]
                pos1 = (
                    QPointF(*self._synapse_pos_1) + self._pixmap_offset
                )  # [p + self._pixmap_offset for p in pos1]

                # DRAW LINE HERE!!!
                line = QLineF(pos0, pos1)
                p.setPen(line_pen)
                p.drawLine(line)

                p.setPen(green_pen)
                p.drawPoint(pos0)

                p.setPen(red_pen)
                p.drawPoint(pos1)

    def _paint_active_synapse(self):
        """
        Paints the *active* synapse a different color than all others.

        :return: None
        """
        if self.active_piece is not None:
            p = self.painter

            _pen = p.pen()

            pen = QPen()
            pen.setColor(QColor(0, 250, 250))
            pen.setWidthF(10 / self.scale)
            pen.setCapStyle(Qt.RoundCap)

            font = QFont()
            font.setPointSizeF(font.pointSize() / self.scale)

            p.setPen(pen)
            p.setFont(font)

            offset = 10 / self.scale
            child = self.active_piece.get_selected_child()

            if isinstance(child, Cell):
                synapse = child.get_selected_child()
            elif isinstance(child, Synapse):
                synapse = child
            else:
                synapse = None

            if synapse is not None:
                _x, _y = synapse.pos0[0], synapse.pos0[1]
                loc = QPointF(_x, _y) + self._pixmap_offset
                p.drawPoints([loc])
                p.drawText(loc.x() + offset, loc.y(), f"Synapse {synapse.id}")

            p.setPen(_pen)

    def _paint_freq_region(self):
        if not (
            self._freq_path_buffer or self._freq_path or self.active_piece.freq_path
        ):
            return
        p = self.painter
        if not self.active_piece.freq_path:
            path: List[Tuple[float, float]] = copy(
                self._freq_path_buffer if self._freq_path_buffer else self._freq_path
            )
        else:
            path = self.active_piece.freq_path

        if self._freq_mouse_pos:
            path += [self._freq_mouse_pos]

        path = [QPointF(*p) + self._pixmap_offset for p in path]

        basal_point = path[0]
        apical_point = path[-1]
        lines = [QLineF(path[i - 1], path[i]) for i in range(1, len(path))]

        basal_pen = QPen()
        basal_pen.setColor(
            self.freq_line_color
            if not self._freq_path_buffer
            else QColor(200, 200, 200)
        )
        basal_pen.setWidthF(self.freq_line_stroke / self.scale)
        p.setPen(basal_pen)
        p.drawLines(lines)

        basal_pen.setWidthF(self.freq_line_stroke * 3 / self.scale)
        basal_pen.setCapStyle(Qt.RoundCap)

        font = QFont()
        font.setPointSizeF(font.pointSize() / self.scale)
        p.setFont(font)

        p.drawText(basal_point, " BASAL")
        p.drawText(apical_point, " APICAL")

        basal_pen.setColor(self.basal_point_color)
        p.setPen(basal_pen)
        p.drawPoint(basal_point)

        basal_pen.setColor(self.apex_point_color)
        p.setPen(basal_pen)
        p.drawPoint(apical_point)

        if self.edit_freq_mode:
            pen = QPen()
            pen.setColor(QColor(255, 255, 255))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidthF(4 / self.scale)
            p.setPen(pen)

            p.drawPoints(path)

            pen.setColor(QColor(0, 0, 0))
            pen.setCapStyle(Qt.RoundCap)
            pen.setWidthF(3 / self.scale)

            p.setPen(pen)
            p.drawPoints(path)

    def _paint_cell_to_freq_path(self):
        if 1 / self.scale > 10:
            return

        p = self.painter
        pen = QPen()
        pen.setColor(QColor(255, 255, 255, 100))
        pen.setWidthF(1 / self.scale)
        p.setPen(pen)

        cells = [c for c in self.active_piece.children.values() if isinstance(c, Cell)]
        lines = [c.line_to_closest_path for c in cells if c.line_to_closest_path]
        lines = [
            QLineF(
                QPointF(x0, y0) + self._pixmap_offset,
                QPointF(x1, y1) + self._pixmap_offset,
            )
            for x0, y0, x1, y1 in lines
        ]
        p.drawLines(lines)

    def _paint_cell_diagnostic(self):
        p = self.painter
        ohc_pen = QPen()
        ohc_pen.setColor(self.ohc_color)
        ohc_pen.setWidthF(4 / self.scale)

        ihc_pen = QPen()
        ihc_pen.setColor(self.ihc_color)
        ihc_pen.setWidthF(4 / self.scale)

        # ohc = [QPointF(*c.get_cell_center()) + self._pixmap_offset
        #            for c in self.active_piece.children.values() if isinstance(c, Cell) and c.type == 'OHC']
        #
        # ihc = [QPointF(*c.get_cell_center()) + self._pixmap_offset
        #        for c in self.active_piece.children.values() if isinstance(c, Cell) and c.type == 'IHC']
        #
        #
        #
        # p.setPen(ohc_pen)
        # p.drawPoints(ohc)
        #
        # p.setPen(ihc_pen)
        # p.drawPoints(ihc)
        font = QFont()
        font.setPointSizeF(font.pointSize() / self.scale)
        p.setFont(font)

        cells = [c for c in self.active_piece.children.values() if isinstance(c, Cell)]
        window = self.getCurrentWindowCoords()
        cells = [c for c in cells if c.is_visible(window)]

        for c in cells:
            freq = c.frequency if c.frequency else -1
            text = f" ID:{c.id}\n {c.type}\n {freq:0.2f}kHz"
            cx, cy = c.get_cell_center()
            cx, cy = cx + self._pixmap_offset.x(), cy + self._pixmap_offset.y()
            rect = QRectF(cx, cy, cx + 10, cy + 10)

            pen = ohc_pen if c.type == "OHC" else ihc_pen
            p.setPen(pen)
            p.drawPoint(QPointF(cx, cy))
            p.drawText(rect, text)

    def _paint_pixmap(self):
        """paints the image to QLabel"""
        if self.pixmap is not None:
            p = self.painter
            p.drawPixmap(self._pixmap_offset, self.pixmap)
            p.setClipping(True)
