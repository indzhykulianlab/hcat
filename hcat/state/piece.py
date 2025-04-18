from copy import copy
from hashlib import sha256
from typing import List, Tuple, Dict

import numpy as np
import torch
import torchvision.ops  # nms
from PySide6.QtCore import QPointF, QLineF, Qt
from PySide6.QtGui import QImage

import hcat.lib.frequency
import hcat.lib.roi
from hcat.gui.tree_widget_item import TreeWidgetItem
from hcat.lib.adjust import _adjust
from hcat.lib.histogram import histogram
from hcat.state import Cell, StateItem, Synapse


class Piece(StateItem):
    def __init__(
        self,
        image: np.ndarray,
        filename: str,
        id: int,
        parent,
    ):
        """
        Data class for a piece of tissue... All pieces of tissue should come from
        the same cochlea, therefore the cochlea contains info on species, but
        tissue may be imaged with different metadata such as imaging params,
        zoom, px size, etc...

        :param image:
        :type image:
        :param filename:
        :type filename:
        :param id:
        :type id:
        :param parent:
        :type parent:
        """
        super(Piece, self).__init__(id=id, parent=parent, score=1.0)

        self.image = image
        self.timeseries = []
        self.sha256_hash_of_image: str = sha256(image.data).hexdigest()
        self.tree_item = TreeWidgetItem((f"Piece",), index=self.id)
        self.tree_item.setFlags(self.tree_item.flags() | Qt.ItemIsEditable)

        # list of RGB adjustment dicts, see self.set_rgb for more info
        self.adjustments: List[Dict[str, int] | None] = [
            {"brightness": 0, "contrast": 1, 'channel': 0},
            {"brightness": 0, "contrast": 1, 'channel': 1},
            {"brightness": 0, "contrast": 1, 'channel': 2},
        ]
        self.live_adjustments: List[Dict[str, int] | None] = [
            {"brightness": 0, "contrast": 1, 'channel': 0},
            {"brightness": 0, "contrast": 1, 'channel': 1},
            {"brightness": 0, "contrast": 1, 'channel': 2},
        ]

        self.dtype = None
        self.pixel_size_xy = None  # nm
        self.pixel_size_z = None  # IF 3D
        self.filename = filename  # just the name...
        self.filepath = None  # full path
        self.device = 'CPU'

        self.freq_path: List[Tuple[float, float]] | None = None
        self.basal_frequency: float | None = None
        self.apical_frequency: float | None = None

        self.microscopy_type = None
        self.microscopy_zoom = None
        self.stain_label: Dict[str, str | None] = {
            "red": None,
            "green": None,
            "blue": None,
        }
        self.animal: str | None = None
        self.relative_order: int | None = None

        self.eval_region = []
        self.eval_region_boxes = []
        self.nms_thr = 0.5
        self.cell_thr = 0.5
        self.live_mode = False
        self.live_area = []

        # children - we have multiple be
        self.live_children = (
            {}
        )  # similar to children, except for holding temporary children
        self._candidate_live_children = {}

        self._display_image_buffer = image.copy()

        # when youre in live mode, we edit a small crop for just random bullshit
        self._previous_live_image_buffer = None  # for previously altered images

        # keep a frequency array
        # base -> apex
        self.cell_connections = None

        self._get_bbox()

        self.histogram: List[Dict[str, np.array]] = []
        for c in range(image.shape[-1]):
            _vals, _pos = histogram(self.image[:, :, c])
            self.histogram.append({"values": _vals, "positions": _pos})

        # precompile _adjust
        for _ in range(10):
            _adjust(np.random.randint(0, 255, (200, 200)).astype(np.float64), 0.0, 1.0)

    # self.microscopy_type = None
    # self.microscopy_zoom = None
    # self.stain_label: Dict[str, str | None] = {'red': None, 'green': None, 'blue': None}
    # self.animal: str | None = None
    # self.relative_order: int | None = None

    def set_timeseries(self, timeseries: List[np.ndarray]):
        self.timeseries = timeseries
        return self

    def get_timeseries(self):
        return self.timeseries

    def set_xy_pixel_size(self, size: int):
        self.pixel_size_xy = size
        if self.eval_region_boxes:
            scale = 289 / self.pixel_size_xy if self.pixel_size_xy is not None else 1
            _verts = [(v.x(), v.y()) for v in self.eval_region]
            self.eval_region_boxes = hcat.lib.roi.get_squares(
                _verts, round(256 * scale), 0.1
            )

    def clear_eval_region(self):
        self.eval_region_boxes = []
        self.eval_region = []

    def set_microscopy_type(self, type: str):
        print(f'SET MICROSCOPY TYPE: {type}')
        self.microscopy_type = type

    def set_microscopy_zoom(self, zoom: int):
        self.microscopy_zoom = zoom

    def set_stain_labels(self, stains: Dict[str, str | None]):
        self.stain_label = stains

    def set_animal(self, animal: str):
        self.animal = animal

    def set_relative_order(self, order: int):
        self.relative_order = order

    def clear_xy_pixel_size(self):
        self.pixel_size_xy = None

    def clear_microscopy_type(self):
        self.microscopy_type = None

    def clear_microscopy_zoom(self):
        self.microscopy_zoom = None

    def clear_stain_labels(self):
        self.stain_label = None

    def clear_animal(self):
        self.animal = None

    def clear_relative_order(self):
        self.relative_order = None

    def clear_freq_path(self):
        self.freq_path = None
        for c in self._candidate_children.values():
            if isinstance(c, Cell):
                c.clear_line_to_path()

    def set_basal_freq(self, basal_freq: float):
        self.basal_frequency = basal_freq

    def clear_basal_freq(self):
        self.basal_frequency = None

    def set_apical_freq(self, apical_freq: float):
        self.apical_frequency = apical_freq

    def clear_apical_freq(self):
        self.apical_frequency = None

    def set_freq_path(self, path: List[Tuple[float, float]] | None):
        if path is None:
            return

        if not isinstance(path, list):
            raise RuntimeError("frequency path must be a list")
        for p in path:
            assert len(p) == 2, f"{len(p)} must have two coordinates in each point"

        self.freq_path = copy(path)
        interp = hcat.lib.frequency.interpolate_path(path, as_array=True)
        cells = [c for c in self._candidate_children.values() if isinstance(c, Cell)]


        if not cells:
            return

        closest = hcat.lib.frequency.closest_point(cells, interp)
        for c, line in zip(cells, closest):
            c.set_line_to_path(line)

    def reverse_freq_path(self):
        if self.freq_path is not None:
            self.freq_path.reverse()

    def enable_live_mode(self, area):
        self.live_area = area
        print(f"set area: {self.live_area}")
        self.reset_live_mode_adjustment()
        area = [round(a) for a in area]
        y0, x0, y1, x1 = area
        self._previous_live_image_buffer = self._display_image_buffer[
            x0:x1, y0:y1, :
        ].copy()
        self.live_adjustments = copy(self.adjustments)
        self.live_mode = True
        self.clear_live_mode_children()

    def clear_live_mode_children(self):
        self.live_children = {}
        self._candidate_live_children = {}

    def reset_live_mode_adjustment(self):
        """resets the adjustments done in live mode"""
        if (
            self.live_mode
            and self.live_area
            and self._previous_live_image_buffer is not None
        ):  # live mode already set
            y0, x0, y1, x1 = self.live_area
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # for c in self.adjustments:
            #     brightness = float(c['brightness'])  # , dtype=np.float64)
            #     contrast = float(c['contrast'])  # , dtype=np.float64)
            print("Resetting to previous")
            print(f"{self._previous_live_image_buffer.mean()}")
            self._display_image_buffer[
                x0:x1, y0:y1, :
            ] = self._previous_live_image_buffer

            self._previous_live_image_buffer = None

    def disable_live_mode(self):
        self.live_mode = False
        self.live_area = []
        self._previous_live_image_buffer = None

        self._cell_rejection_from_thr()

        self.live_children = {}
        self._candidate_live_children = {}

    def get_histogram(self):
        return self.histogram

    def set_eval_region(self, region: List[QPointF | Tuple[float, float]]):
        for i, a in enumerate(region):
            if not isinstance(a, QPointF):
                region[i] = QPointF(*a)

        self.eval_region = region

    def set_eval_region_boxes(self, boxes):
        self.eval_region_boxes = boxes

    def remove_eval_roi(self):
        """
        sets:
        self.eval_region=[]
        self.eval_region_boxes=[]
        """
        self.eval_region = []
        self.eval_region_boxes = []

    def apply_adjustment(self, adjustment: Dict[str, int] | List[Dict[str, int]]):
        """c -> Dict['channel': 0, 'brightness': -255:255, 'contrast':-255:255]"""
        if not isinstance(adjustment, list):
            adjustment = [adjustment]

        for c in adjustment:
            brightness = float(c["brightness"])  # , dtype=np.float64)
            contrast = float(c["contrast"])  # , dtype=np.float64)

            if self.live_mode:
                print(self.live_area)
                self.live_adjustments[c["channel"]] = c
                area = [round(a) for a in self.live_area]
                y0, x0, y1, x1 = area

                _image = self.image[x0:x1, y0:y1, c["channel"]].astype(np.float64)

                self._display_image_buffer[x0:x1, y0:y1, c["channel"]] = _adjust(
                    _image, brightness, contrast
                )

            else:
                self.adjustments[c["channel"]] = c
                _image = self.image[..., c["channel"]].astype(np.float64)
                self._display_image_buffer[..., c["channel"]] = _adjust(
                    _image, brightness, contrast
                )

    def set_path(self, path):
        self.filepath = path

    def _get_bbox(self):
        self.bbox = [0, 0, self.image.shape[0], self.image.shape[1]]

    def set_live_children(self, items: List[Cell]):
        """for adding temp children to live mode..."""
        if len(items) == 0:
            return
        self._candidate_live_children = {}
        self.live_children = {}
        for item in items:
            self._candidate_live_children[item.id] = item

        self._live_cell_rejection_from_thr()

    def add_children(self, items: List[StateItem]):
        if len(items) == 0:
            return

        for item in items:
            self._candidate_children[item.id] = item

        self._cell_rejection_from_thr()

        for item in items:
            if item.id in self.children:
                self.tree_item.addChild(item.tree_item)
                self._children_changed = True

        # if we have a frequency path, we need to auto calculate for the cell
        cells = [c for c in items if isinstance(c, Cell)]
        if self.freq_path and cells:
            interp = hcat.lib.frequency.interpolate_path(self.freq_path, as_array=True)
            closest = hcat.lib.frequency.closest_point(cells, interp)
            for c, line in zip(cells, closest):
                c.set_line_to_path(line)

    def add_child(self, item: Cell):  # Overloaded StateItem Implementation
        """
        add a cell to the parent
        This is the only StateItem which can potentially have two types of children: Synapses and Cells

        We overloaded the add_child method from StateItem such that we can auto assign any free synapses
        to a cell if a cell is ultimately added.

        """
        if item is None:
            raise ValueError("child item must not be none!!!")
            return

        self._candidate_children[item.id] = item

        # this function shuffle around cells from self._candidate_children -> self.children
        # we never `remove` a cell from self._candidate_ch
        self._cell_rejection_from_thr()
        self._synapse_rejection_from_thr()

        if item.id in self.children:
            self.tree_item.addChild(item.tree_item)
            self._children_changed = True

        cells = [c for c in [item] if isinstance(c, Cell)]
        if self.freq_path and cells:
            interp = hcat.lib.frequency.interpolate_path(self.freq_path, as_array=True)
            closest = hcat.lib.frequency.closest_point(cells, interp)
            for c, line in zip(cells, closest):
                c.set_line_to_path(line)

    def remove_child(self, id: int):
        """removes an id from the selected children"""

        if id in self.selected:
            for i, _id in enumerate(self.selected):
                if _id == id:
                    self.selected.pop(i)
                    break
            # self.set_selected_to_none()
        if id in self._candidate_children:
            self._candidate_children.pop(id)  # remove from candidate children

        return self.children.pop(id)  # doesnt delete!!

    def _live_cell_rejection_from_thr(self):
        """moves and deletes items from self.children by threshold values"""
        if len(self._candidate_live_children) == 0:
            return

        cells = [
            c for c in self._candidate_live_children.values() if isinstance(c, Cell)
        ]
        boxes = torch.tensor(
            [
                c.bbox
                for c in self._candidate_live_children.values()
                if isinstance(c, Cell)
            ],
            dtype=torch.float,
        )
        scores = torch.tensor(
            [
                c.score
                for c in self._candidate_live_children.values()
                if isinstance(c, Cell)
            ],
            dtype=torch.float,
        )
        id = torch.tensor(
            [
                _id
                for _id, c in self._candidate_live_children.items()
                if isinstance(c, Cell)
            ],
            dtype=torch.int,
        )

        score_ind = scores > self.cell_thr
        _nms_ind = torchvision.ops.nms(boxes, scores, iou_threshold=self.nms_thr)
        nms_ind = torch.zeros_like(score_ind)
        nms_ind[_nms_ind] = 1

        ind = torch.logical_and(score_ind, nms_ind)  # mask
        # nms_ind is a tensor of INDICIES not a mask

        # print(f'{self.nms_thr=}, {self.cell_thr=}, {score_ind=}, {nms_ind=}')
        for c, _id, i in zip(cells, id, ind):
            if (i > 0) and (_id.item() not in self.live_children):
                self.live_children[_id.item()] = c
                # self.tree_item.addChild(c.tree_item)

            elif (i == 0) and (_id.item() in self.live_children):  # remove
                # self.tree_item.removeChild(c.tree_item)
                item = self.live_children.pop(_id.item())

    def _synapse_rejection_from_thr(self):
        synapses = [
            c for c in self._candidate_children.values() if isinstance(c, Synapse)
        ]
        scores = torch.tensor(
            [
                c.score
                for c in self._candidate_children.values()
                if isinstance(c, Synapse)
            ],
            dtype=torch.float,
        )
        id = torch.tensor(
            [
                _id
                for _id, c in self._candidate_children.items()
                if isinstance(c, Synapse)
            ],
            dtype=torch.int,
        )

        if len(synapses) == 0:
            return

        ind = scores > self.cell_thr
        # nms_ind is a tensor of INDICIES not a mask

        # print(f'{self.nms_thr=}, {self.cell_thr=}, {score_ind=}, {nms_ind=}')
        for c, _id, i in zip(synapses, id, ind):
            if (i > 0) and (_id.item() not in self.children):
                self.children[_id.item()] = c
                self.tree_item.addChild(c.tree_item)

            elif (i == 0) and (_id.item() in self.children):  # remove
                self.tree_item.removeChild(c.tree_item)
                item = self.children.pop(_id.item())
                if item.id in self.selected:  # remove if it was selected
                    for selected_index, v in enumerate(self.selected):
                        if v == item.id:
                            break
                    self.selected.pop(selected_index)

    def _cell_rejection_from_thr(self):
        """moves and deletes items from self.children by threshold values"""

        cells = [c for c in self._candidate_children.values() if isinstance(c, Cell)]
        boxes = torch.tensor(
            [c.bbox for c in self._candidate_children.values() if isinstance(c, Cell)],
            dtype=torch.float,
        )
        scores = torch.tensor(
            [c.score for c in self._candidate_children.values() if isinstance(c, Cell)],
            dtype=torch.float,
        )
        id = torch.tensor(
            [_id for _id, c in self._candidate_children.items() if isinstance(c, Cell)],
            dtype=torch.int,
        )

        if len(cells) == 0:
            return

        score_ind = scores > self.cell_thr
        _nms_ind = torchvision.ops.nms(boxes, scores, iou_threshold=self.nms_thr)
        nms_ind = torch.zeros_like(score_ind)
        nms_ind[_nms_ind] = 1

        ind = torch.logical_and(score_ind, nms_ind)  # mask
        # nms_ind is a tensor of INDICIES not a mask

        # print(f'{self.nms_thr=}, {self.cell_thr=}, {score_ind=}, {nms_ind=}')
        for c, _id, i in zip(cells, id, ind):
            # add to children if nms and thr are good AND the cell hasnt been flagged as a GT cell..
            if (i > 0 or c.groundTruth()) and (_id.item() not in self.children):
                self.children[_id.item()] = c
                self.tree_item.addChild(c.tree_item)

            # remove from children if nms and thr are bad AND the cell hasnt been flagged as a GT cell..
            elif (i == 0 and not c.groundTruth()) and (
                _id.item() in self.children
            ):  # remove
                self.tree_item.removeChild(c.tree_item)
                item = self.children.pop(_id.item())
                if item.id in self.selected:  # remove if it was selected
                    for selected_index, v in enumerate(self.selected):
                        if v == item.id:
                            break
                    self.selected.pop(selected_index)

    def set_thresholds(self, thr: Tuple[float, float]):
        """
        Sets the piece thresholds.
        :param nms: float 0-1
        :param thr:  float 0-1
        :return: None
        """
        nms = thr[0]
        thr = thr[1]

        assert nms <= 1.0
        assert thr <= 1.0
        self.nms_thr = nms
        self.cell_thr = thr
        self._cell_rejection_from_thr() if not self.live_mode else self._live_cell_rejection_from_thr()
        self._children_changed = True  # always i guess...

    def get_adjustments(self):
        return self.adjustments

    def get_image_buffer_as_qimage(self):
        """this will probably break becuase Qt and numpy share different base views of image data..."""
        """
        we want to be able to adjust the pixmap if necessary,
        in this case, it might be a good idea, for adjustment speed
        to keep an additional copy of the data in memory as sort of a temporary buffer
        this way, we can ajdust the pixel values of that guy, without adjusting the px values of the other...
        this buffer can always be uint8 because we're just displaying it to the screen
        
        """
        h, w, _ = self._display_image_buffer.shape
        _image = self._display_image_buffer[:, :, 0:3]

        _image = np.ascontiguousarray(_image.astype(np.uint8))
        return QImage(_image.data, w, h, 3 * w, QImage.Format_RGB888)

    def get_cells_under_mouse(self, mouse_x, mouse_y) -> List[Cell]:
        """returns a list of all cells which a mouse pointer is hovering over"""
        cells_under_mouse = []
        for id, c in self.children.items():
            if isinstance(c, Cell):
                if c.mouse_in_cell(mouse_x, mouse_y):
                    cells_under_mouse.append(c)

        return cells_under_mouse

    # SYNAPSE
    def set_selected_from_regionbox(
        self, bottom_left: QPointF, top_right: QPointF, state_type: StateItem
    ):
        x0, y0 = bottom_left.x(), bottom_left.y()
        x1, y1 = top_right.x(), top_right.y()
        # print(f"SELECTED REGION BOX: {x0=} {y0=} {x1=} {y1=}")

        # adjust to propper box ordering
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        self.deselect_all()
        for k, v in self.children.items():
            # have to check both to ensure that v is a cell and
            # cell was passed as the "kind"
            if isinstance(v, Cell) and isinstance(v, state_type):
                v._correct_bbox_order()
                _bl = (v.bbox[0] > x0) and (v.bbox[1] > y0)
                _tr = (x1 > v.bbox[2]) and (y1 > v.bbox[3])
                if _bl and _tr:
                    v.select()
            elif isinstance(v, Synapse) and isinstance(v, state_type):
                _bl = (v.pos0[0] > x0) and (v.pos0[1] > y0)
                _tr = (x1 > v.pos1[0]) and (y1 > v.pos1[1])
                if _bl and _tr:
                    v.select()

    def get_synapse_pos_as_list(self) -> List[List[int]]:
        """returns all of the synapses for each cell in the piece"""
        all_positions: List[List[int, int, int, int]] = []
        for c in self:
            if isinstance(c, Synapse):
                all_positions.append(c.pos0 + c.pos1)

        return all_positions

    def get_synapse_pos_as_QPoints(self) -> Tuple[List[QPointF], List[QPointF]]:
        """returns all of the synapses for each cell in the piece"""
        all_positions = self.get_synapse_pos_as_list()

        return [QPointF(x, y) for x, y, _, _ in all_positions], [
            QPointF(x, y) for _, _, x, y in all_positions
        ]

    def get_synapse_connections_as_QLines(self):
        return [QLineF(a, b) for a, b in zip(*self.get_synapse_pos_as_QPoints())]

    def get_all_synapsese(self) -> List[Synapse]:
        _synapses = []
        for c in self:
            if isinstance(c, Cell):
                _synapses += c.children.values()
            elif isinstance(c, Synapse):
                _synapses.append(c)
            else:
                raise RuntimeError("Neither Cell nor Synapse a child of Piece")
        return _synapses
