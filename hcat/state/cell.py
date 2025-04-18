from typing import *

from PySide6.QtCore import QPointF

from hcat.gui.tree_widget_item import TreeWidgetItem
from hcat.state import StateItem


class Cell(StateItem):
    def __init__(self,
                 id: int | None,
                 parent,
                 score,
                 type: str | None = None,
                 frequency: float | None = None,
                 mask_verticies: List[QPointF] = None,
                 bbox: List[int] | None = None,
                 ):
        super(Cell, self).__init__(id=id, parent=parent, score=score)

        self.type = type
        self.frequency = frequency
        self.percentage_total_length = None
        self.distance_from_base_in_nm = None
        self.mask_verticies = mask_verticies
        self.bbox: List[int] = bbox
        self.rotation = None
        self.line_to_closest_path: List[float] | None = None  # x0, y0, x1, y1

        self.creator: str | None = None  # who created it? model or user?
        self.image_adjustments_at_creation: Dict[str, float] | None = None
        self.has_been_edited_by_user = False


        # is ground truth makes a cell exempt from removal by threhsolding or nms
        self._ground_truth = False

        self.tree_item = TreeWidgetItem((f'Cell {self.id}',), index=self.id)
        # self.tree_item.setCheckState(0, Qt.Checked)

        self._correct_bbox_order()

    def set_creator(self, creator: str):
        self.creator = creator
    def get_creator(self):
        return self.creator

    def set_image_adjustments_at_creation(self, adjustments):
        self.image_adjustments_at_creation = adjustments
    def get_image_adjustments_at_creation(self):
        return self.image_adjustments_at_creation

    def set_as_edited(self):
        self.has_been_edited_by_user = True

    def was_edited(self):
        return self.has_been_edited_by_user

    def setGroundTruth(self, val: bool):
        self._ground_truth = val

    def groundTruth(self):
        return self._ground_truth

    def set_line_to_path(self, line: List[float] | None):
        if line is None:
            return

        assert len(line) == 4
        self.line_to_closest_path = line

    def clear_line_to_path(self):
        self.line_to_closest_path = None

    def set_percentage(self, percentage: float):
        self.percentage_total_length = percentage

    def set_distance(self, distance_in_nm: float):
        self.distance_from_base_in_nm = distance_in_nm

    def set_frequency(self, frequency: float):
        self.frequency = frequency

    def set_type(self, celltype: str ):
        print('attempting to change child celltype')
        if isinstance(celltype, str) and celltype in ['OHC', 'IHC']:
            self.type = celltype
            print(self.type)
        else:
            raise ValueError(
                f'kwarg {celltype} with type {type(celltype)}, is not a valid input. Must be "OHC" or "IHC"')

    def toggle_type(self):
        if self.type == 'OHC':
            self.type = 'IHC'
        elif self.type == 'IHC':
            self.type = 'OHC'

    def _correct_bbox_order(self):
        """ bbox might be in wrong order... """
        if self.bbox:
            x0, y0, x1, y1 = self.bbox

            self.bbox = [
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1)
            ]

    def mouse_in_cell(self, x, y):
        x0, y0, x1, y1 = self.bbox
        return x >= x0 and x <= x1 and y >= y0 and y <= y1

    def get_cell_center(self):
        x0, y0, x1, y1 = self.bbox
        cx = (abs(x1 - x0) / 2) + x0
        cy = (abs(y1 - y0) / 2) + y0
        return cx, cy

    def dist_to_closest_corner(self, x, y) -> Tuple[float, int]:
        corners = self._get_bbox_corners()
        distance = [(x0 - x) ** 2 + (y0 - y) ** 2 for (x0, y0) in corners]
        min_dist = min(distance)
        for i, d in enumerate(distance):
            if d == min(distance):
                return min_dist, i

    def dist_to_bbox_center(self, x, y):
        cx, cy = self.get_cell_center()
        return ((cx - x) ** 2 + (cy - y) ** 2)

    def dist_to_closest_edge(self, x, y) -> Tuple[float, int]:
        """
        returns the distance to a close edge, and the index of the relevant bbox side to adjust if any

        will return dist=='inf' if nothing is reasonable
        """
        # x0, y0, x1, y1 = self.bbox
        # distance = [float('inf'), float('inf'), float('inf'), float('inf')]
        #
        # if y1 >= y and y >= y0:
        #     distance[0] = abs(x0 - x) # adjust x0
        #
        # if  y1 >= y and y >= y0:
        #     distance[2] = abs(x1 - x) # adjust x0
        #
        # if x1 >= x and x >= x0:
        #     distance[1] = abs(y0 - y)  # adjust x0
        #
        # if x1 >= x and x >= x0:
        #     distance[3] = abs(y1 - y)  # adjust x0
        #
        # min_dist = min(distance)
        # for i, d in enumerate(distance):
        #     if d == min(distance):
        #         return min_dist, i
        corners = self._get_bbox_edge_centers()
        distance = [(x0 - x) ** 2 + (y0 - y) ** 2 for (x0, y0) in corners]
        min_dist = min(distance)
        for i, d in enumerate(distance):
            if d == min(distance):
                return min_dist, i

    def _get_bbox_corners(self) -> List[Tuple[float, float]]:
        """
        Returns corners in clockwize order starting at bottom left
        Implies from bbox

        :return: [bl, tl, tr, br]
        """
        if self.bbox:
            x0, y0, x1, y1 = self.bbox
            bl = (x0, y0)
            tl = (x0, y1)
            tr = (x1, y1)
            br = (x1, y0)
            return [bl, tl, tr, br]

    def _get_bbox_edge_centers(self) -> List[Tuple[float, float]]:
        """
        Returns corners in clockwize order starting at bottom left
        Implies from bbox

        :return: [bl, tl, tr, br]
        """
        if self.bbox:
            x0, y0, x1, y1 = self.bbox
            w = (x1 - x0) / 2
            h = (y1 - y0) / 2

            bl = (x0, y0 + h)
            tl = (x0 + w, y0)
            tr = (x0 + w, y1)
            br = (x1, y0 + h)

            return [bl, tl, br, tr]  # [bl, tl, tr, br]

    def get_opposite_corner_by_index(self, corner_index: int):
        bl, tl, tr, br = self._get_bbox_corners()
        opposite_corner = None

        if corner_index == 0:
            # bl = new_corner_position
            opposite_corner = tr
        elif corner_index == 1:
            # tl = new_corner_position
            opposite_corner = br
        elif corner_index == 2:
            # tr = new_corner_position
            opposite_corner = bl
        elif corner_index == 3:
            # br = new_corner_position
            opposite_corner = tl

        return opposite_corner

    def _adjust_box_by_corner(self, new_corner_position: Tuple[float, float],
                              corner_index: int):
        """
        Lets one change a single corner and infers the rest of the positions via the
        corner which has been altered.

        :param new_corner_position: tuple x/y index of a new corner
        :param corner_index:
        :return:
        """
        opposite_corner = self.get_opposite_corner_by_index(corner_index)

        # print(f'{corner_index} | {opposite_corner}')
        self.bbox = [new_corner_position[0], new_corner_position[1], opposite_corner[0], opposite_corner[1]]
        self._correct_bbox_order()

    def is_visible(self, window: Tuple[float, float, float, float]) -> bool:
        """
        Returns true if any part of the cell bbox is visible from the window...

        :param window: [x0, y0, x1, y1]
        :return: true if cell bbox is within the window
        """
        x0,y0,x1,y1 = self.bbox
        x = window[2] > x0 > window[0] or window[2] > x1 > window[0]
        y = window[3] > y0 > window[1] or window[3] > y1 > window[1]

        return x and y
