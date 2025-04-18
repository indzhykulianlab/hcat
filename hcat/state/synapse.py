from typing import Tuple

import numpy as np
from PySide6.QtCore import QPointF, Qt

from hcat.gui.tree_widget_item import TreeWidgetItem
from hcat.state import StateItem


class Synapse(StateItem):
    def __init__(self, x0: int, y0: int, x1: int, y1: int,
                 id: int,
                 parent,
                 mask: np.ndarray | None = None,
                 score: float = 1.0):
        super(Synapse, self).__init__(id=id, parent=parent, score=score)
        """
        Holds information about a synapse in hcat. These will either be predicted, or manually created

        :param x0: x position of first label (Green)
        :param y0: y positino of first label (Green)
        :param x1: x position of second label (Red)
        :param y1: y position of second label (Red)
        :param mask: [C=2, X, Y] mask of the synapses
        """

        # These
        self.pos0 = [x0, y0]
        self.pos1 = [x1, y1]
        self.mask = mask

        self.area = None  # should only be set by self.get_synapse_area

        self.tree_item = TreeWidgetItem((f'Synapse {self.id}',), index=self.id)
        self.tree_item.setCheckState(0, Qt.Checked)
        self._get_bbox()

    def _get_bbox(self):
        self.bbox = self.pos0 + self.pos1

    def make_selected(self):
        """ weird naming here. We basically want this synapse to be the selected child of its parent """
        self.parent.set_selected_children(self.id)

    def as_QPointF(self):
        return QPointF(*self.pos0), QPointF(*self.pos1)

    def get_synapse_area(self):
        raise NotImplementedError

    def set_synapse_mask(self, mask):
        self.mask = mask

    def dist_to_closest_keypoint(self, x: float, y: float) -> Tuple[float, int]:
        corners = [self.pos0, self.pos1]
        distance = [(x0-x) ** 2 + (y0 - y) ** 2 for (x0, y0) in corners]
        min_dist = min(distance)
        for i, d in enumerate(distance):
            if d == min(distance):
                return min_dist, i

    def __len__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError

    def add_child(self, item):
        raise RuntimeError('cannot add to a synapse')

    def set_selected_children(self, id: int):
        raise RuntimeError('Setting a selected synapse is nonsensicle')

    def get_selected_children(self):
        raise RuntimeError('Synapse may not have a child.')

    def set_selected_to_none(self):
        raise RuntimeError
