from typing import List, Dict, Tuple

from PySide6.QtCore import QPointF

from hcat.gui.tree_widget_item import TreeWidgetItem


class StateItem:
    def __init__(self, id: int, parent, score: float):
        """
        Parent class for all state objects. This includese
        synapses, cells, pieces, and the whole cochlea.
        """
        self.id: int = id
        self.children: Dict[int, StateItem] = {}
        self.selected: List[int] = []
        self._candidate_children: Dict[int, StateItem] = {}
        self._candidate_selected: List[int] = []
        self.tree_item: TreeWidgetItem = None
        self.parent = parent
        self.bbox: List[float] = []
        self.visible = True

        self.score = score

        # Private items
        self._children_changed = False  # it is sometimes usefull to track when a child has been added

    def get_child_at_indices(self, indicies: List[int]):
        """
        given a list of id values starting at the self, gets the lowest child
        For instance, given the data:
        Cochlea
            - Piece 0
                - Cell 0
                    - Synapse 0
                    - Synapse 1
            - Piece 1
                - Cell 1
                    - Synapse 2
                    - Synapse 3
                    - Synapse 4

        And the indicies: [0, 1, 2]

        cochlea.get_child_at_indicies would return Synapse 4
        Piece 0 -> Cell 1 -> Synapse 4
        """
        child = self
        for i in indicies:
            if i in child.children:
                child = child.children[i]
            else:
                _outstr = [(k, v) for k, v in child.children.items()]
                raise ValueError(f'Attempting to index a {self} with index {indicies} but {i} is not in {_outstr}')

        return child

    def set_invisible(self):
        self.visible = False

    def set_visible(self):
        self.visible = True

    def _get_bbox(self) -> List[float]:
        raise NotImplementedError

    def get_bbox_as_QPointF(self) -> Tuple[QPointF, QPointF]:
        return QPointF(self.bbox[0], self.bbox[1]), QPointF(self.bbox[2], self.bbox[3])

    def get_bbox(self):
        return self.bbox

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        yield from self.children.values()

    def get_parent(self):
        return self.parent

    def add_child(self, item):
        """ add a child to the parent """
        if item is not None:
            self.children[item.id] = item
            self.tree_item.addChild(item.tree_item)
            self._children_changed = True

    def set_id(self, id):
        """ have to set id with this method to update parent's children dict """
        old_id = self.id
        self.id = id

        if self.parent is not None:
            self.parent.children[self.id] = self
            del self.parent[old_id]

    def set_selected_children(self, id: int | List[int]):
        """ sets which child is the active child """

        if isinstance(id, int):
            id = [id]

        id = [_id for i, _id in enumerate(id) if _id in self.children.keys()]
        self.selected = id

    def get_selected_children(self) -> List:
        """ returns the active child """
        if self.selected:
            selected = [self.children[i] for i in self.selected]
        else:
            selected = []

        return selected

    def set_selected_to_none(self):
        """ sets no active children """
        self.selected = []

    def deselect_all(self):
        """ sets no active children """
        self.selected = []
        for k, v in self.children.items():
            v.deselect()

    def deselect(self):
        """ removes self from parent selected """
        if self.id in self.parent.selected:
            for i, _id in enumerate(self.parent.selected):
                if _id == id:
                    self.parent.selected.pop(i)  # only pops id from selected list

    def select(self):
        """ adds self to parents selected list """
        self.parent.add_selected(self.id)

    def add_selected(self, id):
        """ adds id to selected list """
        self.selected.append(id)

    def remove_child(self, id: int):
        """ removes an id from the selected children """
        if id not in self.children:
            print(self.children.keys(), id)
            raise KeyError('Cannot delete child id!')


        if id in self.selected:
            for i, _id in enumerate(self.selected):
                if _id == id:
                    self.selected.pop(i)
                    break
            # self.set_selected_to_none()

        return self.children.pop(id)  # doesnt delete!!

    def take_tree_item(self, index: int):
        self.tree_item.takeChild(index)
