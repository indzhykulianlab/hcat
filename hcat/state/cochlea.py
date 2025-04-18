from typing import List

from hcat.state import Piece, StateItem


class Cochlea(StateItem):
    """
    Each Cochlea must have a piece. This is created implicitly when the first piece is created.
    A piece must be selected.

    """
    def __init__(self, id: int, parent):
        super(Cochlea, self).__init__(id=id, parent=parent, score=1.0)
        # StateItem creates: self.id, self.selected, self.children
        # StatItem method: self.add_child(), self.set_selected(), self.get_selected()

        # tree representation
        # self.top_level_piece_tree_items: List[TreeWidgetItem] = []

        self.top_level_tree_items = []

    def delete_selected_piece(self) -> Piece:
        """ deletes a selected piece """
        child = self.get_selected_child()
        removed_item: Piece = self.remove_child(child.id)
        return child

        # try to select any other piece
        for c in self.children.values():
            if isinstance(c, Piece):
                c.select()
                break
        return to_remove[0]

    def get_selected_children(self) -> List:
        raise NotImplementedError

    def _get_bbox(self):
        pass

    def add_child(self, item):
        """ add a child to the parent """
        if item is not None:
            self.children[item.id] = item
            self.top_level_tree_items.append(item.tree_item)

    def getLatestTreeItem(self):
        return self.top_level_tree_items[-1]

    def set_selected_for_all_children(self, index: int | List[int]):
        """
        User can pass a list of id values which cascade from parent to child,
        setting the selected value for each.

        If the datastructure looks like:
            Cochlea:
                Piece 0:
                Piece 1:
                Piece 2:
                Piece 3:
                    Cell 0:
                    Cell 1:

        given an input index of [0, 3, 1], Cell 1 of Piece 3 would be set selected.

        :param index: List of id's for each level
        :return: None
        """
        index = [index] if isinstance(index, int) else index
        item = self
        for i in index:
            item.set_selected_children([i])
            child = item.get_selected_child() if isinstance(item, Cochlea) else item.get_selected_children()
            child = child[0] if isinstance(child, list) else child
            item = child # returns a list of selected children.

        return None

    def get_selected_child(self):
        """ returns the active child """
        if self.selected:
            return self.children[self.selected[0]]
