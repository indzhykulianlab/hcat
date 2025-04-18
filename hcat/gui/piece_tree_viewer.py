from typing import List

from PySide6.QtCore import Qt, QSize, Slot, Signal
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from hcat.gui.edit_cell_params_widget import EditCellParamsWidget
from hcat.gui.tree_widget_item import TreeWidgetItem
from hcat.state import Cochlea, Piece, StateItem, Cell
from hcat.widgets.push_button import WPushButton

"""
This widget does two things. 
    1. Reflects accurately the state of the application.
        - This means we update every time a synapse, cell, or piece is added/modifies
    2. Allow for the selection of cell / piece objects. 
        - if we select a piece, we emit a signal and change the active piece in the cochlea
        - if we select a cell, we emit a signal ang change the active cell...
        - if we select an already selected cell, we DE-select an active thing...
"""


class PieceTreeViewerWidget(QWidget):

    selected_changed = Signal(list)
    cell_changed = Signal(None)
    active_piece_changed = Signal(None)

    def __init__(self, state: Cochlea):
        super(PieceTreeViewerWidget, self).__init__()
        self.state = state
        self.tree = QTreeWidget()
        # self.tree.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.tree.setHeaderLabel('Cochlea')
        self._last_active_indicies = []

        self.delete_piece_button = WPushButton('DEL PIECE')
        self.delete_piece_button.setDangerButton(True)
        self.delete_piece_button.setFixedSize(QSize(75, 18))

        self.add_piece_button = WPushButton('UPDATE')
        self.add_piece_button.setFixedSize(QSize(75, 18))

        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)
        button_layout.setContentsMargins(2,2,2,2)
        button_layout.addWidget(self.add_piece_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        button_layout.addWidget(self.delete_piece_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        button_layout.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        layout.addLayout(button_layout)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        # generate the first tree from the state!
        # hcat.state.cochlea.create_tree() will generate a list of top level TreeWidgetItems from its existing
        # data
        for p in self.state.top_level_tree_items:
            self.tree.addTopLevelItem(p)

        # self.populateTree()

        self.tree.itemClicked.connect(self.getIndiciesToItem)
        self.tree.currentItemChanged.connect(self.getIndiciesToItem)
        # self.tree.itemDoubleClicked.connect(self.editCellProperties)
        # self.tree.itemSelectionChanged.connect(self.getIndiciesToItem)

        self.edit_widget = None

    @Slot(TreeWidgetItem)
    def editCellProperties(self, item: TreeWidgetItem):

        # hide and delete the widget. Lets us open a new one if one is already open.
        if self.edit_widget is not None:
            self.edit_widget.hide()
            del self.edit_widget

        indicies = self.getIndiciesToItem(item, 0) # get the indicies to the item
        child = self.state.get_child_at_indices(indicies)
        if isinstance(child, Cell):
            self.edit_widget = EditCellParamsWidget(child)
            self.edit_widget.setWindowTitle(str(child))
            self.edit_widget.setWindowFlag(Qt.WindowStaysOnTopHint)
            self.edit_widget.type_changed.connect(child.set_type)
            self.edit_widget.update_canvas.connect(self.cell_changed.emit)

            self.edit_widget.show()


    @Slot(TreeWidgetItem)
    def addTopLevelItem(self, item: TreeWidgetItem):
        self.tree.addTopLevelItem(item)
        item.isSelected()

    @Slot(TreeWidgetItem)  # what is this? The typing is all fucked
    def removeItems(self, children: StateItem):
        """
        removes a selected item. An item should ALWAYS be selected. God help me
        """
        for c in children:
            parent_item = c.tree_item.parent()
            if parent_item is not None:
                parent_item.removeChild(c.tree_item)
            else:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(c.tree_item))

            # self.tree.removeItemWidget(c.tree_item, 0)
        self.update()

    def removeTopLevelItem(self, child: Piece):
        """ delete a piece """
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(child.tree_item))
        self.update()

    @Slot(TreeWidgetItem, int)
    def getIndiciesToItem(self,
                          item: TreeWidgetItem | None = None,
                          column: int = 0) -> List[int]:
        """
        Returns the indicies TOP down in order to get back to the actual item.

        For instance, if self.getIndiciesToItem() = [0, 3, 2]
        then the clicked item can be accessed by

        self.cochlea.set_active(0)
        piece: Piece = self.cochela.get_active()
        piece.set_active(3)
        cell: Cell = piece.get_active()
        cell.set_active(2)
        synapse: Synapse = cell.get_active()

        :param item:
        :param column:
        :return:
        """
        # We need this becuase the signal self.tree.selectedItems does not retrun a TreeViewerItem
        # instead we need to get the *selected* items from the tree. This can be weird, as the tree can
        # technically have multiple items selected...
        item: List[TreeWidgetItem] | None = item if item is not None else self.tree.selectedItems()
        item: List[TreeWidgetItem] | None = item[-1] if isinstance(item, list) and item else item
        if not item: # none or empty list
            return []  # nothing is selected...

        # pass
        active_item_indicies = []
        while item:  # eventually item is none as the parent of the Cochlea object is None
            active_item_indicies.append(item.get_object_index())
            item = item.parent()
        active_item_indicies.reverse()

        # print(active_item_indicies)
        self.selected_changed.emit(active_item_indicies)


        # why?~?!?!
        if active_item_indicies and self._last_active_indicies:
            if active_item_indicies[0] != self._last_active_indicies[0]:
                self.active_piece_changed.emit()
        else:
            self.active_piece_changed.emit()

        self._last_active_indicies = active_item_indicies

        return active_item_indicies

    def updateItemOrder(self):
        """ updates tree order based on relative ordering base to apex of each piece... """
        while self.tree.takeTopLevelItem(0) is not None:
            continue

        pieces = [p for p in self.state.children.values() if isinstance(p, Piece)]
        pieces.sort(key=lambda p: p.relative_order)
        for p in pieces:
            self.tree.addTopLevelItem(p.tree_item)

        self.update()


    def selectPieceWidgetItem(self, index: int) -> TreeWidgetItem:
        raise NotImplementedError

    def clear(self):
        """ sets tree and piece to none """
        self.tree.clear()
        self.state = None

    def setNewSate(self, state: Cochlea):
        """ re-draws the treeview with a new state """
        self.clear()
        self.state = state
        self.update()

if __name__ == '__main__':
    cochlea = Cochlea(0, None)
    app = QApplication()
    w = PieceTreeViewerWidget(cochlea)
    # w = ModelFileSelector()
    w.show()
    app.exec()
