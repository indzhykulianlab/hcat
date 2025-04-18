from PySide6.QtWidgets import *


class TreeWidgetItem(QTreeWidgetItem):
    def __init__(self, *args, index: int):
        super(TreeWidgetItem, self).__init__(*args)
        self.index: int = index

    def get_object_index(self):
        return self.index
