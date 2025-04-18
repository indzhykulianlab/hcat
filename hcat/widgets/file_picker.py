from PySide6.QtCore import *
from PySide6.QtWidgets import *
import os.path


class WFilePickerWidget(QWidget):
    def __init__(self, label: str = 'test'):
        super().__init__()

        # Create the widgets
        self.add_button = QPushButton("+")
        # self.add_button.setMinimumSize(QSize(50, 30))
        self.add_button.setMaximumSize(QSize(30, 30))
        self.remove_button = QPushButton("-")
        # self.remove_button.setMinimumSize(QSize(50, 30))
        self.remove_button.setMaximumSize(QSize(30, 30))
        self.add_folder_button = QPushButton("Add Folder")
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Files"])
        self.tree_widget.itemClicked.connect(self.test)

        # Create the button group layout

        plusminus_layout = QHBoxLayout()
        plusminus_layout.addWidget(self.add_button)
        plusminus_layout.addWidget(self.remove_button)
        plusminus_layout.setContentsMargins(0,0,20,0)
        plusminus_layout.setSpacing(0)

        button_layout = QHBoxLayout()
        button_layout.addLayout(plusminus_layout)
        # button_layout.addWidget(self.add_button)
        # button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.add_folder_button)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Create the layout
        layout = QVBoxLayout()
        layout.addWidget(self.tree_widget)
        layout.addLayout(button_layout)
        layout.setSpacing(0)
        layout.setContentsMargins(2,2,2,2)

        self.group = QGroupBox(label)
        self.group.setLayout(layout)

        group_layout = QVBoxLayout()
        group_layout.addWidget(self.group)
        # group_layout.setSpacing(0)
        # group_layout.setContentsMargins(1,1,1,1)

        self.setLayout(group_layout)

        # Connect signals and slots
        self.add_button.clicked.connect(self.add_files)
        self.add_folder_button.clicked.connect(self.add_folder_files)
        self.remove_button.clicked.connect(self.remove_selected)

        # Store the file paths
        self.file_paths = []

    def add_files(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        if dialog.exec():
            new_file_paths = dialog.selectedFiles()
            for path in new_file_paths:
                if path not in self.file_paths:
                    self.file_paths.append(path)
                    item = QTreeWidgetItem([path])
                    self.tree_widget.addTopLevelItem(item)

    def add_folder_files(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec():
            folder_path = dialog.selectedFiles()[0]
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith(".tif"):
                    if file_path not in self.file_paths:
                        self.file_paths.append(file_path)
                        item = QTreeWidgetItem([file_path])
                        self.tree_widget.addTopLevelItem(item)

    def remove_selected(self):
        for item in self.tree_widget.selectedItems():
            index = self.tree_widget.indexOfTopLevelItem(item)
            self.tree_widget.takeTopLevelItem(index)
            self.file_paths.pop(index)


    def get_files(self):
        return self.file_paths

    def sizeHint(self):
        return QSize(10, 150)

    def test(self, item):
        print(item.text(0))
