from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog,
                               QTreeWidget, QTreeWidgetItem, QLabel, QHBoxLayout)
from PySide6.QtCore import Qt
from pathlib import Path
from collections import defaultdict


class FileBrowserWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pair_count = 0

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(2)
        self.select_btn = QPushButton("Select Folder")
        self.select_btn.clicked.connect(self.select_folder)
        self.stats_label = QLabel("No folder selected")
        self.stats_label.setWordWrap(True)
        top_layout.addWidget(self.select_btn)
        top_layout.addWidget(self.stats_label, 1)
        layout.addLayout(top_layout)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Size", "Path"])
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

    def get_xml_paths(self):
        selected_items = self.tree.selectedItems()
        xml_paths = []

        for item in selected_items:
            if not item.childCount():  # If it's a file, not a folder
                file_name = item.text(0)
                relative_path = item.text(2)
                xml_path = Path(self.current_folder) / relative_path / f"{file_name}.xml"
                xml_paths.append(xml_path)

        return xml_paths

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.populate_tree(folder)

    def populate_tree(self, folder_path):
        self.tree.clear()
        self.pair_count = 0
        self.current_folder = folder_path

        folder = Path(folder_path)

        # Find all files and organize by folder
        png_files = {f.stem: f for f in folder.rglob("*.png")}
        xml_files = {f.stem: f for f in folder.rglob("*.xml")}

        # Find matching pairs
        pairs = defaultdict(list)
        total_size = 0

        for stem in set(png_files) & set(xml_files):
            png_file = png_files[stem]
            xml_file = xml_files[stem]
            pairs[png_file.parent].append((png_file, xml_file))
            total_size += png_file.stat().st_size + xml_file.stat().st_size
            self.pair_count += 1

        # Create folder tree
        folder_items = {}
        for folder_path, file_pairs in pairs.items():
            # Create parent folders if they don't exist
            current_folder = folder_path
            while current_folder != folder:
                if current_folder not in folder_items:
                    if current_folder.parent in folder_items:
                        parent_item = folder_items[current_folder.parent]
                    else:
                        parent_item = self.tree
                    folder_items[current_folder] = QTreeWidgetItem(parent_item)
                    folder_items[current_folder].setText(0, current_folder.name)
                    folder_items[current_folder].setText(2, str(current_folder.relative_to(folder)))
                current_folder = current_folder.parent

            # Add file pairs
            for png_file, xml_file in sorted(file_pairs):
                item = QTreeWidgetItem(folder_items[folder_path])
                item.setText(0, png_file.stem)
                combined_size = png_file.stat().st_size + xml_file.stat().st_size
                item.setText(1, self.format_size(combined_size))
                item.setText(2, str(png_file.parent.relative_to(folder)))

        self.stats_label.setText(f"Found {self.pair_count} PNG/XML pairs | Total Size: {self.format_size(total_size)}")

        for i in range(3):
            self.tree.resizeColumnToContents(i)

    @staticmethod
    def format_size(size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    widget = FileBrowserWidget()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())