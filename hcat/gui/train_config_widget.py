from PySide6.QtCore import QSize, QDir
import os

import torch
from PySide6 import QtWidgets
from PySide6.QtCore import QSize, QDir
from PySide6.QtWidgets import *

from hcat.backends.backend import init_model
from hcat.config.config import get_cfg_defaults


class TrainWidget(QtWidgets.QWidget):
    """
    This widget contains ALL logic for training and evaluation of these images...

    We are a stage based approach -
        - Stage 1: Pick a task
        - Stage 2: set the parameters
        - Stage 3: choose your data
        - Stage 4: run a training loop

    We need to allow access to trainng pannel from the start:
        - Perform all actions without impacting other screenview
        - popup another persistent widget for visualization, monitoring (eh.)
            - have this run in a separate thread so that other work may be done

    This sets the configuration of training.
    When the user hits "run" we launch another training script for full approach
    Just get the feature workign, and try it out

    """
    def __init__(self, cfg):
        super().__init__()

        # Load the config file
        self.config = cfg

        self.train_task_combobox = QComboBox()
        self.train_task_combobox.addItems(['Cell Detection', 'Synapse Detection'])

        # Create the widgets
        self.learning_rate_label = QtWidgets.QLabel("Learning Rate")
        self.learning_rate_spinbox = QtWidgets.QDoubleSpinBox()
        self.learning_rate_spinbox.setMinimum(0)
        self.learning_rate_spinbox.setMaximum(1)
        # self.learning_rate_spinbox.setMinimumSize(QSize(80, 20))
        self.learning_rate_spinbox.setSingleStep(1e-4)
        self.learning_rate_spinbox.setDecimals(5)
        self.learning_rate_spinbox.setValue(self.config.TRAIN.LEARNING_RATE)

        self.weight_decay_label = QtWidgets.QLabel("Weight Decay")
        self.weight_decay_spinbox = QtWidgets.QDoubleSpinBox()
        self.weight_decay_spinbox.setMinimum(0)
        self.weight_decay_spinbox.setMaximum(1)
        # self.weight_decay_spinbox.setMinimumSize(QSize(80, 20))
        self.weight_decay_spinbox.setSingleStep(1e-4)
        self.weight_decay_spinbox.setDecimals(5)
        self.weight_decay_spinbox.setValue(self.config.TRAIN.WEIGHT_DECAY)

        self.optimizer_label = QtWidgets.QLabel("Optimizer")
        self.optimizer_combobox = QtWidgets.QComboBox()
        self.optimizer_combobox.addItems(['AdamW', 'Adam', 'SGD', 'SDGM'])
        self.optimizer_combobox.setCurrentText(self.config.TRAIN.OPTIMIZER)

        self.scheduler_label = QtWidgets.QLabel("Scheduler")
        self.scheduler_combobox = QtWidgets.QComboBox()
        self.scheduler_combobox.addItems(['cosine_annealing'])
        self.scheduler_combobox.setCurrentText(self.config.TRAIN.SCHEDULER)

        self.n_epochs_label = QtWidgets.QLabel("Number of Epochs")
        self.n_epochs_spinbox = QtWidgets.QSpinBox()
        self.n_epochs_spinbox.setMinimum(0)
        self.n_epochs_spinbox.setMaximum(1e6)
        self.n_epochs_spinbox.setValue(self.config.TRAIN.N_EPOCHS)

        self.mixed_precision_checkbox = QtWidgets.QCheckBox("Mixed Precision")
        self.mixed_precision_checkbox.setChecked(self.config.TRAIN.MIXED_PRECISION)

        self.save_path_label = QtWidgets.QLabel("Save Path")
        self.save_path_edit = QtWidgets.QLineEdit()
        self.save_path_edit.setText(self.config.TRAIN.SAVE_PATH)


        self.train_picker = FilePickerWidget('Training Data')
        self.validation_picker = FilePickerWidget('Validation Data')

        # Add a final train button
        self.start_train = QPushButton('Start Training')
        self.start_train.clicked.connect(self.start_training)

        # Create the layout
        master_layout = QVBoxLayout()
        master_layout.addWidget(self.train_task_combobox)

        hyperparam_layout = QFormLayout()
        hyperparam_layout.addRow(self.learning_rate_label, self.learning_rate_spinbox)
        hyperparam_layout.addRow(self.weight_decay_label, self.weight_decay_spinbox)
        hyperparam_layout.addRow(self.optimizer_label, self.optimizer_combobox)
        hyperparam_layout.addRow(self.scheduler_label, self.scheduler_combobox)
        hyperparam_layout.addRow(self.n_epochs_label, self.n_epochs_spinbox)
        hyperparam_layout.addRow(None, self.mixed_precision_checkbox)
        hyperparam_layout.addRow(self.save_path_label, self.save_path_edit)

        hyperparam_group = QGroupBox('Hyperparameters')
        hyperparam_group.setLayout(hyperparam_layout)
        layout = QVBoxLayout()
        layout.addWidget(hyperparam_group)

        master_layout.addLayout(layout)
        master_layout.addWidget(self.train_picker)
        master_layout.addWidget(self.validation_picker)
        master_layout.addWidget(self.start_train)

        self.setLayout(master_layout)

        # Connect signals and slots
        # self.learning_rate_spinbox.valueChanged.connect(self.update_config)
        # self.weight_decay_spinbox.valueChanged.connect(self.update_config)
        # self.optimizer_combobox.currentTextChanged.connect(self.update_config)
        # self.scheduler_combobox.currentTextChanged.connect(self.update_config)
        # self.n_epochs_spinbox.valueChanged.connect(self.update_config)
        # self.train_dir_edit.textChanged.connect(self.update_config)
        # self.validation_dir_edit.textChanged.connect(self.update_config)
        # self.mixed_precision_checkbox.stateChanged.connect(self.update_config)

    def get_updated_train_cfg(self):
        raise NotImplementedError

    def start_training(self):

        model = init_model()

        train_data = self.train_picker.get_files()
        val_data = self.validation_picker.get_files()
        train = lambda x, y: None
        cfg = self.get_updated_train_cfg()

        outputs = train(model, cfg)
        filename, type = QFileDialog.getSaveFileName(self, "Save", QDir.currentPath(), 'Whorl Model File (*.hcatmdl)')
        torch.save(filename, outputs)




class FilePickerWidget(QtWidgets.QWidget):
    def __init__(self, label: str = 'test'):
        super().__init__()



        # Create the widgets
        self.add_button = QtWidgets.QPushButton("+")
        # self.add_button.setMinimumSize(QSize(50, 30))
        self.add_button.setMaximumSize(QSize(30, 30))
        self.remove_button = QtWidgets.QPushButton("-")
        # self.remove_button.setMinimumSize(QSize(50, 30))
        self.remove_button.setMaximumSize(QSize(30, 30))
        self.add_folder_button = QtWidgets.QPushButton("Add Folder")
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Files"])
        self.tree_widget.itemClicked.connect(self.test)

        # Create the button group layout

        plusminus_layout = QHBoxLayout()
        plusminus_layout.addWidget(self.add_button)
        plusminus_layout.addWidget(self.remove_button)
        plusminus_layout.setContentsMargins(0,0,20,0)
        plusminus_layout.setSpacing(0)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addLayout(plusminus_layout)
        # button_layout.addWidget(self.add_button)
        # button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.add_folder_button)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # Create the layout
        layout = QtWidgets.QVBoxLayout()
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
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dialog.exec():
            new_file_paths = dialog.selectedFiles()
            for path in new_file_paths:
                if path not in self.file_paths:
                    self.file_paths.append(path)
                    item = QtWidgets.QTreeWidgetItem([path])
                    self.tree_widget.addTopLevelItem(item)

    def add_folder_files(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if dialog.exec():
            folder_path = dialog.selectedFiles()[0]
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith(".tif"):
                    if file_path not in self.file_paths:
                        self.file_paths.append(file_path)
                        item = QtWidgets.QTreeWidgetItem([file_path])
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
    #
    # def mini




if __name__=='__main__':
    cfg = get_cfg_defaults()
    app = QApplication()
    w = TrainWidget(cfg)
    w.show()
    app.exec()
