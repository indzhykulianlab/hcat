import pickle
import xml
from typing import Dict
from typing import Tuple, Callable, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

import hcat.state.load
from hcat.state import Cell, Cochlea

Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class dataset(Dataset):
    def __init__(self,
                 files: List[str],
                 transforms: Callable = lambda x: x,
                 sample_per_image: int = 1,
                 device='cpu',
                 metadata: Optional[Dict[str, str]] = None):

        super(Dataset, self).__init__()

        # Reassigning variables
        self.image = []
        self.boxes = []
        self.labels = []
        self.files = files
        self.transforms = transforms

        self.metadata = metadata
        self.device: str = device
        self.sample_per_image: int = sample_per_image

        for f in self.files:  # Loop over all cochlea files

            # Load the file - its just a pickle...
            with open(f, 'rb') as open_file:
                cochlea: Cochlea = hcat.state.load.load(pickle.load(open_file))

            # a cochlea might have a bunch of pieces, we iterate through them here
            for id, piece in cochlea.children.items(): # piece contains images and cells

                if piece.image is None: continue # silently continue here...

                scale: int = 255 if piece.image.max() <= 256 else 2 ** 16
                image: Tensor = torch.from_numpy(piece.image/scale)  # ensure its always a float

                boxes: List[int] = []
                labels: List[List[int]] = []

                for id, cell in piece.children.items():
                    if not isinstance(cell, Cell): continue
                    label: int = 1 if cell.type == 'OHC' else 2
                    boxes: List[int] = cell.bbox

                # Make sure there is something in label and boxes
                if len(labels) == len(boxes) and len(boxes) > 0:

                    image = image.transpose(2, 0, 1) if image.shape[-1] <= 3 else image

                    self.image.append(image.div(scale).half().to(memory_format=torch.channels_last))
                    self.boxes.append(boxes)
                    self.labels.append(labels)


    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Tuple[str, Tensor]:

        item = item // self.sample_per_image  # We might artificailly want to sample more times per image...

        with torch.no_grad():
            data_dict = {'image': self.image[item].squeeze(-1),
                         'boxes': self.boxes[item],
                         'labels': self.labels[item]}

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        return data_dict

    def get_file_at_index(self, item):
        return self.files[item // self.sample_per_image]

    def get_labels_at_index(self, item):
        return self.labels[item // self.sample_per_image]

    def to(self, device: str):
        self.device = device
        self.image = [x.to(device) for x in self.image]
        self.boxes = [x.to(device) for x in self.boxes]
        self.labels = [x.to(device) for x in self.labels]

        return self

    def cuda(self):
        self.device = 'cuda:0'
        return self.to('cuda:0')

    def cpu(self):
        self.device = 'cpu'
        return self.to('cpu')

    @staticmethod
    def prepare_image(image: Tensor, device: str = 'cpu'):
        c, x, y = image.shape

        for i in range(c):
            max = image[i, ...].max()
            max = max if max != 0 else 1
            image[i, ...] = image[i, ...].div(max)

        if c < 3:
            image = torch.cat((torch.zeros((1, x, y), device=image.device), image), dim=0)

        return image.to(device).float()

    @staticmethod
    def data_from_xml(f: str) -> Tuple[List[List[int]], List[int]]:

        root = xml.etree.ElementTree.parse(f).getroot()

        boxes: List[List[int]] = []
        labels: List[int] = []
        for c in root.iter('object'):
            for box, cls in zip(c.iter('bndbox'), c.iter('name')):
                label: str = cls.text

                if label in ['OHC', 'IHC']:
                    label: int = 1 if label == 'OHC' else 2
                    box = [int(box.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]

                    boxes.append(box)
                    labels.append(label)

        return boxes, labels


class MultiDataset(Dataset):
    def __init__(self, *args):
        self.datasets: List[Dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]
        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self):
        return len(self._mapped_indicies)

    def __getitem__(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i][item - _offset]
        except RuntimeError:
            print(i, _offset, item - _offset, item, len(self.datasets[i]), self.datasets[i].files[item])
            raise RuntimeError

    def get_file_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i].get_file_at_index(item - _offset)
        except IndexError:
            print(i, _offset, item - _offset, item, len(self.datasets[i]), self.datasets[i])
            raise RuntimeError

    def get_cells_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        labels = self.datasets[i].get_labels_at_index(item - _offset)
        ohc = sum([1 for l in labels if l == 1])
        ihc = sum([1 for l in labels if l == 2])

        return ohc, ihc

    def get_metadata_at_index(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offseimport hcat.validate.utilst
        # print(i, _offset, item-_offset, item, len(self.datasets[i]))
        # assert (item - _offset) < len(self.datasets[i]), 'Trying to index outside of dataset'
        try:
            return self.datasets[i].metadata
        except IndexError:
            print(i, _offset, item - _offset, item, len(self.datasets[i]), self.datasets[i])
            raise RuntimeError

    def to(self, device: str):
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cuda:0')
        return self

    def cpu(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cpu')
        return self


def colate(data_dict: List[Dict[str, Tensor]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images = [dd.pop('image').squeeze(-1) for dd in data_dict]

    return images, data_dict