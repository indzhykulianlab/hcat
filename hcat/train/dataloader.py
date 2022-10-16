import warnings

import skimage.scripts.skivi
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import glob
import os.path
import shutil
from tqdm import tqdm
import skimage.io as io
from typing import Dict, List, Union
from skimage.morphology import ball
import xml
from torch.utils.data import Dataset
import hcat.lib.utils
from typing import Callable

import hcat.train.transforms as t


class dataset(DataLoader):
    def __init__(self, path: Union[List[str], str], transforms=None, random_ind: bool = False, min_cell_volume: int = 4000,
                 duplicate_poor_data: bool = False):
        super(DataLoader, self).__init__()

        # Functions to correct for Bad Masks
        self.colormask_to_mask = t.colormask_to_mask()
        self.transformation_correction = t.transformation_correction(min_cell_volume)  # removes bad masks

        # Reassigning variables
        self.mask = []
        self.image = []
        self.centroids = []
        self.transforms = transforms
        self.dpd = duplicate_poor_data

        path: List[str] = [path] if isinstance(path, str) else path
        print('WARNING: Memory PINNED!!!')

        # Find only files
        for p in path:
            self.files = glob.glob(os.path.join(p, '*.labels.tif'))
            for f in self.files:
                image_path = os.path.splitext(f)[0]
                image_path = os.path.splitext(image_path)[0] + '.tif'

                if image_path.find(
                        '-DUPLICATE') > 0 and not self.dpd:  # If -DUPLICATE flag is there and we want extra copies continue
                    continue

                image = torch.from_numpy(io.imread(image_path).astype(np.uint16) / 2 ** 16).unsqueeze(0)
                image = image.transpose(1, 3).transpose(0, -1).squeeze()[[0, 2, 3], ...]

                mask = torch.from_numpy(io.imread(f)).transpose(0, 2).unsqueeze(0)

                self.mask.append(mask.float())
                self.image.append(image.float())
                self.centroids.append(torch.tensor([0, 0, 0]))

            # implement random permutations of the indexing
            self.random_ind = random_ind

            if self.random_ind:
                self.index = torch.randperm(len(self.mask))
            else:
                self.index = torch.arange((len(self.mask)))

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            item = self.index[item]
            data_dict = {'image': self.image[item], 'masks': self.mask[item], 'centroids': self.centroids[item]}
            did_we_get_an_output = False

            while not did_we_get_an_output:
                try:
                    if self.transforms is not None:
                        data_dict = self.transforms(data_dict)
                        did_we_get_an_output = True
                except RuntimeError:
                    continue

            data_dict = self.colormask_to_mask(data_dict)
            assert torch.isnan(data_dict['centroids']).sum() == 0
            data_dict = self.transformation_correction(data_dict)  # removes bad masks

        return data_dict

    def step(self) -> None:
        if self.random_ind:
            self.index = torch.randperm(len(self.mask))


class detection_dataset(Dataset):
    def __init__(self, path: Union[List[str], str], transforms, random_ind: bool = False, min_cell_volume: int = 20*10,
                 duplicate_poor_data: bool = False, simple_class: bool = True, batch_size: int = 1, pad_size: int = 200):
        super(Dataset, self).__init__()

        # Functions to correct for Bad Masks
        self.transformation_correction = t.transformation_correction(min_cell_volume)  # removes bad masks

        if transforms is None: raise RuntimeError('Transformations Necessary for data loading.')

        # Reassigning variables
        self.mask = []
        self.image = []
        self.centroids = []
        self.boxes = []
        self.labels = []
        self.files = []
        self.transforms = transforms
        self.dpd = duplicate_poor_data
        self.simple_class = simple_class
        self.pad_size = pad_size
        self.device = 'cuda:0'

        path: List[str] = [path] if isinstance(path, str) else path

        # Find only files
        for p in path:
            self.files.extend(glob.glob(f'{p}{os.sep}*.xml'))

        print('WARNING: Image hacks in src.train.dataloader.detection_dataset')
        for f in self.files:
            image_path = f[:-4:]+ '.tif'

            if image_path.find(
                    '-DUPLICATE') > 0 and not self.dpd:  # If -DUPLICATE flag is there and we want extra copies continue
                continue

            # [c, X, Y] -> [c, x, y, z=1]
            image: np.array = io.imread(image_path)
            scale: int = hcat.lib.utils.get_dtype_offset(image.dtype)
            image: Tensor = TF.pad(torch.from_numpy(image / scale), self.pad_size).unsqueeze(-1)

            if image.max() < 0.2:
                image = image.div(image.max()).mul(0.8)
            if image[1, ...].max() < 0.3:
                image[1, ...] = image[1, ...].mul(2)

            mask = torch.empty((0,0,0,0))

            self.mask.append(mask)
            self.image.append(image.float().to(self.device, memory_format=torch.channels_last))
            self.centroids.append(torch.tensor([0, 0, 0]))

            tree = xml.etree.ElementTree.parse(f)
            root = tree.getroot()
            bbox_loc = []
            class_labels = []

            for c in root.iter('object'):
                for cls in c.iter('name'):
                    class_labels.append(cls.text)

                for a in c.iter('bndbox'):
                    x1 = int(a.find('xmin').text)
                    y1 = int(a.find('ymin').text)
                    x2 = int(a.find('xmax').text)
                    y2 = int(a.find('ymax').text)
                    bbox_loc.append([x1, y1, x2, y2])

            for i, s in enumerate(class_labels):
                if s == 'OHC':
                    class_labels[i] = 1
                elif s == 'OHC1':
                    class_labels[i] = 1
                elif s == 'OHC2':
                    class_labels[i] = 2
                elif s == 'OHC3':
                    class_labels[i] = 3
                elif s == 'IHC':
                    class_labels[i] = 4
                else:
                    class_labels[i] = 999
                    print('Unidentified Label in XML file of ' + f, '\n Assigning Value 999')

            class_labels = torch.tensor(class_labels)

            # For testing - reduce row specific labels in cell specific (IHC or OHC)
            if self.simple_class:
                class_labels[class_labels == 2] = 1  # OHC2 -> OHC
                class_labels[class_labels == 3] = 1  # OHC3 -> OHC
                class_labels[class_labels == 4] = 2  # IHC

            b = torch.tensor(bbox_loc).to(self.device) + self.pad_size
            c = class_labels.to(self.device)
            b = b[c < 10, :]
            c = c[c < 10]

            # ind = b[:, 0:2] < b[:, 2:]
            # ind = torch.logical_and(ind[:,0], ind[:,1])
            #
            # b = b[ind, :]
            # c = c[ind]

            self.boxes.append(b)
            self.labels.append(c)

        print(f'TOTAL DATASET SIZE: {len(self.files)}')

        # implement random permutations of the indexing
        self.random_ind = random_ind
        self.batch_size = batch_size

        self.index = self._batch_ind(batch_size=self.batch_size, dataset_len=len(self.image), random_ind=self.random_ind)

    @staticmethod
    def _batch_ind( batch_size: int, dataset_len: int, random_ind: bool = False) -> List[List[int]]:
        """

        :param batch_size:
        :param dataset_len:
        :return:
        """

        end = dataset_len + batch_size - (dataset_len % batch_size)
        base_ind: Tensor = torch.arange(0, end) # should make evenly

        if not base_ind.shape[0] % batch_size == 0:
            print(base_ind.shape, batch_size, dataset_len, dataset_len % batch_size, base_ind.shape[0] % batch_size)
            raise ValueError

        indicies = []
        _ind = []
        for i, ind in enumerate(base_ind):
            if ind >= dataset_len:
                base_ind[i] -= dataset_len

        # Shuffle the indicies if we do a random permutation
        if random_ind: base_ind = base_ind[torch.randperm(base_ind.shape[0])]

        for i, ind in enumerate(base_ind):
            _ind.append(base_ind[i].item())
            if len(_ind) == batch_size:
                indicies.append(_ind)
                _ind = []

        return indicies

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Union[Dict[str, torch.Tensor], Dict[str, Union[List[torch.Tensor], torch.Tensor]]]:

        out = []

        with torch.no_grad():
            item_list: List[int] = self.index[item]

            for index, item in enumerate(item_list):
                data_dict = {'image': self.image[item],
                             'masks': self.mask[item],
                             'centroids': self.centroids[item],
                             'boxes': self.boxes[item],
                             'labels': self.labels[item]}

                # Transformation pipeline
                data_dict = self.transforms(data_dict) # Apply transforms

                data_dict['boxes'] = torch.round(data_dict['boxes'].float())
                c, x, y, z = data_dict['image'].shape
                data_dict['image'] = torch.cat((torch.zeros((1, x, y, z), device=data_dict['image'].device),
                                                data_dict['image']), dim=0)[:, :, :, 0]
                data_dict.pop('masks', torch.tensor([]))

                for key in data_dict:
                    data_dict[key].detach()

                out.append(data_dict)

            return out  # List[Dict[str, Tensor]]

    def step(self) -> None:
        if self.random_ind:
            self.index = self._batch_ind(batch_size=self.batch_size, dataset_len=len(self.image), random_ind=self.random_ind)
        else:
            warnings.warn('Stepping when random_ind is set to false has no effect.')

    def cuda(self):
        self.image = [x.cuda() for x in self.image]
        self.mask = [x.cuda() for x in self.mask]
        self.centroids = [x.cuda() for x in self.centroids]
        self.boxes = [x.cuda() for x in self.boxes]
        self.labels = [x.cuda() for x in self.labels]

        return self


