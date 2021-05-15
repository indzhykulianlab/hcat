import skimage.scripts.skivi
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os.path
import skimage.io as io
from typing import Dict
from skimage.morphology import ball

import hcat.transforms as t


class dataset(DataLoader):
    def __init__(self, path, transforms=None, random_ind: bool = False, min_cell_volume: int = 4000,
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

        # Find only files
        self.files = glob.glob(os.path.join(path, '*.labels.tif'))
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


class nul_dataset(DataLoader):
    def __init__(self, path, transforms=None, random_ind: bool = False):

        super(DataLoader, self).__init__()

        # Find only files
        files = glob.glob(os.path.join(path, '*.tif'))

        self.image = []
        self.centroids = []
        self.transforms = transforms

        for f in files:
            image = torch.from_numpy(io.imread(f).astype(np.uint16) / 2 ** 16).unsqueeze(0)
            image = image.transpose(1, 3).transpose(0, -1).squeeze()[[0, 2, 3], ...]
            self.centroids.append(torch.tensor([0, 0, 0]))
            self.image.append(image.float())

        # implement random permutations of the indexing
        self.random_ind = random_ind

        if self.random_ind:
            self.index = torch.randperm(len(self.image))
        else:
            self.index = torch.arange((len(self.image)))

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:

        item = self.index[item]
        shape = self.image[item].shape
        shape[0] = 1  # override the channel shape (4 -> 1)

        data_dict = {'image': self.image[item], 'masks': torch.zeros(shape), 'centroids': self.centroids[item]}
        did_we_get_an_output = False

        while not did_we_get_an_output:
            try:
                if self.transforms is not None:
                    data_dict = self.transforms(data_dict)
                    did_we_get_an_output = True
            except RuntimeError:
                continue

        return data_dict

    def step(self) -> None:
        if self.random_ind:
            self.index = torch.randperm(len(self.image))


class em_dataset(DataLoader):
    def __init__(self, path, transforms=None, random_ind: bool = False):
        super(DataLoader, self).__init__()

        # Find only files
        files = glob.glob(os.path.join(path, '*.labels.tif'))

        self.mask = []
        self.image = []
        self.centroids = []
        self.transforms = transforms

        for f in files:
            image_path = os.path.splitext(f)[0]
            image_path = os.path.splitext(image_path)[0] + '.tif'
            image = torch.from_numpy(io.imread(image_path).astype(np.uint16) / 2 ** 16).unsqueeze(-1)

            if image.shape[-1] > 1:
                image = image.transpose(1, 3).transpose(0, -1).squeeze()[[0, 2, 3], ...]
            else:
                image = image.transpose(1, 3).transpose(0, -1).squeeze().unsqueeze(0)

            mask = torch.from_numpy(io.imread(f)).transpose(0, 2).unsqueeze(0)

            self.mask.append(mask.float())
            self.image.append(image.float())
            self.centroids.append(torch.tensor([0]))

        # implement random permutations of the indexing
        self.random_ind = random_ind

        if self.random_ind:
            self.index = torch.randperm(len(self.mask))
        else:
            self.index = torch.arange((len(self.mask)))

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:

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

        return data_dict

    def step(self) -> None:
        if self.random_ind:
            self.index = torch.randperm(len(self.mask))


class synthetic():
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        ball = torch.from_numpy(skimage.morphology.ball(15))
        mask = torch.zeros((1, 156, 156, 31))
        image = torch.zeros((4, 156, 156, 31))

        x = torch.randint(156 - 32, (1,))
        y = torch.randint(156 - 32, (1,))

        mask[0, x:x + 31, y:y + 31, :] = ball
        image[1, x:x + 31, y:y + 31, :] = ball
        image = (2 * image / 3) + (torch.rand((4, 156, 156, 31)) / 3)
        centroids = torch.zeros((1, 3))
        centroids[0, :] = torch.tensor([x + 31 // 2, y + 31 // 2, 31 // 2])

        return {'masks': mask.cuda(), 'image': image.cuda(), 'centroids': centroids.cuda()}

    def step(self):
        pass


@torch.jit.script
def colormask_to_torch_mask(colormask: torch.Tensor) -> torch.Tensor:
    """

    :param colormask: [C=1, X, Y, Z]
    :return:
    """
    uni = torch.unique(colormask)
    uni = uni[uni != 0]
    num_cells = len(uni)

    shape = (num_cells, colormask.shape[1], colormask.shape[2], colormask.shape[3])
    mask = torch.zeros(shape)

    for i, u in enumerate(uni):
        mask[i, :, :, :] = (colormask[0, :, :, :] == u)

    return mask


@torch.jit.script
def colormask_to_centroids(colormask: torch.Tensor) -> torch.Tensor:
    uni = torch.unique(colormask)
    uni = uni[uni != 0]
    num_cells = len(uni)  # cells are denoted by integers 1->max_cell
    shape = (num_cells, 3)
    centroid = torch.zeros(shape)

    for i, u in enumerate(uni):
        indexes = torch.nonzero(colormask[0, :, :, :] == u).float()
        centroid[i, :] = torch.mean(indexes, dim=0)

    # centroid[:, 0] /= colormask.shape[1]
    # centroid[:, 1] /= colormask.shape[2]
    # centroid[:, 2] /= colormask.shape[3]

    return centroid


if __name__ == "__main__":
    data = synthetic()

    out = data[1]
    print(out['centroids'].shape)
