import src.dataloader
import src.loss
import src.functional
# import torch
# from src.models.RDCNet import RDCNet
from src.models.HCNet import HCNet
# from src.models.unet import Unet_Constructor as unet
# from src.models.RecurrentUnet import RecurrentUnet
# import torch
# import torch.optim
import torch.nn
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast
import time
import numpy as np
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter

import src.transforms as t
import skimage.io as io

from tqdm import trange

epochs = 150

model = torch.jit.script(HCNet(in_channels=3, out_channels=6, complexity=10)).cuda()
model.train()
# model.load_state_dict(torch.load('./trained_model_hcnet_long.mdl'))

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fun = src.loss.jaccard_loss()

transforms = torchvision.transforms.Compose([
    t.nul_crop(),
    t.random_crop(shape=(256, 256, 23)),
    # t.to_cuda(),
    # t.random_h_flip(),
    # t.random_v_flip(),
    # t.random_affine(),
    # t.adjust_brightness(),
    t.adjust_centroids(),
])
data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test',
                              transforms=transforms)
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

transforms = torchvision.transforms.Compose([
    t.adjust_centroids()
])

val = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/validate',
                             transforms=transforms)
val = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)

epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)

for e in epoch_range:
    time_1 = time.clock_gettime_ns(1)
    epoch_loss = []
    model.train()
    for data_dict in data:
        image = data_dict['image']
        image = (image - 0.5) / 0.5
        mask = data_dict['masks'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()

        out = model(image.cuda(), 5)
        assert out.requires_grad

        sigma = torch.sigmoid(out[:, -3::, ...])
        assert sigma.requires_grad

        out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])
        assert out.requires_grad

        out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)
        assert out.requires_grad

        # This is jank
        loss = loss_fun(out, mask.cuda())

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().cpu().item())

    epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())

    del out, sigma, image, mask, centroids, loss

    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)

    time_2 = time.clock_gettime_ns(1)
    delta_time = np.round((np.abs(time_2 - time_1) / 1e9) / 60, decimals=2)

    with torch.no_grad():
        val_loss = []
        model.eval()
        for data_dict in val:
            image = data_dict['image']
            image = (image - 0.5) / 0.5
            mask = data_dict['masks'] > 0.5
            centroids = data_dict['centroids']

            out = model(image.cuda(), 5)
            sigma = out[:, -1, ...]
            out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])
            out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)
            loss = loss_fun(out, mask.cuda())

            val_loss.append(loss.item())
        val_loss = torch.tensor(val_loss).mean()
    writer.add_scalar('Loss/validate', val_loss.item(), e)
    del loss, image, mask, val_loss, sigma

torch.save(model.state_dict(), 'overtrained_model_hcnet_long.mdl')

render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i, :, :, :] = render[i, :, :, :] * (i + 1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2, 1, 0)) / i + 1)
