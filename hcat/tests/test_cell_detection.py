import torch
from hcat.lib.functional import VectorToEmbedding, EstimateCentroids
from torchvision.utils import draw_keypoints, flow_to_image, make_grid

import numpy as np
import matplotlib.pyplot as plt


def make_embedding_image(embedding):
    x = embedding.detach().cpu().numpy()[0, 0, ...].flatten()
    y = embedding.detach().cpu().numpy()[0, 1, ...].flatten()
    histogram, _, _ = np.histogram2d(x, y, bins=(embedding.shape[2], embedding.shape[3]))
    return torch.from_numpy(histogram)


def progress(embedding, vector, centroids):
    keypoint = centroids[0, :, [1, 0]].unsqueeze(0)


    _overlay = make_embedding_image(embedding).unsqueeze(0)[[0, 0, 0], ...]
    _overlay = _overlay.div(10).clamp(0, 1)

    _c = draw_keypoints(_overlay.mul(255).round().type(torch.uint8),
                        keypoint, colors='red', radius=3).cpu()
    _d = draw_keypoints(flow_to_image(vector[0, [1, 0], :, :, 15].float()),
                        keypoint, colors='red', radius=3).cpu()

    img_list = [_c, _d]

    for i, im in enumerate(img_list):
        assert isinstance(im, torch.Tensor), f'im {i} is not a Tensor instead it is a {type(im)}, {img_list[i]}'
        assert img_list[0].shape == img_list[
            i].shape, f'im {i} is has shape {im.shape}, not {img_list[0].shape}'

    _img = make_grid(img_list, nrow=1, normalize=False, scale_each=True)
    return _img



vector = torch.load('/home/chris/Dropbox (Partners HealthCare)/HairCellInstance/hcat/tests/vector_test.trch').cuda()[[0], ...]
prob = torch.load('/home/chris/Dropbox (Partners HealthCare)/HairCellInstance/hcat/tests/prob_map.trch').cuda()[[0], ...]

centroids = torch.empty((1, 0, 3))
num = torch.tensor((60, 60, 20))

embedding = VectorToEmbedding(num, device='cuda')(vector, n=100)

"""
    def __init__(self, downsample: int = 1,
                 n_erode: int = 1,
                 eps: float = 0.5,
                 min_samples: int = 20,
                 scale: Tensor = torch.tensor([25]),
                 device='cpu'):
"""
centroids = EstimateCentroids(downsample=2, n_erode=1, eps=0.5, min_samples=20, scale=num, device='cuda:0')(embedding, prob)
print(centroids)

torch.save(embedding, 'embedding_test.trch')

img = progress(embedding, vector, centroids)
plt.imshow(img.permute((2,1,0)))
plt.show()

