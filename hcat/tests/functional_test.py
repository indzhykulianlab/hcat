import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import torch
import numpy as np
import pytest
from pytest import approx
from hcat.exceptions import ShapeError
import hcat.functional as functional

def test_vector_to_embedding():
    input = torch.ones((1, 3, 25, 25, 25))
    v2e = torch.jit.script(functional.VectorToEmbedding())

    # Test functionality in eval mode
    v2e = v2e.eval()
    out = v2e(input)
    assert isinstance(out, torch.Tensor)
    assert out[0, 0, -1, -1, -1] == 2
    assert out[0, 0, 0, 0, 0] == 1
    assert out.shape == input.shape

    # Test functionality in train mode
    v2e = v2e.train()
    out = v2e(input)
    assert out[0, 0, -1, -1, -1] == 2
    assert out[0, 0, 0, 0, 0] == 1

    # Test functionality with cuda
    if torch.cuda.is_available():
        out = v2e(input.cuda())
        assert out[0, 0, -1, -1, -1] == 2
        assert out[0, 0, 0, 0, 0] == 1
        assert out.device == input.cuda().device


def test_embedding_to_probability():
    torch.manual_seed(1)

    e2p = torch.jit.script(functional.EmbeddingToProbability())

    centroids = torch.rand((1, 10, 3))
    embedding = torch.rand((1, 3, 100, 100, 10))
    sigma = torch.rand((3))

    e2p = e2p.eval()
    prob_1 = e2p(embedding, centroids, sigma)
    assert isinstance(prob_1, torch.Tensor)
    assert prob_1.shape[2::] == embedding.shape[2::]
    assert prob_1.shape[1] == centroids.shape[1]

    e2p = e2p.train()
    prob_2 = e2p(embedding, centroids, sigma)
    assert prob_2.shape[1] == centroids.shape[1]
    assert torch.all(prob_1 == prob_2)

    # if centroids is in pixel space we correct for it by internally divinding
    e2p = e2p.eval()
    prob_2 = e2p(embedding, centroids.mul(25), sigma)
    assert prob_2.shape[1] == centroids.shape[1]
    assert torch.abs(prob_1 - prob_2).sum() < 1

    if torch.cuda.is_available():
        prob_cuda = e2p(embedding.cuda(), centroids.cuda(), sigma.cuda())
        assert prob_cuda.device == embedding.cuda().device


def test_estimate_centroids():
    torch.manual_seed(1)
    np.random.seed(1)

    ec = functional.EstimateCentroids()
    embedding = torch.rand((1, 3, 100, 100, 10))
    prob = torch.rand((1, 1, 100, 100, 10)).gt(0.5).float()
    out = ec(embedding, prob)
    # should detect nothing because inputs are random
    assert out.shape == torch.empty((1, 0, 3)).shape
    assert out.device == torch.empty((1, 0, 3), device='cpu').device

    with pytest.raises(Exception):
        prob = torch.rand((1, 2, 100, 100, 10))
        ec(embedding, prob)
        prob = torch.rand((2, 1, 100, 100, 10))
        ec(embedding, prob)
        prob = torch.rand((1, 100, 100, 10))
        ec(embedding, prob)

    ec = functional.EstimateCentroids(min_samples=1, n_erode=0)
    embedding = torch.rand((1, 3, 100, 100, 10))
    prob = torch.rand((1, 1, 100, 100, 10)).gt(0.5).float()
    out = ec(embedding.cuda(), prob.cuda())
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 1
    assert out.shape[2] == 3
    assert out.cpu().squeeze().numpy() == approx(np.array([12.4864, 12.4506, 12.5686]), rel=1e-3)
    assert out.device == torch.empty((0), device='cuda').device


def test_iou():
    torch.manual_seed(1)
    a = torch.rand((1, 100, 100, 10)).gt(0.5).float()
    b = torch.rand((1, 100, 100, 10)).gt(0.5).float()
    iou = functional._iou(a, b)
    assert iou.item() == approx(0.3341, rel=1e-3)

    with pytest.raises(Exception):
        a = torch.rand((1, 100, 100, 10)).gt(0.5)
        b = torch.rand((1, 100, 100, 10)).gt(0.5)
        functional._iou(a, b)


