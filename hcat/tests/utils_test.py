import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import torch
import pytest
import numpy as np
import hcat.utils as utils

def test_calculate_indexes():
    assert utils.calculate_indexes(0, 100, 130, 130) == [[0, 99], [30, 129]]
    assert utils.calculate_indexes(0, 140, 130, 130) == [[0, 129]]
    assert utils.calculate_indexes(30, 100, 500, 500) == [[0, 159], [100, 259], [200, 359], [300, 459], [340, 499]]


def test_remove_edge_cells():
    zeros = torch.zeros((1, 10, 10, 10))
    assert torch.all(utils.remove_edge_cells(zeros) == 0)

    # Should only work on pytorch tensors
    with pytest.raises(RuntimeError):
        utils.remove_edge_cells(np.zeros((1, 10, 10, 10)))

    # Throw error when input is wrong dimmension
    did_succeed = False
    try:
        utils.remove_edge_cells(torch.rand((10, 10, 10)))
        did_succeed = True
    except:
        if did_succeed: raise ValueError('Did not throw error.')

    # Does it preserve the device
    if torch.cuda.is_available():
        rand = torch.rand((1, 100, 100, 10), device='cuda')
        out = utils.remove_edge_cells(rand)
        assert rand.device == out.device


def test_remove_wrong_sized_cells():
    input = torch.ones((1, 1, 512, 512, 30)).int()
    assert torch.all(utils.remove_wrong_sized_cells(input) == 0)

    input = torch.ones((1, 1, 50, 50, 30)).int()
    input[..., 1::] *= 0
    assert torch.all(utils.remove_wrong_sized_cells(input) == 0)


def test_crop_to_identical_size():
    a = torch.rand((1, 1, 500, 500, 30))
    b = torch.rand((1, 1, 400, 400, 20))
    a, b = utils.crop_to_identical_size(a, b)
    assert a.shape == b.shape

    a = torch.rand((1, 1, 400, 400, 20))
    b = torch.rand((1, 1, 400, 400, 20))
    a, b = utils.crop_to_identical_size(a, b)
    assert a.shape == b.shape

    a = torch.rand((1, 1, 0, 500, 30))
    b = torch.rand((1, 1, 100, 400, 20))
    a, b = utils.crop_to_identical_size(a, b)
    assert a.shape == b.shape

    did_complete = False
    try:
        a = torch.rand((1, 0, 500, 30))
        b = torch.rand((1, 1, 100, 400, 20))
        a, b = utils.crop_to_identical_size(a, b)
        did_complete = True
    except Exception:
        if did_complete:
            raise ValueError('Should have thrown error about ndim.')
