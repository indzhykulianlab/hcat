import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import torch
from hcat.exceptions import ShapeError
import hcat.transforms as transforms
import pytest

# Common Transform Tests ----------
def does_transform_work(transform,
                        image=torch.rand((3, 100, 100, 30)),
                        mask=torch.rand((1, 100, 100, 30)).gt(0.5),
                        centroids=torch.rand((1, 10, 3))):

    dd = {'image': image.float(),
          'masks': mask.float(),
          'centroids': centroids.float()}

    out = transform(dd)

    assert out['image'].ndim == 4
    assert out['masks'].ndim == 4
    assert out['masks'].shape[0] == 1
    assert out['image'].shape[1] == out['masks'].shape[1]
    assert out['image'].shape[2] == out['masks'].shape[2]
    assert out['image'].shape[3] == out['masks'].shape[3]

    with pytest.raises(ValueError):
        transform([])
        transform({'image': []})

    with pytest.raises(KeyError):
        transform({'test': torch.Tensor([])})

    with pytest.raises(Exception):
        transform({'image': torch.rand((3, 100, 100, 10)), 'masks': torch.rand((1, 200, 200, 20)), 'centroids': torch.tensor([])})
        transform({'image': torch.rand((100, 100, 10)), 'masks': torch.rand((100, 100, 10)), 'centroids': torch.tensor([])})

    return out


def does_cuda_work(transform):
    if torch.cuda.is_available():
        image = torch.rand((3, 100, 100, 30), device='cuda')
        mask = torch.rand((1, 100, 100, 30), device='cuda').gt(0.5).float()
        centroids = torch.rand((1, 10, 3), device='cuda')

        dd = {'image': image, 'masks': mask, 'centroids': centroids}
        out = transform(dd)

    assert out['image'].device == image.device

    try: transform({'image': image, 'masks': mask.cpu(), 'centroids': centroids})
    except RuntimeError: pass

# Tests -------------
def test_adjust_brightness():
    transform = transforms.adjust_brightness(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_adjust_gamma():
    transform = transforms.adjust_gamma(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_affine3d():
    transform = transforms.affine3d(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_colormask_to_mask():
    torch.manual_seed(1)
    image = torch.rand((3,100,100,10))
    mask = torch.rand((1,100,100,10)).mul(10).round()
    centroids = torch.rand((1,10,3))

    dd = {'image': image.float(),
          'masks': mask.float(),
          'centroids': centroids.float()}

    transform = transforms.colormask_to_mask()


    out = transform(dd)

    assert out['image'].ndim == 4
    assert out['masks'].ndim == 4
    assert out['masks'].shape[0] == 10
    assert out['image'].shape[1] == out['masks'].shape[1]
    assert out['image'].shape[2] == out['masks'].shape[2]
    assert out['image'].shape[3] == out['masks'].shape[3]

    try: transform([])
    except ValueError: pass

    try: transform({'test': []})
    except ValueError: pass

    try: transform({'test': torch.Tensor([])})
    except KeyError: pass

    try: transform({'test': []})
    except ValueError: pass

    try: transform({'image': torch.rand((3, 100, 100, 10)), 'masks': torch.rand((1, 200, 200, 20)),
                    'centroids': torch.tensor([])})
    except ShapeError: pass

    try: transform({'image': torch.rand((100, 100, 10)), 'masks': torch.rand((100, 100, 10)), 'centroids': torch.tensor([])})
    except ShapeError: pass

def test_elastic_deformation():
    transform = transforms.elastic_deformation(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_erosion():
    transform = transforms.erosion(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_median_filter():
    transform = transforms.median_filter(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_normalize():
    transform = transforms.normalize()
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_affine():
    transform = transforms.random_affine(rate=1)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_nul_crop():
    transform = transforms.nul_crop(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_affine():
    transform = transforms.random_affine(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_crop():
    transform = transforms.random_crop()
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_h_flip():
    transform = transforms.random_h_flip(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_v_flip():
    transform = transforms.random_v_flip(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_random_noise():
    transform = transforms.random_noise(rate=1.0)
    does_transform_work(transform)
    does_cuda_work(transform)

def test_to_cuda():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA not available.')

    transform = transforms.to_cuda()
    out = does_transform_work(transform)
    a = torch.tensor([], device='cuda')

    for key in out:
        assert out[key].device == a.device

# def test_to_tensor():
#     transform = src.transforms.to_tensor()
#     does_transform_work(transform)
#     out = transform({'image': np.ones((1, 1, 100, 100))})
#     assert isinstance(out['image'], torch.Tensor)

def test_transfomration_correction():
    transform = transforms.transformation_correction()
    does_transform_work(transform)
    does_cuda_work(transform)
