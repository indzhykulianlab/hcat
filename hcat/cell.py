import torch
from typing import Tuple, Dict
from hcat.transforms import _crop


@torch.jit.script
def crop_to_identical_size(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops Tensor a to the shape of Tensor b, then crops Tensor b to the shape of Tensor a.

    :param a: torch.
    :param b:
    :return:
    """
    if a.ndim != b.ndim:
        raise RuntimeError('Number of dimensions of tensor "a" does not equal tensor "b".')

    a = _crop(a, x=0, y=0, z=0, w=b.shape[2], h=b.shape[3], d=b.shape[4])
    b = _crop(b, x=0, y=0, z=0, w=a.shape[2], h=a.shape[3], d=a.shape[4])
    return a, b


@torch.jit.script
def _stat(asses: torch.Tensor, index: int) -> Dict[str, torch.Tensor]:
    x = asses[index, :]
    numel = x.numel()

    return {'mean': x.mean(), 'median': torch.median(x),
            'std': x.std(), 'var': x.var(), 'min': x.min(), 'max': x.max(),
            '%saturated': torch.sum(x == 1).div(numel), '%zero': torch.sum(x == 0).div(numel)}


class Cell:
    def __init__(self, image: torch.Tensor, mask: torch.Tensor,
                 loc: torch.Tensor,
                 id: int = None):
        """

        :param im: [B, 4, X, Y, Z]
        :param mask: [B, C=1, X, Y, Z]
        :param location:
        """

        # Ensure image is in [B, C, X, Y, Z]
        assert image.ndim == 5
        if not image.min() >= 0:
            raise ValueError(image.min())

        # Crop both tensors to the same shape
        image, mask = crop_to_identical_size(image, mask)

        self.id = id
        self.loc = loc.cpu() # [C, X, Y, Z]
        self.frequency = None  # Set by self.calculate_frequency
        self.percent_loc = None  # Set by self.calculate_frequency

        self.volume = mask.gt(0.5).sum().cpu()

        # 0:DAPI
        # 1:GFP
        # 2:MYO7a
        # 3:Actin
        mask = mask[0, 0, ...].gt(0.5).cpu()
        image = image.squeeze(0).reshape(image.shape[1], -1)[:, mask.flatten()].cpu()

        self.dapi: Dict[str, torch.Tensor]  = _stat(image, 0)
        self.gfp: Dict[str, torch.Tensor]   = _stat(image, 1)
        self.myo7a: Dict[str, torch.Tensor] = _stat(image, 2)
        self.actin: Dict[str, torch.Tensor] = _stat(image, 3)

    def calculate_frequency(self, curvature: torch.Tensor,
                            A: float = 385.0, a: float = 0.3, K: float = 1) -> None:
        """
        https://en.wikipedia.org/wiki/Greenwood_function

        Values of greenwood function taken from:
        Moore, B C. (1974). Relation between the critical bandwidth and the frequency-difference limen.
        The Journal of the Acoustical Society of America, 55(2), 359.


        """
        if curvature is not None:
            ind = torch.argmin(curvature[1, :].sub(self.loc[1]).pow(2) + curvature[0, :].sub(self.loc[2]).pow(2))
            self.percent_loc = ind/curvature.shape[1] # float
            self.frequency = A * (10**(a*self.percent_loc) - K)


if __name__ == '__main__':
    a = torch.rand((1, 4, 100, 100, 10))
    b = torch.rand((1, 1, 100, 100, 10)).gt(0.5)
    for i in range(1000):
        c = Cell(a, b)
