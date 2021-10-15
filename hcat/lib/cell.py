import torch
from torchvision.ops.boxes import box_convert
from typing import Tuple, Dict, Optional
from hcat.train.transforms import _crop

# DOCUMENTED

@torch.jit.script
def crop_to_identical_size(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops Tensor a to the shape of Tensor b, then crops Tensor b to the shape of Tensor a.


    :param a: input 1
    :param b: input 2
    :raises RuntimeError: If n_dim of tensor a and b are different
    :return:
    """
    if a.ndim != b.ndim:
        raise RuntimeError('Number of dimensions of tensor "a" does not equal tensor "b".')

    a = _crop(a, x=0, y=0, z=0, w=b.shape[2], h=b.shape[3], d=b.shape[4])
    b = _crop(b, x=0, y=0, z=0, w=a.shape[2], h=a.shape[3], d=a.shape[4])
    return a, b


@torch.jit.script
def _stat(asses: torch.Tensor, index: int) -> Dict[str, torch.Tensor]:
    """
    Calculate various statistics from an input cell tensor.

    :param asses: [C, 1] flattened input image.
    :param index: channel index to run analysis on.
    :return:
    """
    x = asses[index, :]
    numel = x.numel()

    return {'mean': x.mean(), 'median': torch.median(x),
            'std': x.std(), 'var': x.var(), 'min': x.min(), 'max': x.max(),
            '%saturated': torch.sum(x == 1).div(numel), '%zero': torch.sum(x == 0).div(numel)}


class Cell:
    def __init__(self,
                 image: Optional[torch.Tensor],
                 mask: Optional[torch.Tensor],
                 loc: torch.Tensor,
                 id: Optional[int] = None,
                 scores: Optional[torch.Tensor] = None,
                 boxes: Optional[torch.Tensor] = None,
                 cell_type: Optional[str] = None,
                 channel_name: Optional[Tuple[str]] = ('dapi', 'gfp', 'myo7a', 'actin')):
        """
        Dataclass of a single detected cell object.

        :param image: [B, C, X, Y ,Z] image crop of a *single* cell
        :param mask: [B, C, X, Y ,Z] segmentation mask of the *same* cell as image with identical size
        :param loc: [C, X, Y, Z] center location of cell
        :param id: unique cell identification number
        :param scores: cell detection likelihood
        :param boxes: [x0, y0, x1, y1] cell detection boxes
        :param cell_type: cell classification ID: 'OHC' or 'IHC'
        :param channel_name: image ordered channel dye names
        """

        self.id = id
        self.loc = loc.cpu()  # [C, X, Y, Z]
        self.frequency = None  # Set by self.calculate_frequency
        self.percent_loc = None  # Set by self.calculate_frequency
        self.type = cell_type # 'OHC' or 'IHC'
        self.channel_names = channel_name
        self.volume = None
        self.summed = None
        self.distance = None
        self.boxes = box_convert(torch.tensor([self.loc[1], self.loc[2], 30, 30]), 'cxcywh', 'xyxy') if boxes is None else boxes


        self.scores = scores # only for faster rcnn
        self.channel_stats = None

        self._curve_ind = None
        self._distance_from_curvature = None
        self._distance_is_far_away = False


        if mask is not None and image is not None:
            # Ensure image is in [B, C, X, Y, Z]
            assert image.ndim == 5
            if not image.min() >= 0:
                raise ValueError(image.min())

            # Crop both tensors to the same shape
            image, mask = crop_to_identical_size(image, mask)

            self.volume = mask.gt(0.5).sum().cpu()


            mask = mask[0, 0, ...].gt(0.5).cpu()

            # 0:DAPI
            # 1:GFP
            # 2:MYO7a
            # 3:Actin
            self.summed = image[0,0,...][mask].mul(2**12).sum(-1).mean()
            image = image.squeeze(0).reshape(image.shape[1], -1)[:, mask.flatten()].cpu()

            self.channel_stats = {}

            for i in range(image.shape[0]):
                self.channel_stats[channel_name[i]] = _stat(image, i)


    def calculate_frequency(self, curvature: torch.Tensor, distance: torch.Tensor) -> None:
        """
        Calculates cell's best frequency from its place along the cochlear curvature.
        Assigns values to properties: percent_loc, frequency

        Values of greenwood function taken from:
        Moore, B C. (1974). Relation between the critical bandwidth and the frequency-difference limen.
        The Journal of the Acoustical Society of America, 55(2), 359.

        https://en.wikipedia.org/wiki/Greenwood_function

        public double fMouse(double d){ // d is fraction total distance
            //f(Hz) = (10 ^((1-d)*0.92) - 0.680)* 9.8
            return (Math.pow(10, (1-d)*0.92) - 0.680) * 9.8;
        }

        d = d * 100;
        // f(KHz) = (10 ^(((100-d)/100)*2) - 0.4)*200/1000

        Example:
        --------

        >>> hcat.lib.cell import Cell
        >>> from hcat.lib.functional import PredictCurvature
        >>> import torch
        >>>
        >>> cells = torch.load('array_of_cells.trch')
        >>> masks = torch.load('predicted_segmentation_mask.trch') # torch.shape = [C=1, X, Y, Z]
        >>> curvature, distance, apex = PredictCurvature()(masks)
        >>> for c in cells:
        >>>     c.calculate_frequency(curvature, distance)
        >>> print(f'Best Frequency: {cells[0].frequency}, Hz') # Best Frequency: 1.512 kHz

        :param curvature: 2D curvature array from src.lib.functional.PredictCurvature
        :param distance: distance tensor from src.lib.functional.PredictCurvature
        :return: None
        """
        # confocal voxel size: 288.88 nm...

        dist = curvature[0, :].sub(self.loc[1]).pow(2) + curvature[1, :].sub(self.loc[2]).pow(2)

        self._curve_ind = torch.argmin(dist)
        self._distance_from_curvature = torch.sqrt(dist[self._curve_ind])
        self._distance_is_far_away = self._distance_from_curvature > 100
        self.distance = distance[self._curve_ind]
        self.percent_loc = distance[self._curve_ind].div(distance.max())
        self.frequency = (10 **((1-self.percent_loc)*0.92) - 0.680) * 9.8




if __name__ == '__main__':
    pass