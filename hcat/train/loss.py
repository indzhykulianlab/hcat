import torch
import torch.nn as nn
from typing import Optional

from hcat.lib.utils import crop_to_identical_size


class jaccard(nn.Module):
    def __init__(self):
        super(jaccard, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                smooth: float = 1e-10) -> torch.Tensor:
        """
        Returns jaccard index of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: jaccard_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        intersection = (predicted * ground_truth).sum().add(smooth)
        union = (predicted + ground_truth).sum().sub(intersection).add(smooth)

        return 1.0 - (intersection / union)


class dice(nn.Module):
    def __init__(self):
        super(dice, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                smooth: float = 1e-10) -> torch.Tensor:
        """
        Returns dice index of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: dice_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """


        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        intersection = (predicted * ground_truth).sum().add(smooth)
        denominator = (predicted + ground_truth).sum().add(smooth)
        loss = 2 * intersection / denominator

        return 1 - loss


class tversky(nn.Module):
    def __init__(self):
        super(tversky, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-10,
                alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.0, weight: bool = False) -> torch.Tensor:
        """
        Returns dice index of two torch.Tensors

        :param predicted: [B, N, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, N: instances in image
        :param ground_truth: [B, N, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (N).
        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :param alpha: float
                - Value which penalizes False Positive Values
        :param beta: float
                - Value which penalizes False Negatives
        :param gamma: float
                - Focal loss term
        :return: dice_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        if gamma != 0:
            # true_positive = (predicted * ground_truth).flatten(start_dim=2).sum(-1)
            # false_positive = (torch.logical_not(ground_truth) * predicted).flatten(start_dim=2).sum(-1).add(1e-10) * alpha
            # false_negative = ((1 - predicted) * ground_truth).flatten(start_dim=2).sum(-1) * beta
            # tversky = (true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)
            # tversky = tversky.add(-1).mul(-1).pow(1 / gamma).mean()
            true_positive = (predicted * ground_truth).sum()
            false_positive = (torch.logical_not(ground_truth) * predicted).sum().add(1e-10) * alpha
            false_negative = ((1 - predicted) * ground_truth).sum() * beta
            tversky = (true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)
            tversky = 1 - tversky
            tversky = tversky.pow(1/gamma)

        else:
            true_positive = (predicted * ground_truth).sum()
            false_positive = (torch.logical_not(ground_truth) * predicted).sum().add(1e-10) * alpha
            false_negative = ((1 - predicted) * ground_truth).sum() * beta
            tversky = (true_positive + smooth) / (true_positive + false_positive + false_negative + smooth)
            tversky = 1 - tversky

        return tversky


class l1(nn.Module):
    def __init__(self):
        super(l1, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Returns mean l1 distance of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :return: l1: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)
        loss = torch.nn.L1Loss()

        return loss(predicted, ground_truth)

class L1_CE(nn.Module):
    def __init__(self):
        super(L1_CE, self).__init__()

        self.l1 = torch.nn.L1Loss()
        self.cel = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predicted, ground_truth) -> torch.Tensor:
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        return self.l1(predicted.float(), ground_truth.float()).mean() + self.cel(predicted.float(), ground_truth.float()).mean()



# class cross_entropy(nn.Module):
#     def __init__(self):
#         super(cross_entropy, self).__init__()
#
#     def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
#         """
#
#         :param predicted: [B, I, X, Y, Z] torch.Tensor of probabilities calculated from hcat.utils.embedding_to_probability
#                           where B: is batch size, I: instances in image
#         :param ground_truth: [B, I, X, Y, Z] segmentation mask for each instance (I).
#         :return:
#         """
#         raise RuntimeError('Borked. Dont Use. Cuda Error due to not functioning bixel by pixel analysis.')
#
#         # Crop both tensors to the same shape
#         predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)
#
#         loss = torch.nn.CrossEntropyLoss()
#
#         for i in range(ground_truth.shape[1]):
#             ground_truth[:, i, ...] = ground_truth[:, i, ...] * (i + 1)
#
#         ground_truth = ground_truth.sum(1)
#
#         return loss(predicted, ground_truth)

class cross_entropy(nn.Module):
    def __init__(self, method: str = 'pixel', N: Optional[int] = None):
        super(cross_entropy, self).__init__()

        _methods = ['pixel', 'worst_z', 'random']
        if method not in _methods:
            raise ValueError(f'Viable methods for cross entropy loss are {_methods}, not {method}.')

        self.method = method
        self.N = N

        if method == 'random':
            if self.N is None:
                raise ValueError(f'the number of random pixels to draw is not defined. Please set num_random_pixels to a ' +
                                 f'value larger than 1.')
            if self.N <= 1:
                raise ValueError(f'num_random_pixels should be greater than 1 not {self.N}.')

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor, pwl: Optional[torch.Tensor] =None) -> torch.Tensor:
        """
        Pytorch Implementation of Cross Entropy Loss with a pixel by pixel weighting as described in U-NET



        :param pred: torch.Tensor | probability map of shape [B,C,X,Y,Z] predicted by hcat.unet
        :param mask: torch.Tensor | ground truth probability map of shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped
                     to identical size of pred
        :param pwl:  torch.Tensor | weighting map pf shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped to be identical in
                     size to pred and will be used to multiply the cross entropy loss at each pixel.
        :param method: ['pixel', 'worst_z', 'random'] | method by which to weight loss
                            - pixel: weights each pixel size by values in pwl
                            - worst_z: adds additional exponentially decaying weight to z planes on with the largest weight
                                       to the worst performing z plane
                            - random: randomly chooses
        :param num_random_pixels: int or None | Number of randomly selected pixels to draw when method='random'
        :return: torch.float | Average cross entropy loss of all pixels of the predicted mask vs ground truth
        """

        if (ground_truth == 0).sum() == 0:
            raise ValueError(f'There are no background pixels in mask.\n\t(mask==0).sum() == 0 -> True')

        pred_shape = predicted.shape
        n_dim = len(pred_shape)

        is_pwl_none = False
        if pwl is None:
            pwl = torch.ones(predicted.shape, device=predicted.device)
            is_pwl_none = True

        # Crop mask and pwl to the same size as pred
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)
        predicted, pwl = crop_to_identical_size(predicted, pwl)
        ground_truth, pwl = crop_to_identical_size(ground_truth, pwl)


        if not is_pwl_none:
            # Hacky way to do this:
            pwl[ground_truth > 0.5] += 2

        cel = nn.BCEWithLogitsLoss(reduction='none')
        loss = None

        if self.method == 'pixel':
            """
            Calculates BCE on ALL pixels of image
            """
            loss = cel(predicted.float(), ground_truth.float())
            loss = (loss * (pwl + 1))

        elif self.method == 'worst_z':
            """
            Calculates BCE on each pixel but weights poor performing z planes more highly
            """
            loss = cel(predicted.float(), ground_truth.float())
            loss = (loss * (pwl + 1))
            scaling = torch.linspace(1, 2, predicted.shape[4]) ** 2
            loss, _ = torch.sort(loss.sum(dim=[0, 1, 2, 3]))
            loss *= scaling.to(loss.device)
            loss /= (predicted.shape[2] * predicted.shape[3])

        elif self.method == 'random':
            """
            Randomly samples an equal number of positive and negative pixels for calculating BCE
            """
            pred = predicted.reshape(-1)
            mask = ground_truth.reshape(-1)

            if (mask == 1).sum() == 0:
                loss = cel(pred.float(), mask.float())
            else:
                pos_ind = torch.randint(low=0, high=int((mask == 1).sum()), size=(1, self.N))[0, :]
                neg_ind = torch.randint(low=0, high=int((mask == 0).sum()), size=(1, self.N))[0, :]

                pred = torch.cat([pred[mask == 1][pos_ind], pred[mask == 0][neg_ind]]).unsqueeze(0)
                mask = torch.cat([mask[mask == 1][pos_ind], mask[mask == 0][neg_ind]]).unsqueeze(0)

                loss = cel(pred.float(), mask.float())

        return loss.mean()


if __name__ == '__main__':
    a = torch.rand((1, 1, 200, 200, 30))
    b = torch.rand((1, 1, 200, 200, 30))
    loss = tversky()
    l = loss((a>0.5).float(), (a > 0.5).long())
    print(l)
