from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from yacs.config import CfgNode


def write_progress(writer: SummaryWriter, tag: str, epoch: int, images: Tensor, masks: Tensor, target: Tensor, out: Tensor, cfg: CfgNode=None):
    """
    for writing training images to tensorboard

    :param writer:
    :param tag:
    :param epoch:
    :param images:
    :param masks:
    :param lsd:
    :param out:
    :return:
    """

    _a = images[0, [0, 0, 0], :, :, 7].cpu()
    _b = masks[0, [0, 0, 0], :, :, 7].gt(0.5).float().cpu()

    img_list = [_a, _b]

    if cfg is not None and cfg.TRAIN.TARGET in ['lsd', 'mtlsd']:
        _, c, _, _, _ = target.shape
        img_list.append(target[0, 0:3, ..., 7].float().cpu())
        img_list.append(out[0, 0:3, ..., 7].float().cpu())

        img_list.append(target[0, 3:6, ..., 7].float().cpu())
        img_list.append(out[0, 3:6, ..., 7].float().cpu())

        img_list.append(target[0, 6:9, ..., 7].float().cpu())
        img_list.append(out[0, 6:9, ..., 7].float().cpu())

        img_list.append(target[0, 9, ..., 7].expand(3, -1, -1).float().cpu())
        img_list.append(out[0, 9, ..., 7].expand(3, -1, -1).float().cpu())


    if cfg is not None and cfg.TRAIN.TARGET in ['affinities', 'mtlsd']:
        _, c, _, _, _ = target.shape

        img_list.append(target[0, c-3 : c, ..., 7].float().cpu())
        img_list.append(out[0, c-3 : c, ..., 7].float().cpu())

    else:
        c = target.shape[1]

        for index in range(c):
            img_list.append(target[0, [index], ..., 7].expand(3, -1, -1).float().cpu())
            img_list.append(out[0, [index], ..., 7].expand(3, -1, -1).float().cpu())



    _img = make_grid(img_list, nrow=2, normalize=True, scale_each=True)

    writer.add_image(tag, _img, epoch, dataformats='CWH')
