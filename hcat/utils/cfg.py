import torch.nn as nn
from yacs.config import CfgNode


def cfg_to_model(cfg: CfgNode) -> nn.Module:
    raise NotImplementedError('Should return a torch model thing...')

