import torch
from functools import partial

# import bism.loss.cl_dice
# import bism.loss.tversky
# import bism.loss.dice
# import bism.loss.jaccard
# import bism.loss.omnipose
# import bism.loss.torchvision
# import bism.loss.iadb
#
# from bism.targets.affinities import affinities
# from bism.targets.local_shape_descriptors import lsd
# from bism.targets.aclsd import aclsd
# from bism.targets.mtlsd import mtlsd
# import warnings
# try:
#     from bism.targets.omnipose import omnipose
# except:
#     warnings.warn('triton not found, likely because this is not a CUDA enabled machine. skipping validation for omnipose')
#     omnipose = None
#
# from bism.targets.semantic import semantic
# from bism.targets.maskrcnn import maskrcnn
# from bism.targets.iadb import IADBTarget
#
# import bism.backends
# import bism.backends.unet_conditional_difusion
# from bism.models.generic import Generic
# from bism.models.lsd import LSDModel
# from bism.models.spatial_embedding import SpatialEmbedding


"""
 --- Idea --- 
 This is not an awful way to do it, but requires us to import everything, also makes it hard to validate 
 Could implement a class to do this. Worth the added complexity? Probably not.
"""

_valid_optimizers = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamax': torch.optim.Adamax
}

_valid_lr_schedulers = {
    'cosine_annealing_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}