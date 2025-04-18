from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Training config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
# Define a BISM Model
_C.MODEL = CN()

_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 1e-4   # any float, use log scale, start on this value
_C.TRAIN.WEIGHT_DECAY = 1e-4    # any float, use log scale, start on this value
_C.TRAIN.OPTIMIZER = 'adamw'  # could also do 'adam', 'sgd', 'sgdm'
_C.TRAIN.SCHEDULER = 'cosine_annealing' # this is the only option
_C.TRAIN.N_EPOCHS = 300 # any integer
_C.TRAIN.TRAIN_DIR = []  # button which selects multiple folders and finds pairs of images, *pngs and *xml files with the same name
_C.TRAIN.VALIDATION_DIR = []  # same as above
_C.TRAIN.SAVE_PATH = '' # button that selects a folder and saves its path
_C.TRAIN.MIXED_PRECISION = False # button that selects a folder and saves its path


_C.EVAL = CN()
_C.EVAL.CELL_THRESHOLD = 0.5 # between 0 - 1, QDoubleSpinBox
_C.EVAL.NMS_THRESHOLD = 0.5 # between 0 - 1, QDoubleSpinBox

# group these into checkboxes in a single line
_C.EVAL.USE_RED = True
_C.EVAL.USE_GREEN = True
_C.EVAL.USE_BLUE = True

_C.EVAL.LIVE_UPDATE = False
_C.EVAL.EVAL_PATCH_SIZE = 512  # between 128 - 1024
_C.EVAL.QUICK_EVAL = False


# constants = {
#     'model': model,
#     'lr': 1e-4,
#     'wd': 1e-4,
#     'optimizer': partial(torch.optim.AdamW, eps=1e-16),
#     'scheduler': partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=3000 + 1),
#     # partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=500),
#     'epochs': 3000,  # int(.3 * (60 * 60 / 18)),  # 1.5hrs worth of training...
#     'device': device,
#     'train_data': dataloader,
#     'val_data': valdiation_dataloader,
#     'train_sampler': train_sampler,
#     'test_sampler': test_sampler,
#     'distributed': True,
#     'mixed_precision': True,
#     'savepath': '/home/chris/Dropbox (Partners HealthCare)/trainHairCellDetection/models'
# }


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
