# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Rao Fu, RainbowSecret
# --------------------------------------------------------'

import os

import argparse
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ""
# Dataset name
_C.DATA.DATASET = "imagenet"
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "swin"
# Model name
_C.MODEL.NAME = "swin_tiny_patch4_window7_224"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
_C.MODEL.RESUME_ONLY_MODEL = False
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# High Resolution Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# DEIT Transformer parameters
_C.MODEL.DEIT = CN()
_C.MODEL.DEIT.PATCH_SIZE = 16
_C.MODEL.DEIT.IN_CHANS = 3
_C.MODEL.DEIT.EMBED_DIM = 768
_C.MODEL.DEIT.DEPTHS = 12
_C.MODEL.DEIT.NUM_HEADS = 12
_C.MODEL.DEIT.MLP_RATIO = 4
_C.MODEL.DEIT.QKV_BIAS = True
_C.MODEL.DEIT.QK_SCALE = None
_C.MODEL.DEIT.REPRESENTATION_SIZE = None
_C.MODEL.DEIT.DROP_RATE = 0.0
_C.MODEL.DEIT.ATTN_DROP_RATE = 0.0
_C.MODEL.DEIT.DROP_PATH_RATE = 0.0
_C.MODEL.DEIT.HYBRID_BACKBONE = None
_C.MODEL.DEIT.NORM_LAYER = None
_C.MODEL.DEIT.PATCH_NORM_LAYER = None

# PVT Transformer parameters
_C.MODEL.PVT = CN()
_C.MODEL.PVT.PATCH_SIZE = 16
_C.MODEL.PVT.IN_CHANS = 3
_C.MODEL.PVT.EMBED_DIMS = [64, 128, 256, 512]
_C.MODEL.PVT.NUM_HEADS = [1, 2, 4, 8]
_C.MODEL.PVT.MLP_RATIOS = [4, 4, 4, 4]
_C.MODEL.PVT.QKV_BIAS = True
_C.MODEL.PVT.QK_SCALE = None
_C.MODEL.PVT.DROP_RATE = 0
_C.MODEL.PVT.ATTN_DROP_RATE = 0.0
_C.MODEL.PVT.DROP_PATH_RATE = 0.1
_C.MODEL.PVT.DEPTHS = [3, 4, 6, 3]
_C.MODEL.PVT.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.PVT.NUM_STAGES = 4
_C.MODEL.PVT.LINEAR = False

# HRNet parameters
_C.MODEL.HRNET = CN()
_C.MODEL.HRNET.DROP_PATH_RATE = 0.2

_C.MODEL.HRNET.STAGE1 = CN()
_C.MODEL.HRNET.STAGE1.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE1.NUM_BRANCHES = 1
_C.MODEL.HRNET.STAGE1.NUM_BLOCKS = [2]
_C.MODEL.HRNET.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.HRNET.STAGE1.BLOCK = "BOTTLENECK"

_C.MODEL.HRNET.STAGE2 = CN()
_C.MODEL.HRNET.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.STAGE2.NUM_BLOCKS = [2, 2]
_C.MODEL.HRNET.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.HRNET.STAGE2.BLOCK = "BASIC"

_C.MODEL.HRNET.STAGE3 = CN()
_C.MODEL.HRNET.STAGE3.NUM_MODULES = 3
_C.MODEL.HRNET.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.STAGE3.NUM_BLOCKS = [2, 2, 2]
_C.MODEL.HRNET.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.HRNET.STAGE3.BLOCK = "BASIC"

_C.MODEL.HRNET.STAGE4 = CN()
_C.MODEL.HRNET.STAGE4.NUM_MODULES = 2
_C.MODEL.HRNET.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
_C.MODEL.HRNET.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
_C.MODEL.HRNET.STAGE4.BLOCK = "BASIC"

# HRT parameters
_C.MODEL.HRT = CN()
_C.MODEL.HRT.DROP_PATH_RATE = 0.2

_C.MODEL.HRT.STAGE1 = CN()
_C.MODEL.HRT.STAGE1.NUM_MODULES = 1
_C.MODEL.HRT.STAGE1.NUM_BRANCHES = 1
_C.MODEL.HRT.STAGE1.NUM_BLOCKS = [2]
_C.MODEL.HRT.STAGE1.NUM_CHANNELS = [64]
_C.MODEL.HRT.STAGE1.NUM_HEADS = [2]
_C.MODEL.HRT.STAGE1.NUM_MLP_RATIOS = [4]
_C.MODEL.HRT.STAGE1.NUM_WINDOW_SIZES = [7]
_C.MODEL.HRT.STAGE1.ATTN_TYPES = [[["msw", "msw"]]]
_C.MODEL.HRT.STAGE1.FFN_TYPES = [[["conv_mlp", "conv_mlp"]]]
_C.MODEL.HRT.STAGE1.NUM_RESOLUTIONS = [[56, 56]]
_C.MODEL.HRT.STAGE1.BLOCK = "BOTTLENECK"

_C.MODEL.HRT.STAGE2 = CN()
_C.MODEL.HRT.STAGE2.NUM_MODULES = 1
_C.MODEL.HRT.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRT.STAGE2.NUM_BLOCKS = [2, 2]
_C.MODEL.HRT.STAGE2.NUM_CHANNELS = [18, 36]
_C.MODEL.HRT.STAGE2.NUM_HEADS = [1, 2]
_C.MODEL.HRT.STAGE2.NUM_MLP_RATIOS = [4, 4]
_C.MODEL.HRT.STAGE2.NUM_WINDOW_SIZES = [7, 7]
_C.MODEL.HRT.STAGE2.ATTN_TYPES = [[["msw", "msw"], ["msw", "msw"]]]
_C.MODEL.HRT.STAGE2.FFN_TYPES = [[["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"]]]
_C.MODEL.HRT.STAGE2.NUM_RESOLUTIONS = [[56, 56], [28, 28]]
_C.MODEL.HRT.STAGE2.BLOCK = "BASIC"

_C.MODEL.HRT.STAGE3 = CN()
_C.MODEL.HRT.STAGE3.NUM_MODULES = 3
_C.MODEL.HRT.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRT.STAGE3.NUM_BLOCKS = [2, 2, 2]
_C.MODEL.HRT.STAGE3.NUM_CHANNELS = [18, 36, 72]
_C.MODEL.HRT.STAGE3.NUM_HEADS = [1, 2, 4]
_C.MODEL.HRT.STAGE3.NUM_MLP_RATIOS = [4, 4, 4]
_C.MODEL.HRT.STAGE3.NUM_WINDOW_SIZES = [7, 7, 7]
_C.MODEL.HRT.STAGE3.ATTN_TYPES = [
    [["msw", "msw"], ["msw", "msw"], ["msw", "msw"]],
    [["msw", "msw"], ["msw", "msw"], ["msw", "msw"]],
    [["msw", "msw"], ["msw", "msw"], ["msw", "msw"]],
]
_C.MODEL.HRT.STAGE3.FFN_TYPES = [
    [["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"]],
    [["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"]],
    [["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"], ["conv_mlp", "conv_mlp"]],
]
_C.MODEL.HRT.STAGE3.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14]]
_C.MODEL.HRT.STAGE3.BLOCK = "BASIC"

_C.MODEL.HRT.STAGE4 = CN()
_C.MODEL.HRT.STAGE4.NUM_MODULES = 2
_C.MODEL.HRT.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRT.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
_C.MODEL.HRT.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
_C.MODEL.HRT.STAGE4.NUM_HEADS = [1, 2, 4, 8]
_C.MODEL.HRT.STAGE4.NUM_MLP_RATIOS = [4, 4, 4, 4]
_C.MODEL.HRT.STAGE4.NUM_WINDOW_SIZES = [7, 7, 7, 7]
_C.MODEL.HRT.STAGE4.ATTN_TYPES = [
    [["msw", "msw"], ["msw", "msw"], ["msw", "msw"], ["msw", "msw"]],
    [["msw", "msw"], ["msw", "msw"], ["msw", "msw"], ["msw", "msw"]],
]
_C.MODEL.HRT.STAGE4.FFN_TYPES = [
    [
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
    ],
    [
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
        ["conv_mlp", "conv_mlp"],
    ],
]
_C.MODEL.HRT.STAGE4.NUM_RESOLUTIONS = [[56, 56], [28, 28], [14, 14], [7, 7]]
_C.MODEL.HRT.STAGE4.BLOCK = "BASIC"

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = "O1"
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    # if args.amp_opt_level:
    #     config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # support more datasets
    if args.data_set == "CIFAR":
        config.DATA.DATASET = "cifar"
    elif args.data_set == "IMNET":
        config.DATA.DATASET = "imagenet"

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_default(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    return config


def parse_option():
    parser = argparse.ArgumentParser(
        "HighResolution Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        default='/root/autodl-tmp/ou/DSCNet-main/DSCNet-main/DSCNet_2D_opensource/Code/DRIVE/DSCNet/model/HREFNet_base.yaml',
        type=str,
        # required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    # distributed training
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        # required=True,
        help="local rank for DistributedDataParallel",
    )

    # set training dataset
    parser.add_argument(
        "--data-set", default="IMNET", choices=["CIFAR", "IMNET"], type=str
    )

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config