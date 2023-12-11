import torch

from .height_estimation import HeightEstimationTask
from .util import get_ckpt_callback, get_early_stop_callback, get_tb_logger, get_csv_logger

def get_task(args):
    if args.get("task") == "height_estimation":
        return HeightEstimationTask(args)
    else:
        print('The value entered for task is invalid')
        return None

def load_task(ckpt_path, **kwargs):
    if kwargs.get("task") == "height_estimation":
        return HeightEstimationTask.load_from_checkpoint(ckpt_path, **kwargs, strict=False)
   