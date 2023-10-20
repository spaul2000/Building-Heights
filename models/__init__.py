import argparse

from util import Args
from .estimation import *



def get_model(model_args):
    model_args_ = model_args

    if isinstance(model_args, argparse.Namespace):
        model_args_ = Args(vars(model_args))

    return globals().copy()[model_args_.get("model")](model_args_)

def get_est_model(params: dict, n_bands: int
                 ) -> nn.Module:
    return SegNet(params['seg_architecture'], params['seg_backbone'], n_bands)