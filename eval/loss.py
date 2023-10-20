import torch
import argparse

def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "MSE":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")