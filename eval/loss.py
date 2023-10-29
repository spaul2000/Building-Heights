import torch
import argparse

def custom_mse_loss(output, target):
    # Check for NaN and -1 values in target and ignore them
    mask_nan = torch.isnan(target) | (target == -1)
    
    # Apply the mask to both output and target
    output_masked = output[~mask_nan]
    target_masked = target[~mask_nan]
    
    # Calculate the MSE loss on the filtered values
    mse_loss = torch.nn.functional.mse_loss(output_masked, target_masked)
    
    return mse_loss

def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "MSE":
        return custom_mse_loss  # Return the custom loss function
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")