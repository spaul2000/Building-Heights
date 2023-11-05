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

def custom_weighted_mse_loss(output, target, weight_factor=1.2):
    # Check for NaN and -1 values in target and ignore them
    mask_nan = torch.isnan(target) | (target == -1)
    
    # Apply the mask to both output and target
    output_masked = output[~mask_nan]
    target_masked = target[~mask_nan]
    
    # Calculate the squared error
    squared_error = (output_masked - target_masked) ** 2
    
    # Apply weights to values greater than 0 in the target
    weights = torch.where(target_masked > 0, target_masked * weight_factor, torch.tensor(1.0))
    
    # Calculate the weighted MSE loss on the filtered values
    weighted_mse_loss = torch.sum(weights * squared_error) / torch.sum(weights)
    
    return weighted_mse_loss

def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "MSE":
        return custom_weighted_mse_loss  # Return the custom loss function
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")