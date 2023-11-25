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

def custom_weighted_mse_loss(output, target, weight_factor=0.3):
    # Check for NaN and -1 values in target and ignore them
    mask_nan = torch.isnan(target) | (target == -1)
    
    # Apply the mask to both output and target
    output_filtered = output[~mask_nan]
    target_filtered = target[~mask_nan]

    # Check if filtering resulted in NaN values
    if torch.isnan(output_filtered).any() or torch.isnan(target_filtered).any():
        raise ValueError("NaN values detected in filtered output or target")

    mask_buildings = target_filtered > 0
    mask_no_building = target_filtered == 0

    output_buildings = output_filtered[mask_buildings]
    target_buildings = target_filtered[mask_buildings]

    output_no_buildings = output_filtered[mask_no_building]
    target_no_buildings = target_filtered[mask_no_building]
    
    # Check for empty tensors and compute MSE loss
    if output_buildings.nelement() == 0 or target_buildings.nelement() == 0:
        mse_loss_buildings = torch.tensor(0.0, device=output.device)
    else:
        mse_loss_buildings = torch.nn.functional.mse_loss(output_buildings, target_buildings)
        if torch.isnan(mse_loss_buildings).any():
            raise ValueError("NaN detected in mse_loss_buildings")

    if output_no_buildings.nelement() == 0 or target_no_buildings.nelement() == 0:
        mse_loss_no_buildings = torch.tensor(0.0, device=output.device)
    else:
        mse_loss_no_buildings = torch.nn.functional.mse_loss(output_no_buildings, target_no_buildings)
        if torch.isnan(mse_loss_no_buildings).any():
            raise ValueError("NaN detected in mse_loss_no_buildings")
    
    # Calculate the weighted MSE loss on the filtered values
    weighted_mse_loss = weight_factor * mse_loss_buildings + (1 - weight_factor) * mse_loss_no_buildings

    return weighted_mse_loss

def building_mse_loss(output, target):
    # Check for NaN and -1 values in target and ignore them
    mask_valid = ~torch.isnan(target) & (target > 0)
    
    # Apply the mask to both output and target
    output_masked = output[mask_valid]
    target_masked = target[mask_valid]
    
    # Check if the resulting tensors are empty
    if output_masked.numel() == 0 or target_masked.numel() == 0:
        return None

    # Calculate the MSE loss on the filtered values
    mse_loss = torch.nn.functional.mse_loss(output_masked, target_masked, reduction='mean')
    
    # Check for NaN in loss
    if torch.isnan(mse_loss):
        raise RuntimeError("NaN value encountered in mse_loss.")
    return mse_loss



def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    if loss_fn == "MSE":
        return custom_weighted_mse_loss  # Return the custom loss function
    elif loss_fn == "BuildingMSE":
        return building_mse_loss
    elif loss_fn == "WeightedMSE":
        return custom_weighted_mse_loss
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")