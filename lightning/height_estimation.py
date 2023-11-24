import logging
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import csv
from data import EstimationDataset
from eval import get_loss_fn
from models import *
from util import constants as C
from util import util
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision import transforms


class HeightEstimationTask(pl.LightningModule):
    """Interface for trainer to work with semantic segmentation"""

    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        ds_name = self.hparams['seg_dataset']
        self.ds_meta, n_bands = C.EST_DS[ds_name]
        n_s1 = C.NUM_S1
        n_s2 = C.NUM_S2
        print(params)
        # Unet traditional Architecture
        # self.model = get_est_model(params, n_bands=n_bands)
        # Dual Branch Unet Architecture
        self.model = get_dual_model(params, n_s1=n_s1, n_s2=n_s2)
        self.current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.current_device)
        self.supervised_loss = get_loss_fn(params)

        self.training_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []   # save targets in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []        # save targets in each batch to compute metric overall epoch

        self.test_loss_outputs = []
        self.test_path_outputs = []

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x)

    def configure_optimizers(self) -> list:
        lr = float(self.hparams['learning_rate'])
        optimizer = self.hparams['optimizer']
        if optimizer == 'Adam':
            return [torch.optim.Adam(self.parameters(), lr)]
        elif optimizer == 'SGD':
            return [torch.optim.SGD(self.parameters(), lr)]
        elif optimizer == 'AdamW':
            return [torch.optim.AdamW(self.parameters(), lr)]
        else:
            raise ValueError(f"optimizer {optimizer} not supported.")

    def training_step(self, batch: dict, batch_nb: int
                      ) -> torch.FloatTensor:
        x = batch['im'].to(self.device)
        y = batch['mask'].to(self.device)


        logits = self.forward(x)
        
        loss = self.supervised_loss(logits, y)
<<<<<<< HEAD
        if loss is not None:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True )
=======

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True )
>>>>>>> 0ae3e64f82d3a409ef9b21f1be9cfb7dfc76a57a

        return loss

    def log_images_to_tensorboard(self, original_images, masks, predictions, global_step):
        grid = []
        for im, mask, pred in zip(original_images, masks, predictions):
            im_rgb = im[-3:]  # Extract the last 3 channels for RGB
            mask = mask.squeeze(0)
            pred = pred.squeeze(0)

            # Normalize the RGB image to [0, 1] for better visualization
            im_rgb = (im_rgb - im_rgb.min()) / (im_rgb.max() - im_rgb.min())

            # Use a colormap to color the mask and predictions
            cm = plt.get_cmap('jet')
            mask_np = mask.cpu().numpy()
            mask_colored_np = cm(mask_np)[..., :3]  # This gives a numpy array
            mask_colored = torch.tensor(mask_colored_np).to(device=mask.device)
            pred_np = pred.cpu().numpy()
            pred_colored_np = cm(pred_np)[..., :3]  # This gives a numpy array
            pred_colored = torch.tensor(pred_colored_np).to(device=pred.device)

            # Stack the original RGB image, mask, and prediction along the width dimension
            triplet = torch.cat([
                im_rgb,
                mask_colored.permute(2, 0, 1),
                pred_colored.permute(2, 0, 1)
            ], dim=2)  # Concatenate along the width

            grid.append(triplet)

        # Make a grid of images and log to Tensorboard
        grid_tensor = make_grid(grid, nrow=1)  # Make a grid with 1 row
        self.logger.experiment.add_image('Validation Image-Mask-Prediction', grid_tensor, global_step)


    def validation_step(self, batch: dict, batch_nb: int):
        x = batch['im']
        y = batch['mask']


        logits = self.forward(x)
      
        loss = self.supervised_loss(logits, y)
<<<<<<< HEAD
        if loss is not None:
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True )

            self.val_step_outputs.append(loss)
=======
        self.log('val_loss', loss)
        self.val_step_outputs.append(loss)
>>>>>>> 0ae3e64f82d3a409ef9b21f1be9cfb7dfc76a57a

        

        # self.log_images_to_tensorboard(x[:1], y[:1], logits[:1], self.current_epoch)

        return {
            'val_loss': loss
        }

    def on_validation_epoch_end(self):
        
        val_loss = torch.stack([x for x in self.val_step_outputs]).mean()
        
        self.log('val_loss', val_loss)
     
        self.eval_results = {
            'val_loss': val_loss,
        }

        self.val_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        x = batch['im']  # Assume x has shape (batch_size, channels, height, width)
        y = batch['mask']  # Assume y has shape (batch_size, channels, height, width)

        logits = self.forward(x)
        logits = logits.reshape(y.shape)
        
        loss = self.supervised_loss(logits, y)

        output_paths = []  # List to hold paths of saved files for each item in the batch
        
        for i in range(x.size(0)):  # Loop through each item in the batch
            # Define paths
<<<<<<< HEAD
            img_path = f"test_results3/images/img_{batch_nb}_{i}.png"
            mask_path = f"test_results3/masks/mask_{batch_nb}_{i}.npy"
            logits_path = f"test_results3/logits/logits_{batch_nb}_{i}.npy"
=======
            img_path = f"/home/Duke/test_results/images/img_{batch_nb}_{i}.png"
            mask_path = f"/home/Duke/test_results/masks/mask_{batch_nb}_{i}.npy"
            logits_path = f"/home/Duke/test_results/logits/logits_{batch_nb}_{i}.npy"
>>>>>>> 0ae3e64f82d3a409ef9b21f1be9cfb7dfc76a57a
            
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(logits_path), exist_ok=True)
            
            # Save image, mask, and logits
            transforms.ToPILImage()(x[i, -3:, :, :]).save(img_path)
            np.save(mask_path, y[i, 0, :, :].cpu().numpy())
            np.save(logits_path, logits[i].cpu().numpy())
            # breakpoint()
            # Append paths to list
            output_paths.append({
                'img_path': img_path,
                'mask_path': mask_path,
                'logits_path': logits_path,
            })
        self.test_loss_outputs.append(loss)
        self.test_path_outputs.extend(output_paths)
        return {
            'output_paths': output_paths,
            'test_loss': loss.cpu().item(),
        }

    def on_test_epoch_end(self):
        test_loss = torch.tensor([x for x in self.test_loss_outputs]).mean().item()
        self.log('test_loss', test_loss)
        
        # Writing to CSV
        with open('/home/Duke/test_results/results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "Logits Path", "Mask Path", "Error"])  # Writing header
            for output in self.test_path_outputs:
                    writer.writerow([output['img_path'], output['logits_path'], output['mask_path']])
        
        print(test_loss)
        self.eval_results = {
            'test_loss': test_loss,
        }

    def train_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'train')
        return DataLoader(ds,
                          batch_size=self.hparams['batch_size'],num_workers=0)

    def val_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'val')
        return DataLoader(ds, 
                          batch_size=self.hparams['batch_size'],num_workers=0)

    def test_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'test')
        ds_subset = Subset(ds, indices=range(20))
        return DataLoader(ds_subset, batch_size=self.hparams['batch_size'],num_workers=0)