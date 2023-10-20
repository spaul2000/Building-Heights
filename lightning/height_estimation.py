import logging
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
from data import EstimationDataset
from eval import get_loss_fn
from models import get_est_model
from util import constants as C
from util import util


class HeightEstimationTask(pl.LightningModule):
    """Interface for trainer to work with semantic segmentation"""

    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        ds_name = self.hparams['seg_dataset']
        self.ds_meta, n_bands = C.EST_DS[ds_name]
        print(params)
        self.model = get_est_model(params, n_bands=n_bands)
        self.supervised_loss = get_loss_fn(params)


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
        x = batch['im']
        y = batch['mask']

        y = y.unsqueeze(1)

        logits = self.forward(x)
        
        loss = self.supervised_loss(logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: dict, batch_nb: int):
        x = batch['im']
        y = batch['mask']
        y = y.unsqueeze(1)

        logits = self.forward(x)
      
        loss = self.supervised_loss(logits, y)
        


        return {
            'val_loss': loss,
        
        }

    def on_validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss']
                                 for x in outputs]).mean()
        
        self.log('val_loss', val_loss)
     
        self.eval_results = {
            'val_loss': val_loss,
        }

    def test_step(self, batch, batch_nb):
        x = batch['im']
        y = batch['mask']
        y = torch.unsqueeze(y, dim=0)

        logits = self.forward(x)
        logits = logits.reshape(y.shape)
        
        loss = self.supervised_loss(logits, y)
        

        return {
            'test_loss': loss,
          
        }

    def on_test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss']
                                 for x in outputs]).mean()
       
        self.log('test_loss', test_loss)
       

        self.eval_results = {
            'test_loss': test_loss,
        }

    def train_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'train')
        return DataLoader(ds,
                          batch_size=self.hparams['batch_size'])

    def val_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'val')
        return DataLoader(ds, 
                          batch_size=self.hparams['batch_size'])

    def test_dataloader(self) -> DataLoader:
        ds = EstimationDataset(self.ds_meta, 'test')
        return DataLoader(ds, batch_size=self.hparams['batch_size'])