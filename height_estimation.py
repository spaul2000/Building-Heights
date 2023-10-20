import logging
import os
import fire
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from util import constants as C
from util import init_exp_folder, Args

from lightning import (get_task,
                       load_task,
                       get_ckpt_callback,
                       get_early_stop_callback,
                       get_tb_logger,
                       get_csv_logger)

def train(save_dir=str(C.SANDBOX_PATH),
          tb_path=str(C.TB_PATH),
          task='height_estimation',
          exp_name="inital_model_run",
          seg_architecture="UNet",
          seg_backbone="resnet18",
          seg_dataset="inital_test",
	      learning_rate=0.0001,
	      batch_size=16,
          loss_fn='MSE',
          optimizer='Adam',  # Options: 'Adam', 'SGD', 'AdamW'
          patience=10,
         ):
    """
    Run the training experiment.
    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        seg_architecture: Architecture to use if task is segmentation
        seg_backbone: Backbone to use if task is segmentation
        seg_dataset: Dataset to use if task is segmentation
        tb_path: Path to global tb folder
    Returns:
        Path to where the best checkpoint is saved
    """
    logging.getLogger().setLevel(logging.INFO)

    args = Args(locals())
    # init_exp_folder(args)
    task = get_task(args)

    trainer = Trainer(logger=[get_tb_logger(save_dir, exp_name), get_csv_logger(save_dir, exp_name)],
                      callbacks=[get_early_stop_callback(patience, monitor="train_loss", mode="min"),
                                 get_ckpt_callback(save_dir, exp_name, monitor="train_loss", mode="min")],
                      log_every_n_steps=1,
                      max_epochs=20
                     )
    trainer.fit(task)

    return trainer.checkpoint_callback.best_model_path

if __name__ == "__main__":
    fire.Fire()