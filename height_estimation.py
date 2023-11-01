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
          exp_name="deletesoon2",
          seg_architecture="UNet",
          seg_backbone="resnet18",
          seg_dataset="inital_test",
	      learning_rate=0.0001,
	      batch_size=4,
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
    init_exp_folder(args)  
    task = get_task(args)
    # task.to('cuda')

    trainer = Trainer(logger=[get_tb_logger(save_dir, exp_name), get_csv_logger(save_dir, exp_name)],
                      callbacks=[get_early_stop_callback(patience, monitor="val_loss", mode="min"),
                                 get_ckpt_callback(save_dir, exp_name, monitor="val_loss", mode="min")],
                      log_every_n_steps=1,
                      max_epochs=1000,accelerator="gpu", devices=1
                     )
    trainer.fit(task)

    return trainer.checkpoint_callback.best_model_path

def test(ckpt_path='/home/Duke/group/main/sandbox/real_initial_go/ckpts/model_checkpoint.ckpt',
         eval_split='test',
         **kwargs):
    """
    Run the testing experiment.
    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        eval_split: use 'val' or 'test' split
        eval_target_resolution: desired resolution (in cm) to evaluate
                                at
        n_batches_inspect: number of batches to save predictions for
    Returns:
        Metrics for the input ckpt evaluated on the specified split
    """
    logging.getLogger().setLevel(logging.INFO)
    task = load_task(
        ckpt_path,
        eval_split=eval_split,
        task="height_estimation",
        **kwargs
    )
    trainer = Trainer()
    trainer.test(task)
    return task.eval_results
if __name__ == "__main__":
    fire.Fire()