import logging
import os
import fire
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from util import constants as C
from util import init_exp_folder, Args

from lightning import (get_task,
                       load_task,
                       get_ckpt_callback,
                       get_early_stop_callback,
                       get_tb_logger,
                       get_csv_logger)

def train_tune(config, save_dir=str(C.SANDBOX_PATH),
          tb_path=str(C.TB_PATH),
          task='height_estimation',
          exp_name=None,
          seg_architecture="UNet",
          seg_backbone=None,
          learning_rate=None,
          batch_size=None,
          seg_dataset="inital_test",
          loss_fn='MSE',
          optimizer='Adam',  # Options: 'Adam', 'SGD', 'AdamW'
          patience=5,
         ):
    """
    Run the training experiment.
    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        seg_architecture: Architecture to use if task is segmentation
        seg_dataset: Dataset to use if task is segmentation
        tb_path: Path to global tb folder
    Returns:
        Path to where the best checkpoint is saved
    """
    logging.getLogger().setLevel(logging.INFO)

    seg_backbone = config["seg_backbone"] if seg_backbone is None else seg_backbone
    learning_rate = config["learning_rate"] if learning_rate is None else learning_rate
    batch_size = config["batch_size"] if batch_size is None else batch_size
    exp_name = f"{seg_backbone}_lr{learning_rate:.3e}_bs{batch_size}" if exp_name is None else exp_name

    args = Args(locals())
    init_exp_folder(args)
    task = get_task(args)

    trainer = Trainer(logger=[get_tb_logger(save_dir, exp_name), get_csv_logger(save_dir, exp_name)],
                      callbacks=[get_early_stop_callback(patience, monitor="val_loss", mode="min"),
                                 get_ckpt_callback(save_dir, exp_name, monitor="val_loss", mode="min")],
                      log_every_n_steps=1,
                      max_epochs=1000
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

def tune_hyperparameters(num_samples=10, max_num_epochs=100, gpus_per_trial=1):
    config = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([4, 8, 16]),
    "seg_backbone": tune.choice(["resnet50",'resnet34']),
    }
    search_alg = HyperOptSearch(metric="val_loss", mode="min")
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["learning_rate", "batch_size", "seg_backbone"],
        metric_columns=["val_loss", "training_iteration"])
    result = tune.run(
        train_tune,
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_acc"]))

if __name__ == "__main__":
    tune_hyperparameters()
