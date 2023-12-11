"""Define Logger class for logging information to stdout and disk."""
import json
import os
from os.path import join
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name, "ckpts")


def get_ckpt_callback(save_path, exp_name, monitor="val_loss", mode="min"):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    return ModelCheckpoint(dirpath=ckpt_dir,
                           save_top_k=1,
                           verbose=True,
                           monitor=monitor,
                           mode=mode)


def get_early_stop_callback(patience=10, monitor="val_loss", mode="min"):
    return EarlyStopping(monitor=monitor,
                         patience=patience,
                         verbose=True,
                         mode=mode)


def get_tb_logger(save_path, exp_name):
    exp_dir = os.path.join(save_path, exp_name)
    return TensorBoardLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version="0")

def get_csv_logger(save_path, exp_name):
    exp_dir = os.path.join(save_path, exp_name)
    return CSVLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version="0")