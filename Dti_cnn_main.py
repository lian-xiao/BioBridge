from multiprocessing import freeze_support

import torch
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse

from config import get_train_args, get_model_args
from py_lighting_model.light_Dti_cnn import LightDti, LightDta
from Data.DataMoudle_base_emb import BEDataModule, BEDataModule_Da
import os
import json
from utils_callback import ProgressBar
class CustomFilenameCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        test_metrics = trainer.callback_metrics  
        test_auroc = test_metrics['test_auroc_avg']
        metrics = {key: float(value) for key, value in test_metrics.items()}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        new_path = os.path.join("/".join(components[:-1]), f"auroc{test_auroc :.4f}")
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()
        os.rename(original_dir_name, new_path)
class CustomFilenameCallbackrmse(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        test_metrics = trainer.callback_metrics  
        test_auroc = test_metrics['test_rmse_avg']
        metrics = {key: float(value) for key, value in test_metrics.items()}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        new_path = os.path.join("/".join(components[:-1]), f"rmse{test_auroc :.4f}")
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()
        os.rename(original_dir_name, new_path)
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    freeze_support()

    torch.cuda.empty_cache()
    train_config = get_train_args()
    model_config = get_model_args()

    config = argparse.ArgumentParser(description='Combined configuration')
    config.__dict__.update(train_config.__dict__)
    config.__dict__.update(model_config.__dict__)

    train_config.dataFolder = os.path.join(train_config.data_root, train_config.dataset_name)
    seed_everything(train_config.seed)
    if train_config.early_stop == True:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=10)
    else:
        early_stop_callback = EarlyStopping(monitor='train_loss', patience=train_config.max_epochs)
    p_emb, d_emb = None, None
    alphabet, mol_tokenizer = None, None
    if train_config.Da:
        model_config.binary = 2
        datamodule = BEDataModule_Da(config, tokenizer=mol_tokenizer, alphabet=alphabet)
        datamodule.prepare_data()
    else:
        datamodule = BEDataModule(config, tokenizer=mol_tokenizer, alphabet=alphabet)
        datamodule.prepare_data()

    if train_config.dataset_name == 'pdb2020':
        model = LightDta(model_config, train_config)
    else:

        model = LightDti(d_emb, p_emb, model_config, train_config)

    checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset_name)
    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        default_hp_metric=False,
    )

    if train_config.dataset_name == 'pdb2020':
        cf = CustomFilenameCallbackrmse()
        checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, monitor='val_rmse_avg', save_top_k=1,
                                          mode='min', filename='{epoch}-{val_rmse_avg:.4f}')
    else:
        cf = CustomFilenameCallback()
        checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, monitor='val_auroc_avg', save_top_k=1,
                                          mode='max', filename='{epoch}-{val_auroc_avg:.4f}')
    if train_config.DDP:
        device = torch.cuda.device_count()
        num_nodes = int(device / 4)
    else:
        device = 1 if is_cuda_available() else None
        num_nodes = 1

    print(model)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback,  early_stop_callback,ProgressBar()],
        strategy="ddp_find_unused_parameters_false" if train_config.DDP else 'auto',
        max_epochs=train_config.max_epochs,
        default_root_dir=checkpoint_root,
        accelerator="gpu",
        devices=1,
        logger=logger,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)

