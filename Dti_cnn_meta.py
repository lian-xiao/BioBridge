import torch
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse
from Data.DataMoudle_Meta import get_datamoudle
import os

from config_meta import get_train_args, get_model_args
from py_lighting_model.light_Dti_cnn_meta import LightningPrototypicalNetworks
from utils_callback import CustomFilenameCallback, My_ModelCheckpoint, ProgressBar




if __name__ == '__main__':
    torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    train_config = get_train_args()
    model_config = get_model_args()

    config = argparse.ArgumentParser(description='Combined configuration')
    config.__dict__.update(train_config.__dict__)
    config.__dict__.update(model_config.__dict__)

    train_config.dataFolder = os.path.join(train_config.data_root, train_config.dataset_name)
    if 'meta_unseen_protein' in train_config.dataset_name:
        train_config.domain_index = 3
    elif 'meta_unseen_drug' in train_config.dataset_name:
        train_config.domain_index = 4

    seed_everything(train_config.seed)
    if train_config.load == True:
        if 'meta_unseen_protein' in train_config.dataset_name:
            if 'bindingdb' in train_config.dataset_name:
                train_config.load = 'pretrain_checkpoint/dti_multicnn/bindingdb/model.pth'
            elif 'biosnap' in train_config.dataset_name:
                train_config.load = 'pretrain_checkpoint/dti_multicnn/biosnap/model.pth'
    else:
        train_config.load = None

    p_emb, d_emb = None, None
    alphabet, mol_tokenizer = None, None

    datamodule = get_datamoudle(train_config)
    model = LightningPrototypicalNetworks(model_config=model_config,train_config=train_config)
    checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset_name,'shot'+str(train_config.train_shots))
    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        default_hp_metric=False,
    )

    checkpoint_callback = My_ModelCheckpoint(dirpath=logger.log_dir,
                                                   monitor=f'val_{train_config.main_metric}', save_top_k=1,
                                                   mode='max',
                                                   filename=f'{{epoch}}-{{val_{train_config.main_metric}:.4f}}')
    if train_config.DDP:
        device = torch.cuda.device_count()
        num_nodes = int(device / 4)
    else:
        device = 1 if is_cuda_available() else None
        num_nodes = 1

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback,CustomFilenameCallback(),ProgressBar()],
        strategy="ddp_find_unused_parameters_false" if train_config.DDP else 'auto',
        max_epochs=train_config.max_epochs,
        accumulate_grad_batches=train_config.batch_size,
        default_root_dir=checkpoint_root,
        accelerator="gpu",
        devices=1,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)
