from multiprocessing import freeze_support

import dgl
import numpy as np
import torch
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse


from Data.DataMoudle_Meta import get_datamoudle
import os

import json

from Dti.Dti_meta_cnn import Dti_mamlpp_mulcnn
from Meta_utils.MAMLpp import MAMLpp, test, test_anil
from py_lighting_model.light_Dti_cnn_meta import LightningPrototypicalNetworks, LightningMamlNetworks, \
    LightningAnilNetworks, LightningMetaoptNetworks
from utils_callback import CustomFilenameCallback, My_ModelCheckpoint, ProgressBar




if __name__ == '__main__':

    #for seed in [2024,2025,3407]:
    torch.cuda.empty_cache()

    train_config = {'data_root': 'Data', 'dataset_name': 'bindingdb/meta_unseen_protein/', 'measure_name': ['Y'], 'Da':False,
                    'optimizer_type': 'Adam', 'optimizer_momentum': 0.9, 'weight_decay': 2.5e-5, 'dropout': 0.1,
                    'early_stop': False,'loss':'FL','data_parallel':False,'load':True,'label_index':2,'domain_index':3,
                    'finetune_load_path': None, 'batch_size': 32, 'max_epochs': 50,'num_tasks':20000,'main_metric':'auroc',
                    'method':'prototypes',
                    'mamlpp':False,'adaptation_lr':1e-2,'adaptation_steps':5,'Da_epoch':25,'msl_epochs':13,
                    'adaptive': True,'normalize':True,
                    'num_workers': 0, 'lr': 1e-4,'scheduler_step':5,'scheduler_decay':1.0,'distance_metric':'cosine',
                    'train_ways':2,'train_queries':1,'train_shots':5,'test_ways':2,'test_queries':1,'test_shots':5,
                    'checkpoints_folder': 'checkpoints/', 'seed': 2025, 'train_dataset_length': None,
                    'test_dataset_length': None, 'eval_dataset_length': None, 'DDP': False}

    model_config = {'drugban':False,'muti_out': True, 'p_ems2_emb': False, 'd_molformer_emb': False, 'p_emb': 128, 'd_emb': 128,
                    'd_stem_channel': 128,'stem_kernel':1,'p_stem':True,'d_stem':True,'gate':True,'norm':True,
                    'p_stem_channel': 128, 'd_channels': [128, 128, 128], 'p_channels': [128, 128, 128],
                    'd_out_channel': 128, 'p_out_channel': 128, 'out_hidden_size': 256, 'layers_num': 3, 'binary': 2}
    train_config = argparse.Namespace(**train_config)
    model_config = argparse.Namespace(**model_config)
    config = argparse.Namespace(**vars(train_config), **vars(model_config))
    train_config.dataFolder = os.path.join(train_config.data_root, train_config.dataset_name)
    if 'meta_unseen_protein' in train_config.dataset_name:
        train_config.domain_index = 3
    elif 'meta_unseen_drug' in train_config.dataset_name:
        train_config.domain_index = 4
    #train_config.seed = seed
    seed_everything(train_config.seed)
    if train_config.load == True:
        if not getattr(model_config,'drugban',False):
            if model_config.muti_out:
                if 'meta_unseen_protein' in train_config.dataset_name:
                    if 'bindingdb' in train_config.dataset_name:
                        #train_config.load = 'pretrain_checkpoint/dti_multicnn/bindingdb/meta_unseen_protein/epoch_42_0_6357.pth'
                        train_config.load = 'pretrain_checkpoint/dti_multicnn/bindingdb/version2/model_epoch_100.pth'
                    elif 'biosnap' in train_config.dataset_name:
                        #train_config.load = 'pretrain_checkpoint/dti_multicnn/biosnap/meta_unseen_protein/epoch_42_0_6358.pth'
                        train_config.load = 'pretrain_checkpoint/dti_multicnn/biosnap/version3/best_model_epoch_66.pth'
            else:
                if 'bindingdb' in train_config.dataset_name:
                    train_config.load = 'pretrain_checkpoint/dti_multi/bindingdb/epoch_45_0_587.pth'
                elif 'biosnap' in train_config.dataset_name:
                    train_config.load = 'pretrain_checkpoint/dti_multi/biosnap/epoch_33_0_6303.pth'
        else:
            if 'meta_unseen_protein' in train_config.dataset_name:
                if 'bindingdb' in train_config.dataset_name:
                    train_config.load = 'pretrain_checkpoint/dti_multicnn/bindingdb/version2/model_epoch_100.pth'
                elif 'biosnap' in train_config.dataset_name:
                    train_config.load = 'pretrain_checkpoint/dti_sacnn/biosnap/version1/model_epoch_100.pth'
    else:
        train_config.load = None

    #'pretrain_checkpoint/dti_multicnn/bindingdb/epoch_42_0_6357.pth'
    p_emb, d_emb = None, None
    alphabet, mol_tokenizer = None, None

    datamodule = get_datamoudle(train_config)

    if train_config.method == 'maml':
        train_config.lr = 1e-3
        model = LightningMamlNetworks(model_config=model_config, train_config=train_config)
    elif train_config.method == 'prototypes':
        model = LightningPrototypicalNetworks(model_config=model_config,train_config=train_config)
    elif train_config.method == 'metaopt':
        model = LightningMetaoptNetworks(model_config=model_config,train_config=train_config)
    else:
        train_config.lr = 1e-3
        model = LightningAnilNetworks(model_config=model_config, train_config=train_config)
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
    # print(model)
    # checkpoint_callback,
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
    if train_config.method == 'maml':
        print(trainer.checkpoint_callback.best_model_path)
        main_metrice = test(trainer,model,trainer.checkpoint_callback.best_model_path,datamodule,model_config,train_config)
    elif train_config.method == 'anil':
        print(trainer.checkpoint_callback.best_model_path)
        main_metrice = test_anil(trainer,model,trainer.checkpoint_callback.best_model_path,datamodule,model_config,train_config)
    else:
        trainer.test(ckpt_path='best', datamodule=datamodule)
        test_metrics = trainer.callback_metrics  # 获取测试集指标
        main_metrice = test_metrics[f'test_{trainer.model.main_metrice}']
    #with torch.enable_grad():
    #trainer.validate(model,ckpt_path='checkpoints/bindingdb/meta_unseen_protein/shot5/lightning_logs/version_3/epoch=42-val_auroc=0.6805.ckpt',datamodule=datamodule)

    # else:
    # trainer.test(model,ckpt_path='checkpoints/bindingdb/meta_unseen_protein/shot5/lightning_logs/version_3/epoch=42-val_auroc=0.6805.ckpt', datamodule=datamodule)

