from multiprocessing import freeze_support

import torch
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import esm
import argparse

from py_lighting_model.light_Dti_cnn import LightDti, LightDta
from Data.DataMoudle_base_emb import BEDataModule, BEDataModule_Da
import os

import json

from utils_callback import ProgressBar


class CustomFilenameCallback(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        # 获取测试集指标信息
        test_metrics = trainer.callback_metrics  # 获取测试集指标
        test_auroc = test_metrics['test_auroc_avg']
        # 修改文件名，将测试集损失信息添加到文件名中
        metrics = {key: float(value) for key, value in test_metrics.items()}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        new_path = os.path.join("/".join(components[:-1]), f"auroc{test_auroc :.4f}")
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()
        os.rename(original_dir_name, new_path)  # 重命名文件夹或文件路径

class CustomFilenameCallbackrmse(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        # 获取测试集指标信息
        test_metrics = trainer.callback_metrics  # 获取测试集指标
        test_auroc = test_metrics['test_rmse_avg']
        # 修改文件名，将测试集损失信息添加到文件名中
        metrics = {key: float(value) for key, value in test_metrics.items()}
        original_dir_name = trainer.logger.log_dir
        components = original_dir_name.split("/")
        new_path = os.path.join("/".join(components[:-1]), f"rmse{test_auroc :.4f}")
        f = open(os.path.join(trainer.logger.log_dir, 'metrics.json'), 'w')
        json.dump(metrics, f)
        f.close()
        os.rename(original_dir_name, new_path)  # 重命名文件夹或文件路径


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    freeze_support()
    # datasets = ['bindingdb/meta_unseen_protein/','bindingdb/meta_unseen_drug/']
    # for name in datasets:
    torch.cuda.empty_cache()
    train_config = {'data_root': 'Data', 'dataset_name': 'pdb2020', 'measure_name': ['Y'], 'Da':False,
                    'gamma': 10, 'MDa': True,
                    'optimizer_type': 'Adam', 'optimizer_momentum': 0.9, 'weight_decay': 2.5e-5, 'dropout': 0.1,
                    'early_stop': False,'main_metric': 'rmse',
                    'finetune_load_path': None, 'Da_warm_epochs': 0, 'batch_size': 64, 'max_epochs': 100,
                    'num_workers': 0, 'lr': 5e-5,
                    'checkpoints_folder': 'checkpoints_allstem/', 'seed': 2023, 'train_dataset_length': None,
                    'test_dataset_length': None, 'eval_dataset_length': None, 'DDP': False}

    model_config = {'drugban':False,'muti_out': True, 'p_ems2_emb': False, 'd_molformer_emb': False, 'p_emb': 128, 'd_emb': 128,
                    'd_stem_channel': 128,'stem_kernel':1,'p_stem':True,'d_stem':True,'gate':True,
                    'p_stem_channel': 128, 'd_channels': [128, 128, 128], 'p_channels': [128, 128, 128],
                    'd_out_channel': 128, 'p_out_channel': 128, 'out_hidden_size': 256, 'layers_num': 3, 'binary': 1}
    train_config = argparse.Namespace(**train_config)
    model_config = argparse.Namespace(**model_config)
    config = argparse.Namespace(**vars(train_config), **vars(model_config))
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
        #model = LightDti_Da(d_emb, p_emb, model_config, train_config)
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
    #cf,
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
    #trainer.fit(model, datamodule=datamodule)
    #trainer.test(ckpt_path='best', datamodule=datamodule)
    trainer.test(model,ckpt_path="/home/pan/LXQ/DTI_work/checkpoints_allstem/pdb2020/lightning_logs/rmse1.4328/epoch=74-val_rmse_avg=1.2418.ckpt", datamodule=datamodule)

# compound = pcp.get_compounds('9,10-Dimethyl-1,2-benzanthracene', 'name')
