from multiprocessing import freeze_support
from pytorch_lightning.cli import ReduceLROnPlateau
#from apex import optimizers
from torch.optim import Adam

import esm
from MolFormer.MolFormer import *
from MolFormer.tokenizer import MolTranBertTokenizer
import argparse
from MolFormer.DataMoudle import PropertyPredictionDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch import nn, optim, mean, Tensor
from lightning_fabric.accelerators.cuda import is_cuda_available
from torchmetrics import AUROC,AveragePrecision,Accuracy,Specificity,Recall
from pytorch_lightning import seed_everything
import os
import torch
from torch import Tensor
from typing import Any, Callable, Optional
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from Dti.Dti_model import Dti_Cross_Model

class LightDti(pl.LightningModule):
    def __init__(self,molformer,esm2,model_config,train_config):
        super(LightDti, self).__init__()
        self.model = Dti_Cross_Model(molformer,esm2,model_config)
        self.model.freeze_backbone()
        self.train_config = train_config
        self.loss = nn.BCELoss()

        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_spec = Specificity(task="binary")
        self.val_spec = Specificity(task="binary")
        self.test_spec = Specificity(task="binary")

        self.train_sen = Recall(task="binary")
        self.val_sen = Recall(task="binary")
        self.test_sen = Recall(task="binary")

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.train_config.lr)
        return opt


    def log_metrics_step(self,preds,labels,mode):
        avg = 'step'
        labels = (labels).long()
        if mode == 'train':
            self.train_auc(preds, labels)
            self.log(f'train_auroc_{avg}', self.train_auc)

            self.train_auprc(preds,labels)
            self.log(f'train_auprc_{avg}', self.train_auprc)

            self.train_acc(preds, labels)
            self.log(f'train_acc_{avg}', self.train_acc)

            self.train_spec(preds, labels)
            self.log(f'train_spec_{avg}', self.train_spec)

            self.train_sen(preds, labels)
            self.log(f'train_sen_{avg}', self.train_sen)

        elif mode == 'val':
            self.val_auc(preds, (labels).long())
            self.log(f'val_auroc_{avg}', self.val_auc)

            self.val_auprc(preds,labels)
            self.log(f'val_auprc_{avg}', self.val_auprc)
            self.val_acc(preds, labels)
            self.log(f'val_acc_{avg}', self.val_acc)

            self.val_spec(preds, labels)
            self.log(f'val_spec_{avg}', self.val_spec)

            self.val_sen(preds, labels)
            self.log(f'val_sen_{avg}', self.val_sen)
        else:
            self.test_auc(preds, (labels).long())
            self.log(f'test_auroc_{avg}', self.test_auc)

            self.test_auprc(preds,labels)
            self.log(f'test_auprc_{avg}', self.test_auprc)

            self.test_acc(preds, labels)
            self.log(f'test_acc_{avg}', self.test_acc)

            self.test_spec(preds, labels)
            self.log(f'test_spec_{avg}', self.test_spec)

            self.test_sen(preds, labels)
            self.log(f'test_sen_{avg}', self.test_sen)


    def log_metrics_epoch(self,mode = 'train'):
        avg = 'avg'
        if mode == 'train':
            self.log(f'train_auroc_{avg}', self.train_auc)
            self.log(f'train_auprc_{avg}', self.train_auprc)
            self.log(f'train_acc_{avg}', self.train_acc)
            self.log(f'train_spec_{avg}', self.train_spec)
            self.log(f'train_sen_{avg}', self.train_sen)
        elif mode == 'val':
            self.log(f'val_auroc_{avg}', self.val_auc)
            self.log(f'val_auprc_{avg}', self.val_auprc)
            self.log(f'val_acc_{avg}', self.val_acc)
            self.log(f'val_spec_{avg}', self.val_spec)
            self.log(f'val_sen_{avg}', self.val_sen)
        else:
            self.log(f'test_auroc_{avg}', self.test_auc)
            self.log(f'test_auprc_{avg}', self.test_auprc)
            self.log(f'test_acc_{avg}', self.test_acc)
            self.log(f'test_spec_{avg}', self.test_spec)
            self.log(f'test_sen_{avg}', self.test_sen)


    def _calculate_loss(self,batch,mode='train'):
        mol, mol_mask, protein, labels = batch

        preds = self.model(mol, protein, mol_mask)

        labels = labels.float()
        loss = self.loss(preds, labels)
        self.log_metrics_step(preds,labels,mode)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,mode='train')
        self.log('train_loss',loss,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,mode='val')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,mode='test')
        self.log('test_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='val')
    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')
    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')