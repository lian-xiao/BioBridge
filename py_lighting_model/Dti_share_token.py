# from apex import optimizers
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
from torchmetrics import AUROC, AveragePrecision, Accuracy, Specificity, Recall
import torch
from Dti.Dti_share_token import Dti_ShareT_Model

class LightDti(pl.LightningModule):
    def __init__(self, molformer,esm2,model_config, train_config,gpu_tracker=None):
        super(LightDti, self).__init__()
        self.model = Dti_ShareT_Model(molformer,esm2,model_config,gpu_tracker)
        self.model.freeze_backbone()
        self.sync = train_config.DDP
        # 跨域
        self.train_config = train_config
        self.gpu_tracker = gpu_tracker
        self.Da = train_config.Da
        if self.Da:
            self.loss = nn.CrossEntropyLoss()
            self.train_ways = train_config.train_ways
            self.train_shot = train_config.train_shot
            self.train_queries = train_config.train_queries
            self.test_ways = train_config.test_ways
            self.test_shot = train_config.test_shot
            self.test_queries = train_config.test_queries
            #self.classifier = self.model.classifier
            #self.classifier = MAML(self.classifier, lr=self.train_config.adaptation_lr)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.train_epoch_preds = torch.tensor([]).detach()
        self.train_epoch_labels = torch.tensor([]).detach()
        self.val_epoch_preds = torch.tensor([]).detach()
        self.val_epoch_labels = torch.tensor([]).detach()

        self.train_auc = AUROC(task='multiclass', num_classes=2)
        self.val_auc = AUROC(task='multiclass', num_classes=2)
        self.test_auc = AUROC(task='multiclass', num_classes=2)

        self.train_auprc = AveragePrecision(task='multiclass', num_classes=2)
        self.val_auprc = AveragePrecision(task='multiclass', num_classes=2)
        self.test_auprc = AveragePrecision(task='multiclass', num_classes=2)

        self.train_acc = Accuracy(task='multiclass', num_classes=2)
        self.val_acc = Accuracy(task='multiclass', num_classes=2)
        self.test_acc = Accuracy(task='multiclass', num_classes=2)

        self.train_spec = Specificity(task='multiclass', num_classes=2)
        self.val_spec = Specificity(task='multiclass', num_classes=2)
        self.test_spec = Specificity(task='multiclass', num_classes=2)

        self.train_sen = Recall(task='multiclass', num_classes=2)
        self.val_sen = Recall(task='multiclass', num_classes=2)
        self.test_sen = Recall(task='multiclass', num_classes=2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_config.lr,weight_decay=0.05)
        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=self.train_config.scheduler_step,
        #     gamma=self.train_config.scheduler_decay,
        # )
        return optimizer

    def log_metrics_step(self, preds, labels,mode='train'):
        labels = (labels).long()
        if mode == 'train':
            self.train_epoch_preds = torch.cat((self.train_epoch_preds.to(preds),preds),dim=0).detach()
            self.train_epoch_labels = torch.cat((self.train_epoch_labels.to(labels),labels),dim=0).detach()
        else:
            self.val_epoch_preds = torch.cat((self.val_epoch_preds.to(preds),preds),dim=0).detach()
            self.val_epoch_labels = torch.cat((self.val_epoch_labels.to(labels),labels),dim=0).detach()
    def log_metrics_epoch(self, mode='train'):
        avg = 'avg'
        if mode == 'train':
            self.train_auc(self.train_epoch_preds,self.train_epoch_labels)
            self.train_auprc(self.train_epoch_preds,self.train_epoch_labels)
            self.train_acc(self.train_epoch_preds,self.train_epoch_labels)
            self.train_sen(self.train_epoch_preds,self.train_epoch_labels)
            self.train_spec(self.train_epoch_preds,self.train_epoch_labels)
            self.log(f'train_auroc_{avg}', self.train_auc,sync_dist=self.sync)
            self.log(f'train_auprc_{avg}', self.train_auprc,sync_dist=self.sync)
            self.log(f'train_acc_{avg}', self.train_acc,sync_dist=self.sync)
            self.log(f'train_spec_{avg}', self.train_spec,sync_dist=self.sync)
            self.log(f'train_sen_{avg}', self.train_sen,sync_dist=self.sync)
            self.train_epoch_preds = torch.tensor([]).detach()
            self.train_epoch_labels = torch.tensor([]).detach()
        elif mode == 'eval':
            self.val_auc(self.val_epoch_preds,self.val_epoch_labels)
            self.val_auprc(self.val_epoch_preds,self.val_epoch_labels)
            self.val_acc(self.val_epoch_preds,self.val_epoch_labels)
            self.val_sen(self.val_epoch_preds,self.val_epoch_labels)
            self.val_spec(self.val_epoch_preds,self.val_epoch_labels)
            self.log(f'val_auroc_{avg}', self.val_auc,sync_dist=self.sync)
            self.log(f'val_auprc_{avg}', self.val_auprc,sync_dist=self.sync)
            self.log(f'val_acc_{avg}', self.val_acc,sync_dist=self.sync)
            self.log(f'val_spec_{avg}', self.val_spec,sync_dist=self.sync)
            self.log(f'val_sen_{avg}', self.val_sen,sync_dist=self.sync)
            self.val_epoch_preds = torch.tensor([]).detach()
            self.val_epoch_labels = torch.tensor([]).detach()
        else:
            self.test_auc(self.epoch_preds,self.epoch_labels)
            self.test_auprc(self.epoch_preds,self.epoch_labels)
            self.test_acc(self.epoch_preds,self.epoch_labels)
            self.test_sen(self.epoch_preds,self.epoch_labels)
            self.test_spec(self.epoch_preds,self.epoch_labels)

            self.log(f'test_auroc_{avg}', self.test_auc,sync_dist=self.sync)
            self.log(f'test_auprc_{avg}', self.test_auprc,sync_dist=self.sync)
            self.log(f'test_acc_{avg}', self.test_acc,sync_dist=self.sync)
            self.log(f'test_spec_{avg}', self.test_spec,sync_dist=self.sync)
            self.log(f'test_sen_{avg}', self.test_sen,sync_dist=self.sync)
            self.val_epoch_preds = torch.tensor([]).detach()
            self.val_epoch_labels = torch.tensor([]).detach()

    #@torch.enable_grad()
    def _calculate_loss_maml(self, batch, mode='train'):
        drug, _, protein, labels, sort = batch
        if mode == 'train':
            ways = self.train_ways
            shot = self.train_shot
            query_num = self.train_queries
        else:
            ways = self.test_ways
            shot = self.test_shot
            query_num = self.test_queries
        # Sort data samples by labels
        # TODO: Can this be replaced by ConsecutiveLabels ?
        support_indices = np.zeros(labels.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + query_num)
        for offset in range(shot):
            support_indices[selection + offset] = True
        # Compute support and query embeddings
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_labels = labels[support_indices]
        query_labels = labels[query_indices]
        if mode == 'train':
            _, _, fea,preds = self.model(drug, protein, 'train')
            support_fea = fea[support_indices]
            query_fea = fea[query_indices]
        loss = self.loss(preds, query_labels.long())
        self.log_metrics_step(preds, query_labels, mode)
        return loss


    def _calculate_loss(self,batch,mode='train'):
        if self.gpu_tracker:
            self.gpu_tracker.track()
        mol, mol_mask, protein, labels = batch
        labels = labels.squeeze(-1)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        preds = self.model(mol, protein, mol_mask,mode)
        loss = self.loss(preds, labels.long())
        self.log_metrics_step(preds,labels.long(),mode)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self._calculate_loss(batch, mode='train')

        if self.gpu_tracker:
            self.gpu_tracker.track()
        self.log('t_l', loss.item(), prog_bar=True,sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='eval')
        self.log('v_l', loss.item(), prog_bar=True,sync_dist=self.sync)
        torch.cuda.empty_cache()
        return loss.item()

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='test')
        self.log('test_loss', loss.item())
        torch.cuda.empty_cache()
        return loss.item()

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='eval')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')
