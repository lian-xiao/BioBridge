# from apex import optimizers
import dgl
import numpy as np
import pytorch_lightning as pl
from learn2learn.algorithms import MAML
from learn2learn.nn import PrototypicalClassifier
from torch import nn, optim
from torchmetrics import AUROC, AveragePrecision, Accuracy, Specificity, Recall
import torch
from Drug_ban.models import DrugBAN, cross_entropy_logits, binary_cross_entropy, RandomLayer, entropy_logits
from My_nets.AmsLoss import AdMSoftmaxLoss


class LightDti(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super(LightDti, self).__init__()
        self.model = DrugBAN(**model_config)
        # 跨域
        self.train_config = train_config
        self.Da = train_config.Da
        if self.Da:
            #self.loss = AdMSoftmaxLoss(2)
            self.loss = nn.CrossEntropyLoss()
            self.train_ways = train_config.train_ways
            self.train_shot = train_config.train_shot
            self.train_queries = train_config.train_queries
            self.test_ways = train_config.test_ways
            self.test_shot = train_config.test_shot
            self.test_queries = train_config.test_queries

            self.classifier = self.model.mlp_classifier
            self.classifier = MAML(self.classifier, lr=self.train_config.adaptation_lr)
            # self.classifier = PrototypicalClassifier(distance=train_config.distance_metric)
        else:
            self.loss = nn.CrossEntropyLoss()

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
        optimizer = optim.Adam(self.parameters(), lr=self.train_config.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.train_config.scheduler_step,
            gamma=self.train_config.scheduler_decay,
        )
        return [optimizer], [lr_scheduler]

    def log_metrics_step(self, preds, labels, mode):
        avg = 'step'
        labels = (labels).long()
        if mode == 'train':
            self.train_auc(preds, labels)
            self.log(f'train_auroc_{avg}', self.train_auc)

            self.train_auprc(preds, labels)
            self.log(f'train_auprc_{avg}', self.train_auprc)

            self.train_acc(preds, labels)
            self.log(f'train_acc_{avg}', self.train_acc)

            self.train_spec(preds, labels)
            self.log(f'train_spec_{avg}', self.train_spec)

            self.train_sen(preds, labels)
            self.log(f'train_sen_{avg}', self.train_sen)

        elif mode == 'eval':
            self.val_auc(preds, (labels).long())
            self.log(f'val_auroc_{avg}', self.val_auc)

            self.val_auprc(preds, labels)
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

            self.test_auprc(preds, labels)
            self.log(f'test_auprc_{avg}', self.test_auprc)

            self.test_acc(preds, labels)
            self.log(f'test_acc_{avg}', self.test_acc)

            self.test_spec(preds, labels)
            self.log(f'test_spec_{avg}', self.test_spec)

            self.test_sen(preds, labels)
            self.log(f'test_sen_{avg}', self.test_sen)

    def log_metrics_epoch(self, mode='train'):
        avg = 'avg'
        if mode == 'train':
            self.log(f'train_auroc_{avg}', self.train_auc)
            self.log(f'train_auprc_{avg}', self.train_auprc)
            self.log(f'train_acc_{avg}', self.train_acc)
            self.log(f'train_spec_{avg}', self.train_spec)
            self.log(f'train_sen_{avg}', self.train_sen)
        elif mode == 'eval':
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

    @torch.enable_grad()
    def _calculate_loss(self, batch, mode='train'):
        self.model.train()
        #learner = self.classifier.clone()
        # learner = self.model.clone()
        # learner.train()
        drug, c, protein, labels, sort = batch

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
            _, _, fea,_ = self.model(drug, protein, 'train')
            support_fea = fea[support_indices]
            query_fea = fea[query_indices]
        else:
            with torch.no_grad():
                _, _, fea, _ = self.model(drug, protein, 'train')
                support_fea = fea[support_indices]
                query_fea = fea[query_indices]
        # Adapt the classifier
        for step in range(self.train_config.adaptation_steps):
            preds = learner(support_fea)
            train_error = self.loss(preds, support_labels.long())
            learner.adapt(train_error)
            #,allow_nograd=True,allow_unused=True
        torch.cuda.empty_cache()
        preds = learner(query_fea)

        loss = self.loss(preds, query_labels.long())
        self.log_metrics_step(preds, query_labels, mode)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        self.log('t_l', loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='eval')
        self.log('v_l', loss.item(), prog_bar=True)
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
