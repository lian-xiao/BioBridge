import math

import numpy as np
#from sklearn.metrics import roc_auc_score

from Dti.Dti_cnn import Dti_cnn, Dti_cnn_mutiout, Dti_DrugBAN
from torch.optim import Adam
import dgl
import pytorch_lightning as pl
from torch import nn
from torchmetrics import AUROC, AveragePrecision, Accuracy, Specificity, Recall,MeanSquaredError,MeanAbsoluteError,PearsonCorrCoef,SpearmanCorrCoef,R2Score
import torch
from collections import OrderedDict
import torch.nn.functional as F
from My_nets.GRL import ReverseLayerF
from lifelines.utils import concordance_index
# Prepare for rm2
def get_cindex(Y, P):
    return concordance_index(Y, P)

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


# Prepare for rm2
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)

    return 1 - (upp / down)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)

    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))

class LightDti(pl.LightningModule):
    def __init__(self, d_embedding, p_embedding, model_config, train_config):
        super(LightDti, self).__init__()
        drugban = getattr(model_config, 'drugban', False)
        if not drugban:
            if model_config.muti_out:
                self.model = Dti_cnn_mutiout(None, None, model_config)
            else:
                self.model = Dti_cnn(None, None, model_config)
        else:
            self.model = Dti_DrugBAN()
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.train_config = train_config
        self.binary = model_config.binary
        self.loss = nn.BCEWithLogitsLoss()
        #if model_config.binary == 1:
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        self.train_auprc = AveragePrecision(task='binary')
        self.val_auprc = AveragePrecision(task='binary')
        self.test_auprc = AveragePrecision(task='binary')

        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        self.train_spec = Specificity(task='binary')
        self.val_spec = Specificity(task='binary')
        self.test_spec = Specificity(task='binary')

        self.train_sen = Recall(task='binary')
        self.val_sen = Recall(task='binary')
        self.test_sen = Recall(task='binary')
        self.test_preds = []
        self.test_labels = []
        self.test_emb = torch.tensor([]).to('cpu')
    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.train_config.lr)
        return opt

    def log_metrics_step(self, preds, labels, mode):
        avg = 'step'
        if mode == 'train':
            self.log(f'train_auroc_{avg}', self.train_auc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_auprc_{avg}', self.train_auprc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_acc_{avg}', self.train_acc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_spec_{avg}', self.train_spec(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_sen_{avg}', self.train_sen(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
        elif mode == 'eval':
            self.log(f'val_auroc_{avg}', self.val_auc(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
            self.log(f'val_auprc_{avg}', self.val_auprc(preds, labels), on_step=True, on_epoch=True,
                     sync_dist=self.sync)
            self.log(f'val_acc_{avg}', self.val_acc(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
            self.log(f'val_spec_{avg}', self.val_spec(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
            self.log(f'val_sen_{avg}', self.val_sen(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
        else:
            # self.test_preds = torch.cat((self.test_preds.to(preds), preds), dim=0).detach()
            # self.test_labels = torch.cat((self.test_labels.to(preds), (labels).long()), dim=0).detach()
            self.test_auc(preds, labels)
            self.test_auprc(preds, labels)
            self.test_acc(preds, labels)
            self.test_spec(preds, labels)
            self.test_sen(preds, labels)


    def log_metrics_epoch(self, mode='train'):
        avg = 'avg'
        if mode == 'train':
            self.log(f'train_auroc_{avg}', self.train_auc, sync_dist=self.sync)
            self.log(f'train_auprc_{avg}', self.train_auprc, sync_dist=self.sync)
            self.log(f'train_acc_{avg}', self.train_acc, sync_dist=self.sync)
            self.log(f'train_spec_{avg}', self.train_spec, sync_dist=self.sync)
            self.log(f'train_sen_{avg}', self.train_sen, sync_dist=self.sync)
        elif mode == 'eval':
            self.log(f'val_auroc_{avg}', self.val_auc, sync_dist=self.sync)
            self.log(f'val_auprc_{avg}', self.val_auprc, sync_dist=self.sync)
            self.log(f'val_acc_{avg}', self.val_acc, sync_dist=self.sync)
            self.log(f'val_spec_{avg}', self.val_spec, sync_dist=self.sync)
            self.log(f'val_sen_{avg}', self.val_sen, sync_dist=self.sync)
        else:
            self.log(f'test_auroc_{avg}', self.test_auc, sync_dist=self.sync)
            self.log(f'test_auprc_{avg}', self.test_auprc, sync_dist=self.sync)
            self.log(f'test_acc_{avg}', self.test_acc, sync_dist=self.sync)
            self.log(f'test_spec_{avg}', self.test_spec, sync_dist=self.sync)
            self.log(f'test_sen_{avg}', self.test_sen, sync_dist=self.sync)
            # print(self.acc(self.test_preds,self.test_labels))
            # print(self.sen(self.test_preds,self.test_labels))
            # print(self.spec(self.test_preds,self.test_labels))

    def cross_entropy_logits(self,linear_output, label, weights=None):
        class_output = F.log_softmax(linear_output, dim=1)
        n = F.softmax(linear_output, dim=1)[:, 1]
        max_class = class_output.max(1)
        y_hat = max_class[1]  # get the index of the max log-probability
        if weights is None:
            loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
        else:
            losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
            loss = torch.sum(weights * losses) / torch.sum(weights)
        return n, loss

    def _calculate_loss(self, batch, mode='train'):
        drug, protein, labels = batch
        preds, fs = self.model(drug, protein)
        if self.binary == 1:
            loss = self.loss(preds, labels.float())
            self.log_metrics_step(preds, labels.long(), mode)
        else:
            labels = labels.squeeze(-1)
            preds,loss = cross_entropy_logits(preds, labels.long())
            self.log_metrics_step(preds, labels.long(), mode)
            if mode == 'test':
                self.test_preds = self.test_preds + preds.to("cpu").tolist()
                self.test_labels = self.test_labels + labels.to("cpu").tolist()
                self.test_emb = torch.concat([self.test_emb,fs.to('cpu')],dim=0)
        return loss

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        drug, protein, labels = batch
        preds, _ = self.model(drug, protein)
        return preds.sigmoid()

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='eval')
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='test')
        self.log('test_loss', loss, sync_dist=self.sync)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='eval')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')



class LightDta(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super(LightDta, self).__init__()
        model_config.binary = 1
        drugban = getattr(model_config, 'drugban', False)
        if not drugban:
            if model_config.muti_out:
                self.model = Dti_cnn_mutiout(None, None, model_config)
            else:
                self.model = Dti_cnn(None, None, model_config)
        else:
            self.model = Dti_DrugBAN()
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.train_config = train_config
        self.loss = nn.MSELoss()
        #if model_config.binary == 1:
        self.train_auc = MeanSquaredError(squared=False)
        self.val_auc = MeanSquaredError(squared=False)
        self.test_auc = MeanSquaredError(squared=False)

        self.train_auprc = MeanAbsoluteError()
        self.val_auprc = MeanAbsoluteError()
        self.test_auprc = MeanAbsoluteError()

        self.train_acc = PearsonCorrCoef()
        self.val_acc = PearsonCorrCoef()
        self.test_acc = PearsonCorrCoef()

        self.train_spec = SpearmanCorrCoef()
        self.val_spec = SpearmanCorrCoef()
        self.test_spec =SpearmanCorrCoef()

        self.test_preds = torch.tensor([])
        self.test_labels = torch.tensor([])
        self.test_emb = torch.tensor([]).to('cpu')

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.train_config.lr)
        return opt

    def log_metrics_step(self, preds, labels, mode):
        avg = 'step'
        if mode == 'train':
            self.log(f'train_rmse_{avg}', self.train_auc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_mae_{avg}', self.train_auprc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_pearson_{avg}', self.train_acc(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
            self.log(f'train_spearman_{avg}', self.train_spec(preds, labels), on_step=True, on_epoch=False,
                     sync_dist=self.sync)
        elif mode == 'eval':
            self.log(f'val_rmse_{avg}', self.val_auc(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
            self.log(f'val_mae_{avg}', self.val_auprc(preds, labels), on_step=True, on_epoch=True,
                     sync_dist=self.sync)
            self.log(f'val_pearson_{avg}', self.val_acc(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
            self.log(f'val_spearman_{avg}', self.val_spec(preds, labels), on_step=True, on_epoch=True, sync_dist=self.sync)
        else:
            self.test_preds = torch.cat((self.test_preds.to(preds), preds), dim=0).detach()
            self.test_labels = torch.cat((self.test_labels.to(preds), labels), dim=0).detach()
            self.test_auc(preds, labels)
            self.test_auprc(preds, labels)
            self.test_acc(preds, labels)
            self.test_spec(preds, labels)


    def log_metrics_epoch(self, mode='train'):
        avg = 'avg'
        if mode == 'train':
            self.log(f'train_rmse_{avg}', self.train_auc, sync_dist=self.sync,prog_bar=True)
            self.log(f'train_mae_{avg}', self.train_auprc, sync_dist=self.sync)
            self.log(f'train_pearson_{avg}', self.train_acc, sync_dist=self.sync)
            self.log(f'train_spearman_{avg}', self.train_spec, sync_dist=self.sync)
        elif mode == 'eval':
            self.log(f'val_rmse_{avg}', self.val_auc, sync_dist=self.sync,prog_bar=True)
            self.log(f'val_mae_{avg}', self.val_auprc, sync_dist=self.sync)
            self.log(f'val_pearson_{avg}', self.val_acc, sync_dist=self.sync)
            self.log(f'val_spearman_{avg}', self.val_spec, sync_dist=self.sync)
        else:
            self.log(f'test_rmse_{avg}', self.test_auc, sync_dist=self.sync)
            self.log(f'test_mae_{avg}', self.test_auprc, sync_dist=self.sync)
            self.log(f'test_pearson_{avg}', self.test_acc, sync_dist=self.sync)
            self.log(f'test_spearman_{avg}', self.test_spec, sync_dist=self.sync)
            rm2 = get_rm2(self.test_labels.to('cpu').numpy(),self.test_preds.to('cpu').numpy())
            cindex = get_cindex(self.test_labels.to('cpu').numpy(),self.test_preds.to('cpu').numpy())
            self.log(f'test_rm2_{avg}', rm2[0], sync_dist=self.sync)
            self.log(f'test_cindex_{avg}', cindex, sync_dist=self.sync)

    def _calculate_loss(self, batch, mode='train'):
        drug, protein, labels = batch
        preds, fs = self.model(drug, protein)
        loss = self.loss(preds, labels.float())
        self.log_metrics_step(preds, labels, mode)
        return loss

    def predict_step(self, batch, batch_idx):
        drug, protein, labels = batch
        preds, _ = self.model(drug, protein)
        result = {
            "predictions": preds,
        }
        return result

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='eval')
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='test')
        self.log('test_loss', loss, sync_dist=self.sync)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='eval')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')



class Discriminator(nn.Module):
    def __init__(self, input_size=128, n_class=2, bigger_discrim=True):

        super(Discriminator, self).__init__()
        output_size = 256 if bigger_discrim else 128

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size) if bigger_discrim else nn.Linear(output_size, n_class)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(output_size, n_class)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        if self.bigger_discrim:
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):

        return_list = [torch.mm(input_list[i], self.random_matrix[i].to(input_list[i])) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

# class LightDti_Da(pl.LightningModule):
#     def __init__(self, d_embedding, p_embedding, model_config, train_config):
#         super(LightDti_Da, self).__init__()
#         if not model_config.muti_out:
#             self.model = Dti_cnn(d_embedding, p_embedding, model_config)
#             #self.model = Dti_DrugBAN()
#         else:
#             self.model = Dti_cnn_mutiout(d_embedding, p_embedding, model_config)
#         # 跨域
#         self.train_config = train_config
#         self.gamma = train_config.gamma
#         self.num_classes = 2
#         self.sync = train_config.DDP
#         self.MDa = train_config.MDa
#         #self.loss = nn.BCEWithLogitsLoss()
#         self.val_preds = torch.Tensor([])
#         self.val_labels = torch.Tensor([])
#         self.train_auc = AUROC(task='binary')
#         self.val_auc = AUROC(task='binary')
#         self.test_auc = AUROC(task='binary')
#
#         self.train_auprc = AveragePrecision(task='binary')
#         self.val_auprc = AveragePrecision(task='binary')
#         self.test_auprc = AveragePrecision(task='binary')
#
#         self.train_acc = Accuracy(task='binary')
#         self.val_acc = Accuracy(task='binary')
#         self.test_acc = Accuracy(task='binary')
#         self.train_spec = Specificity(task='binary')
#         self.val_spec = Specificity(task='binary')
#         self.test_spec = Specificity(task='binary')
#
#         self.train_sen = Recall(task='binary')
#         self.val_sen = Recall(task='binary')
#         self.test_sen = Recall(task='binary')
#         #self.automatic_optimization = False
#         self.random_layer = RandomLayer([model_config.out_hidden_size, 2], model_config.out_hidden_size)
#         if self.MDa:
#             self.domain_classifiers = nn.ModuleList(
#                 [nn.Sequential(Discriminator(model_config.out_hidden_size,2)) for _ in range(self.num_classes)])
#             #self.daloss = nn.CrossEntropyLoss()
#         else:
#             self.domain_classifiers = Discriminator(model_config.out_hidden_size,2)
#             #self.daloss = nn.BCEWithLogitsLoss()
#
#
#     def da_forward(self,drug,portein, mode='train'):
#         #lbda = self.get_lambda_p(self.get_p()) if mode == 'train' else 0
#         lbda = 1 if mode == 'train' else 0
#         class_logits, features = self.model(drug, portein)
#         if self.current_epoch >= self.train_config.Da_warm_epochs and mode == 'train':
#             class_predictions = torch.nn.Softmax(dim=1)(class_logits)
#             class_predictions = class_predictions.detach()
#             reverse_features = ReverseLayerF.apply(features, lbda)
#             random_out = self.random_layer.forward([reverse_features, class_predictions])
#             if self.MDa:
#                 domain_logits = []
#                 for class_idx in range(self.num_classes):
#                     weighted_reverse_features = class_predictions[:, class_idx].unsqueeze(1) * random_out
#                     domain_logits.append(
#                         self.domain_classifiers[class_idx](weighted_reverse_features).cuda()
#                     )
#             else:
#                 domain_logits = self.domain_classifiers(random_out.view(-1, random_out.size(1)))
#             return class_logits, domain_logits
#         else:
#             return class_logits
#
#     def _calculate_loss(self, batch, mode='train'):
#         if mode == 'train':
#             if self.current_epoch >= self.train_config.Da_warm_epochs:
#                 (d_s, p_s, ys), (d_t, p_t, _) = batch
#                 ys = ys.squeeze(-1).long()
#                 # Generate fake labels for domains (0's for source and 1's for target)
#                 ys_domain = torch.zeros(len(ys), device=self.device, dtype=torch.long)
#                 yt_domain = torch.ones(len(p_t), device=self.device, dtype=torch.long)
#
#                 class_logit_src, domain_logit_src = self.da_forward(d_s, p_s)
#
#                 _, domain_logit_tgt = self.da_forward(d_t, p_t)
#                 n,loss_class_src = cross_entropy_logits(class_logit_src, ys)
#                 self.log_metrics_step(n, ys, mode)
#                 if self.MDa:
#                     losses_domain_src = torch.Tensor(0).to(ys)
#                     losses_domain_tgt = torch.Tensor(0).to(ys)
#                     for class_idx in range(self.num_classes):
#                         _,loss_domain_src = cross_entropy_logits(domain_logit_src[class_idx], ys_domain)
#                         _,loss_domain_tgt = cross_entropy_logits(domain_logit_tgt[class_idx], yt_domain)
#                         losses_domain_tgt = losses_domain_tgt+loss_domain_tgt
#                         losses_domain_src = losses_domain_src+loss_domain_src
#                     losses_domain = sum(losses_domain_src + losses_domain_tgt) / self.num_classes
#                 else:
#                     n,losses_domain_src = cross_entropy_logits(domain_logit_src, ys_domain)
#                     n,losses_domain_tgt = cross_entropy_logits(domain_logit_tgt, yt_domain)
#                     # Aggregate losses
#                     #lbda = self.get_lambda_p(self.get_p())
#                     losses_domain = losses_domain_src + losses_domain_tgt
#
#                 loss_tot = loss_class_src + losses_domain
#                 return loss_tot
#             else:
#                 (d_s, p_s, ys), (d_t, p_t, _) = batch
#                 ys = ys.squeeze(-1).long()
#                 class_logit_src = self.da_forward(d_s, p_s, mode='test')
#                 n,loss = cross_entropy_logits(class_logit_src, ys)
#                 self.log_metrics_step(n, ys, mode)
#                 return loss
#         else:
#             d_s, p_s, ys = batch
#             ys = ys.squeeze(-1).long()
#             class_logit_src = self.da_forward(d_s, p_s,mode='test')
#             n,loss = cross_entropy_logits(class_logit_src, ys)
#             self.val_preds = torch.cat((self.val_preds.to(class_logit_src), class_logit_src), dim=0).detach()
#             self.val_labels = torch.cat((self.val_labels.to(ys), ys), dim=0).detach()
#             self.log_metrics_step(n, ys, mode)
#             return loss
#
#     def training_step(self, batch, batch_idx):
#         loss = self._calculate_loss(batch, mode='train')
#         self.log('train_loss', loss, sync_dist=self.sync)
#         return loss
#         # Unpack source/target batch of images and labels
#
#     def validation_step(self, batch, batch_idx):
#         loss = self._calculate_loss(batch, mode='eval')
#         self.log('val_loss', loss, sync_dist=self.sync)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         loss = self._calculate_loss(batch, mode='test')
#         self.log('test_loss', loss, sync_dist=self.sync)
#         return loss
#
#     def on_validation_epoch_end(self):
#         self.log_metrics_epoch(mode='eval')
#
#     def on_train_epoch_end(self):
#         self.log_metrics_epoch(mode='train')
#
#     def on_test_epoch_end(self):
#         self.log_metrics_epoch(mode='test')
#
#
#     def configure_optimizers(self):
#         model_parameter = [
#             {
#                 "params": self.model.parameters(),
#                 "lr_mult": 1,
#                 'decay_mult': 1,
#             },
#             {
#                 "params": self.domain_classifiers.parameters(),
#                 "lr_mult": 0.1,
#                 'decay_mult': 1,
#             },
#             # *[
#             #     {
#             #         "params": self.domain_classifiers[class_idx].parameters(),
#             #         "lr_mult": 1.0,
#             #         'decay_mult': 2,
#             #     } for class_idx in range(self.num_classes)
#             # ]
#         ]
#
#         opt = torch.optim.Adam(model_parameter, lr=self.train_config.lr)
#         #opt_da = torch.optim.Adam(self.domain_classifiers.parameters(), lr=self.train_config.da_lr)
#         return opt
#         #,opt_da
#
#     def log_metrics_step(self,preds,labels,mode):
#         avg = 'step'
#         labels = (labels).long()
#         if mode == 'train':
#             self.train_auc(preds, labels)
#             self.log(f'train_auroc_{avg}', self.train_auc)
#
#             self.train_auprc(preds,labels)
#             self.log(f'train_auprc_{avg}', self.train_auprc)
#
#             self.train_acc(preds, labels)
#             self.log(f'train_acc_{avg}', self.train_acc)
#
#             self.train_spec(preds, labels)
#             self.log(f'train_spec_{avg}', self.train_spec)
#
#             self.train_sen(preds, labels)
#             self.log(f'train_sen_{avg}', self.train_sen)
#
#         elif mode == 'eval':
#             self.val_auc(preds, (labels).long())
#             self.log(f'val_auroc_{avg}', self.val_auc)
#
#             self.val_auprc(preds,labels)
#             self.log(f'val_auprc_{avg}', self.val_auprc)
#             self.val_acc(preds, labels)
#             self.log(f'val_acc_{avg}', self.val_acc)
#
#             self.val_spec(preds, labels)
#             self.log(f'val_spec_{avg}', self.val_spec)
#
#             self.val_sen(preds, labels)
#             self.log(f'val_sen_{avg}', self.val_sen)
#
#     def log_metrics_epoch(self,mode = 'train'):
#         avg = 'avg'
#         if mode == 'train':
#             self.log(f'train_auroc_{avg}', self.train_auc)
#             self.log(f'train_auprc_{avg}', self.train_auprc)
#             self.log(f'train_acc_{avg}', self.train_acc)
#             self.log(f'train_spec_{avg}', self.train_spec)
#             self.log(f'train_sen_{avg}', self.train_sen)
#         elif mode == 'eval':
#             self.log(f'val_auroc_{avg}', self.val_auc)
#             self.log(f'val_auprc_{avg}', self.val_auprc)
#             self.log(f'val_acc_{avg}', self.val_acc)
#             self.log(f'val_spec_{avg}', self.val_spec)
#             self.log(f'val_sen_{avg}', self.val_sen)
#             n = F.softmax(self.val_preds, dim=1)[:, 1]
#             self.log('t_roc', roc_auc_score(self.val_labels.to('cpu').tolist(), n.to('cpu').tolist()), prog_bar=True)
#             self.val_preds = torch.Tensor([])
#             self.val_labels = torch.Tensor([])
#         else:
#             self.log(f'test_auroc_{avg}', self.test_auc)
#             self.log(f'test_auprc_{avg}', self.test_auprc)
#             self.log(f'test_acc_{avg}', self.test_acc)
#             self.log(f'test_spec_{avg}', self.test_spec)
#             self.log(f'test_sen_{avg}', self.test_sen)
#             n = F.softmax(self.val_preds, dim=1)[:, 1]
#             self.log('t_roc', roc_auc_score(self.val_labels.to('cpu').tolist(), n.to('cpu').tolist()), prog_bar=True)
#             self.val_preds = torch.Tensor([])
#             self.val_labels = torch.Tensor([])
#     def get_p(self):
#         current_iterations, current_epoch, len_dataloader = self.global_step, self.current_epoch, self.trainer.datamodule.train_ds.__len__()
#         return float(
#             current_iterations + current_epoch * len_dataloader) / self.train_config.max_epochs / len_dataloader
#
#     def get_lambda_p(self, p):
#         return 2. / (1. + np.exp(-self.gamma * p)) - 1

