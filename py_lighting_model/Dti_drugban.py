
#from apex import optimizers
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import dgl
import pytorch_lightning as pl
from torch import nn
from torchmetrics import AUROC,AveragePrecision,Accuracy,Specificity,Recall
import torch
from Drug_ban.models import DrugBAN, cross_entropy_logits, binary_cross_entropy, RandomLayer, entropy_logits

from Drug_ban.domain_adaptator import ReverseLayerF,Discriminator
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score



class LightDti(pl.LightningModule):
    def __init__(self,config,train_config):
        super(LightDti, self).__init__()
        self.model = DrugBAN(**config)
        # 跨域
        self.train_config = train_config
        self.Da = train_config.Da
        print('Da',self.Da)
        # if self.Da:
        #     self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.BCEWithLogitsLoss()

        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.is_da = config["DA"]["USE"]
        self.alpha = 1
        self.n_class = config["DECODER"]["BINARY"]
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = Discriminator(input_size=config["DA"]["RANDOM_DIM"], n_class=config["DECODER"]["BINARY"])
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                                              out_features=config["DA"]
                                              ["RANDOM_DIM"], bias=False)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

        self.val_preds = torch.Tensor([])
        self.val_labels = torch.Tensor([])

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


    def configure_optimizers(self):
        model_parameter = [
            {
                "params": self.model.parameters(),
                "lr_mult": 1,
                'decay_mult': 1,
            },
            {
                "params": self.domain_dmm.parameters(),
                "lr_mult": 0.1,
                'decay_mult': 1,
            },
        ]

        opt = torch.optim.Adam(model_parameter, lr=self.train_config.lr)
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

        elif mode == 'eval':
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

    def log_metrics_epoch(self,mode = 'train'):
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
            n = F.softmax(self.val_preds, dim=1)[:, 1]
            self.log('t_roc', roc_auc_score(self.val_labels.to('cpu').tolist(), n.to('cpu').tolist()), prog_bar=True)
            self.val_preds = torch.Tensor([])
            self.val_labels = torch.Tensor([])
        else:
            self.log(f'test_auroc_{avg}', self.test_auc)
            self.log(f'test_auprc_{avg}', self.test_auprc)
            self.log(f'test_acc_{avg}', self.test_acc)
            self.log(f'test_spec_{avg}', self.test_spec)
            self.log(f'test_sen_{avg}', self.test_sen)
            n = F.softmax(self.val_preds, dim=1)[:, 1]
            self.log('t_roc', roc_auc_score(self.val_labels.to('cpu').tolist(), n.to('cpu').tolist()), prog_bar=True)
            self.val_preds = torch.Tensor([])
            self.val_labels = torch.Tensor([])
    def _calculate_loss(self,batch,mode='train'):
        drug,protein,labels = batch
        _,_,_,preds = self.model(drug,protein,'train')
        #loss = self.loss(preds, labels.long())
        n, loss = cross_entropy_logits(preds, labels)
        if mode != 'train':
            self.val_preds = torch.cat((self.val_preds.to(preds), preds), dim=0).detach()
            self.val_labels = torch.cat((self.val_labels.to(preds), labels), dim=0).detach()
        self.log_metrics_step(n,labels,mode)
        return loss

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def training_step(self, batch, batch_idx):
        if self.Da:
            (batch_s, batch_t) = batch
            v_d, v_p, labels = batch_s[0], batch_s[1], batch_s[2].float()
            v_d_t, v_p_t = batch_t[0], batch_t[1]
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)
                #model_loss = self.loss(score,labels)
            else:
                n, model_loss = cross_entropy_logits(score, labels)

                #model_loss = self.loss(score, labels.long())
            self.log_metrics_step(n, labels, 'train')
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
                                                                tgt_weight)
                    # loss_cdan_src = self.loss(adv_output_src_score, torch.zeros(len(adv_output_src_score)).to(adv_output_src_score).long())
                    # loss_cdan_tgt = self.loss(adv_output_tgt_score, torch.ones(len(adv_output_tgt_score)).to(adv_output_tgt_score).long())
                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            self.log('train_loss', loss, prog_bar=True)
            return loss
        else:
            loss = self._calculate_loss(batch,mode='train')
            self.log('train_loss',loss,prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,mode='eval')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch,mode='test')
        self.log('test_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='eval')
    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')
    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')
