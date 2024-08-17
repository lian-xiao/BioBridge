#!/usr/bin/env python3

"""
"""
import dgl
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics.functional import accuracy,auroc,average_precision,specificity,recall
from torchmetrics import AUROC, AveragePrecision, Accuracy, Specificity, Recall
from torch import nn
#from learn2learn.utils import accuracy
from learn2learn.nn import PrototypicalClassifier, SVClassifier
import learn2learn as l2l

from Dti.Dti_cnn import Dti_cnn_mutiout, Dti_cnn
from Dti.Dti_meta_cnn import Dti_meta_mulcnn, Dti_meta_cnn, AdaptivePrototypicalClassifier, Dti_maml_mulcnn, \
    Dti_mamlpp_mulcnn, Dti_meta_drugban, Dti_anil_mulcnn
import torch.nn.functional as F
from torch.autograd import Variable

from Meta_utils.MAMLpp import MAMLpp


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class LightningMamlNetworks(LightningModule):

    def __init__(self, model_config, train_config):
        super(LightningMamlNetworks, self).__init__()
        if train_config.loss == 'CE':
            loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            loss = FocalLoss(class_num=2)
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.loss = loss
        self.train_ways = train_config.train_ways
        self.train_shots = train_config.train_shots
        self.train_queries = train_config.train_queries
        self.test_ways = train_config.test_ways
        self.test_shots = train_config.test_shots
        self.test_queries = train_config.test_queries


        self.lr = train_config.lr
        self.scheduler_step = train_config.scheduler_step
        self.scheduler_decay = train_config.scheduler_decay
        self.distance_metric = train_config.distance_metric
        self.save_hyperparameters(train_config)
        self.data_parallel = train_config.data_parallel
        self.adaptive = train_config.adaptive
        self.mamlpp = train_config.mamlpp

        if model_config.muti_out:
            if self.mamlpp:
                self.model = Dti_mamlpp_mulcnn(None, None, model_config,train_config.adaptation_steps)
            else:

                self.model = Dti_maml_mulcnn(None, None, model_config)
        else:
            self.model = Dti_cnn(None, None, model_config)

        # if train_config.load is not None:
        #     pretrain = torch.load(train_config.load)
        #     print(self.model.load_state_dict(pretrain,strict=False))
        if self.data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.adaptation_steps = train_config.adaptation_steps
        self.adaptation_lr = train_config.adaptation_lr

        if self.mamlpp:
            # Multi-Step Loss
            self.batch_size = train_config.batch_size
            self._msl_epochs = train_config.msl_epochs
            self._step_weights = torch.ones(self.adaptation_steps, device=self._device) * (1.0 / self.adaptation_steps)
            self._msl_decay_rate = 1.0 / self.adaptation_steps / self._msl_epochs
            self._msl_min_value_for_non_final_losses = torch.tensor(0.03 / self.adaptation_steps)
            self._msl_max_value_for_final_loss = 1.0 - (
                (self.adaptation_steps - 1) * self._msl_min_value_for_non_final_losses
            )

            # Derivative-Order Annealing (when to start using second-order opt)
            self._derivative_order_annealing_from_epoch = train_config.Da_epoch

            self.model = MAMLpp(
                self.model,
                lr=self.adaptation_lr, # Initialisation LR for all layers and steps
                adaptation_steps=self.adaptation_steps, # For LSLR
                first_order=False,
                allow_nograd=True, # For the parameters of the MetaBatchNorm layers
            )
        else:
            self.model = l2l.algorithms.MAML(self.model, lr=self.adaptation_lr, first_order=False,allow_unused=True)

        self.train_pred = torch.tensor([]).to(self.device)
        self.train_labels = torch.tensor([]).to(self.device)

        self.pred = torch.tensor([]).to(self.device)
        self.labels = torch.tensor([]).to(self.device)



    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self.adaptation_steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def log_metrics_step(self, preds, labels, mode):
        if mode == 'train':
            self.train_pred = torch.concat([self.train_pred.to(preds), preds], dim=0)
            self.train_labels = torch.concat([self.train_labels.to(labels), labels], dim=0)
        else:
            self.pred = torch.concat([self.pred.to(preds),preds],dim=0)
            self.labels = torch.concat([self.labels.to(labels),labels],dim=0)

    def log_metrics_epoch(self, mode='train'):
        if mode == 'train':
            metrics_auroc = auroc(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            self.train_pred = torch.tensor([]).to(self.device)
            self.train_labels = torch.tensor([]).to(self.device)
        else:
            metrics_auroc = auroc(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.pred,self.labels,task='multiclass',num_classes=2)
            self.pred = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        self.log(f'{mode}_auroc', metrics_auroc, prog_bar=True,on_epoch=True,sync_dist=self.sync)
        self.log(f'{mode}_auprc', metrics_ap, sync_dist=self.sync)
        self.log(f'{mode}_acc', metrics_acc, sync_dist=self.sync)
        self.log(f'{mode}_spec', metrics_spec, sync_dist=self.sync)
        self.log(f'{mode}_sen', metrics_recall, sync_dist=self.sync)

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, ways, shots, queries,mode='eval'):
        learner = self.model.clone()

        learner.train()
        if self.data_parallel:
            learner.module = torch.nn.DataParallel(learner.module)
        #data, labels = batch
        drug, protein, labels, trg_cluster, drug_cluster, index = batch
        labels = labels.long()
        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(labels.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        if len(support_indices) != 0:
            support_drug = dgl.batch([drug[i] for i in range(len(drug)) if support_indices[i]])
            #support_drug = (drug[support_indices])
            support_prot = protein[support_indices]
            support_labels = labels[support_indices]

        query_drug = dgl.batch([drug[i] for i in range(len(drug)) if query_indices[i]])
        query_prot = protein[query_indices]
        query_labels = labels[query_indices]

        if self.mamlpp:
            query_loss = torch.tensor(.0, device=self._device)
            msl = (self.current_epoch < self._msl_epochs),
            second_order = True
            if self.current_epoch < self._derivative_order_annealing_from_epoch:
                second_order = False
            # Adapt the model on the support set
            for step in range(self.adaptation_steps):
                if mode == 'train':
                    # forward + backward + optimize
                    pred,_ = learner(support_drug, support_prot, False, step)
                    support_loss = self.loss(pred, support_labels)
                    learner.adapt(support_loss, first_order=not second_order, step=step)
                    # Multi-Step Loss
                    if msl:
                        predictions, _ = learner(query_drug, query_prot, False, step)
                        query_loss += self._step_weights[step] * self.loss(
                            predictions, query_labels
                        )
                else:
                    pred,_ = learner(support_drug, support_prot, False, step)
                    support_loss = self.loss(pred, support_labels)
                    learner.adapt(support_loss, step=step)
            if not msl:
                if mode == 'train':
                    predictions, _ = learner(query_drug, query_prot, False, self.adaptation_steps - 1)
                    query_loss = self.loss(predictions,query_labels)

            if mode != 'train':
            # Evaluate the adapted model on the query set
                with torch.no_grad():
                    predictions, _ = learner(query_drug, query_prot, False, self.adaptation_steps - 1)
                    query_loss = self.loss(predictions, query_labels)

            valid_error = query_loss
        else:
            # Adapt the model
            # learner = self.adapt(learner,support_drug,support_prot,support_labels)
            for step in range(self.adaptation_steps):
                logit,_ = learner(support_drug, support_prot, False)
                train_error = self.loss(logit, support_labels)
                learner.adapt(train_error)

            #torch.cuda.empty_cache()
            if mode != 'train':
                with torch.no_grad():
                    predictions, _ = learner(query_drug, query_prot, False)
                    valid_error = self.loss(predictions, query_labels)
            else:
                predictions,_ = learner(query_drug,query_prot,False)
                valid_error = self.loss(predictions, query_labels)

        self.log_metrics_step(predictions, query_labels, mode)
        return valid_error

    def training_step(self, batch, batch_idx):
        #torch.cuda.empty_cache()
        loss = self.meta_learn(batch, batch_idx, self.train_ways, self.train_shots, self.train_queries,'train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'eval'
        )
        #torch.cuda.empty_cache()
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss.item()


    def test_step(self, batch, batch_idx):
        #torch.cuda.empty_cache()
        #with torch.set_grad_enabled(True):
        # loss = self.meta_learn(
        #         batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'eval'
        #     )
        # self.log('test_loss', loss, sync_dist=self.sync)
        loss = self.validation_step(batch,batch_idx)
        return loss.item()

    def configure_optimizers(self):
        if self.mamlpp:
            opt = torch.optim.AdamW(self.parameters(), self.lr, betas=(0, 0.999))

            iter_per_epoch = len(self.trainer.datamodule.train_dataloader())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=50 * iter_per_epoch,
                eta_min=0.00001,
            )
            return [opt], [scheduler]
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr,betas=(0, 0.999))
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step,
                gamma=self.scheduler_decay,
            )
            return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):
        if self.mamlpp:
            self.model.restore_backup_stats()
        self.log_metrics_epoch(mode='val')

    def on_train_epoch_end(self):
        if self.mamlpp:
            self._anneal_step_weights()
        self.log_metrics_epoch(mode='train')

    def on_validation_epoch_start(self) -> None:
        if self.mamlpp:
            self.model.backup_stats()

    def on_before_optimizer_step(self,optimizer):
        if self.mamlpp:
            with torch.no_grad():
                for p in self.parameters():
                    # Remember the MetaBatchNorm layer has parameters that don't require grad!
                    if p.requires_grad:
                        p.grad.data.mul_(1.0 / self.train_shots)

    # def on_test_epoch_start(self) -> None:
    #     opt = torch.optim.AdamW(self.parameters(), self.lr, betas=(0, 0.999))
    #     self.model = MAMLpp(
    #         self.model,
    #         lr=self.adaptation_lr,
    #         adaptation_steps=self.adaptation_steps,
    #         first_order=False,
    #         allow_nograd=True,
    #     )

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')


class LightningPrototypicalNetworks(LightningModule):

    distance_metric = "euclidean"

    def __init__(self,model_config,train_config):
        super(LightningPrototypicalNetworks, self).__init__()
        if train_config.loss == 'CE':
            loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            loss = FocalLoss(class_num=2)
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.loss = loss
        self.train_ways = train_config.train_ways
        self.train_shots = train_config.train_shots
        self.train_queries = train_config.train_queries
        self.test_ways = train_config.test_ways
        self.test_shots = train_config.test_shots
        self.test_queries = train_config.test_queries
        self.lr = train_config.lr
        self.scheduler_step = train_config.scheduler_step
        self.scheduler_decay = train_config.scheduler_decay
        self.distance_metric = train_config.distance_metric
        self.save_hyperparameters(train_config)
        self.data_parallel = train_config.data_parallel
        self.adaptive = train_config.adaptive
        drugban = getattr(model_config, 'drugban', False)
        if not drugban:
            if model_config.muti_out:
                self.model = Dti_meta_mulcnn(None, None, model_config)
            else:
                self.model = Dti_meta_cnn(None, None, model_config)
        else:
            self.model = Dti_meta_drugban()

        if train_config.load is not None:
            pretrain = torch.load(train_config.load)
            print(self.model.load_state_dict(pretrain,strict=False))

        if self.data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.adaptive:
            self.classifier = AdaptivePrototypicalClassifier(distance=self.distance_metric,normalize=train_config.normalize)
        else:
            self.classifier = PrototypicalClassifier(distance=self.distance_metric,normalize=train_config.normalize)

        self.train_pred = torch.tensor([]).to(self.device)
        self.train_labels = torch.tensor([]).to(self.device)

        self.pred = torch.tensor([]).to(self.device)
        self.labels = torch.tensor([]).to(self.device)


    def log_metrics_step(self, preds, labels, mode):
        if mode == 'train':
            self.train_pred = torch.concat([self.train_pred.to(preds), preds], dim=0)
            self.train_labels = torch.concat([self.train_labels.to(labels), labels], dim=0)
        else:
            self.pred = torch.concat([self.pred.to(preds),preds],dim=0)
            self.labels = torch.concat([self.labels.to(labels),labels],dim=0)

    def log_metrics_epoch(self, mode='train'):
        if mode == 'train':
            metrics_auroc = auroc(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            self.train_pred = torch.tensor([]).to(self.device)
            self.train_labels = torch.tensor([]).to(self.device)
        else:
            metrics_auroc = auroc(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.pred,self.labels,task='multiclass',num_classes=2)
            self.pred = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        self.log(f'{mode}_auroc', metrics_auroc, prog_bar=True,on_epoch=True,sync_dist=self.sync)
        self.log(f'{mode}_auprc', metrics_ap, sync_dist=self.sync)
        self.log(f'{mode}_acc', metrics_acc, sync_dist=self.sync)
        self.log(f'{mode}_spec', metrics_spec, sync_dist=self.sync)
        self.log(f'{mode}_sen', metrics_recall, sync_dist=self.sync)


    def meta_learn(self, batch, batch_idx, ways, shots, queries,mode):
        self.model.train()
        self.model.classifier.eval()
        drug,protein,labels,trg_cluster,drug_cluster,index = batch
        labels = labels.long()
        embeddings = self.model.features(drug,protein,False)
        support_indices = np.zeros(protein.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]
        if not self.adaptive:
            self.classifier.fit_(support, support_labels)
            logits = self.classifier(query)
        else:
            logits = self.classifier(query,support, support_labels)
        eval_loss = self.loss(logits, query_labels)
        self.log_metrics_step(logits,query_labels,mode)

        return eval_loss

    # def predict_step(self, batch,batch_idx):
    #     self.model.train()
    #     self.model.classifier.eval()
    #     drug,protein,labels,trg_cluster,drug_name,target_name,cid,index = batch
    #     labels = labels.long()
    #     embeddings = self.model.features(drug,protein,False)
    #     support_indices = np.zeros(protein.size(0), dtype=bool)
    #     selection = np.arange(self.test_ways) * (self.test_shots + self.test_queries)
    #     for offset in range(self.test_shots):
    #         support_indices[selection + offset] = True
    #     query_indices = torch.from_numpy(~support_indices)
    #     support_indices = torch.from_numpy(support_indices)
    #     support = embeddings[support_indices]
    #     support_labels = labels[support_indices]
    #     query = embeddings[query_indices]
    #     query_labels = labels[query_indices]
    #     logits,attns = self.classifier(query,support, support_labels,True)
    #     logits = F.softmax(logits, -1)
    #     result = {
    #         "logits": logits,
    #         "predictions": logits.argmax(dim=1),
    #         "query_target": [target_name[i] for i in range(len(query_indices)) if query_indices[i]],
    #         "query_drug":[cid[i] for i in range(len(query_indices)) if query_indices[i]],
    #         "query_labels": query_labels,
    #         "support_target": [target_name[i] for i in range(len(support_indices)) if support_indices[i]],
    #         "support_drug": [cid[i] for i in range(len(support_indices)) if support_indices[i]],
    #         "support_labels": support_labels,
    #         "attentions": attns,
    #     }
    #     return result


    def predict_step(self, batch,batch_idx):
        self.model.train()
        self.model.classifier.eval()
        drug,protein,labels,trg_cluster,drug_name,target_name,cid,index = batch
        labels = labels.long()
        embeddings = self.model.features(drug,protein,False)
        support_indices = np.zeros(protein.size(0), dtype=bool)
        selection = np.arange(self.test_ways) * (self.test_shots + self.test_queries)
        for offset in range(self.test_shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]
        logits,attns = self.classifier(query,support, support_labels,True)
        logits = F.softmax(logits, -1)
        result = {
            "logits": logits,
            "predictions": logits.argmax(dim=1),
            "query_labels": query_labels,
        }
        return result

    def training_step(self, batch, batch_idx):
        loss = self.meta_learn(batch, batch_idx, self.train_ways, self.train_shots, self.train_queries,'train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'eval'
        )
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'test'
        )
        self.log('test_loss', loss, sync_dist=self.sync)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_decay,
        )
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='val')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')



class LightningMetaoptNetworks(LightningModule):

    distance_metric = "euclidean"

    def __init__(self,model_config,train_config):
        super(LightningMetaoptNetworks, self).__init__()
        loss = nn.CrossEntropyLoss(reduction="mean")
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.loss = loss
        self.train_ways = train_config.train_ways
        self.train_shots = train_config.train_shots
        self.train_queries = train_config.train_queries
        self.test_ways = train_config.test_ways
        self.test_shots = train_config.test_shots
        self.test_queries = train_config.test_queries
        self.lr = train_config.lr
        self.scheduler_step = train_config.scheduler_step
        self.scheduler_decay = train_config.scheduler_decay
        self.distance_metric = train_config.distance_metric
        self.save_hyperparameters(train_config)
        self.data_parallel = train_config.data_parallel
        self.svm_C_reg = 0.1
        self.svm_max_iters = 15
        drugban = getattr(model_config, 'drugban', False)
        if not drugban:
            if model_config.muti_out:
                self.model = Dti_meta_mulcnn(None, None, model_config)
            else:
                self.model = Dti_meta_cnn(None, None, model_config)
        else:
            self.model = Dti_meta_drugban()
        self.classifier = SVClassifier(
            C_reg=self.svm_C_reg,
            max_iters=self.svm_max_iters,
            normalize=False,
        )
        self.train_pred = torch.tensor([]).to(self.device)
        self.train_labels = torch.tensor([]).to(self.device)

        self.pred = torch.tensor([]).to(self.device)
        self.labels = torch.tensor([]).to(self.device)


    def log_metrics_step(self, preds, labels, mode):
        if mode == 'train':
            self.train_pred = torch.concat([self.train_pred.to(preds), preds], dim=0)
            self.train_labels = torch.concat([self.train_labels.to(labels), labels], dim=0)
        else:
            self.pred = torch.concat([self.pred.to(preds),preds],dim=0)
            self.labels = torch.concat([self.labels.to(labels),labels],dim=0)

    def log_metrics_epoch(self, mode='train'):
        if mode == 'train':
            metrics_auroc = auroc(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            self.train_pred = torch.tensor([]).to(self.device)
            self.train_labels = torch.tensor([]).to(self.device)
        else:
            metrics_auroc = auroc(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.pred,self.labels,task='multiclass',num_classes=2)
            self.pred = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        self.log(f'{mode}_auroc', metrics_auroc, prog_bar=True,on_epoch=True,sync_dist=self.sync)
        self.log(f'{mode}_auprc', metrics_ap, sync_dist=self.sync)
        self.log(f'{mode}_acc', metrics_acc, sync_dist=self.sync)
        self.log(f'{mode}_spec', metrics_spec, sync_dist=self.sync)
        self.log(f'{mode}_sen', metrics_recall, sync_dist=self.sync)


    def meta_learn(self, batch, batch_idx, ways, shots, queries,mode):
        self.model.train()
        self.model.classifier.eval()
        drug,protein,labels,trg_cluster,drug_cluster,index = batch
        labels = labels.long()
        embeddings = self.model.features(drug,protein,False)
        support_indices = np.zeros(protein.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]
        self.classifier.fit_(support, support_labels, ways=ways)
        logits = self.classifier(query)
        eval_loss = self.loss(logits, query_labels)
        self.log_metrics_step(logits,query_labels,mode)
        return eval_loss


    def training_step(self, batch, batch_idx):
        loss = self.meta_learn(batch, batch_idx, self.train_ways, self.train_shots, self.train_queries,'train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'eval'
        )
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'test'
        )
        self.log('test_loss', loss, sync_dist=self.sync)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_decay,
        )
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):
        self.log_metrics_epoch(mode='val')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')


class LightningAnilNetworks(LightningModule):

    def __init__(self, model_config, train_config):
        super(LightningAnilNetworks, self).__init__()
        if train_config.loss == 'CE':
            loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            loss = FocalLoss(class_num=2)
        self.main_metrice = train_config.main_metric
        self.sync = train_config.DDP
        self.loss = loss
        self.train_ways = train_config.train_ways
        self.train_shots = train_config.train_shots
        self.train_queries = train_config.train_queries
        self.test_ways = train_config.test_ways
        self.test_shots = train_config.test_shots
        self.test_queries = train_config.test_queries


        self.lr = train_config.lr
        self.scheduler_step = train_config.scheduler_step
        self.scheduler_decay = train_config.scheduler_decay
        self.distance_metric = train_config.distance_metric
        self.save_hyperparameters(train_config)
        self.data_parallel = train_config.data_parallel
        self.adaptive = train_config.adaptive
        self.mamlpp = train_config.mamlpp

        if model_config.muti_out:
            self.model = Dti_anil_mulcnn(None, None, model_config)
        if self.data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.adaptation_steps = train_config.adaptation_steps
        self.adaptation_lr = train_config.adaptation_lr
        self.features = self.model.features
        self.classifer = self.model.pred
        self.classifer = l2l.algorithms.MAML(self.classifer, lr=self.adaptation_lr,allow_unused=True)

        self.train_pred = torch.tensor([]).to(self.device)
        self.train_labels = torch.tensor([]).to(self.device)

        self.pred = torch.tensor([]).to(self.device)
        self.labels = torch.tensor([]).to(self.device)



    def _anneal_step_weights(self):
        self._step_weights[:-1] = torch.max(
            self._step_weights[:-1] - self._msl_decay_rate,
            self._msl_min_value_for_non_final_losses,
        )
        self._step_weights[-1] = torch.min(
            self._step_weights[-1] + ((self.adaptation_steps - 1) * self._msl_decay_rate),
            self._msl_max_value_for_final_loss,
        )

    def log_metrics_step(self, preds, labels, mode):
        if mode == 'train':
            self.train_pred = torch.concat([self.train_pred.to(preds), preds], dim=0)
            self.train_labels = torch.concat([self.train_labels.to(labels), labels], dim=0)
        else:
            self.pred = torch.concat([self.pred.to(preds),preds],dim=0)
            self.labels = torch.concat([self.labels.to(labels),labels],dim=0)

    def log_metrics_epoch(self, mode='train'):
        if mode == 'train':
            metrics_auroc = auroc(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.train_pred,self.train_labels,task='multiclass',num_classes=2)
            self.train_pred = torch.tensor([]).to(self.device)
            self.train_labels = torch.tensor([]).to(self.device)
        else:
            metrics_auroc = auroc(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_acc = accuracy(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_ap = average_precision(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_recall = recall(self.pred,self.labels,task='multiclass',num_classes=2)
            metrics_spec = specificity(self.pred,self.labels,task='multiclass',num_classes=2)
            self.pred = torch.tensor([]).to(self.device)
            self.labels = torch.tensor([]).to(self.device)
        self.log(f'{mode}_auroc', metrics_auroc, prog_bar=True,on_epoch=True,sync_dist=self.sync)
        self.log(f'{mode}_auprc', metrics_ap, sync_dist=self.sync)
        self.log(f'{mode}_acc', metrics_acc, sync_dist=self.sync)
        self.log(f'{mode}_spec', metrics_spec, sync_dist=self.sync)
        self.log(f'{mode}_sen', metrics_recall, sync_dist=self.sync)

    @torch.enable_grad()
    def meta_learn(self, batch, batch_idx, ways, shots, queries,mode='eval'):
        self.features.train()
        learner = self.classifer.clone()
        learner.train()
        #data, labels = batch
        drug, protein, labels, trg_cluster, drug_cluster, index = batch
        data = self.features(drug,protein)

        labels = labels.long()
        # Separate data into adaptation and evaluation sets
        support_indices = np.zeros(labels.size(0), dtype=bool)
        selection = np.arange(ways) * (shots + queries)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        support = data[support_indices]
        support_labels = labels[support_indices]
        query = data[query_indices]
        query_labels = labels[query_indices]

        # Adapt the classifier
        for step in range(self.adaptation_steps):
            preds = learner(support)
            train_error = self.loss(preds, support_labels)
            learner.adapt(train_error)
        # Evaluating the adapted model

        if mode != 'train':
            with torch.no_grad():
                predictions = learner(query)
                valid_error = self.loss(predictions, query_labels)
        else:
            predictions = learner(query)
            valid_error = self.loss(predictions, query_labels)

        self.log_metrics_step(predictions, query_labels, mode)
        return valid_error

    def training_step(self, batch, batch_idx):
        #torch.cuda.empty_cache()
        loss = self.meta_learn(batch, batch_idx, self.train_ways, self.train_shots, self.train_queries,'train')
        self.log('train_loss', loss, prog_bar=True, sync_dist=self.sync)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.meta_learn(
            batch, batch_idx, self.test_ways, self.test_shots, self.test_queries,'eval'
        )
        #torch.cuda.empty_cache()
        self.log('val_loss', loss, sync_dist=self.sync)
        return loss.item()


    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch,batch_idx)
        return loss.item()

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr,betas=(0, 0.999))
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_decay,
        )
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):

        self.log_metrics_epoch(mode='val')

    def on_train_epoch_end(self):
        self.log_metrics_epoch(mode='train')

    def on_validation_epoch_start(self) -> None:
        if self.mamlpp:
            self.model.backup_stats()

    def on_before_optimizer_step(self,optimizer):
        if self.mamlpp:
            with torch.no_grad():
                for p in self.parameters():
                    # Remember the MetaBatchNorm layer has parameters that don't require grad!
                    if p.requires_grad:
                        p.grad.data.mul_(1.0 / self.train_shots)

    # def on_test_epoch_start(self) -> None:
    #     opt = torch.optim.AdamW(self.parameters(), self.lr, betas=(0, 0.999))
    #     self.model = MAMLpp(
    #         self.model,
    #         lr=self.adaptation_lr,
    #         adaptation_steps=self.adaptation_steps,
    #         first_order=False,
    #         allow_nograd=True,
    #     )

    def on_test_epoch_end(self):
        self.log_metrics_epoch(mode='test')