import argparse
import os
import dgl
import numpy as np
from config_meta import get_train_args, get_model_args
import torch
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics.functional import accuracy,auroc,average_precision,specificity,recall
from torch import nn
from learn2learn.nn import PrototypicalClassifier
from torch.utils.data import Dataset
from functools import partial
import pubchempy as pcp
from Data.DataMoudle_base_emb import get_dataset
from Dti.Dti_meta_cnn import Dti_meta_mulcnn, Dti_meta_cnn, AdaptivePrototypicalClassifier,Dti_meta_drugban
import pytorch_lightning as pl
import pandas as pd
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from learn2learn.utils.lightning import EpisodicBatcher
from lightning import seed_everything
from rdkit import Chem
import learn2learn as l2l
from torch.autograd import Variable
from Dti.utils import integer_label_protein
from Meta_utils.CrossFewshotData import CrossMetaDataset, FusedNwaysKShotsCross
from py_lighting_model.light_Dti_cnn import LightDta
from utils_callback import ProgressBar
import torch.nn.functional as F

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

class drug_transfer(object):
    def __init__(self,max_drug_nodes=290):
        self.max_drug_nodes = max_drug_nodes
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
    def convert(self,v_ds):
        for index,v_d in enumerate(v_ds):
            v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            actual_node_feats = v_d.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            v_d = v_d.add_self_loop()
            v_ds[index] = v_d
        return v_ds
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
        self.drug_transfer = drug_transfer()

        support_df = pd.read_csv(train_config.support_file)

        df = pd.read_csv(train_config.query_file)
        self.drug_data = list(df['SMILES'].to_list())
        self.protein_data = list(df['Protein'].to_list())
        self.drug_index = 0
        self.protein_index = 0
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

    def predict_step(self, batch,batch_idx):
        self.model.train()
        self.model.classifier.eval()
        drug,protein,labels,trg_cluster,drug_name,target_name,cid,trg_seq,drug_smile,index = batch
        support_indices = np.zeros(protein.size(0), dtype=bool)
        selection = np.arange(self.test_ways) * (self.test_shots + self.test_queries)
        for offset in range(self.test_shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        if self.drug_index+2 > len(self.drug_data):
            self.drug_index = 0
            self.protein_index = 0

        q_protein = self.protein_data[self.protein_index:self.protein_index+2]
        q_drug = self.drug_data[self.drug_index:self.drug_index+2]
        self.protein_index = self.protein_index+2
        self.drug_index = self.drug_index+2
        drug_smile = list(drug_smile)
        drug_smile[self.test_shots] = q_drug[0]
        drug_smile[self.test_shots*2+1] = q_drug[1]
        q_drug = self.drug_transfer.convert(q_drug)
        drug[self.test_shots] = q_drug[0].to('cuda')
        drug[self.test_shots*2+1] = q_drug[1].to('cuda')
        drug = dgl.batch(drug)
        labels = labels.long()

        embeddings = self.model.features(drug,protein,False)

        support = embeddings[support_indices]
        support_labels = labels[support_indices]
        query = embeddings[query_indices]
        query_labels = labels[query_indices]

        query_trgseq = [value for value, condition in zip(list(trg_seq), query_indices) if condition]
        query_drugsmiles = [value for value, condition in zip(list(drug_smile), query_indices) if condition]
        logits,_ = self.classifier(query,support, support_labels,True)
        logits = F.softmax(logits, -1)
        result = {
            "logits": logits[:,-1],
            "predictions": logits.argmax(dim=1),
            "query_target": [target_name[i] for i in range(len(query_indices)) if query_indices[i]],
            "query_drug":[cid[i] for i in range(len(query_indices)) if query_indices[i]],
            "support_target": [target_name[i] for i in range(len(support_indices)) if support_indices[i]],
            "support_drug": [cid[i] for i in range(len(support_indices)) if support_indices[i]],
            "support_labels": support_labels,
            'SMILES':query_drugsmiles,
            'Proteins':query_trgseq,
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

def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:  # 确保SMILES字符串是有效的
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None  # 如果无效，返回None或其他占位符

def get_common_name(smiles):
    # 使用PubChemPy通过SMILES查找化合物
    compounds = pcp.get_compounds(smiles, 'smiles')
    if not compounds:
        return "Compound not found"
    # 获取化合物的CID
    cid = compounds[0].molecular_formula
    return cid




class DTIDataset(Dataset):
    def __init__(self, data_csv, max_drug_nodes=290):

        self.df = pd.read_csv(data_csv)
        if 'cid' not in self.df.columns:
            self.df['cid'] = self.df['Smiles'].apply(get_common_name)
        self.max_drug_nodes = max_drug_nodes
        self.list_IDs = self.df.index.values
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self._bookkeeping_path = data_csv.replace('.csv','.json')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['Smiles']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)

        y = self.df.iloc[index]["Y"]
        trg_cluster = self.df.iloc[index]['target_cluster']
        trg_name = self.df.iloc[index]['Protein name']
        trg_seq = self.df.iloc[index]['Protein']
        drug_name = self.df.iloc[index]['Drug name']
        drug_cid = self.df.iloc[index]['cid']
        drug_smile = self.df.iloc[index]['Smiles']
        return v_d, v_p, y, trg_cluster,drug_name,trg_name,drug_cid,trg_seq,drug_smile

def graph_collate_func(x):
    d, p, y,c,d_n,t_n,d_c,trg_seq,smiles = zip(*x)
    y = torch.tensor(y)
    sort = torch.sort(y)
    c = torch.tensor(c)
    p = torch.tensor(np.array(p))
    d = list(d)
    d = [d[i] for i in sort.indices.tolist()]
    p = p[sort.indices]
    return d, p, y,c,d_n,t_n,d_c, trg_seq,smiles,sort.indices

def get_domaindataset(data_csv,domian_ways=1,ways=2,query=1,shot=5,num_tasks=100,label_index=2,domain_index=3):

    dataset = DTIDataset(data_csv)

    dataset = CrossMetaDataset(dataset,label_index=label_index,domain_index=domain_index)
    task_transforms = [
                          FusedNwaysKShotsCross(dataset, domain_n=domian_ways, n=ways, k=query+shot),
                          l2l.data.transforms.LoadData(dataset),
                          l2l.data.transforms.ConsecutiveLabels(dataset),
                      ]

    data = l2l.data.Taskset(dataset, task_transforms=task_transforms,num_tasks=num_tasks,task_collate=graph_collate_func)
    return data


train_config = get_train_args()
model_config = get_model_args()


config = argparse.ArgumentParser(description='Combined configuration')
config.__dict__.update(train_config.__dict__)
config.__dict__.update(model_config.__dict__)
train_config = argparse.Namespace(**train_config)
model_config = argparse.Namespace(**model_config)
config = argparse.Namespace(**vars(train_config), **vars(model_config))
train_config.dataFolder = os.path.join(train_config.data_root, train_config.dataset_name)
if 'meta_unseen_protein' in train_config.dataset_name:
    train_config.domain_index = 3
elif 'meta_unseen_drug' in train_config.dataset_name:
    train_config.domain_index = 4
# train_config.seed = seed
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

model = LightningPrototypicalNetworks(model_config=model_config, train_config=train_config)

checkpoint_root = os.path.join(train_config.checkpoints_folder, train_config.dataset_name,
                               'shot' + str(train_config.train_shots))

trainer = pl.Trainer(
    callbacks=[ProgressBar()],
    strategy="ddp_find_unused_parameters_false" if train_config.DDP else 'auto',
    max_epochs=train_config.max_epochs,
    accumulate_grad_batches=train_config.batch_size,
    default_root_dir=checkpoint_root,
    accelerator="gpu",
    devices=1,
)

test = get_domaindataset(train_config.support_file, domian_ways=1, ways=2,
                         query=1,
                         shot=5, num_tasks=train_config.num_tasks,
                         label_index=2, domain_index=3)
dataloader = EpisodicBatcher.epochify(test,train_config.num_tasks)

predictions = trainer.predict(model, dataloaders=dataloader,ckpt_path="checkpoints/bindingdb_shot5.ckpt")

#
def collate(batch):
    smiles, proteins, labels = zip(*batch)
    smiles = dgl.batch(smiles)
    proteins = torch.Tensor(np.array(proteins))
    return smiles, proteins, torch.tensor(labels)

data = predictions
# 创建一个空的DataFrame
df = pd.DataFrame()
# 遍历列表中的每个任务
for task in data:
    logits = task['logits']
    support_target = task['support_target']
    support_drug = task['support_drug']
    query_target = task['query_target']
    query_drug = task['query_drug']
    predicted_labels = task['predictions']
    query_target_seq = task['Proteins']
    query_drug_smiles = task['SMILES']
    # 将支持集中的数据转换为字符串，以便在表格中显示
    support_target_str = ', '.join(support_target)
    support_drug_str = ', '.join(map(str, support_drug))  # 将数值转换为字符串
    # 为每个查询创建一个新行
    for i in range(len(query_target)):
        row = {
            'Support Target Positive': support_target[:5],
            'Support Drug Positive': support_drug[:5],
            'Support Target Negative': support_target[5:],
            'Support Drug Negative': support_drug[5:],
            'Query Target': query_target[i],
            'Query Drug': str(query_drug[i]),  # 将数值转换为字符串
            'Predicted Label': int(predicted_labels[i]),
            'SMILES':query_drug_smiles[i],
            'Protein':query_target_seq[i],
            'logits': logits.numpy()[i],
        }
        # 将行添加到DataFrame
        df = df._append(row, ignore_index=True)
df.to_csv(train_config.out_file,index=False)


train_config = {'data_root': 'Data', 'dataset_name': 'pdb2020', 'measure_name': ['Y'], 'Da':False,
                'gamma': 10, 'MDa': True,
                'optimizer_type': 'Adam', 'optimizer_momentum': 0.9, 'weight_decay': 2.5e-5, 'dropout': 0.1,
                'early_stop': False,'main_metric': 'rmse',
                'finetune_load_path': None, 'Da_warm_epochs': 0, 'batch_size': 64, 'max_epochs': 100,
                'num_workers': 0, 'lr': 5e-5,
                'checkpoints_folder': 'checkpoints_allstem/', 'seed': 2026, 'train_dataset_length': None,
                'test_dataset_length': None, 'eval_dataset_length': None, 'DDP': False}

model_config = {'drugban':False,'muti_out': True, 'p_ems2_emb': False, 'd_molformer_emb': False, 'p_emb': 128, 'd_emb': 128,
                'd_stem_channel': 128,'stem_kernel':1,'p_stem':True,'d_stem':True,'gate':True,
                'p_stem_channel': 128, 'd_channels': [128, 128, 128], 'p_channels': [128, 128, 128],
                'd_out_channel': 128, 'p_out_channel': 128, 'out_hidden_size': 256, 'layers_num': 3, 'binary': 1}
train_config = argparse.Namespace(**train_config)
model_config = argparse.Namespace(**model_config)

df = pd.read_csv(train_config.out_file)
test_rmse = get_dataset('','scd1_fdatask.csv',measure_name=['logits'])
dataloader = torch.utils.data.DataLoader(test_rmse,shuffle=False,collate_fn=collate)
model = LightDta(model_config, train_config)

predictions2 = trainer.predict(model,dataloaders=dataloader,ckpt_path='checkpoints/pdb202.ckpt')
predictions2 = [float(d['predictions']) for d in predictions2]
df['Aff Pred'] = predictions2
df['score'] = df['logits']**2*df['Aff Pred']
drugs = pd.read_csv(train_config.query_file)
df = pd.merge([df,drugs],on='SMILES',how='left')
df.to_csv(train_config.out_file,index=False)
