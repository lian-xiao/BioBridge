import numpy as np
import pandas as pd
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import Dataset
import learn2learn as l2l
import os
from learn2learn.utils.lightning import EpisodicBatcher
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import dgl
from Drug_ban.utils import integer_label_protein
from operator import itemgetter
def PartitionByProtein(data,data_root,train_size=0.8,val_size=0.1,test_size=0.1):
    assert test_size+train_size+val_size == 1
    total_size = len(data)
    train_size,val_size,test_size = int(total_size * train_size),int(total_size * val_size),int(total_size * test_size)
    grouped_data = data.groupby('target_cluster')
    n_samples_per_category = grouped_data.size().reset_index(name='counts')
    n_samples_per_category = n_samples_per_category.sample(frac=1).reset_index()
    # 将计数列转换为数值类型
    n_samples_per_category['counts'] = n_samples_per_category['counts'].astype(int)
    # 创建一个新列用于存储累加结果
    n_samples_per_category['cumulative_count'] = 0
    n_samples_per_category.loc[0, 'cumulative_count'] = n_samples_per_category.loc[0, 'counts']
    # 对计数列进行累加
    for i in range(1, len(n_samples_per_category)):
        n_samples_per_category.loc[i, 'cumulative_count'] = n_samples_per_category.loc[i - 1, 'cumulative_count'] + n_samples_per_category.loc[i, 'counts']
    train_category = list(n_samples_per_category.loc[n_samples_per_category['cumulative_count']<=train_size,'target_cluster'])
    test_category = list(n_samples_per_category.loc[
        train_size + val_size < n_samples_per_category['cumulative_count'],'target_cluster'])
    validation_category = list(set(n_samples_per_category.loc[:,'target_cluster'])-set(train_category)-set(test_category))
    val = data[data['target_cluster'].isin(validation_category)]
    test = data[data['target_cluster'].isin(test_category)]
    train = data[data['target_cluster'].isin(train_category)]
    train.to_csv(os.path.join(data_root,'train.csv'), index=False)
    test.to_csv(os.path.join(data_root,'test.csv'),index=False)
    val.to_csv(os.path.join(data_root,'val.csv'), index=False)



class DTIDataset(Dataset):
    def __init__(self, df, max_drug_nodes=290):

        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.list_IDs = df.index.values
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
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
        cluster = self.df.iloc[index]['target_cluster']

        return v_d, cluster, v_p, y


def graph_collate_func(x):
    d, c, p, y = zip(*x)
    c = torch.tensor(c)
    sort = torch.sort(c)
    y = torch.tensor(y)
    p = torch.tensor(np.array(p))
    d = list(d)
    d = [d[i] for i in sort.indices.tolist()]
    p = p[sort.indices]
    d = dgl.batch(d)
    return d,c,p,y,sort.indices


class graph_collate_func_maml:
    def __init__(self, ways,shot,query_num):
        self.ways = ways
        self.shot = shot
        self.query_num = query_num

    def __call__(self, x):
        '''在这里重写collate_fn函数'''
        d, c, p, y = zip(*x)
        c = torch.tensor(c)
        sort = torch.sort(c)
        y = torch.tensor(y)
        p = torch.tensor(np.array(p))
        d = list(d)
        d = [d[i] for i in sort.indices.tolist()]
        p = p[sort.indices]
        support_indices = np.zeros(y.size(0), dtype=bool)
        selection = np.arange(self.ways) * (self.shot + self.query_num)
        for offset in range(self.shot):
            support_indices[selection + offset] = True
        # Compute support and query embeddings
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support_labels = y[support_indices]
        query_labels = y[query_indices]

        support_d = [d_g for d_g, s in zip(d, support_indices.tolist()) if s]
        query_d = [d_g for d_g, s in zip(d, support_indices.tolist()) if not s]
        support_p = p[support_indices]
        query_p = p[query_indices]

        support_d = dgl.batch(support_d)
        query_d = dgl.batch(query_d)

        return (support_d,support_p,support_labels),c,(query_d,query_p,query_labels)



def get_dataset(data_csv,train_ways,train_query,shot,alg='anil'):
    df = pd.read_csv(data_csv)
    value_counts = df['target_cluster'].value_counts()
    # 过滤出现次数少于3次的值的索引
    indices_to_remove = value_counts[value_counts < 5].index
    # 根据条件删除行
    df = df[~df['target_cluster'].isin(indices_to_remove)].reset_index()
    dataset = DTIDataset(df)
    train_dataset = l2l.data.MetaDataset(dataset)
    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_query + shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    if alg == 'anil':
        data = l2l.data.Taskset(train_dataset, task_transforms=train_transforms,num_tasks=10,task_collate=graph_collate_func)
    else:
        graph_collate = graph_collate_func_maml(train_ways,shot,train_query)
        data = l2l.data.Taskset(train_dataset, task_transforms=train_transforms, num_tasks=10000,
                                   task_collate=graph_collate)
    return data


def get_datamoudle(data_root,split_name,batch_size,train_config):
    def get_split_dataset_filename(split):
        return split + ".csv"
    dataset_name = os.path.join(data_root,split_name)
    train_filename = os.path.join(dataset_name,get_split_dataset_filename("train"))

    valid_filename = os.path.join(dataset_name,get_split_dataset_filename("val"))

    test_filename = os.path.join(dataset_name,get_split_dataset_filename("test"))
    train = get_dataset(train_filename,train_config.train_ways,train_config.train_queries,train_config.train_shot,train_config.Da)
    val = get_dataset(valid_filename,train_config.test_ways,train_config.test_queries,train_config.test_shot,train_config.Da)
    test = get_dataset(test_filename,train_config.test_ways,train_config.test_queries,train_config.test_shot,train_config.Da)
    datamoudle = EpisodicBatcher(
        train,
        val,
        test,
        epoch_length=batch_size*10,
    )

    return datamoudle
