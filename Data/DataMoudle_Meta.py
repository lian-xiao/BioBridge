import numpy as np
import pandas as pd
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from learn2learn.data import MetaDataset
from learn2learn.data.samplers import TasksetSampler
from torch.utils.data import Dataset
import learn2learn as l2l
import os
from learn2learn.utils.lightning import EpisodicBatcher
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import dgl
from Drug_ban.utils import integer_label_protein
from Meta_utils.CrossFewshotData import *
def PartitionByProtein(data,data_root,base_size=0.5,train_size=0.8,val_size=0.1,test_size=0.1):
    assert test_size+train_size+val_size == 1
    total_size = len(data)
    base_size = int(total_size * base_size)
    total_size = total_size-base_size
    train_size,val_size,test_size = int(total_size * train_size),int(total_size * val_size),int(total_size * test_size)
    grouped_data = data.groupby('Category')
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

    base_category = list(n_samples_per_category.loc[n_samples_per_category['cumulative_count']<=base_size,'Category'])
    train_category = list(n_samples_per_category.loc[(base_size + train_size >= n_samples_per_category['cumulative_count']) &( n_samples_per_category['cumulative_count'] > base_size), 'Category'])
    test_category = list(n_samples_per_category.loc[
         (base_size + train_size + val_size >= n_samples_per_category['cumulative_count']) &(n_samples_per_category['cumulative_count'] >base_size + train_size),'Category'])
    validation_category = list(set(n_samples_per_category.loc[:,'Category'])-set(train_category)-set(test_category)-set(base_category))
    base = data[data['Category'].isin(base_category)]
    val = data[data['Category'].isin(validation_category)]
    test = data[data['Category'].isin(test_category)]
    train = data[data['Category'].isin(train_category)]
    base.to_csv(os.path.join(data_root,'base.csv'),index=False)
    train.to_csv(os.path.join(data_root,'train.csv'), index=False)
    test.to_csv(os.path.join(data_root,'test.csv'),index=False)
    val.to_csv(os.path.join(data_root,'val.csv'), index=False)

class DTIDataset(Dataset):
    def __init__(self, data_csv,domain_index=3, max_drug_nodes=290):

        self.df = pd.read_csv(data_csv)
        if domain_index == 3:
            value_counts = self.df['target_cluster'].value_counts()
        #
        # 过滤出现次数少于3次的值的索引
            indices_to_remove = value_counts[value_counts <= 3].index
        #
        # # 根据条件删除行
            self.df = self.df[~self.df['target_cluster'].isin(indices_to_remove)].reset_index()
        else:
            value_counts = self.df['drug_cluster'].value_counts()
            #
            # 过滤出现次数少于3次的值的索引
            indices_to_remove = value_counts[value_counts <= 5].index
            #
            # # 根据条件删除行
            self.df = self.df[~self.df['drug_cluster'].isin(indices_to_remove)].reset_index()
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
        trg_cluster = self.df.iloc[index]['target_cluster']
        drug_cluster = self.df.iloc[index]['drug_cluster']
        return v_d, v_p, y, trg_cluster,drug_cluster




class MyTaskset(l2l.data.Taskset):
    def __init__(self, dataset, task_transforms=None, num_tasks=-1, task_collate=None):
        super(MyTaskset, self).__init__(dataset, task_transforms=task_transforms, num_tasks=num_tasks, task_collate=task_collate)

    def sample_task_description(self):
        #  Samples a new task description.
        # cdef list description = fast_allocate(len(self.dataset))
        description = None
        if callable(self.task_transforms):
            return self.task_transforms(description)
        for transform in self.task_transforms:
            description = transform(description)
        return description

    def get_task(self, task_description):
        #  Given a task description, creates the corresponding batch of data.
        all_data = []
        for data_description in task_description:
            data = data_description.index
            for transform in data_description.transforms:
                data = transform(data)
            all_data.append(data)
        return self.task_collate(all_data)

    def sample(self):
        """
        **Description**

        Randomly samples a task from the TaskDataset.

        **Example**
        ~~~python
        X, y = taskset.sample()
        ~~~
        """
        i = random.randint(0, len(self) - 1)
        return self[i]

    def __len__(self):
        if self.num_tasks == -1:
            # Ok to return 1, since __iter__ will run forever
            # and __getitem__ will always resample.
            return 1
        return self.num_tasks

    def __getitem__(self, i):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())
        if i not in self.sampled_descriptions:
            self.sampled_descriptions[i] = self.sample_task_description()
        task_description = self.sampled_descriptions[i]
        return self.get_task(task_description)

    def __iter__(self):
        self._task_id = 0
        return self

    def __next__(self):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())

        if self._task_id < self.num_tasks:
            task = self[self._task_id]
            self._task_id += 1
            return task
        else:
            raise StopIteration

    def __add__(self, other):
        msg = 'Adding datasets not yet supported for TaskDatasets.'
        raise NotImplementedError(msg)

def graph_collate_func(x):
    d, p, y,c,d_c = zip(*x)
    y = torch.tensor(y)
    sort = torch.sort(y)
    c = torch.tensor(c)
    d_c = torch.tensor(d_c)
    p = torch.tensor(np.array(p))
    d = list(d)
    d = [d[i] for i in sort.indices.tolist()]
    p = p[sort.indices]
    d = dgl.batch(d)
    return d, p, y,c,d_c, sort.indices

def graph_collate_func_maml(x):
    d, p, y,c,d_c = zip(*x)
    y = torch.tensor(y,dtype=torch.int32)
    sort = torch.sort(y)
    c = torch.tensor(c,dtype=torch.float32)
    d_c = torch.tensor(d_c,dtype=torch.float32)
    p = torch.tensor(np.array(p),dtype=torch.float32)
    d = list(d)
    d = [d[i] for i in sort.indices.tolist()]
    p = p[sort.indices]
    #d = dgl.batch(d)
    return d, p, y,c,d_c, sort.indices

def get_domaindataset(data_csv,maml=False,domian_ways=1,ways=2,query=1,shot=2,num_tasks=100,label_index=2,domain_index=3):

    dataset = DTIDataset(data_csv,domain_index)

    dataset = CrossMetaDataset(dataset,label_index=label_index,domain_index=domain_index)
    task_transforms = [
                          FusedNwaysKShotsCross(dataset, domain_n=domian_ways, n=ways, k=query+shot),
                          l2l.data.transforms.LoadData(dataset),
                          l2l.data.transforms.ConsecutiveLabels(dataset),
                      ]
    if maml:
        data = l2l.data.Taskset(dataset, task_transforms=task_transforms, num_tasks=num_tasks,
                                task_collate=graph_collate_func_maml)
    else:
        data = l2l.data.Taskset(dataset, task_transforms=task_transforms,num_tasks=num_tasks,task_collate=graph_collate_func)
    return data


class MyEpisodicBatcher(EpisodicBatcher):

    """
    nc
    """

    def __init__(self, train_tasks, validation_tasks=None, test_tasks=None, train_epoch_length=1,val_epoch_length=1, test_epoch_length=1):
        super().__init__(train_tasks, validation_tasks, test_tasks)
        self.train_tasks = train_tasks
        if validation_tasks is None:
            validation_tasks = train_tasks
        self.validation_tasks = validation_tasks
        if test_tasks is None:
            test_tasks = validation_tasks
        self.test_tasks = test_tasks
        self.train_epoch_length = train_epoch_length
        self.test_epoch_length = test_epoch_length
        self.val_epoch_length = val_epoch_length

    @staticmethod
    def epochify(taskset, epoch_length):
        class Epochifier(object):
            def __init__(self, tasks, length):
                self.tasks = tasks
                self.length = length

            def __getitem__(self, *args, **kwargs):
                return self.tasks.sample()

            def __len__(self):
                return self.length

        return Epochifier(taskset, epoch_length)

    def train_dataloader(self):
        return EpisodicBatcher.epochify(
            self.train_tasks,
            self.train_epoch_length,
        )

    def val_dataloader(self):
        return EpisodicBatcher.epochify(
            self.validation_tasks,
            self.val_epoch_length,
        )

    def test_dataloader(self):
        return EpisodicBatcher.epochify(
            self.test_tasks,
            self.test_epoch_length,
        )



def get_datamoudle(train_config):
    def get_split_dataset_filename(split):
        return split + ".csv"
    dataset_name = os.path.join(train_config.data_root, train_config.dataset_name)
    train_filename = os.path.join(dataset_name,get_split_dataset_filename("target_train"))

    #valid_filename = os.path.join(dataset_name,get_split_dataset_filename("target_test"))
    maml = train_config.method == 'maml'
    test_filename = os.path.join(dataset_name,get_split_dataset_filename("target_test"))
    train = get_domaindataset(train_filename,maml=maml,domian_ways=1,ways=train_config.train_ways,query=train_config.train_queries,
                              shot=train_config.train_shots,num_tasks=train_config.num_tasks,label_index=train_config.label_index,domain_index=train_config.domain_index)
    test = get_domaindataset(test_filename,maml=maml,domian_ways=1,ways=train_config.test_ways,query=train_config.test_queries,
                             shot=train_config.test_shots,num_tasks=train_config.num_tasks,label_index=train_config.label_index,domain_index=train_config.domain_index)
    train_epoch_length = int(len(train.dataset)/1000)*train_config.batch_size
    val_epoch_length =  int(len(test.dataset)/1000)*train_config.batch_size
    if val_epoch_length < 100:
        val_epoch_length = 128
    test_epoch_length = int(len(test.dataset)/train_config.batch_size)*(train_config.batch_size*2)
    datamoudle = MyEpisodicBatcher(
        train,
        test,
        test,
        train_epoch_length=train_epoch_length,
        val_epoch_length = val_epoch_length,
        test_epoch_length=test_epoch_length
    )

    return datamoudle
