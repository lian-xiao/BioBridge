import pickle

import numpy as np
import pandas as pd
import torch
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
#from learn2learn.data.samplers import TasksetSampler
from learn2learn.data import DataDescription
from torch.utils.data import Dataset
import learn2learn as l2l
import os
from learn2learn.utils.lightning import EpisodicBatcher
from learn2learn.data.transforms import NWays
from collections import defaultdict
import random

from tqdm import tqdm


class CrossMetaDataset(l2l.data.MetaDataset):
    def __init__(self, dataset, labels_to_indices=None, indices_to_labels=None,label_index=1,domain_index =2):
        self.label_index = label_index
        self.domain_index = domain_index
        self.dataset = dataset
        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping(
                labels_to_indices=labels_to_indices,
                indices_to_labels=indices_to_labels,
            )

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self, labels_to_indices=None, indices_to_labels=None,true_labels_to_indices=None,indices_to_true_labels=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)
            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
            labels_to_indices = defaultdict(list)
            true_labels_to_indices = defaultdict(list)
            indices_to_labels = defaultdict(int)
            indices_to_true_labels = defaultdict(int)

            for i in tqdm(range(len(self.dataset))):
                try:
                    label = self.dataset[i][self.domain_index]
                    true_label = self.dataset[i][self.label_index]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(label, 'item'):
                        label = self.dataset[i][self.domain_index].item()
                    if hasattr(true_label, 'item'):
                        true_label = self.dataset[i][self.label_index].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))

                labels_to_indices[label].append(i)
                true_labels_to_indices[true_label].append(i)
                indices_to_labels[i] = label
                indices_to_true_labels[i] = true_label

        self.labels_to_indices = labels_to_indices
        self.true_labels_to_indices = true_labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.indices_to_true_labels = indices_to_true_labels
        self.labels = list(self.labels_to_indices.keys())
        self.true_labels = list(self.true_labels_to_indices.keys())
        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels,
            'true_labels_to_indices': self.true_labels_to_indices,
            'indices_to_true_labels': self.indices_to_true_labels,
            'true_labels': self.true_labels
        }

    def load_bookkeeping(self, path):
        if not os.path.exists(path):
            self.create_bookkeeping()
            self.serialize_bookkeeping(path)
        else:
            with open(path, 'rb') as f:
                self._bookkeeping = pickle.load(f)
            self.labels_to_indices = self._bookkeeping['labels_to_indices']
            self.indices_to_labels = self._bookkeeping['indices_to_labels']
            self.labels = self._bookkeeping['labels']
            self.true_labels_to_indices = self._bookkeeping['true_labels_to_indices']
            self.indices_to_true_labels = self._bookkeeping['indices_to_true_labels']
            self.true_labels = self._bookkeeping['true_labels']

    def serialize_bookkeeping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._bookkeeping, f, protocol=-1)


class FusedNwaysKShotsCross(l2l.data.transforms.FusedNWaysKShots):
    def __init__(self, dataset, domain_n=1,n=2, k=1, replacement=False, filter_labels=None):
        super(FusedNwaysKShotsCross, self).__init__(
            dataset,
            n=n,
            k=k,
            replacement=replacement,
            filter_labels=filter_labels,
        )
        self.domain_n = domain_n

    def new_task(self):
        task_description = []
        domain_labels = self.filter_labels
        labels = self.dataset.true_labels
        selected_domian_labels = random.sample(domain_labels, k=self.domain_n)
        #首先选择使用几个域的数据
        for sl in selected_domian_labels:
            domain_indices = self.dataset.labels_to_indices[sl]
            selected_labels = random.sample(labels,k=self.n)
            selected_indices = []
            # 在该域下的每个类别取kshot
            for l in selected_labels:
                label_indices = self.dataset.true_labels_to_indices[l]
                indices = list(set(domain_indices) & set(label_indices))
                if self.replacement:
                    selected_indices = selected_indices + [random.choice(indices) for _ in range(self.k)]
                else:
                    if len(indices)>=self.k:
                        selected_indices = selected_indices + random.sample(indices, k=self.k)
                    else:
                        selected_indices = selected_indices + random.choices(indices, k=self.k)
                        # 如果当前的类别的样本数小于k，则进行重复采样
            for idx in selected_indices:
                task_description.append(DataDescription(idx))
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            return self.new_task()
        # Not fused
        return self.kshots(self.nways(self.filter(task_description)))
#
# NUM_TASKS = 10
# NUM_DATA = 128
# X_SHAPE = 16
# Y_SHAPE = 10
# EPSILON = 1e-6
# SUBSET_SIZE = 5
# WORKERS = 4
# META_BSZ = 16
# data = torch.randn(NUM_DATA, X_SHAPE)
# labels = torch.randint(0, Y_SHAPE, (NUM_DATA,))
# true_labels = torch.randint(0,2,(NUM_DATA,))
# dataset = torch.utils.data.TensorDataset(data,true_labels,labels)
# dataset = CrossMetaDataset(dataset,label_index=1,domain_index=2)
# taskset = l2l.data.Taskset(
#     dataset,
#     task_transforms=[
#         FusedNwaysKShotsCross(dataset, domain_n=1,n=2, k=2),
#         l2l.data.transforms.LoadData(dataset),
#         l2l.data.transforms.RemapLabels(dataset),
#         #l2l.data.transforms.ConsecutiveLabels(dataset),
#     ],
#     num_tasks=NUM_TASKS,
# )
#
# sampler = TasksetSampler(taskset)
# dataloader = torch.utils.data.DataLoader(
#     dataset=dataset,
#     batch_sampler=sampler,
# )
# for task in dataloader:
#     print(task)
