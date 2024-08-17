import os
from argparse import Namespace
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Drug_ban.utils import graph_collate_func
from Drug_ban.dataloader import DTIDataset, MultiDataLoader


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.dataset_name = hparams.dataset_name
        self.source_generator = None
        self.target_generator = None
        self.test_generator = None
        self.val_generator = None
    def prepare_data(self):
        if self.hparams.Da:

            train_source_path = os.path.join(self.hparams.dataFolder, 'train.csv')
            train_target_path = os.path.join(self.hparams.dataFolder, 'val.csv')
            test_target_path = os.path.join(self.hparams.dataFolder, 'test.csv')
            df_train_source = pd.read_csv(train_source_path)
            df_train_target = pd.read_csv(train_target_path)
            df_test_target = pd.read_csv(test_target_path)

            train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
            train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
            test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)
            params = {'batch_size': 64, 'shuffle': False, 'num_workers': 24, 'drop_last': True,
                      'collate_fn': graph_collate_func}
            self.source_generator = DataLoader(train_dataset, **params)
            self.target_generator = DataLoader(train_target_dataset, **params)
            n_batches = max(len(self.source_generator), len(self.target_generator))
            self.training_generator = MultiDataLoader(dataloaders=[self.source_generator, self.target_generator], n_batches=n_batches)
            self.test_generator = DataLoader(test_target_dataset, **params)
            self.val_generator = DataLoader(test_target_dataset, **params)
        else:
            train_path = os.path.join(self.hparams.dataFolder, 'train.csv')
            val_path = os.path.join(self.hparams.dataFolder, "val.csv")
            test_path = os.path.join(self.hparams.dataFolder, "test.csv")
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)

            train_dataset = DTIDataset(df_train.index.values, df_train)
            val_dataset = DTIDataset(df_val.index.values, df_val)
            test_dataset = DTIDataset(df_test.index.values, df_test)
            params = {'batch_size': self.hparams.batch_size, 'shuffle': True, 'num_workers': self.hparams.num_workers,
                      'drop_last': True, 'collate_fn': graph_collate_func}
            self.training_generator = DataLoader(train_dataset, **params)
            params['shuffle'] = False
            params['drop_last'] = False
            self.val_generator = DataLoader(val_dataset, **params)
            self.test_generator = DataLoader(test_dataset, **params)
        #

    def val_dataloader(self):
        return self.val_generator

    def train_dataloader(self):
        return self.training_generator

    def test_dataloader(self):
        return self.test_generator
