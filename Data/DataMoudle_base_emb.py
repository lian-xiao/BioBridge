import os
from argparse import Namespace

import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial
from Dti.utils import integer_label_protein


def normalize_smiles(smi, canonical, isomeric):
    try:
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveHs(mol)
        normalized = Chem.MolToSmiles(
            mol, canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def get_dataset(data_root, filename, dataset_len=None, measure_name=['Y'], canonical=True, d_emb=None, p_emb=None):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = BEDataset(df, measure_name, canonical, d_emb=d_emb, p_emb=p_emb)
    return dataset


class BEDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name, canonical=True, max_drug_nodes=360, d_emb=None, p_emb=None,sample=False):
        super(BEDataset, self).__init__()
        self.measure_name = measure_name
        if 'Ligand' in df.columns:
            df = df.rename(columns={'Ligand': 'SMILES'})
        if 'regression_label' in df.columns:
            df = df.rename(columns={'regression_label': 'Y'})
        if 'Smiles' in df.columns:
            df = df.rename(columns={'Smiles': 'SMILES'})
        try:
            df = df[['SMILES', 'Protein'] + measure_name]
            df = df.dropna(subset=['SMILES'])
            df = df.dropna(subset=['Protein'])
        except:
            df = df[['SMILES', 'Target Sequence'] + measure_name]
            df = df.dropna(subset=['SMILES'])
            df = df.dropna(subset=['Target Sequence'])
            df = df.rename(columns={'Target Sequence': 'Protein'})
        if sample:
            df = df.sample(frac=1)
        df['isomeric_smiles'] = df['SMILES'].apply(
            lambda smi: normalize_smiles(smi, canonical=canonical, isomeric=False))
        # df = df[df['Protein'].str.len() <= 2024]
        df_good = df.dropna(subset=['isomeric_smiles'])  # TODO - Check why some rows are na
        self.df = df_good
        self.df = self.df.reset_index(drop=True)
        # 如果没有emb则创建graph
        self.d_emb = d_emb
        self.p_emb = p_emb
        if not d_emb:
            self.max_drug_nodes = max_drug_nodes
            self.atom_featurizer = CanonicalAtomFeaturizer()
            self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
            self.fc = partial(smiles_to_bigraph, add_self_loop=True)


    def __getitem__(self, index):
        v_d = self.df.loc[index, 'isomeric_smiles']
        v_p = self.df.loc[index, 'Protein']
        measures = self.df.loc[index, self.measure_name].values.tolist()
        if not self.d_emb:
            v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            if v_d.number_of_nodes() > self.max_drug_nodes:
                v_d = dgl.node_subgraph(v_d, list(range(self.max_drug_nodes)))
                v_d.ndata.pop('_ID')
                v_d.edata.pop('_ID')
            actual_node_feats = v_d.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats[:num_actual_nodes], virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            v_d = v_d.add_self_loop()
        if not self.p_emb:
            v_p = integer_label_protein(v_p)

        return v_d, v_p, measures

    def __len__(self):
        return len(self.df)


class BEDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer, alphabet, max_len=1200):
        super(BEDataModule, self).__init__()
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.alphabet = alphabet

        if self.alphabet:
            self.batch_converter = self.alphabet.get_batch_converter(max_len)
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(self, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = BEDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = BEDataModule.get_split_dataset_filename(
            self.dataset_name, "val"
        )

        test_filename = BEDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            train_filename,
            self.hparams.train_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        val_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            valid_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        test_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            test_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds

    def collate(self, batch):
        smiles, proteins, labels = zip(*batch)
        if self.alphabet:
            proteins_str = []
            for idx, protein in enumerate(proteins):
                proteins_str.append((f'protens{idx}', protein))
            _, _, proteins = self.batch_converter(proteins_str)
        else:
            proteins = torch.Tensor(np.array(proteins))
        if self.tokenizer:
            smiles_tokens = self.tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
            smiles = torch.tensor(smiles_tokens['input_ids'])
        else:
            smiles = dgl.batch(smiles)
        return smiles, proteins, torch.tensor(labels)

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


class BEDataset_predict(BEDataset):
    def __init__(self, df, measure_name, canonical=True, max_drug_nodes=360, d_emb=None, p_emb=None,sample=False):
        super().__init__(df, measure_name, canonical=canonical, max_drug_nodes=max_drug_nodes, d_emb=None, p_emb=None,sample=False)

    def __getitem__(self, index):
        v_d = self.df.loc[index, 'isomeric_smiles']
        v_p = self.df.loc[index, 'Protein']
        #measures = self.df.loc[index, self.measure_name].values.tolist()
        if not self.d_emb:
            v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            if v_d.number_of_nodes() > self.max_drug_nodes:
                v_d = dgl.node_subgraph(v_d, list(range(self.max_drug_nodes)))
                v_d.ndata.pop('_ID')
                v_d.edata.pop('_ID')
            actual_node_feats = v_d.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats[:num_actual_nodes], virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            v_d = v_d.add_self_loop()
        if not self.p_emb:
            v_p = integer_label_protein(v_p)

        return v_d, v_p,index

    def __len__(self):
        return len(self.df)





class BEDataModule_Da(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer, alphabet, max_len=1200):
        super(BEDataModule_Da, self).__init__()
        self.train_target_ds = None
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.alphabet = alphabet
        if self.alphabet:
            self.batch_converter = self.alphabet.get_batch_converter(max_len)
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(self, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = BEDataModule_Da.get_split_dataset_filename(
            self.dataset_name, "source_train"
        )

        valid_filename = BEDataModule_Da.get_split_dataset_filename(
            self.dataset_name, "target_train"
        )

        test_filename = BEDataModule_Da.get_split_dataset_filename(
            self.dataset_name, "target_test"
        )

        train_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            train_filename,
            self.hparams.train_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        train_target_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            valid_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        test_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            test_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=True,
            d_emb=self.tokenizer,
            p_emb=self.alphabet
        )

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_target_ds = train_target_ds

    def collate(self, batch):
        smiles, proteins, labels = zip(*batch)
        if self.alphabet:
            proteins_str = []
            for idx, protein in enumerate(proteins):
                proteins_str.append((f'protens{idx}', protein))
            _, _, proteins = self.batch_converter(proteins_str)
        else:
            proteins = torch.Tensor(np.array(proteins))
        if self.tokenizer:
            smiles_tokens = self.tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
            smiles = torch.tensor(smiles_tokens['input_ids'])
        else:
            smiles = dgl.batch(smiles)
        return smiles, proteins, torch.tensor(labels)

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )
    #
    # def train_dataloader(self):
    #     source_generator = DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
    #                                   num_workers=self.hparams.num_workers,
    #                                   shuffle=True,
    #                                   collate_fn=self.collate, )
    #     target_generator = DataLoader(self.train_target_ds, batch_size=self.hparams.batch_size,
    #                                   num_workers=self.hparams.num_workers,
    #                                   shuffle=True,
    #                                   collate_fn=self.collate, )
    #     n_batches = max(len(source_generator), len(target_generator))
    #     multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
    #     return multi_generator


    # def train_dataloader(self):
    #     source_generator = DataLoader(self.train_ds, batch_size=self.hparams.batch_size,
    #                                   num_workers=self.hparams.num_workers,
    #                                   shuffle=True,
    #                                   collate_fn=self.collate, )
    #     target_generator = DataLoader(self.train_target_ds, batch_size=self.hparams.batch_size,
    #                                   num_workers=self.hparams.num_workers,
    #                                   shuffle=True,
    #                                   collate_fn=self.collate, )
    #     n_batches = max(len(source_generator), len(target_generator))
    #     multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
    #     return multi_generator

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )
