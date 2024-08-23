import os
from argparse import Namespace
import pandas as pd
import torch
from rdkit import Chem
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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


def get_dataset(data_root, filename, dataset_len, measure_name,canonical):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df, measure_name, canonical)
    return dataset


class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, measure_name,canonical=True):
        df = df[['SMILES','Protein'] + measure_name]
        self.measure_name = measure_name
        df = df.dropna(subset=['SMILES'])
        df = df.dropna(subset = ['Protein'])
        df = df.sample(frac=1)
        df['isomeric_smiles'] = df['SMILES'].apply(lambda smi: normalize_smiles(smi, canonical=canonical, isomeric=False))
        df = df[df['Protein'].str.len() <= 2024]
        df_good = df.dropna(subset=['isomeric_smiles'])  # TODO - Check why some rows are na
        self.df = df_good
        self.df = self.df.reset_index(drop=True)

    def __getitem__(self, index):
        canonical_smiles = self.df.loc[index, 'isomeric_smiles']
        proteins = self.df.loc[index, 'Protein']
        measures = self.df.loc[index, self.measure_name].values.tolist()
        return canonical_smiles, proteins,measures

    def __len__(self):
        return len(self.df)


class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer, alphabet):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        #self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter(800)
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(self, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "val"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            train_filename,
            self.hparams.train_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=self.hparams.canonical,
        )

        val_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            valid_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical = self.hparams.canonical,
        )

        test_ds = get_dataset(
            os.path.join(self.hparams.data_root, self.dataset_name),
            test_filename,
            self.hparams.eval_dataset_length,
            measure_name=self.hparams.measure_name,
            canonical=self.hparams.canonical,
        )

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds

    def collate(self, batch):
        smiles = []
        proteins = []
        labels = []
        for idx,(smile,protein,label) in enumerate(batch):
            smiles.append(smile)
            proteins.append((f'protens{idx}',protein))
            labels.append(label)

        smiles_tokens = self.tokenizer.batch_encode_plus(smiles, padding=True, add_special_tokens=True)
        _,_,proteins_tokens = self.batch_converter(proteins)

        return torch.tensor(smiles_tokens['input_ids']),torch.tensor(smiles_tokens['attention_mask']),\
               proteins_tokens, torch.tensor(labels)

        #

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )