import dgl
import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from rdkit import Chem
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

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):

        self.max_drug_nodes = max_drug_nodes
        df = df[['SMILES', 'Protein','Y']]
        df = df.dropna(subset=['SMILES'])
        df = df.dropna(subset=['Protein'])
        df['isomeric_smiles'] = df['SMILES'].apply(
            lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        df_good = df.dropna(subset=['isomeric_smiles'])# TODO - Check why some rows are na
        self.df = df_good
        self.df = self.df.reset_index(drop=True)# TODO - Check why some rows are na
        self.list_IDs = self.df.index.values
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['isomeric_smiles']

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

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        # y = torch.Tensor([y])
        return v_d, v_p, y


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
