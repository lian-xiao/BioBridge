from typing import Union
import torch
import torch.nn as nn
from fast_transformers.events import QKVEvent

import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer,gelu
from fast_transformers.masking import LengthMask as LM, FullMask, LengthMask
import torch.nn as nn



class Pred_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks, dropout=0.1):
        super().__init__()
        self.desc_skip_connection = True
        self.num_tasks = num_tasks
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x_out = self.fc1(x)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + x

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)
        return torch.sigmoid(z)


class Dti_Cross_Model(nn.Module):

    def __init__(self, molformer, esm2, config):
        super().__init__()
        self.molformer = molformer
        self.esm2 = esm2
        self.start_finetune_layer = config.start_finetune_layer
        self.add_p2 = config.add_p2
        self.add_p1 = config.add_p1
        self.p1_hidden_size = config.p1_hidden_size
        self.p2_hidden_size = config.p2_hidden_size
        self.config = config


        esm2_hidden_list = []
        molformer_hidden_list = []

        for idx_layer, my_blk in enumerate(self.molformer.blocks.layers):
            if idx_layer>= self.start_finetune_layer:
                hidden_d_size = my_blk.linear1.in_features
                molformer_hidden_list.append(hidden_d_size)

        for idx_layer, my_blk in enumerate(self.esm2.layers):
            if idx_layer >= self.start_finetune_layer:
                hidden_d_size = my_blk.embed_dim
                esm2_hidden_list.append(hidden_d_size)

        self.mol_linear = nn.Linear(molformer_hidden_list[-1],config.hidden_size)
        self.protein_linear = nn.Linear(esm2_hidden_list[-1], config.hidden_size)

        self.pred_net = Pred_net(config.hidden_size,config.hidden_size,2,0.1)


    def load_from_pretrain(self, pretrain_file='Pretrain_checkpoints/MolFormer.ckpt'):
        checkpoint = torch.load(pretrain_file)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

    def freeze_backbone(self):

        self.esm2.requires_grad_(False)
        self.molformer.requires_grad_(False)

    def forward(self, mol, protein,mol_mask):

        #ems2
        assert protein.ndim == 2
        padding_mask = protein.eq(self.esm2.padding_idx)  # B, T
        x_protein = self.esm2.embed_scale * self.esm2.embed_tokens(protein)
        if self.esm2.token_dropout:
            x_protein.masked_fill_((protein == self.esm2.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (protein == self.esm2.mask_idx).sum(-1).to(x_protein.dtype) / src_lengths
            x_protein = x_protein * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        if padding_mask is not None:
            x_protein = x_protein * (1 - padding_mask.unsqueeze(-1).type_as(x_protein))
        # (B, T, E) => (T, B, E)
        x_protein = x_protein.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None

        #molformer
        token_embeddings = self.molformer.tok_emb(mol)  # each index maps to a (learnable) vector
        x_mol = self.molformer.drop(token_embeddings)
        # Normalize the masks
        N = x_mol.shape[0]
        L = x_mol.shape[1]
        attn_mask = None
        length_mask = LM(mol_mask.sum(-1)) or \
            LengthMask(x_mol.new_full((N,), L, dtype=torch.int64))


        for i in range(len(self.molformer.blocks.layers)):
            if i>self.start_finetune_layer:
                # (T, B, E) => (B, T, E)
                x_protein = x_protein.transpose(0, 1)

                if self.add_p1:
                    res_protein = self.protein_adapter_blocks_p1_turn1[i-self.start_finetune_layer](x_protein)
                    res_mol = self.mol_adapter_blocks_p1_turn1[i-self.start_finetune_layer](x_mol)
                    res_protein = self.protein_adapter_blocks_p1[i-self.start_finetune_layer](res_protein,res_mol)
                    res_mol = self.mol_adapter_blocks_p1[i-self.start_finetune_layer](res_mol,res_protein)
                    res_protein = self.protein_adapter_blocks_p1_turn2[i-self.start_finetune_layer](res_protein)
                    res_mol = self.mol_adapter_blocks_p1_turn2[i-self.start_finetune_layer](res_mol)
                else:
                    res_protein = x_protein#(B, T, E)
                    res_mol = x_mol
                x_protein = x_protein.transpose(0, 1)# (B, T, E) => (T, B, E)
                x_protein = self.esm2.layers[i].self_attn_layer_norm(x_protein)
                x_protein,_= self.esm2.layers[i].self_attn(query=x_protein,key=x_protein,value=x_protein,key_padding_mask=padding_mask)
                x_protein = x_protein + res_protein.transpose(0, 1)#(T, B, E)

                N, L, _ = x_mol.shape
                _, S, _ = x_mol.shape
                H = self.molformer.blocks.layers[i].attention.n_heads
                x_mol_queries = self.molformer.blocks.layers[i].attention.query_projection(x_mol).view(N, L, H, -1)
                x_mol_keys = self.molformer.blocks.layers[i].attention.key_projection(x_mol).view(N, S, H, -1)
                x_mol_values = self.molformer.blocks.layers[i].attention.value_projection(x_mol).view(N, S, H, -1)
                self.molformer.blocks.layers[i].attention.event_dispatcher.dispatch(QKVEvent(self,x_mol_queries, x_mol_keys, x_mol_values))
                temp_attn_mask = attn_mask or FullMask(L, device=x_mol.device)
                x_mol = self.molformer.blocks.layers[i].attention.inner_attention(
                    x_mol_queries,
                    x_mol_keys,
                    x_mol_values,
                    temp_attn_mask,
                    length_mask,
                    length_mask
                ).view(N, L, -1)
                x_mol = x_mol+res_mol

                x_protein = x_protein.transpose(0, 1)
                if self.add_p2:
                    res_protein = self.protein_adapter_blocks_p2_turn1[i-self.start_finetune_layer](x_protein)
                    res_mol = self.mol_adapter_blocks_p2_turn1[i-self.start_finetune_layer](x_mol)
                    res_protein = self.protein_adapter_blocks_p2[i-self.start_finetune_layer](res_protein,res_mol)
                    res_mol = self.mol_adapter_blocks_p2[i-self.start_finetune_layer](res_mol,res_protein)
                    res_protein = self.protein_adapter_blocks_p2_turn2[i-self.start_finetune_layer](res_protein)
                    res_mol = self.mol_adapter_blocks_p2_turn2[i-self.start_finetune_layer](res_mol)
                else:
                    res_protein = x_protein
                    res_mol = x_mol
                x_protein = x_protein.transpose(0, 1)
                x_protein = self.esm2.layers[i].final_layer_norm(x_protein)
                x_protein = gelu(self.esm2.layers[i].fc1(x_protein))
                x_protein = self.esm2.layers[i].fc2(x_protein)
                x_protein = res_protein.transpose(0, 1) + x_protein

                x_mol = res_mol + self.molformer.blocks.layers[i].attention.out_projection(x_mol)

            else:
                #esm2
                x_protein, attn_protein = self.esm2.layers[i](
                    x_protein,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=False,
                )
                #molformer
                x_mol = self.molformer.blocks.layers[i](x_mol, attn_mask=attn_mask, length_mask=length_mask)

        x_protein = self.esm2.emb_layer_norm_after(x_protein)
        x_protein = x_protein.transpose(0, 1)  # (T, B, E) => (B, T, E)

        x_mol = self.molformer.blocks.norm(x_mol)

        x_mol = self.mol_linear(x_mol)
        x_protein = self.protein_linear(x_protein)

        x = torch.cat((x_mol, x_protein), dim=-2)
        x = torch.mean(x,dim=1)
        x = self.pred_net(x)
        return x

