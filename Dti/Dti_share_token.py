import re
from typing import Union
import torch
import torch.nn as nn
from fast_transformers.events import QKVEvent
from torch.nn.utils import weight_norm

from Drug_ban.ban import BANLayer
from My_nets.utils_net import Residual_Net
import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer, gelu
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
        self.final = nn.Linear(hidden_dim, 2)

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
        return torch.softmax(z, dim=-1)


class Dti_ShareT_Model(nn.Module):

    def __init__(self, molformer, esm2, config,gpu_tracker=None):
        super().__init__()
        self.molformer = molformer
        self.esm2 = esm2
        self.start_finetune_layer = config.start_finetune_layer
        self.num_tokens = config.num_tokens  # number of prompted tokens
        self.tokens_dim = config.tokens_dim
        self.noise_lamda = config.noise_lamda
        self.hidden_size = config.hidden_size
        self.gpu_tracker = gpu_tracker
        self.config = config

        esm2_hidden_list = []
        molformer_hidden_list = []
        esm2_trans_list = []
        molformer_trans_list = []
        for idx_layer, my_blk in enumerate(self.molformer.blocks.layers):
            hidden_d_size = my_blk.linear1.in_features
            molformer_hidden_list.append(hidden_d_size)
            if self.num_tokens != 0:

                molformer_trans_list.append(nn.Linear(self.tokens_dim, hidden_d_size))



        for idx_layer, my_blk in enumerate(self.esm2.layers):
            hidden_d_size = my_blk.embed_dim
            esm2_hidden_list.append(hidden_d_size)
            if self.num_tokens != 0:


                esm2_trans_list.append(nn.Linear(self.tokens_dim, hidden_d_size))

            #if idx_layer >= self.start_finetune_layer:


        if self.num_tokens != 0:
            self.share_tokens = nn.Parameter((torch.zeros(
                12, self.num_tokens, self.tokens_dim)))
            nn.init.kaiming_normal_(self.share_tokens)

            self.tokens_linear = nn.Linear(self.tokens_dim, self.hidden_size)
            self.tokens2mol = nn.ModuleList(molformer_trans_list)
            self.tokens2pro = nn.ModuleList(esm2_trans_list)

        self.mol_linear = nn.Sequential(nn.Linear(molformer_hidden_list[-1], self.hidden_size),
                                        nn.LayerNorm(self.hidden_size))
        self.pro_linear = nn.Sequential(nn.Linear(esm2_hidden_list[-1], self.hidden_size),
                                        nn.LayerNorm(self.hidden_size))
        self.bcn = weight_norm(
            BANLayer(v_dim=self.hidden_size, q_dim=self.hidden_size, h_dim=self.hidden_size, h_out=2),
            name='h_mat', dim=None)
        self.fea_linear = Residual_Net(self.hidden_size*2, self.hidden_size, 'gelu')
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.pred_net = Pred_net(self.hidden_size, self.hidden_size, 2, 0.1)




    def load_from_pretrain(self):
        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            return state_dict

        checkpoint = torch.load('pretrain_checkpoint/MolFormer/MolFormer.ckpt')
        self.molformer.load_state_dict(checkpoint['state_dict'], strict=False)
        checkpoint = torch.load('pretrain_checkpoint/esm2_t12_35M_UR50D/esm2_t12_35M_UR50D.pt')
        state_dict = checkpoint["model"]
        state_dict = upgrade_state_dict(state_dict)
        self.esm2.load_state_dict(state_dict,strict=False)

    def freeze_backbone(self):
        self.load_from_pretrain()
        print('esm2 and molformer have been loaded')

        for idx_layer, my_blk in enumerate(self.molformer.blocks.layers):
            if idx_layer < self.start_finetune_layer:
                my_blk.requires_grad_(False)

        for idx_layer, my_blk in enumerate(self.esm2.layers):
            if idx_layer < self.start_finetune_layer:
                my_blk.requires_grad_(False)
        print('esm2 and molformer have been frozen')
        # self.esm2.requires_grad_(False)
        # self.molformer.requires_grad_(False)

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        B, N, C = size
        feat_var = feat.contiguous().reshape(B, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.contiguous().view(B, -1).mean(dim=1)
        return feat_mean, feat_std

    def add_noise(self, x_protein, x_mol):
        protein_mean, protein_std = self.calc_mean_std(x_protein)
        protein_mean = torch.nn.Parameter(protein_mean).requires_grad_()
        protein_std = torch.nn.Parameter(protein_std).requires_grad_()

        mol_mean, mol_std = self.calc_mean_std(x_protein)
        mol_mean = torch.nn.Parameter(mol_mean).requires_grad_()
        mol_std = torch.nn.Parameter(mol_std).requires_grad_()
        protein_noise = protein_mean.clone().detach().requires_grad_(True).unsqueeze(-1).unsqueeze(
            -1) + protein_std.clone().detach().requires_grad_(True).unsqueeze(-1).unsqueeze(-1) * torch.randn(
            x_mol.size()).to(x_mol)
        mol_noise = mol_mean.clone().detach().requires_grad_(True).unsqueeze(-1).unsqueeze(
            -1) + mol_std.clone().detach().requires_grad_(True).unsqueeze(-1).unsqueeze(-1) * torch.randn(
            x_protein.size()).to(x_protein)
        x_mol = x_mol + self.noise_lamda * protein_noise
        x_protein = x_protein + self.noise_lamda * mol_noise
        return x_protein, x_mol

    def before_in_layers(self, mol, protein, mol_mask):
        # ems2
        if self.gpu_tracker:
            self.gpu_tracker.track()
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
        # if not padding_mask.any():
        #     padding_mask = None
        # molformer
        token_embeddings = self.molformer.tok_emb(mol)  # each index maps to a (learnable) vector
        x_mol = self.molformer.drop(token_embeddings)
        # Normalize the masks
        N = x_mol.shape[0]
        L = x_mol.shape[1]
        attn_mask = None
        length_mask = LM(mol_mask.sum(-1)) or \
                      LengthMask(x_mol.new_full((N,), L, dtype=torch.int64))
        mol_mask_tokens = torch.cat((mol_mask[:, :1],
                                     torch.ones((mol_mask.size(0), self.num_tokens), dtype=torch.int64).to(mol_mask),
                                     mol_mask[:, 1:]), dim=1)
        length_mask_tokens = LM(mol_mask_tokens.sum(-1)) or \
                             LengthMask(x_mol.new_full((N,), L, dtype=torch.int64))

        return x_protein, x_mol, padding_mask, length_mask, attn_mask, mol_mask_tokens, length_mask_tokens

    def forward_normal(self, mol, protein, mol_mask):
        x_protein, x_mol, padding_mask, length_mask, attn_mask, mol_mask_tokens, length_mask_tokens = self.before_in_layers(
            mol, protein, mol_mask)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        for i in range(len(self.molformer.blocks.layers)):
            # esm2
            x_protein, attn_protein = self.esm2.layers[i](
                x_protein,
                self_attn_padding_mask=padding_mask,
                need_head_weights=False,
            )
            # molformer
            x_mol = self.molformer.blocks.layers[i](x_mol, attn_mask=attn_mask, length_mask=length_mask)
            if self.gpu_tracker:
                self.gpu_tracker.track()
        x_protein = self.esm2.emb_layer_norm_after(x_protein)
        x_protein = x_protein.transpose(0, 1)  # (T, B, E) => (B, T, E)

        x_mol = self.molformer.blocks.norm(x_mol)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        # x_mol = self.mol_linear(torch.mean(x_mol,dim=1))
        # x_protein = self.pro_linear(torch.mean(x_protein,dim=1))
        x_mol = self.mol_linear(x_mol)
        x_protein = self.pro_linear(x_protein)

        #x = torch.cat((x_mol, x_protein), dim=2)
        #x = self.fea_linear(x)
        #x = torch.mean(x, dim=1)
        x,attn = self.bcn(x_mol,x_protein)
        x = self.bn(x)
        x = self.pred_net(x)
        if self.gpu_tracker:
            self.gpu_tracker.track()
        return x

    def forward_shareT(self, mol, protein, mol_mask, mode='train'):
        x_protein, x_mol, padding_mask, length_mask, attn_mask, mol_mask_tokens, length_mask_tokens = self.before_in_layers(
            mol, protein, mol_mask)
        for i in range(len(self.molformer.blocks.layers)):
            x_protein = x_protein.transpose(0, 1)
            if mode == 'train' and self.noise_lamda != 0:
                x_protein, x_mol = self.add_noise(x_protein, x_mol)

            # share tokens
            if i == 0:
                x_mol = torch.cat((x_mol[:, :1, :], self.tokens2mol[i](
                    self.share_tokens[i].expand(x_mol.shape[0], -1, -1)).to(x_mol),
                                   x_mol[:, 1:, :]), dim=1)
                x_protein = torch.cat((x_protein[:, :1, :], self.tokens2pro[i](
                    self.share_tokens[i].expand(x_protein.shape[0], -1, -1)).to(
                    x_protein),
                                       x_protein[:, 1:, :]), dim=1)
            else:
                x_mol = torch.cat((x_mol[:, :1, :], self.tokens2mol[i](
                    self.share_tokens[i].expand(x_mol.shape[0], -1, -1)).to(x_mol),
                                   x_mol[:, (1 + self.num_tokens):, :]), dim=1)
                x_protein = torch.cat((x_protein[:, :1, :], self.tokens2pro[i](
                    self.share_tokens[i].expand(x_protein.shape[0], -1, -1)).to(
                    x_protein),
                                       x_protein[:, 1 + self.num_tokens:, :]), dim=1)

            x_protein = x_protein.transpose(0, 1)
            res_protein = x_protein
            res_mol = x_mol

            x_protein = self.esm2.layers[i].self_attn_layer_norm(x_protein)

            padding_mask_tokens = torch.cat((padding_mask[:, :1],
                                             torch.zeros((padding_mask.size()[0], self.num_tokens),
                                                         dtype=torch.bool).to(padding_mask), padding_mask[:, 1:]),
                                            dim=1)
            x_protein, _ = self.esm2.layers[i].self_attn(query=x_protein, key=x_protein, value=x_protein,
                                                         key_padding_mask=padding_mask_tokens)
            N, L, _ = x_mol.shape
            _, S, _ = x_mol.shape
            H = self.molformer.blocks.layers[i].attention.n_heads
            x_mol_queries = self.molformer.blocks.layers[i].attention.query_projection(x_mol).view(N, L, H, -1)
            x_mol_keys = self.molformer.blocks.layers[i].attention.key_projection(x_mol).view(N, S, H, -1)
            x_mol_values = self.molformer.blocks.layers[i].attention.value_projection(x_mol).view(N, S, H, -1)
            self.molformer.blocks.layers[i].attention.event_dispatcher.dispatch(
                QKVEvent(self, x_mol_queries, x_mol_keys, x_mol_values))
            temp_attn_mask = attn_mask or FullMask(L, device=x_mol.device)
            x_mol = self.molformer.blocks.layers[i].attention.inner_attention(
                x_mol_queries,
                x_mol_keys,
                x_mol_values,
                temp_attn_mask,
                length_mask_tokens,
                length_mask_tokens
            ).view(N, L, -1)

            x_mol = x_mol + res_mol
            x_protein = x_protein + res_protein  # (T, B, E)
            # x_protein = x_protein.transpose(0, 1)

            res_protein = x_protein
            res_mol = x_mol
            x_protein = x_protein
            x_protein = self.esm2.layers[i].final_layer_norm(x_protein)
            x_protein = gelu(self.esm2.layers[i].fc1(x_protein))
            x_protein = self.esm2.layers[i].fc2(x_protein)
            x_protein = res_protein + x_protein

            x_mol = res_mol + self.molformer.blocks.layers[i].attention.out_projection(x_mol)

        x_protein = self.esm2.emb_layer_norm_after(x_protein)
        x_protein = x_protein.transpose(0, 1)  # (T, B, E) => (B, T, E)

        x_mol = self.molformer.blocks.norm(x_mol)

        # x_mol = self.mol_linear(x_mol[:, :1, :])
        # x_protein = self.pro_linear(x_protein[:, :1, :])
        # x = self.tokens_linear(self.share_tokens[-1].expand(x_protein.shape[0], -1, -1))
        # x = torch.cat((x_mol, x_protein, x), dim=1)
        # x = self.fea_linear(x)
        # x = torch.mean(x, dim=1)
        x_mol = self.mol_linear(x_mol)
        x_protein = self.pro_linear(x_protein)

        #x = torch.cat((x_mol, x_protein), dim=2)
        #x = self.fea_linear(x)
        #x = torch.mean(x, dim=1)
        x,attn = self.bcn(x_mol,x_protein)
        x = self.bn(x)
        x = self.pred_net(x)
#        x = self.pred_net(x)
        return x

    def forward(self, mol, protein, mol_mask, mode='train'):
        if self.num_tokens == 0:
            x = self.forward_normal(mol, protein, mol_mask)
        else:
            x = self.forward_shareT(mol, protein, mol_mask, mode)
        return x
