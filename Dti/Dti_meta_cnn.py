from copy import deepcopy

import torch
from dgllife.model.gnn.gcn import GCNLayer
from learn2learn.nn import PrototypicalClassifier
from torch.nn.utils import weight_norm
from dgllife.model.gnn import GCN
import dgl.function as fn
from Drug_ban.ban import BANLayer
from Drug_ban.models import MolecularGCN
from Dti.Dti_cnn import Dti_cnn,Dti_cnn_mutiout,GAU,Dti_DrugBAN
from torch import nn, einsum
import torch.nn.functional as F
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v



class Dti_meta_mulcnn(Dti_cnn_mutiout):
    def __init__(self,d_embedding, p_embedding, config):
        super(Dti_meta_mulcnn, self).__init__(d_embedding, p_embedding, config)

        self.classifier = self.pred
        del self.pred

    def features(self,mol, protein,visual):
        protein = protein.to(torch.float32)
        fs = torch.Tensor([]).to(protein)
        attns = torch.Tensor([]).to(protein)
        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein)
        if self.muti:
            for i in range(len(mol)):
                f,attn = self.bcn[i](mol[i],prot[i])
                fs = torch.cat([fs,f.unsqueeze(1)],dim=1)
                attns = torch.cat([attns,attn.unsqueeze(1)],dim=1)
        else:
            f, attn = self.bcn[0](mol[0], prot[0])
            fs = torch.cat([fs, f.unsqueeze(1)], dim=1)
            attns = torch.cat([attns, attn.unsqueeze(1)], dim=1)
        if visual:
            fs,gate_attn,gate = self.bcn_gate(fs,visual)  # b,3,256
        else:
            fs = self.bcn_gate(fs)#b,3,256

        #fs = self.bcn_gate(fs)
        return torch.mean(fs, dim=1).squeeze(1)


    def forward(self, mol, protein,visual=False):
        fs = self.extract_feat(mol, protein,visual)
        return self.classifier(fs)


class Dti_meta_cnn(Dti_cnn):
    def __init__(self,d_embedding, p_embedding, config):
        super(Dti_meta_cnn, self).__init__(d_embedding, p_embedding, config)
        self.classifier = self.pred

    def features(self, mol, protein,visual=False):
        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein)
        f,attn = self.bcn(mol,prot)
        if visual:
            return f,attn
        else:
            return f

class Dti_meta_drugban(Dti_DrugBAN):
    def __init__(self):
        super(Dti_meta_drugban, self).__init__()
        self.classifier = self.mlp_classifier

    def features(self, bg_d, v_p,visual=False):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        return f


def check_nan(x):
    if torch.isnan(x).any():
        raise ValueError("NaN detected in loss")


class AdaptiveGate(nn.Module):
    def __init__(        self,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qk = nn.Sequential(
            weight_norm(nn.Linear(dim, query_key_dim)),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x)  # (bs,seq_len,dim)
        Z = self.to_qk(normed_x)  # (bs,seq_len,query_key_dim)
        QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len
        # A = sim
        #A = F.relu(sim) ** 2
        # A = self.dropout(A)
        return sim

class AdaptivePrototypicalClassifier(PrototypicalClassifier):
    def __init__(self,
                support=None,
                labels=None,
                distance="euclidean",
                normalize=True):
        super(AdaptivePrototypicalClassifier, self).__init__(support,labels,distance,normalize)
        self.attn = AdaptiveGate(dim=256)

    def fit_(self, support, labels):
        """
        **Description**

        Computes and updates the prototypes given support embeddings and
        corresponding labels.

        """
        # TODO: Make a differentiable version? (For Proto-MAML style algorithms)

        # Compute new prototypes
        prototypes = self._compute_prototypes(support, labels)

        # Normalize if necessary
        if self.normalize:
            prototypes = PrototypicalClassifier.normalize(prototypes)

        # Assign prototypes and return them
        self.prototypes = prototypes
        return prototypes

    def forward(self, x,support,labels,visual=False):

        if self.normalize:
            x = PrototypicalClassifier.normalize(x)
            support = PrototypicalClassifier.normalize(support)
        x = x.unsqueeze(1)
        support = support.unsqueeze(0)
        support_expanded = support.repeat(x.size(0), 1, 1)
        fea = torch.concat([x,support_expanded],dim=1)
        attn = self.attn(fea)

        # 初始化原型数组，假设有 num_classes 个类别
        classes = torch.unique(labels)
        prototypes = torch.zeros(x.size(0), classes.size(0),
            *support.shape[2:], device=support.device, dtype=support.dtype)#(query_num,class_num,fea)
        # 扩展支持集特征以匹配注意力矩阵的批次维度
        #support_expanded = support.unsqueeze(0).repeat(2, 1, 1)  # 维度变为 [2, num_samples, feature_length]
        # 对每个类别计算原型
        attn_querys = torch.Tensor([]).to(attn)
        for i, cls in enumerate(classes):
            # 选择当前类别的样本索引
            class_indices = (labels == cls).nonzero(as_tuple=False).squeeze()  # 获取索引的张量形式
            # 如果当前类别没有样本，则跳过
            if class_indices.numel() == 0:
                continue
            # 对每个查询样本，使用注意力加权当前类别的样本特征
            attn_querys2 = torch.Tensor([]).to(attn)
            for query_idx in range(2):
                attention_query = F.softmax(attn[query_idx, 0, 1:],dim=-1).unsqueeze(1)  # 忽略对自身的注意力
                if support.shape[1] / x.shape[0] == 1.0:
                    weighted_features = support_expanded[query_idx, class_indices] * attention_query[class_indices].unsqueeze(1)
                else:
                    weighted_features = support_expanded[query_idx, class_indices] * attention_query[class_indices]
                # 计算加权特征的均值作为该类别的原型
                prototypes[query_idx, int(cls)].add_(torch.mean(weighted_features, dim=0))
                attn_querys2 = torch.concat([attn_querys2, attention_query],dim=1)
            attn_querys = torch.concat([attn_querys,attn_querys2],dim=-1)
        #F.cosine_similarity(x.expand(-1, 2, -1), prototypes, dim=-1)
        if visual:
            return F.cosine_similarity(prototypes,x.expand(-1, 2, -1),dim=-1),attn_querys
        else:
            return F.cosine_similarity(prototypes, x.expand(-1, 2, -1), dim=-1)
class Stem(nn.Module):
    """
    Use CMTStem module to process input image and overcome the limitation of the
    non-overlapping patches.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (B, C, T)
    Output:
        - result: (B, 128, T)
    """

    def __init__(self, in_channels, out_channels,kernel_size =1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_channels,eps=1e-1,momentum=0.9)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels,eps=1e-1,momentum=0.9)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        result = self.bn2(x)
        return result


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=1, groups=in_channels, bias=True
                                   )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class Aggregate(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(Aggregate, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              stride=2, padding=0, bias=True)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, c, t = x.size()
        result = nn.functional.layer_norm(x, (c, t),eps=1e-1)
        return result


class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """

    def __init__(self, in_channels, R):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, exp_channels, kernel_size=1),
            nn.BatchNorm1d(exp_channels,eps=1e-1,momentum=0.9),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm1d(exp_channels,eps=1e-1,momentum=0.9),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(exp_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels,eps=1e-1,momentum=0.9)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """

    def __init__(self, in_channels, out_channels):
        super(LPU, self).__init__()
        self.Conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        result = self.Conv(x) + x
        return result


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, R=3.6):
        super(Block, self).__init__()
        self.lpu = LPU(in_channels, in_channels)
        self.DWConv1 = DWCONV(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.DWConv2 = DWCONV(out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.DWConv3 = DWCONV(out_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_channels,eps=1e-1,momentum=0.9)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels,eps=1e-1,momentum=0.9)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_channels,eps=1e-1,momentum=0.9)
        self.ffn = IRFFN(out_channels, R)

    def forward(self, x):
        x = self.lpu(x)
        x = self.bn1(self.gelu1(self.DWConv1(x)))
        x = self.bn2(self.gelu2(self.DWConv2(x)))
        x = self.bn3(self.gelu3(self.DWConv3(x)))
        x = self.ffn(x)
        return x


class fea_Cnn(nn.Module):
    def __init__(self, embedding, emb_dim, stem_channel, channels, out_channel, R=3.6):
        super(fea_Cnn, self).__init__()
        if embedding:
            self.embedding = embedding
            emb_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(26, emb_dim)
        self.stem = Stem(emb_dim, stem_channel)
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.aggregate1 = Aggregate(stem_channel, channels[0])
        self.aggregate2 = Aggregate(channels[0], channels[1])
        self.aggregate3 = Aggregate(channels[1], channels[2])
        self.aggregate4 = Aggregate(channels[2], channels[3])

        self.stage1 = Block(channels[0], channels[0], 3, R)
        self.stage2 = Block(channels[1], channels[1], 3, R)
        self.stage3 = Block(channels[2], channels[2], 3, R)
        self.stage4 = Block(channels[3], channels[3], 3, R)

        self.top_layer4 = nn.Conv1d(in_channels=channels[3], out_channels=out_channel, kernel_size=1)
        self.top_layer3 = nn.Conv1d(in_channels=channels[2], out_channels=out_channel, kernel_size=1)
        self.top_layer2 = nn.Conv1d(in_channels=channels[1], out_channels=out_channel, kernel_size=1)
        self.top_layer1 = nn.Conv1d(in_channels=channels[0], out_channels=out_channel, kernel_size=1)

        self.smooth1 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.smooth2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.smooth3 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        # self.smooth4 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(2, 1)
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.aggregate1(x)
        result1 = self.stage1(x)
        x = self.aggregate2(result1)
        result2 = self.stage2(x)
        x = self.aggregate3(result2)
        result3 = self.stage3(x)
        x = self.aggregate4(result3)
        result4 = self.stage4(x)

        result4 = self.top_layer4(result4)
        result3 = self.top_layer3(result3)
        result3 = F.interpolate(result4, size=result3.shape[2:]) + result3
        result2 = self.top_layer2(result2)
        result2 = F.interpolate(result3, size=result2.shape[2:]) + result2
        result1 = self.top_layer1(result1)
        result1 = F.interpolate(result2, size=result1.shape[2:]) + result1

        result = self.smooth1(result1)  # +self.smooth2(result2)+self.smooth3(result3)#+self.smooth4(result4)
        result = result.view(result.size(0), result.size(2), -1)
        return result


class ProCnn_muti(nn.Module):
    def __init__(self, embedding, emb_dim, stem_channel, channels, out_channel,stem_kernel=1,stem=True):
        super(ProCnn_muti, self).__init__()
        if embedding:
            self.embedding = embedding
            emb_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(26, emb_dim)
        if stem:
            self.stem = Stem(emb_dim, stem_channel,stem_kernel)
        else:
            self.stem = nn.Identity()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=stem_channel, out_channels=channels[0], kernel_size=3, stride=1), nn.ReLU()
            , nn.BatchNorm1d(channels[0],eps=1e-1,momentum=0.9))
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=6, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[1],eps=1e-1,momentum=0.9))
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=9, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[2],eps=1e-1,momentum=0.9))
        self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.mpool2 = nn.MaxPool1d(kernel_size=6, stride=1)
        self.mpool3 = nn.MaxPool1d(kernel_size=9, stride=1)
        self.smooth1 = nn.Conv1d(in_channels=channels[0], out_channels=out_channel, kernel_size=1)
        self.smooth2 = nn.Conv1d(in_channels=channels[1], out_channels=out_channel, kernel_size=1)
        self.smooth3 = nn.Conv1d(in_channels=channels[2], out_channels=out_channel, kernel_size=1)
        self.smooth = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.ffn = IRFFN(out_channel, 3.6)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(2, 1)
        x = self.stem(x)
        x = self.block1(x)
        result1 = self.mpool1(x)
        x = self.block2(x)
        result2 = self.mpool2(x)
        x = self.block3(x)
        result3 = self.mpool3(x)
        result = self.smooth1(result1) + self.smooth2(F.interpolate(result2, size=result1.shape[2:])) + self.smooth3(
            F.interpolate(result3, size=result1.shape[2:]))
        result = self.smooth(result)
        result = self.ffn(result)
        result = result.view(result.size(0), result.size(2), -1)
        return result

class ProCnn_muti_out(nn.Module):
    def __init__(self, embedding, emb_dim, stem_channel, channels, out_channel,stem_kernel = 1,stem=True,layer_num=3):
        super(ProCnn_muti_out, self).__init__()
        if embedding:
            self.embedding = embedding
            emb_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(26, emb_dim)
        if stem:
            self.stem = Stem(emb_dim, stem_channel,stem_kernel)
        else:
            self.stem = nn.Identity()
        if layer_num == 1:
            self.muti = False
            self.mpool3 = nn.MaxPool1d(kernel_size=9, stride=1)
            self.smooth3 = nn.Conv1d(in_channels=channels[2], out_channels=out_channel, kernel_size=1)
        else:
            self.muti = True
            self.mpool1 = nn.MaxPool1d(kernel_size=3, stride=1)
            self.mpool2 = nn.MaxPool1d(kernel_size=6, stride=1)
            self.mpool3 = nn.MaxPool1d(kernel_size=9, stride=1)
            self.smooth1 = nn.Conv1d(in_channels=channels[0], out_channels=out_channel, kernel_size=1)
            self.smooth2 = nn.Conv1d(in_channels=channels[1], out_channels=out_channel, kernel_size=1)
            self.smooth3 = nn.Conv1d(in_channels=channels[2], out_channels=out_channel, kernel_size=1)

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=stem_channel, out_channels=channels[0], kernel_size=3, stride=1), nn.ReLU()
            , nn.BatchNorm1d(channels[0],eps=1e-1,momentum=0.99))
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=6, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[1],eps=1e-1,momentum=0.99))
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=9, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[2],eps=1e-1,momentum=0.99))

    def forward(self, x):
        result = []
        x = self.embedding(x.long())
        x = x.transpose(2, 1)
        x = self.stem(x)
        if self.muti:
            x = self.block1(x)
            result1 = self.mpool1(x)
            x = self.block2(x)
            result2 = self.mpool2(x)
            x = self.block3(x)
            result3 = self.mpool3(x)
            result.append(self.smooth1(result1).view(result1.size(0), result1.size(2), -1))
            result.append(self.smooth2(F.interpolate(result2, size=result1.shape[2:])).view(result1.size(0), result1.size(2), -1))
            result.append(self.smooth3(
                F.interpolate(result3, size=result1.shape[2:])).view(result1.size(0), result1.size(2), -1))
        else:
            x = self.block1(x)
            #result1 = self.mpool1(x)
            x = self.block2(x)
            #result2 = self.mpool2(x)
            x = self.block3(x)
            result3 = self.mpool3(x)
            result.append(self.smooth3(result3).view(result3.size(0), -1,self.embedding.embedding_dim))
        return result

class MaxPoolLayer(nn.Module):
    def forward(self, graph, node_feats):
        with graph.local_scope():
            graph.ndata['h'] = node_feats
            # 最大池化操作
            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h_max'))
            return graph.ndata['h_max']


class GCN_muti(GCN):
    def __init__(self, in_feats,
                 hidden_feats=None,
                 gnn_norm=None,
                 activation=None,
                 residual=None,
                 batchnorm=None,
                 dropout=None,
                 allow_zero_in_degree=None, ):
        super(GCN_muti, self).__init__(in_feats,
                                       hidden_feats,
                                       gnn_norm,
                                       activation,
                                       residual,
                                       batchnorm,
                                       dropout,
                                       allow_zero_in_degree)
        self.n_layers = len(hidden_feats)
        self.pool_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.pool_layers.append(MaxPoolLayer())
            self.smooth_layers.append(GCNLayer(
                hidden_feats[i],
                hidden_feats[i],
                "none",
                F.relu,
                True,
                True,
                0
            ))
        self.smooth = GCNLayer(hidden_feats[-1],
                               hidden_feats[-1],
                               "none",
                               F.relu,
                               True,
                               True,
                               0)

    def forward(self, g, feats):
        result = torch.zeros_like(feats)
        for i in range(self.n_layers):
            feats = self.gnn_layers[i](g, feats)
            result = result + self.smooth_layers[i](g, self.pool_layers[i](g, feats))
        result = self.smooth(g, result)
        return result


class GCN_muti_out(GCN):
    def __init__(self, in_feats,
                 hidden_feats=None,
                 gnn_norm=None,
                 activation=None,
                 residual=None,
                 batchnorm=None,
                 dropout=None,
                 allow_zero_in_degree=None,
                 layer_num=3):
        super(GCN_muti_out, self).__init__(in_feats,
                                           hidden_feats,
                                           gnn_norm,
                                           activation,
                                           residual,
                                           batchnorm,
                                           dropout,
                                           allow_zero_in_degree)
        self.n_layers = layer_num
        self.pool_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.pool_layers.append(MaxPoolLayer())
            self.smooth_layers.append(GCNLayer(
                hidden_feats[i],
                hidden_feats[i],
                "none",
                F.relu,
                True,
                True,
                0
            ))

    def forward(self, g, feats):
        result = []
        if self.n_layers == 3:
            for i in range(self.n_layers):
                feats = self.gnn_layers[i](g, feats)
                result.append(self.smooth_layers[i](g, self.pool_layers[i](g, feats)))
        else:
            for gnn in self.gnn_layers:
                feats = gnn(g, feats)
            result.append(self.smooth_layers[0](g,self.pool_layers[0](g,feats)))
        return result

class GCNStem(nn.Module):
    """
    Use CMTStem module to process input image and overcome the limitation of the
    non-overlapping patches.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (B, C, T)
    Output:
        - result: (B, 128, T)
    """

    def __init__(self, in_feature):
        super().__init__()
        self.linear1 = nn.Linear(in_feature,in_feature)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(in_feature,eps=1e-1,momentum=0.99)
        self.linear2 = nn.Linear(in_feature,in_feature)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(in_feature,eps=1e-1,momentum=0.99)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.gelu2(x)
        result = self.bn2(x)
        return result+x

class MolecularGCN_muti(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None, out_feats=None,stem=True):
        super(MolecularGCN_muti, self).__init__()
        self.init_transform = nn.Linear(75, dim_embedding, bias=False)
        if stem:
            self.stem = GCNStem(dim_embedding)
        else:
            self.stem = nn.Identity()
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN_muti(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = out_feats

    def forward(self, batch_graph):
        temp = batch_graph.ndata.pop('h')
        node_feats = temp
        node_feats = self.init_transform(node_feats)
        node_feats = self.stem(node_feats)
        muti_feats = self.gnn(batch_graph, node_feats)
        batch_graph.ndata['h'] = temp
        batch_size = batch_graph.batch_size
        # node_feats = node_feats.view(batch_size, -1, self.output_feats)
        muti_feats = muti_feats.view(batch_size, -1, self.output_feats)
        return muti_feats


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim,eps=1e-1,momentum=0.99)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim,eps=1e-1,momentum=0.99)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim,eps=1e-1,momentum=0.99)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class MolecularGCN_mutiout(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None, out_feats=None,stem=True,layer_num=3):
        super(MolecularGCN_mutiout, self).__init__()
        self.init_transform = nn.Linear(75, dim_embedding, bias=False)
        if stem:
            self.stem = GCNStem(dim_embedding)
        else:
            self.stem = nn.Identity()

        if layer_num == 1:
            self.muti = False
        else:
            self.muti = True

        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN_muti_out(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation,layer_num=layer_num)
        self.output_feats = out_feats

    def forward(self, batch_graph):
        temp = batch_graph.ndata.pop('h')
        node_feats = temp
        node_feats = self.init_transform(node_feats)
        node_feats = self.stem(node_feats)
        muti_feats = self.gnn(batch_graph, node_feats)
        batch_graph.ndata['h'] = temp
        batch_size = batch_graph.batch_size
        # node_feats = node_feats.view(batch_size, -1, self.output_feats)
        feats = []
        if self.muti:
            for i in muti_feats:
                feats.append(i.view(batch_size, -1, self.output_feats))
        else:
            feats.append(muti_feats[-1].view(batch_size, -1, self.output_feats))
        return feats


class Dti_DrugBAN(nn.Module):
    def __init__(self):
        super(Dti_DrugBAN, self).__init__()

        self.drug_extractor = MolecularGCN(in_feats=75, dim_embedding=128,
                                           padding=True,
                                           hidden_feats=[128, 128, 128])
        self.protein_extractor = ProteinCNN(128, [128, 128, 128], [3, 6, 9],True)
        self.bcn = weight_norm(
            MetaBANLayer(v_dim=128, q_dim=128, h_dim=256, h_out=2),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=2)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        return score,f
        #if mode == "train":

        # elif mode == "eval":
        #     return score, att

class MamlMLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MamlMLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim,eps=1e-1,momentum=0.99)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim,eps=1e-1,momentum=0.99)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim,eps=1e-1,momentum=0.99)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class MetaMamlBANLayer(BANLayer):
    def __init__(self,v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(MetaMamlBANLayer, self).__init__(v_dim, q_dim, h_dim, h_out, act, dropout, k)
        self.bn = nn.BatchNorm1d(h_dim,eps=1e-1,momentum=0.99)
    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class MamlGAU(GAU):
    def __init__(self,dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.,
        norm = True):
        super(MamlGAU, self).__init__(dim,
        query_key_dim,
        expansion_factor,
        add_residual,
        dropout,
        norm)
        self.norm = nn.LayerNorm(dim,eps=1e-2)

class Dti_anil_mulcnn_features(nn.Module):
    def __init__(self, d_embedding, p_embedding, config):
        super(Dti_anil_mulcnn_features, self).__init__()
        self.drug_extractor = MolecularGCN_mutiout(in_feats=config.d_emb, dim_embedding=config.d_stem_channel,
                                                padding=True,
                                                hidden_feats=config.d_channels,
                                                out_feats=config.d_out_channel,stem=config.d_stem,layer_num=config.layers_num)
        self.prot_extractor = ProCnn_muti_out(p_embedding, config.p_emb, config.p_stem_channel, config.p_channels,
                                             config.p_out_channel,config.stem_kernel,stem=config.p_stem,layer_num=config.layers_num)
        self.bcn = nn.ModuleList([weight_norm(
            MetaMamlBANLayer(v_dim=config.d_out_channel, q_dim=config.p_out_channel, h_dim=config.out_hidden_size, h_out=2),
            name='h_mat', dim=None) for i in range(config.layers_num)])
        self.bcn_gate = MamlGAU(256,128,norm=True)
        self.muti = True
    def forward(self, mol, protein):
        fs = torch.Tensor([]).to(protein.device)
        attns = torch.Tensor([]).to(protein)
        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein)
        if self.muti:
            for i in range(len(mol)):
                f, attn = self.bcn[i](mol[i], prot[i])
                fs = torch.cat([fs, f.unsqueeze(1)], dim=1)
                attns = torch.cat([attns, attn.unsqueeze(1)], dim=1)
        else:
            f, attn = self.bcn[0](mol[0], prot[0])
            fs = torch.cat([fs, f.unsqueeze(1)], dim=1)
            attns = torch.cat([attns, attn.unsqueeze(1)], dim=1)
            fs = self.bcn_gate(fs)  # b,3,256
        fs = torch.mean(fs, dim=1).squeeze(1)
        return fs

class Dti_anil_mulcnn(nn.Module):
    def __init__(self, d_embedding, p_embedding, config):
        super(Dti_anil_mulcnn, self).__init__()
        self.features = Dti_anil_mulcnn_features(d_embedding, p_embedding, config)
        self.pred = MamlMLPDecoder(256,512, 128,2)
    def forward(self,mol, protein,visual=False):
        fs = self.features(mol, protein)
        p = self.pred(fs)
        return p


class Dti_maml_mulcnn(Dti_cnn_mutiout):
    def __init__(self,d_embedding, p_embedding, config):
        super(Dti_maml_mulcnn, self).__init__(d_embedding, p_embedding, config)
        self.drug_extractor = MolecularGCN_mutiout(in_feats=config.d_emb, dim_embedding=config.d_stem_channel,
                                                padding=True,
                                                hidden_feats=config.d_channels,
                                                out_feats=config.d_out_channel,stem=config.d_stem,layer_num=config.layers_num)
        self.prot_extractor = ProCnn_muti_out(p_embedding, config.p_emb, config.p_stem_channel, config.p_channels,
                                             config.p_out_channel,config.stem_kernel,stem=config.p_stem,layer_num=config.layers_num)
        self.bcn = nn.ModuleList([weight_norm(
            MetaMamlBANLayer(v_dim=config.d_out_channel, q_dim=config.p_out_channel, h_dim=config.out_hidden_size, h_out=2),
            name='h_mat', dim=None) for i in range(config.layers_num)])
        self.bcn_gate = MamlGAU(256,128,norm=True)
        self.pred = MamlMLPDecoder(self.pred.fc1.in_features,512, 128,self.pred.fc4.out_features)


class MetaBatchNormLayer(torch.nn.Module):
    """
    An extension of Pytorch's BatchNorm layer, with the Per-Step Batch Normalisation Running
    Statistics and Per-Step Batch Normalisation Weights and Biases improvements proposed in
    MAML++ by Antoniou et al. It is adapted from the original Pytorch implementation at
    https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch,
    with heavy refactoring and a bug fix
    (https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/issues/42).
    """

    def __init__(
        self,
        num_features,
        eps=1e-1,
        momentum=0.999,
        affine=True,
        meta_batch_norm=True,
        adaptation_steps: int = 1,
    ):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.running_mean = torch.nn.Parameter(
            torch.zeros(adaptation_steps, num_features), requires_grad=False
        )
        self.running_var = torch.nn.Parameter(
            torch.ones(adaptation_steps, num_features), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(adaptation_steps, num_features), requires_grad=True
        )
        self.weight = torch.nn.Parameter(
            torch.ones(adaptation_steps, num_features), requires_grad=True
        )
        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)
        self.momentum = momentum

    def forward(
        self,
        input,
        step,
    ):
        """
        :param input: input data batch, size either can be any.
        :param step: The current inner loop step being taken. This is used when to learn per step params and
         collecting per step batch statistics.
        :return: The result of the batch norm operation.
        """
        assert (
            step < self.running_mean.shape[0]
        ), f"Running forward with step={step} when initialised with {self.running_mean.shape[0]} steps!"

        return F.batch_norm(
            input,
            self.running_mean[step],
            self.running_var[step],
            self.weight[step],
            self.bias[step],
            training=True,
            momentum=self.momentum,
            eps=self.eps,
        )

    def backup_stats(self):
        self.backup_running_mean.data = deepcopy(self.running_mean.data)
        self.backup_running_var.data = deepcopy(self.running_var.data)

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.running_mean = torch.nn.Parameter(
            self.backup_running_mean, requires_grad=False
        )
        self.running_var = torch.nn.Parameter(
            self.backup_running_var, requires_grad=False
        )

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}".format(
            **self.__dict__
        )

class MamlppMLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1,step=3):
        super(MamlppMLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = MetaBatchNormLayer(hidden_dim,adaptation_steps=step)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = MetaBatchNormLayer(hidden_dim,adaptation_steps=step)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = MetaBatchNormLayer(out_dim,adaptation_steps=step)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x,step=0):
        x = self.bn1(F.relu(self.fc1(x)),step)
        x = self.bn2(F.relu(self.fc2(x)),step)
        x = self.bn3(F.relu(self.fc3(x)),step)
        x = self.fc4(x)
        return x

class MetaBANLayer(BANLayer):
    def __init__(self,v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3,step=3):
        super(MetaBANLayer, self).__init__(v_dim, q_dim, h_dim, h_out, act, dropout, k)
        self.bn = MetaBatchNormLayer(self.bn.num_features,adaptation_steps=step)
    def forward(self, v, q, softmax=False,step=0):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits,step)
        return logits, att_maps

class Dti_mamlpp_mulcnn(Dti_cnn_mutiout):
    def __init__(self,d_embedding, p_embedding, config,step=3):
        super(Dti_mamlpp_mulcnn, self).__init__(d_embedding, p_embedding, config)
        self.drug_extractor = MolecularGCN_mutiout(in_feats=config.d_emb, dim_embedding=config.d_stem_channel,
                                                padding=True,
                                                hidden_feats=config.d_channels,
                                                out_feats=config.d_out_channel,stem=config.d_stem,layer_num=config.layers_num)
        self.prot_extractor = ProCnn_muti_out(p_embedding, config.p_emb, config.p_stem_channel, config.p_channels,
                                             config.p_out_channel,config.stem_kernel,stem=config.p_stem,layer_num=config.layers_num)
        self.bcn = nn.ModuleList([weight_norm(
            MetaBANLayer(v_dim=config.d_out_channel, q_dim=config.p_out_channel, h_dim=config.out_hidden_size, h_out=2,step=step),
            name='h_mat', dim=None) for i in range(config.layers_num)])
        self.pred = MamlppMLPDecoder(self.pred.fc1.in_features,512, 128,self.pred.fc4.out_features,step=step)
        self.bcn_gate = MamlGAU(256, 128, norm=True)
    def backup_stats(self):
        """
        Backup stored batch statistics before running a validation epoch.
        """
        for name, module in self.named_modules():
            if type(module) is MetaBatchNormLayer:
                module.backup_stats()
        # for layer in self.features.modules():

    def forward(self, mol, protein,visual=False,step=0):
        fs = torch.Tensor([]).to(protein)
        attns = torch.Tensor([]).to(protein)
        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein)
        if self.muti:
            for i in range(len(mol)):
                f,attn = self.bcn[i](mol[i],prot[i],step=step)
                fs = torch.cat([fs,f.unsqueeze(1)],dim=1)
                attns = torch.cat([attns,attn.unsqueeze(1)],dim=1)
        else:
            f, attn = self.bcn[0](mol[0], prot[0],step=step)
            fs = torch.cat([fs, f.unsqueeze(1)], dim=1)
            attns = torch.cat([attns, attn.unsqueeze(1)], dim=1)
        if visual:
            fs,gate_attn,gate = self.bcn_gate(fs,visual)  # b,3,256
        else:
            fs = self.bcn_gate(fs)#b,3,256
        fs = torch.mean(fs,dim=1).squeeze(1)
        pred = self.pred(fs,step=step)
        if visual:
            return pred,fs,attns,gate_attn,gate
        else:
            return pred,fs

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        # for layer in self.features.modules():
        #     if type(layer) is MetaBatchNormLayer:
        #         layer.restore_backup_stats()
        for name, module in self.named_modules():
            if type(module) is MetaBatchNormLayer:
                module.restore_backup_stats()