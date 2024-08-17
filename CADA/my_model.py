import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn.gcn import GCNLayer
from torch import einsum
from torch.nn.utils import weight_norm
from dgllife.model.gnn import GCN
import dgl.function as fn

from ban import BANLayer


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
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels)
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
        result = nn.functional.layer_norm(x, (c, t))
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
            nn.BatchNorm1d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm1d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(exp_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels)
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
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_channels)
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
            , nn.BatchNorm1d(channels[0]))
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=6, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[1]))
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=9, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[2]))
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
            , nn.BatchNorm1d(channels[0]))
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=6, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[1]))
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=9, stride=1), nn.ReLU(),
            nn.BatchNorm1d(channels[2]))

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
        self.bn1 = nn.BatchNorm1d(in_feature)
        self.linear2 = nn.Linear(in_feature,in_feature)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(in_feature)

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
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
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

class GAU(nn.Module):
    def __init__(
        self,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.,
        norm = True
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if norm:
            self.to_hidden = nn.Sequential(
                weight_norm(nn.Linear(dim, hidden_dim * 2)),
                nn.SiLU()
            )

            self.to_qk = nn.Sequential(
                weight_norm(nn.Linear(dim, query_key_dim)),
                nn.SiLU()
            )
            self.to_out = nn.Sequential(
                weight_norm(nn.Linear(hidden_dim, dim)),
                nn.Dropout(dropout)
            )
        else:
            self.to_hidden = nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                nn.SiLU()
            )

            self.to_qk = nn.Sequential(
                nn.Linear(dim, query_key_dim),
                nn.SiLU()
            )
            self.to_out = nn.Sequential(
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)


        self.add_residual = add_residual

    def forward(self, x,visual=False):
        seq_len = x.shape[-2]
        normed_x = self.norm(x) #(bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #(bs,seq_len,seq_len)
        Z = self.to_qk(normed_x) #(bs,seq_len,query_key_dim)
        QK = einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        sim = einsum('b i d, b j d -> b i j', q, k) / seq_len
        A = F.relu(sim) ** 2
        A = self.dropout(A)
        V = einsum('b i j, b j d -> b i d', A, v)
        V = V * gate

        out = self.to_out(V)

        if self.add_residual:
            out = out + x
        if visual:
            return out,A,gate
        else:
            return out
class Dti_cnn(nn.Module):
    def __init__(self,config, d_embedding=None, p_embedding=None):
        super(Dti_cnn, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN_muti(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                                padding=True,
                                                hidden_feats=drug_hidden_feats,
                                                out_feats=drug_embedding)
        self.prot_extractor = ProCnn_muti(p_embedding, 128,128, num_filters,
                                             128)
        self.bcn = weight_norm(
            BANLayer(v_dim=128, q_dim=128, h_dim=256, h_out=2),
            name='h_mat', dim=None)

        self.pred = MLPDecoder(256, 512, 128,out_binary)

    def forward(self, mol, protein,attns =False):

        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein)
        f,attn = self.bcn(mol,prot)
        pred = self.pred(f)
        if attns:
            return mol, prot, pred, attn
        else:
            return mol, prot, f, pred



class Dti_cnn_mutiout(nn.Module):
    def __init__(self,config,d_embedding=None, p_embedding=None):
        super(Dti_cnn_mutiout, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        self.drug_extractor = MolecularGCN_mutiout(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                                padding=True,
                                                hidden_feats=drug_hidden_feats,
                                                out_feats=drug_embedding,stem=True,layer_num=3)
        self.prot_extractor = ProCnn_muti_out(p_embedding, protein_emb_dim,protein_emb_dim, num_filters,
                                             protein_emb_dim,1,stem=True,layer_num=3)

        self.bcn = nn.ModuleList([weight_norm(
            BANLayer(v_dim=drug_embedding, q_dim=protein_emb_dim, h_dim=256, h_out=2),
            name='h_mat', dim=None) for i in range(3)])
        self.bcn_gate = GAU(dim=256, query_key_dim=128)

        self.pred = MLPDecoder(256, 512, 128,out_binary)


    def forward(self, mol, protein):
        fs = torch.Tensor([]).to(protein.device)
        attns = torch.Tensor([]).to(protein.device)
        mol = self.drug_extractor(mol)
        prot = self.prot_extractor(protein.float())

        for i in range(len(mol)):
            f,attn = self.bcn[i](mol[i],prot[i])
            fs = torch.cat([fs,f.unsqueeze(1)],dim=1)
            attns = torch.cat([attns,attn.unsqueeze(1)],dim=1)

        fs = self.bcn_gate(fs)
        fs = torch.mean(fs,dim=1).squeeze(1)
        pred = self.pred(fs)
        return mol,prot,fs,pred
