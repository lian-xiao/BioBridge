import torch
import torch.nn as nn
from My_nets.utils_net import Residual_Net
class Dti_VecModel(nn.Module):
    def __init__(self,model_config,gpu_tracker):
        super(Dti_VecModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=6,dim_feedforward=512)
        self.share_token = nn.Parameter(torch.zeros((1,300)))
        nn.init.kaiming_normal_(self.share_token)
        #self.bcn = nn.BatchNorm1d(300)
        #self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=4)
        self.mol_linear = nn.Linear(in_features=300,out_features=2048)
        self.pro_linear = nn.Linear(in_features=300,out_features=2048)

        #self.dense_net = Residual_Net(in_feature=2048,hidden_size=512,hidden_act='relu')
        self.dense = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,512),nn.Linear(512,512))
        self.pred_net = nn.Linear(in_features=512,out_features=2)

    def forward(self,mol,protein):
        #x = self.bcn(x)

        #x = torch.cat((self.share_token.expand(mol.shape[0], -1, -1),mol.unsqueeze(1),protein.unsqueeze(1)),dim=1)
        #x = self.encoder(x)
        mol = self.mol_linear(mol)
        protein = self.pro_linear(protein)
        x = torch.cat((mol,protein),dim=-1)

        x = self.dense(x)
        x = self.pred_net(x)
        return x
