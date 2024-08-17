import torch.nn as nn
from transformers.activations import ACT2FN

class Residual_Net(nn.Module):
    def __init__(self,in_feature,hidden_size,hidden_act):
        super(Residual_Net, self).__init__()
        self.linear1 = nn.Linear(in_features=in_feature,out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size,out_features=in_feature)

        if isinstance(hidden_act, str):
            self.act_fn = ACT2FN[hidden_act]
        else:
            self.act_fn = hidden_act

    def forward(self,x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x