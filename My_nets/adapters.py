import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
import copy
from My_nets.phmLiner import PHMLinear
from transformers.activations import get_activation
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VisualAdapter(nn.Module):
    """Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind,reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = True
        self.opt = opt
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None
        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor
            phm_dim = 4
            ### -----> attetnion
            self.my_tokens = nn.Parameter(torch.rand((self.opt.num_tokens, input_dim)))
            self.gate_av = nn.Parameter(torch.zeros(1))
            ### <------
            self.activation = nn.ReLU(inplace=True)
            if self.down_sample_size % phm_dim != 0:
                for i in range(3,9):
                    if self.down_sample_size % i == 0:
                        phm_dim = i

            self.down_sampler = PHMLinear(input_dim, self.down_sample_size,phm_dim=phm_dim,bias=True)
            self.up_sampler = PHMLinear(self.down_sample_size, output_dim,phm_dim=phm_dim,bias = True)
            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)
            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------
        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            # nn.init.zeros_(self.down_sampler) # yb:for lora
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)
            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)
            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------
        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)
            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        else:
            raise NotImplementedError

    def forward(self, x, add_token=None):
        if self.adapter_kind == "bottleneck" and self.is_multimodal:
            ### -------> high dim att
            rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
            att_v2tk = torch.bmm(rep_token, add_token.transpose(1,2))
            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, add_token)
            rep_token = rep_token + rep_token_res
            att_tk2x = torch.bmm(x, rep_token.permute(0, 2, 1))
            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token)
            x = x + self.gate_av * x_res.contiguous()
            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x)
            z = self.down_sampler(x)
            ## <----
            if self.use_bn:
                z = self.bn1(z)
            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)
            z = self.down_sampler(x)
            if self.use_bn:
                z = self.bn1(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)
        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C'))
                output = rearrange(output, 'N L C -> N C L')
        if self.opt.is_post_layernorm:
            output = self.ln_post(output)
        if self.gate is not None:
            output = self.gate * output
        return output


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True):
        super().__init__()

        import json

        config = json.load(open('/data/yanbo/ada_av/nets/compacter.json'))
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // reduction_factor

        # self.activation = Activations(self.config['non_linearity'].lower())
        self.activation = get_activation('gelu_new')

        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config['phm_c_init'],
                                      phm_dim=config['hypercomplex_division'],
                                      learn_phm=config['learn_phm'],
                                      w_init=config['hypercomplex_nonlinearity'],
                                      shared_phm_rule=config['shared_phm_rule'],
                                      factorized_phm=config['shared_phm_rule'],
                                      #   shared_W_phm=config['shared_W_phm'],
                                      factorized_phm_rule=config['factorized_phm_rule'],
                                      #   phm_rank=config['phm_rank'],
                                      phm_init_range=config['phm_init_range'],
                                      #   kronecker_prod=config['kronecker_prod']
                                      )

        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim,
                                    bias=True,
                                    c_init=config['phm_c_init'],
                                    phm_dim=config['hypercomplex_division'],
                                    learn_phm=config['learn_phm'],
                                    w_init=config['hypercomplex_nonlinearity'],
                                    shared_phm_rule=config['shared_phm_rule'],
                                    factorized_phm=config['factorized_phm'],
                                    # shared_W_phm=config['shared_W_phm'],
                                    factorized_phm_rule=config['factorized_phm_rule'],
                                    # phm_rank=config['phm_rank'],
                                    phm_init_range=config['phm_init_range'],
                                    # kronecker_prod=config['kronecker_prod']
                                    )

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)
