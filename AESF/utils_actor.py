import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import network_init as init
import numpy as np
from torch.autograd import Function


def np_softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def cfg_read(path):
    with open(path, 'r') as f:
        cfg = json.loads(f.read(), cls=Decoder)
    return cfg


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def build_mlp(input_dim, output_dim, hidden_dims):
    '''
    Not include actiavtion of output layer
    '''
    network = nn.ModuleList()
    dims = [input_dim] + hidden_dims
    for in_dim_, out_dim_ in zip(dims[:-1], dims[1:]):
        network.append(
            nn.Linear(
                in_features=in_dim_,
                out_features=out_dim_
            )
        )
        network.append(nn.ReLU())
    network.append(
        nn.Linear(
            in_features=hidden_dims[-1],
            out_features=output_dim
        )
    )

    return nn.Sequential(*network)


class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func=init.basic_init,
                 last_activation_func=None):
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for i, next_shape in enumerate(hidden_shapes):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
            self.output_shape = next_shape

    def forward(self, x):

        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        out = self.last_activation_func(out)
        return out


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,cfg,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.num_tasks = cfg["num_tasks"]
        # shared feature extractor
        self.feature_share = base_type(activation_func=activation_func,
                                       input_shape=int(cfg["state_dim"])+int(cfg['num_tasks']), hidden_shapes=cfg["hidden_shapes"])
        # unique feature extractor
        self.feature_own = base_type(activation_func=activation_func,input_shape=int(cfg["state_dim"]) + int(cfg["num_tasks"]) ,hidden_shapes=cfg["hidden_shapes"])
        self.classifier = build_mlp(input_dim=self.feature_own.output_shape, output_dim=int(cfg["num_tasks"]),hidden_dims=cfg["hidden_shapes"])
        self.activation_func = activation_func
        self.alpha = 0

        append_input_shape = self.feature_own.output_shape*2
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out_own = self.feature_own(x)
        out_share = self.feature_share(x)
        out = torch.cat([out_share, out_own], dim=1)
        reverse_feature = ReverseLayerF().apply(out_share, self.alpha)
        task_output = F.softmax(self.classifier(reverse_feature), dim=1)
        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out, task_output




class BootstrappedNet(Net):
    def __init__(self, cfg, num_tasks):
        self.head_num = num_tasks
        self.action_dim = cfg['action_dim']
        self.mtobs_dim = int(cfg["state_dim"])+int(cfg["num_tasks"])
        self.num_tasks = cfg["num_tasks"]
        self.action_bound = cfg['action_bound']
        output_shape = int(cfg['action_dim'])*2
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        self.k = (self.action_bound[1]-self.action_bound[0])/2
        super().__init__(output_shape=output_shape,base_type=MLPBase,cfg=cfg)

    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        out, task_output = super().forward(x)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])

        out = out.reshape(out_shape)

        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)

        out = out.gather(-1, idx).squeeze(-1)
        return out, task_output


















