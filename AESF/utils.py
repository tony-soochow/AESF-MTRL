import json
import torch
import torch.nn as nn
import numpy as np

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


# calculate squared Frobenius norm
class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()  # 先求二范数
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss













