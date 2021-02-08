from torch import nn
import torch
from torch.distributions import normal
from .utils import init_weights
from torch.nn import BatchNorm1d


class FlattenBatchNorm1d(BatchNorm1d):
    def forward(self, input):
        return super().forward(input.view(-1, input.shape[-1])).view(input.shape)


class NeuralNet(nn.Module):

    def __init__(self,
                 n_in,
                 n_out,
                 n_neurons,
                 n_layers=3,
                 activation_fn=nn.Tanh,
                 bias_last_layer=True,
                 separate_net=False,
                 batch_norm=False,
                 last_layer=None,
                 ):
        super(NeuralNet, self).__init__()

        assert n_layers >= 1

        self.n_in = n_in
        self.n_out = n_out
        self.bias_last_layer = bias_last_layer
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.last_layer = last_layer

        if not isinstance(n_neurons, (list, tuple)):
            n_neurons = [int(n_neurons)] * (n_layers - 1)

        self.n_neurons = n_neurons + [n_out]
        self.n_layers = len(self.n_neurons)

        if separate_net:
            raise NotImplementedError()
        else:
            self.fc = self._make_layers()

    def _make_seperate_networks(self):
        pass

    def _make_layers(self, ):
        layers = []
        for i, _n_neurons in enumerate(self.n_neurons):
            if i == 0:
                n_in = self.n_in
            else:
                n_in = n_out
            n_out = _n_neurons

            if i == self.n_layers - 1:
                activation = None
                bias = self.bias_last_layer
            else:
                bias = True
                activation = self.activation_fn()

            layers.append(
                nn.Linear(n_in, n_out, bias=bias)
            )
            if activation is not None:
                layers.append(
                    activation
                )
                if self.batch_norm:
                    layers.append(FlattenBatchNorm1d(n_out))

        fc = nn.Sequential(*layers)

        # if use_optimizations:
        return fc.apply(init_weights)

    def forward(self, x):
        if isinstance(self.fc, list):
            x = torch.cat([fc(x) for fc in self.fc], dim=-1)
        else:
            x = self.fc(x)
        if self.last_layer is not None:
            x = self.last_layer(x)
        return x
