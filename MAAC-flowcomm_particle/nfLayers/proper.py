import torch
import numpy as np
import torch.nn as nn
from .layers import AbstractInvertibleLayer
from .neural_net import NeuralNet

class AugmentedLagrangian(nn.Module):
    def __init__(self, rho, lam, beta=2, thr=1):
        super().__init__()
        self.register_buffer('rho', torch.ones(1, dtype=torch.float)*rho)
        lam = (torch.ones(1, dtype=torch.float)*lam).requires_grad_(True)
        self.register_buffer('lam', lam)
        self.register_buffer('beta', torch.ones(1, dtype=torch.float)*beta)

        self.lam_hook = self.lam.register_hook(lambda grad: self.lambda_hook_fn(grad))
        self.backward_count = 0
        self.cumulated_grad_lambda = 0
        self.raw_cost = 0
        self.register_buffer('thr', torch.ones(1)*thr)

    def lambda_hook_fn(self, grad):
        self.backward_count += 1
        self.cumulated_grad_lambda -= grad.item()
        return -grad

    def forward(self, cost):
        self.raw_cost += cost.item()
        cost = self.lam * cost + self.rho/2 * cost**2
        return cost

    def step(self):
        if self.backward_count>0:
            step = abs(self.cumulated_grad_lambda/self.backward_count)
            if step>self.thr or not np.isfinite(step):
                step = self.thr.item()
            self.lam.data += self.rho.data * step
            self.rho.data *= self.beta
            print(f'rho: {self.rho.item()}, '
                  f'lambda: {self.lam.item()}, '
                  f'raw_cost: {self.raw_cost/self.backward_count}')

            self.backward_count = 0
            self.cumulated_grad_lambda = 0
            self.raw_cost = 0
            self.zero_grad()

class InvertibleSparseTranspose(AbstractInvertibleLayer):
    def __init__(self,
                 reduced_latent_size,
                 augmented_latent_size,
                 auxiliary_size=0,
                 sub_network_layers=3,
                 sub_network_cells=32,
                 sub_network_activation=nn.Tanh,
                 sub_network_batch_norm=True,
                 lagrangian_rho = 1,
                 lagrangian_lambda = 0,
                 lagrangian_beta = 2,
                 temperature=1,
                 mapmode='up',
                 ):
        super().__init__()
        self.reduced_latent_size = reduced_latent_size
        self.augmented_latent_size = augmented_latent_size
        self.auxiliary_size = auxiliary_size
        self.register_buffer('temperature', torch.ones(1)*temperature)
        assert mapmode in ['up','down']
        self.mapmode = mapmode

        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm

        self.transform = NeuralNet(
            self.auxiliary_size+self.reduced_latent_size,
            self.augmented_latent_size,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=False,
            batch_norm=self.sub_network_batch_norm
        )

        self.lagrangian = AugmentedLagrangian(rho=lagrangian_rho, lam=lagrangian_lambda, beta=lagrangian_beta)
        self.cost = None

    def forward_transform(self, input, return_log_jacobian=True, auxiliary_input=None):
        if self.mapmode == 'up':
            return self._map_up(input, return_log_jacobian, auxiliary_input)
        elif self.mapmode == 'down':
            return self._map_down(input, return_log_jacobian, auxiliary_input)
        else:
            raise NotImplementedError()

    def backward_transform(self, output, return_log_jacobian=True, auxiliary_input=None):
        if self.mapmode == 'up':
            return self._map_down(output, return_log_jacobian, auxiliary_input)
        elif self.mapmode == 'down':
            return self._map_up(output, return_log_jacobian, auxiliary_input)
        else:
            raise NotImplementedError()

    def _map_up(self, reduced_latent, return_log_jacobian=True, auxiliary_input=None):
        ids = abs(reduced_latent).argsort(-1)
        input = reduced_latent.gather(-1, ids)
        if auxiliary_input is not None:
            input = torch.cat([
                input,
                auxiliary_input], -1)
        l = self.transform(input)
        R = RandomT(l, t=self.temperature, max_s=self.reduced_latent_size)
        T = R()
        augmented_latent = T @ reduced_latent.unsqueeze(-1)
        if return_log_jacobian:
            return augmented_latent.squeeze(-1), 0.0
        else:
            return augmented_latent.squeeze(-1)

    def _map_down(self, augmented_latent, return_log_jacobian=True, auxiliary_input=None):
        ids = abs(augmented_latent).argsort(-1)[..., -self.reduced_latent_size:]
        transpose_net_input = augmented_latent.gather(-1, ids)
        if auxiliary_input is not None:
            transpose_net_input = torch.cat([
                transpose_net_input,
                auxiliary_input], -1)
        l = self.transform(transpose_net_input)
        R = RandomT(l, t=self.temperature, max_s=self.reduced_latent_size)
        T = R()
        Tp = T.transpose(-2, -1)
        reduced_latent = Tp @ augmented_latent.unsqueeze(-1)
        cost = abs(T @ reduced_latent - augmented_latent.unsqueeze(-1)).sum(-1).mean()
        cost = self.lagrangian(cost)
        self.cost = cost

        if return_log_jacobian:
            return reduced_latent.squeeze(-1), 0.0
        else:
            return reduced_latent.squeeze(-1)

    def _get_lagrangian_cost(self):
        if self.cost is not None:
            cost = self.cost
            self.cost = None
            return cost
        return 0

class RandomT:
    def __init__(self, l, t, max_s=None, discretize=False):
        self.l = l
        self.pruned_l = l.clone()
        self.t = t
        self.batch = l.shape[0]
        self.hooks = []
        self.device = l.device
        self.max_s = max_s if max_s is not None else self.l.shape[1]
        self.discretize = discretize

    def _rand(self, ):
        dist = torch.distributions.RelaxedOneHotCategorical(logits=self.pruned_l, temperature=self.t * torch.ones(1, dtype=torch.float, device=self.device))
        s = dist.rsample()
        if self.discretize:
            s_discrete = s
        else:
            s_discrete = s.clone().detach()
        argmax = s_discrete.data.argmax(1)
        s_discrete.data.zero_()
        s_discrete.data.index_add_(1, argmax, torch.eye(self.batch, dtype=torch.float, device=self.device))
        self.pruned_l = self.pruned_l[(1 - s_discrete.data).to(bool)].view(self.batch, -1)

        return s, s_discrete

    def __call__(self):
        while len(self.hooks) > 0:
            hook = self.hooks.pop()
            hook.remove()

        R = []
        filled = torch.zeros(self.l.shape, dtype=torch.bool, device=self.device)
        for i in range(self.l.shape[1]):
            _r, _r_discrete = self._rand()
            r = torch.full_like(self.l, 0, dtype=torch.float, device=self.device)
            r[torch.logical_not(filled)] = _r.view(-1)
            filled[torch.logical_not(filled)] = _r_discrete.to(bool).view(-1)
            R.append(r)

        R = torch.stack(R, 1)
        if self.max_s < self.l.shape[1]:
            R = R[:, :, :self.max_s]
        return R