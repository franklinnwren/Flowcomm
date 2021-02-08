from torch import nn
import sys
import torch
from torch import distributions
import torch.distributions.normal as normal
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .neural_net import NeuralNet
import os
from .proper import InvertibleSparseTranspose, AugmentedLagrangian
from .layers import neg_grad, AbstractInvertibleLayer, MirroredCouplingFlow, PlanarFlow, RadialFlow, \
    HouseHolderFlow, gradient_transfer_backward_transform, sample_given_auxiliary, EraseCache, AffineTransformNF
from .utils import ShiftLayer, tanh5scale, log1p_softplus

import pdb
def set_auxiliary(layer, x):
    if isinstance(layer, AbstractInvertibleLayer):
        layer.set_auxiliary(x)

def inv_softplus(x):
    return torch.log(torch.expm1(x))

class NF(nn.Module):
    def __init__(self,
                 input_size,
                 latent_size,
                 nlayers,
                 embedding_network_layers=3,
                 embedding_network_cells=32,
                 embedding_network_activation=nn.ReLU,
                 embedding_network_batch_norm=False,
                 embedding_sd_init=1.0,
                 layer_type=MirroredCouplingFlow,
                 sub_network_layers=3,
                 sub_network_cells=32,
                 sub_network_activation=nn.ReLU,
                 sub_network_batch_norm=False,
                 augmented_latent_size=0,
                 auxiliary=True,
                 shift_layer=False,
                 requires_embedding=False,
                 tanh5scaling=False,
                 scaled=True,
                 scale_value=0.1,
                 ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.auxiliary = auxiliary
        self.shift_layer = shift_layer
        self.tanh5scaling = tanh5scaling

        self.embedding_network_layers = embedding_network_layers
        self.embedding_network_cells = embedding_network_cells
        self.embedding_network_activation = embedding_network_activation
        self.embedding_network_batch_norm = embedding_network_batch_norm
        self.embedding_sd_init = embedding_sd_init
        self.requires_embedding = requires_embedding
        self._make_embedding()

        self.layer_type = layer_type
        self.augmented_latent_size = augmented_latent_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm

        self.scaled=scaled
        self.scale_value = scale_value
        self._make_layers()

    def reparameterize_as(self, x, y):
        dist = self.__call__(x)
        base_dist = dist.base_dist
        with torch.no_grad():
            epsilon = y
            for trsf in reversed(dist.transforms):
                epsilon = trsf.inv(epsilon)
            epsilon = (epsilon-base_dist.loc)/base_dist.scale
        y_rep = base_dist.loc + base_dist.scale*epsilon
        for trsf in dist.transforms:
            y_rep = trsf(y_rep)
        y_rep.data.copy_(y.data)
        return y_rep

    def _make_embedding(self):
        if self.requires_embedding:
            self.embedding_layer = NeuralNet(self.input_size,
                                         2 * self.latent_size,
                                         n_neurons=self.embedding_network_cells,
                                         activation_fn=self.embedding_network_activation,
                                         batch_norm=self.embedding_network_batch_norm,
                                         n_layers=self.embedding_network_layers,
                                         bias_last_layer=True,
                                         )
            self.embedding_layer.fc[-1].bias.data.fill_(inv_softplus(torch.tensor(self.embedding_sd_init)))
        else:
            self.register_buffer('embedding_sd', torch.full((self.latent_size,),self.embedding_sd_init))
            
    def _make_layers(self):
        if self.augmented_latent_size>0:
            new_latent_size = self.augmented_latent_size
            layers = [InvertibleSparseTranspose(
                self.latent_size,
                self.augmented_latent_size,
                self.input_size if self.auxiliary else 0,
                sub_network_layers=self.sub_network_layers,
                sub_network_cells=self.sub_network_cells,
                sub_network_activation=self.sub_network_activation,
                sub_network_batch_norm=self.sub_network_batch_norm,
                lagrangian_rho=1,
                lagrangian_lambda=0,
                lagrangian_beta=2,
                temperature=0.1,
                mapmode='up',
            )]
        else:
            new_latent_size = self.latent_size
            layers = []
        layers += [
            self.layer_type(
                self.input_size if self.auxiliary else 0,
                new_latent_size,
                self.sub_network_layers,
                self.sub_network_cells,
                self.sub_network_activation,
                self.sub_network_batch_norm,
                scaled=self.scaled,
                scale_value=self.scale_value,
                ) for i in range(self.nlayers)
        ]
        if self.augmented_latent_size>0:
            layers += [InvertibleSparseTranspose(
                self.latent_size,
                self.augmented_latent_size,
                self.input_size if self.auxiliary else 0,
                sub_network_layers=self.sub_network_layers,
                sub_network_cells=self.sub_network_cells,
                sub_network_activation=self.sub_network_activation,
                sub_network_batch_norm=self.sub_network_batch_norm,
                lagrangian_rho=1,
                lagrangian_lambda=0,
                lagrangian_beta=2,
                temperature=0.1,
                mapmode='down',
            )]

        if self.shift_layer:
            layers += [ShiftLayer(self.latent_size)]

        self.layers = nn.Sequential(*layers)

    def get_cost(self):
        costs = []
        self.apply(lambda x: self._get_cost_layer(x, costs))
        return sum(costs) if len(costs)>0 else 0

    def step_lagrangian(self):
        self.apply(self._step_lagrangian)

    def get_cost_generated(self, y):
        prev_mode = self.mode
        dist = self(x)

        cost = self.get_cost()
        self.set_mode(prev_mode)
        return cost

    @staticmethod
    def _step_lagrangian(layer):
        if isinstance(layer, AugmentedLagrangian):
            layer.step()

    @staticmethod
    def _get_cost_layer(layer, costs=[]):
        if hasattr(layer, '_get_lagrangian_cost'):
            costs.append(layer._get_lagrangian_cost())

    def set_auxiliary(self, x):
        for layer in self.layers:
            # if isinstance(layer, AbstractInvertibleLayer):
            layer.set_auxiliary(x)

    def forward(self, x):
        self.layers.apply(EraseCache())

        z = x
        for layer in self.layers:
            z = layer(z)
        return z

    def backward(self, z):
        self.layers.apply(EraseCache())

        for layer in reversed(self.layers):
            z = layer._inverse(z)
        return z

    def _make_dist(self, x):
        if hasattr(self, 'embedding_layer'):
            mu_sigma = self.embedding_layer(x)
            mu, sigma = mu_sigma[..., :self.latent_size], log1p_softplus(mu_sigma[..., self.latent_size:])
            mu = tanh5scale(mu) if self.tanh5scaling else mu
            dist = normal.Normal(mu, sigma)
        else:
            dist = torch.distributions.Normal(
                    torch.zeros(*x.shape[:-1], self.latent_size, device=x.device),
                    self.embedding_sd.view(
                        *[1,]*(x.ndimension()-1), self.latent_size))
        return dist

    def erase_cache(self):
        for l in self.layers:
            l.erase_cache()

    def compose_dist(self, x, return_log_prob=True):
        dist = self._make_dist(x)
        dist = NFTransformedDistribution(
                distributions.Independent(
                    normal.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale)), 1),
                    [*self.layers,
                    AffineTransformNF(
                        dist.loc,
                        dist.scale,
                    )]
                )

        return dist

class NFTransformedDistribution(torch.distributions.TransformedDistribution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._non_invertible_flow = False
        for p in self.transforms:
            if isinstance(p, (PlanarFlow, RadialFlow, HouseHolderFlow)):
                self._non_invertible_flow = True
                break

    def log_prob(self, value):
        if self._non_invertible_flow:
            with torch.no_grad():
                epsilon = reversed_non_inv(self.transforms, value)
            with gradient_transfer_backward_transform():  # ensures that gradient are computed as should be
                y = sample_given_auxiliary(self, epsilon)  # to cache forward values in case analytical backward does not exist
                log_abs_det_jacobian = torch.zeros_like(value[..., 0])
                for t in reversed(self.transforms):
                    x = t.inv(y)
                    log_abs_det_jacobian += t.log_abs_det_jacobian(x, y).view(log_abs_det_jacobian.shape)
                    y = x
                y = neg_grad(y)
                epsilon_lp = self.base_dist.log_prob(y)
                log_p = epsilon_lp - log_abs_det_jacobian.view(x.shape[:-1])
                return log_p
        else:
            return super().log_prob(value)

def reversed_non_inv(seq, value, decay=0.95, maxiter=500, x_guess=None):
    if x_guess is None:
        if not isinit_nf(seq):
            x_guess = torch.zeros_like(value)
        else:
            x_guess = seq[0]._cache['input']
    alpha = 1.0
    epsilon_sim = inv_nf(seq, x_guess, value, alpha, decay=decay, max_iter=maxiter)
    return epsilon_sim.clone()


def isinit_nf(seq):
    return all([t._cache['Jacob'] is not None for t in seq])

def inv_nf2(seq, y, alpha_init=1.0, decay=0.99, max_iter=1000):
    count = 0
    alpha = alpha_init
    with torch.no_grad():
        while True:
            y_prev = seq[-1]._cache['output']
            x_prev = seq[0]._cache['input']
            j = seq[0]._cache['Jacob']
            for t in seq[1:]:
                j = j @ t._cache['Jacob']
            dy = (y-y_prev)
            dx, _ = torch.solve(dy.unsqueeze(-1), j)
            x = x_prev + dx.squeeze(-1)
            y_sim = x.clone()
            for t in seq:
                y_sim = t(y_sim)
            d = (y-y_sim).norm(dim=-1).max()
            if count == 0:
                d_init = d
            if d<1e-6 or count==max_iter:
                print(f'distance: {d} (d init: {d_init}) at count {count}')
                break
            count += 1
            alpha *= decay
    return x

@torch.no_grad()
def inv_nf_upd(t, y: torch.Tensor, y_guess: torch.Tensor, jacob: torch.Tensor, x_guess: torch.Tensor, d: torch.Tensor,
               idx: torch.Tensor, dtype, dtype_orig, alpha):
    y_idx = y[idx]
    y_guess_idx = y_guess[idx]
    dy = y_idx - y_guess_idx
    jacob_idx = jacob[idx]
    x_guess_idx = x_guess[idx]

    dx = torch.solve(dy.unsqueeze(-1).to(dtype), jacob_idx.to(dtype))[0].to(dtype_orig)
    x_guess.masked_scatter_(idx.unsqueeze(-1).expand_as(x_guess), x_guess_idx+alpha * dx.squeeze(-1))

    _y_guess, _, _jacob = t.forward_transform(x_guess[idx], t.auxiliary_input[idx])
    y_guess.masked_scatter_(idx.unsqueeze(-1).expand_as(y_guess), _y_guess)
    jacob.masked_scatter_(idx.unsqueeze(-1).unsqueeze(-1).expand_as(jacob), _jacob)

    d_new = (y_guess[idx] - y_idx).norm(dim=-1)
    dd = abs(d[idx] - d_new).max()
    d.masked_scatter_(idx, d_new)
    return dd

@torch.no_grad()
def inv_nf(seq, x_guess, y, alpha_init=1.0, decay=0.95, max_iter=1000, to_double=True, thr=1e-6):
    dtype_orig = x_guess.dtype
    dtype=torch.double if to_double else torch.float
    forward_nf(seq, x_guess.float())
    for t in reversed(seq):
        if t.invertible:
            y = t.inv(y)
            continue
        alpha = alpha_init
        idx = torch.ones(t._cache['Jacob'].shape[:-2], device=y.device, dtype=torch.bool)
        count = 0
        x_guess = t._cache['input'].clone()
        y_guess = t._cache['output'].clone()
        Jacob = t._cache['Jacob']
        d = torch.full(idx.shape, np.inf, device=y.device, dtype=y.dtype)
        while True:
            dd = inv_nf_upd(t, y, y_guess, Jacob, x_guess, d, idx, dtype, dtype_orig, alpha)
            d_max = d.max()
            if count==0:
                d_init = d_max
            idx = d>thr
            if idx.sum()==0 or dd<thr:
                break
            if count==max_iter:
                print(f'failed at count {count} with d {d_max} (d init: {d_init})')
                break
            count += 1
            alpha = alpha*decay

        y = x_guess.clone()
    eps = y
    return eps.to(torch.float)

def LM_step(J, x_guess, dy, lam):
    dtype = x_guess.dtype
    X,_ = torch.solve(dy, J.transpose(-2,-1))
    I = torch.eye(J.shape[-1], device=J.device,)# dtype=torch.double)
    C = lam.unsqueeze(-1).unsqueeze(-1)*I.expand_as(J)
    JtJ = J.transpose(-2,-1)@J
    dx,_ = torch.solve(X, JtJ+C)
    pred_red = (-dx).transpose(-2,-1) @ (2*X+JtJ@dx)
    return x_guess + dx.to(dtype).view_as(x_guess), pred_red.squeeze(-1).squeeze(-1), dx, X, JtJ

    dtype = x_guess.dtype
    A = J.transpose(-2,-1) @ J
    I = torch.eye(J.shape[-1], device=J.device,)# dtype=torch.double)
    X = (J.transpose(-2,-1) @ dy).to(torch.double)
    C = lam.unsqueeze(-1)*I.expand_as(J)
    dx, _ = torch.solve(X, I+(C @ A).to(torch.double))
    dx = dx.to(dtype).view_as(x_guess)
    pred_red = d.dot(2*JtJ)
    return x_guess + dx, dx


@torch.no_grad()
def inv_nf_lm(seq, x_guess, y, alpha_init=1.0, decay=0.99, max_iter=100, to_double=True, thr=1e-6):
    forward_nf(seq, x_guess.float())
    y_init = y.clone()
    for t in reversed(seq):
        if t.invertible:
            y = t.inv(y)
            continue
        d = np.inf
        count = 0

        r_high = 0.75
        r_low = 0.25

        x_guess = t._cache['input'].clone()
        dy = (y - t._cache['output']).unsqueeze(-1)
        J = t._cache['Jacob'].clone()
        lam = torch.full_like(dy[...,0,0], 1.0)
        lam_c = torch.full_like(dy[..., 0,0], 0.75)
        prev_loss = (t(x_guess.clone())-y).pow(2).sum(-1)
        while True:
            new_x, pred_red, dx, X, JtJ = LM_step(J, x_guess, dy, lam)
            loss = (t(new_x.clone())-y).pow(2).sum(-1)
            ratio = (prev_loss-loss)/pred_red

            idx = ratio > r_high #loss<=prev_loss
            if idx.any():
                lam_upd_high(lam[idx], lam_c[idx])

            idx = ratio < r_low
            if idx.any():
                lam_upd_low(lam[idx], lam_c[idx], loss[idx], prev_loss[idx], dx[idx], X[idx], JtJ[idx])

            idx = loss<prev_loss
            x_guess[idx] = new_x[idx]
            J[idx] = t._cache['Jacob'][idx]
            prev_loss[idx] = loss[idx]
            dy[idx] = (y-t._cache['output']).unsqueeze(-1)[idx]

            dd = np.sqrt(prev_loss.max().item())-d
            d = np.sqrt(prev_loss.max().item())
            print(f'iter {count}, d: {prev_loss.max(), prev_loss.min()}')
            if count==0:
                d_init = d
            if loss.max()<thr: # or abs(dd)<thr:
                break
            if count==max_iter:
                print(f'failed at count {count} with d {d} (d init: {d_init})')
                break
            count += 1
        y = x_guess.clone()
    eps = y
    print('total error:')
    y = forward_nf(seq, y)
    print((y-y_init).norm(dim=-1).max())
    return eps.to(torch.float)

def lam_upd_high(lam, lam_c, idx):
    lam.masked_scatter_(idx, lam[idx]/2)
    lam.masked_fill_(idx & (lam < lam_c), 0.0)

def lam_upd_low(lam, lc, loss, prev_loss, dx, X, JtJ, idx):
    nu = (loss[idx] - prev_loss[idx])
    nu /= (-dx[idx].transpose(-2,-1)@X[idx]).view_as(nu)
    nu += 2
    nu.clamp_(2, 10)
    ids0 = lam == 0
    if (ids0 & idx).any():
        lc.masked_scatter_((ids0 & idx), abs(JtJ[(idx & ids0)].inverse().diagonal()).max(dim=-1)[0].reciprocal())
        lam.masked_scatter_((ids0 & idx), lc[(idx & ids0)])
        nu.masked_scatter_(ids0, nu[ids0] / 2)
    lam.masked_scatter_(idx, lam*nu)

def forward_nf(seq, value):
    for t in seq:
        value = t(value)
    return value

class NFTrainer:
    def __init__(self, data, nf, batch, epochs, niter, device='cuda:0', thr = 0.1, callback=None, lr=1e-2):
        self.data = data
        self.nf = nf
        self.batch = batch
        self.epochs = epochs
        self.niter = niter
        self.device = device
        self.thr = thr
        self.optim = torch.optim.Adam(nf.parameters(), lr=lr, weight_decay=1e-4)
        self.lr_decay = 0.95
        self.callback = callback

    def __call__(self):
        return self.train_nf()

    def train_nf(self):
        # nf has n_states + 1 inputs

        data = self.data

        print('\n')
        for e in range(self.epochs):
            if self.callback is not None:
                self.callback(e)

            nf = self.nf.train()

            data_iter = iter(data)
            pbar = tqdm.tqdm(enumerate(data_iter), total=self.niter, file=sys.stdout)
            for i,(y,x) in pbar:
                if i==self.niter:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                dist = nf(x)
                logp = dist.log_prob(y)
                loss = -logp.mean()
                loss_item = loss.item()
                cost = nf.get_cost() + nf.get_cost_generated(x)
                if cost is not 0:
                    cost_item = cost.item()
                else:
                    cost_item = 0
                loss = loss + cost
                loss.backward()
                nn.utils.clip_grad_norm_(nf.parameters(), 10.0)
                self.optim.step()
                self.optim.zero_grad()
                pbar.set_description(f'epoch: {e}, loss: {loss_item:4.4f}, cost: {cost_item:4.4f}')
            nf.step_lagrangian()

            self.optim.param_groups[0]['lr'] *= self.lr_decay

    def plot_histogram_over_trajectory(self, t_interval, nbins=32, batch=int(1e4), device='cuda:0', dest=f'./out/'):
        """
        Plots a histogram of state-to-state transition probabilities given samples and NF for comparision
        """
        nf = self.nf.eval()
        s = self.state_sampling(1).view(-1)
        s_expand = s.expand((batch, s.numel()))
        s_ = s_expand
        for t in range(self.time_horizon):
            a = self.action_sampling(s_)
            s_ = self.sas_sampling(s_, a)

            total_true_hists = []
            total_nf_hists = []
            if t % t_interval == 0:
                true_hists, nf_hists = self._plot_hist(s_, t, s, nbins)
                total_true_hists.append(true_hists)
                total_nf_hists.append(nf_hists)
                f, ax = plt.subplots(3,2,figsize=(15,10))
                for i,ax_r in enumerate(ax):
                    ax_r[0].imshow(true_hists[i])
                    ax_r[1].imshow(nf_hists[i])
                plt.savefig(os.path.join(dest,f'hist_{t}.png'))
                plt.close(f)

    def _plot_animate(self, true_hists, nf_hists):
        f, ax = plt.subplots(3, 2, figsize=(15, 10))
        def animate(j):
            for i, ax_r in enumerate(ax):
                ax_r[0].imshow(true_hists[j][i])
                ax_r[1].imshow(nf_hists[j][i])

        anim = animation.FuncAnimation(f, animate,
                                       frames=len(true_hists),
                                       interval=1, blit=True)

    def _plot_hist(self, s, t, s0, nbins):
        assert s.shape[-1] == 3, "Only 3d states supported at this moment"
        coords = [[]]*3
        true_hists = []
        for dim_couple in ((0, 1), (0, 2), (1, 2)):
            x,y = s[:,dim_couple[0]].cpu().numpy(), s[:,dim_couple[1]].cpu().numpy()
            _hist, coords[dim_couple[0]], coords[dim_couple[1]] = np.histogram2d(x, y, bins=nbins)
            true_hists.append(_hist)

        X, Y, Z = np.meshgrid(*coords)
        torch_coords = torch.tensor([X.reshape(-1),Y.reshape(-1),Z.reshape(-1), ],
                                    device=self.device, dtype=torch.float).permute(1,0)

        nf_hist = []
        s0 = torch.cat([
            s0.expand((torch_coords.shape[0], s0.numel())),
            t*torch.ones(Z.size,1,device=s0.device)],dim=-1)
        with torch.no_grad():
            lp = self.nf(torch_coords,s0,).reshape(len(X), len(X), len(X))
        for dim_couple in ((0,1),(0,2),(1,2)):
            dim_not = [i for i in range(3) if i not in dim_couple]
            _lp = torch.logsumexp(lp, dim=dim_not)-np.log(len(X))
            nf_hist.append(_lp.exp().cpu().numpy())
        return true_hists, nf_hist

if __name__ == "__main__":
    input_size = 5
    batch = 11
    x = torch.randn(input_size, batch)
    y = torch.randn(input_size, batch)

    nf = NF(
        input_size,
        5,
        2,
    )

    dist = nf(x)
    import pdb
    pdb.set_trace()
