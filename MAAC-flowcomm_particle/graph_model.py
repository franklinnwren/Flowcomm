import torch
import torch.nn.functional as F
import numpy as np
import igraph as ig
import pdb
import torch as th
from torch.distributions import Transform

from torch import nn
from torch import distributions
import numbers
import time
from torch import autograd
try:
    from agents.nfLayers.neural_net import NeuralNet
except:
    from nfLayers.neural_net import NeuralNet

try:
    from agents.nfLayers.nf import NF
except:
    from nfLayers.nf import NF

import sys

try:
    from agents.nfLayers import disc_utils
except:
    from nfLayers import disc_utils

# set hyparameter
rho, alpha = 1., 1. # for h func

NATIVE_GRADIENT = True

# add coupling flow
class GraphFlows(nn.Module):
    def __init__(self, n_s=10, n_agent=5, n_step=0, n_n=0, policy_name='lstm', agent_name='graph_player', n_fc=64,
                 n_lstm=64):
        super(GraphFlows, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_step = n_step
        self.batch_size = self.n_step
        # init for flow
        self.nagt = n_agent # number of agents
        self.use_entropy = True
        self.use_dag = False
        self.out_size = n_agent **2 # size of A
        self.vocab_size = 2 # 0 or 1 for A_ij
        self.sample_num = 1000

        # init parameters for distribution
        self.base_logit_probs = torch.tensor(torch.rand(self.out_size, self.vocab_size), requires_grad=True)

        n_auxiliary = n_s # dim of auxiliary feature, auxiliary should has same size as input.

        self.flow_model = NF(
            n_auxiliary,
            latent_size=self.out_size,
            nlayers=3,
            layer_type=DiscreteCouplingFlow,
            sub_network_layers=3,
            sub_network_cells=10,
            sub_network_activation=th.nn.Tanh,
            sub_network_batch_norm=False,
            scaled=True,
            scale_value=0.9,
        )

        self.graph_optim = torch.optim.Adam([{'params':self.flow_model.parameters(),'lr':1e-2},
                                             {'params':self.base_logit_probs,'lr':1e-2}], weight_decay=1e-2)


    def sample_graphs_from_base(self, batch_size, base_log):

        def sample_gumbel(n, k):
            unif = torch.distributions.Uniform(0, 1).sample((n, k))
            g = -torch.log(-torch.log(unif))
            return g

        def sample_gumbel_softmax(pi, n, temperature):
            k = len(pi)
            g = sample_gumbel(n, k)
            h = (g + torch.log(pi)) / temperature
            h_max = h.max(dim=1, keepdim=True)[0]
            h = h - h_max
            cache = torch.exp(h)
            y = cache / cache.sum(dim=-1, keepdim=True)
            return y

        softmax_func = torch.nn.Softmax(dim=1)
        pi = softmax_func(base_log)
        outs = []
        for idx in range(len(pi)):
            y = sample_gumbel_softmax(pi[idx], batch_size, temperature=1.)
            outs.append(y.unsqueeze(0))
        b = torch.cat(outs, dim=0)
        out = b.transpose(0, 1)
        out1 = torch.argmax(out, dim=2)

        def sample_gumbel_softmax_batch(pi, n, temperature):
            k = len(pi[0])
            g = sample_gumbel(n * len(pi), k).reshape(n, len(pi), k)
            h = (g + torch.log(pi)) / temperature
            h_max = h.max(dim=2, keepdim=True)[0]
            h = h - h_max
            cache = torch.exp(h)
            y = cache / cache.sum(dim=-1, keepdim=True)
            return y

        out = sample_gumbel_softmax_batch(pi, batch_size, temperature=1.)
        out = torch.argmax(out, dim=2)
        return out

    
    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))


    def save(self, path):
        torch.save(self.state_dict(), path)

        
    def load(self, path):
        self.load_state_dict(torch.load(path))

        
    def forward(self,ob, *_args, **_kwargs):

        if isinstance(ob, list):
            ob1=ob[0]
        else:
            ob1 = ob
        ob2 = torch.from_numpy(ob1).float()
        self.flow_model.set_auxiliary(ob2)
        self.prior = self.sample_graphs_from_base(len(ob), self.base_logit_probs)
        self.xs = self.flow_model.forward(self.prior).squeeze()

        priorv =self.prior.reshape(1*self.nagt*self.nagt).long() # convert to int64
        priorv = torch.nn.functional.one_hot(priorv, num_classes=2).reshape(1, self.nagt*self.nagt,2)

        self.base_log_probs_sm = torch.nn.functional.log_softmax(self.base_logit_probs, dim=-1)
        log_As_probs = priorv * self.base_log_probs_sm # zs are onehot so zero out all other logprobs.
        log_As_probs = th.sum(log_As_probs, dim=(1,2))  

        As = self.xs.reshape([-1,self.nagt, self.nagt]).squeeze().float()

        return As, log_As_probs


    def backward(self, obs, qs, As, log_As_probs, nactions=None, acts=None, dones=None, Rs=None, Advs=None,e_coef=5e-3, v_coef=None, summary_writer=None, global_step=None,nbatch=None):

        # if use auxiliary feature
        qs = torch.from_numpy(qs).float()

        As_tensor = torch.flatten(As,start_dim=1)

        log_As_probs = torch.reshape(log_As_probs,(-1,1))

        self.flow_model.set_auxiliary(obs)

        zs = self.flow_model.backward(As_tensor).squeeze()

        zs = torch.cat([zs.unsqueeze(-1), (1 - zs).unsqueeze(-1)], dim=-1).reshape(self.batch_size, self.nagt*self.nagt,2)
        self.base_log_probs_sm = torch.nn.functional.log_softmax(self.base_logit_probs, dim=-1)
        logprobs = zs * self.base_log_probs_sm  # zs are onehot so zero out all other logprobs.
        logprobs = th.sum(logprobs, dim=(1, 2))

        logprobs = torch.reshape(logprobs,(-1,1))

        rho_loss = -((logprobs-log_As_probs).exp() * qs).mean()

        loss = rho_loss
        if self.use_entropy:
            sample_As = self.sample_graphs_from_base(self.sample_num, self.base_logit_probs)
            sample_Asv = sample_As.reshape(self.sample_num * self.nagt * self.nagt).long()  # convert to int64
            sample_Asv = torch.nn.functional.one_hot(sample_Asv, num_classes=2).reshape(self.sample_num,
                                                                                        self.nagt * self.nagt, 2)
            entropy_loss = (logprobs).mean() * e_coef
            loss += entropy_loss

        if self.use_dag:
            h_val = self.h_func(As)
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            loss += penalty

        loss.backward()
        self.graph_optim.step()
        self.graph_optim.zero_grad()

        for params in self.parameters():
            pass
        param_index =0 
        for name, param in self.state_dict().items():
            param_index +=1

        return loss.item()


    def h_func(self, As):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        # batch wise loss
        M0 = th.eye(self.nagt)
        M0 = M0.reshape((1, self.nagt, self.nagt))
        M0 = M0.repeat(self.n_step, 1, 1)
        M = M0 + th.from_numpy(As)/self.nagt
        E =th.matrix_power(M, self.nagt - 1)
        hs = ((th.transpose(E, 1, 2)*M).sum() - self.nagt*self.n_step)/self.nagt
        # pdb.set_trace()
        return hs


class EraseCache():
    def __call__(self, layer):
        if isinstance(layer, AbstractInvertibleLayer):
            layer.erase_cache()


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def sample_given_auxiliary(dist, epsilon):
    global NATIVE_GRADIENT
    if isinstance(dist, distributions.Normal):
        loc = dist.loc
        scale = dist.scale

        return loc + scale * epsilon

    if isinstance(dist, distributions.Uniform):
        low = dist.low
        high = dist.high
        if not NATIVE_GRADIENT:
            low = low.detach()
            high = high.detach()
        return low + (high - low) * epsilon
    elif isinstance(dist, distributions.Independent):
        return sample_given_auxiliary(dist.base_dist, epsilon)
    elif isinstance(dist, distributions.TransformedDistribution):
        sample = sample_given_auxiliary(dist.base_dist, epsilon)
        for t in dist.transforms:
            sample = t(sample)
        return sample
    elif isinstance(dist, MultitaskMultivariateNormal):
        raise NotImplementedError()
    elif isinstance(dist, MultivariateNormal):
        L = dist.lazy_covariance_matrix.root_decomposition().root.transpose(-2, -1)
        epsilon = epsilon.unsqueeze(-1)
        loc = dist.loc
        while loc.ndimension() < epsilon.ndimension():
            loc = loc.unsqueeze(-1)
        result = (loc + L @ epsilon).squeeze(-1)
        return result
    else:
        raise NotImplementedError(f'type {type(dist)} not supported')


class gradient_transfer_backward_transform():
    prev_val = True

    def __enter__(self):
        global NATIVE_GRADIENT
        self.prev_val = NATIVE_GRADIENT
        NATIVE_GRADIENT = False

        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global NATIVE_GRADIENT
        NATIVE_GRADIENT = self.prev_val


def inv_softplus(x):
    if isinstance(x, numbers.Number):
        x = torch.tensor([x])
    return torch.log(torch.expm1(x))


SCALE_VALUE = 1.0


class ScaleLayer(nn.Module):

    def __init__(self, init_value=None, scale_mode='linear'):
        super().__init__()
        if init_value is None:
            init_value = SCALE_VALUE
        self.scale_mode = scale_mode
        if scale_mode == 'linear':
            self.scale = nn.Parameter(torch.tensor([init_value]))
        elif scale_mode == 'softplus':
            self.scale = nn.Parameter(inv_softplus(torch.tensor([init_value])))
        elif scale_mode == 'sigmoid':
            self.scale = nn.Parameter(torch.tensor([init_value / (1 - init_value)]).log())
        else:
            raise NotImplementedError()

        
    def get_scale(self):
        if self.scale_mode == 'linear':
            scale = self.scale
        elif self.scale_mode == 'softplus':
            scale = F.softplus(self.scale)
        elif self.scale_mode == 'sigmoid':
            scale = torch.sigmoid(self.scale)
        return scale

    
    def forward(self, input):
        return self.get_scale() * input


class AbstractInvertibleLayer(Transform, nn.Module):  # , Transform):
    event_dim = 0
    grad_mode = True
    invertible = True

    def __init__(self):
        super(AbstractInvertibleLayer, self).__init__()
        self.auxiliary_input = None
        self._cache = {'input': None, 'output': None, 'ldj': None, 'Jacob': None}

        
    def __hash__(self):
        return super(nn.Module, self).__hash__()


    def set_auxiliary(self, auxiliary):
        # self.register_buffer('auxiliary_input', auxiliary)
        self.auxiliary_input = auxiliary

        
    def forward_transform(self, input, auxiliary_input=None):
        raise NotImplementedError()

    
    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        raise NotImplementedError()

    
    def set_gradient_mode_transform(self, mode=True):
        # for reparameterization in PF
        self.grad_mode = mode
        for c in self.children():
            if isinstance(c, AbstractInvertibleLayer):
                c.set_gradient_mode_transform(mode)

                
    def forward(self, x, logp=None, auxiliary_input=None):
        raise NotImplementedError()

    
    def log_abs_det_jacobian(self, x, y):
        if x is self._cache['input']:
            ldj = self._cache['ldj']
        else:
            if not NATIVE_GRADIENT:
                x = self._cache['input_detach']
                x = neg_grad(x)
            y, ldj, Jacob = self.forward_transform(x, auxiliary_input=self.auxiliary_input)
    
        return ldj.unsqueeze(-1)

    
    def inv(self, y):
        x = self.backward_transform(y, auxiliary_input=self.auxiliary_input)
        ldj = None
        self._cache.update({
            'input': x,
            'output': y,
            'ldj': ldj
        })
        self.set_gradient_mode_transform(True)
        return x

    
    def _inverse(self, y):
        return self.inv(y)

    
    def __call__(self, x):
        if x is self._cache['input']:
            return self._cache['output']
        x_input = x
        if not NATIVE_GRADIENT:
            x_input = x.detach()
    
        y = self.forward_transform(x_input, auxiliary_input=self.auxiliary_input)
        ldj, Jacob = None, None
        if Jacob is None and not NATIVE_GRADIENT:
            print('Wagning: assessing Jacobian using autograd!')
            Jacob = torch.stack([
                torch.autograd.functional.jacobian(
                    lambda x: self.forward_transform(x, auxiliary_input=self.auxiliary_input)[0],
                    (x_input[i].unsqueeze(0),))[0].squeeze()
                for i in range(x_input.shape[0])], 0)
        self._cache.update({
            'input': x,
            'output': y,
            'ldj': ldj,
            'Jacob': Jacob
        })
        return y

    
    def clone_and_copy_data(self, x, y, Jacob):
        out = jacob_mult(y, x, Jacob)
        return out

    
    def erase_cache(self):
        self._cache = {k: None for k in self._cache}


class NegGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    
    def backward(ctx, grad_output):
        return - grad_output


neg_grad = NegGrad.apply


class JacobMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, jacob):
        ctx.save_for_backward(jacob.detach())
        return x

    
    @staticmethod
    def backward(ctx, grad_output):
        jacob, = ctx.saved_tensors
        dtype = grad_output.dtype
        B = grad_output.unsqueeze(-1).to(torch.double)
        A = jacob.transpose(-2, -1).to(torch.double)
        sol, _ = torch.solve(B, A)
        G = sol.squeeze(-1).to(dtype)

        return G, grad_output, None


jacob_mult = JacobMult.apply


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    
class DiscreteCouplingFlow(AbstractInvertibleLayer):
    invertible = True

    def __init__(self,
                 input_size,
                 output_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 scaled=True,
                 partition_size=None,
                 biased=False,
                 scale_mode='linear',
                 scale_value=None,
                 idx_a=None,
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self.biased = biased
        self.temperature = 0.1 # taken from Tran2016
        if partition_size is None:
            partition_size = self.output_size // 2
        self.dim_split_a = partition_size
        self.input_size = input_size + partition_size

        self.dim_split_b = self.output_size - partition_size
        if idx_a is None:
            rp = torch.randperm(self.output_size)
            idx_a = rp[:self.dim_split_a]
            idx_b = rp[self.dim_split_a:]
            inverse_idx = rp.argsort()

        self.register_buffer('idx_a', idx_a)
        self.register_buffer('idx_b', idx_b)
        self.register_buffer('inverse_idx', inverse_idx)

        self._build_layer()

        
    def _build_layer(self):
        self.transform_ft = NeuralNet(
            self.input_size,
            self.dim_split_b * 2 if self.biased else self.dim_split_b,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=True,
            batch_norm=self.sub_network_batch_norm,
            last_layer=ScaleLayer(self.scale_value, self.scale_mode) if self.scaled else None
        )
        self.register_buffer('bias_softplus', inv_softplus(1.0))


    def forward_transform(self, input, auxiliary_input=None):
        input_a, input_b = input[..., self.idx_a], input[..., self.idx_b]
        if auxiliary_input is not None:
            ndimdiff = input_a.ndimension() - auxiliary_input.ndimension()
            if ndimdiff > 0:
                auxiliary_input = auxiliary_input.expand(*input_a.shape[:-1], auxiliary_input.shape[-1])
            input_net = torch.cat([ input_a, auxiliary_input], dim=-1)
        else:
            input_net = input_a

        f = self.transform_ft(input_net)

        if self.biased:
            f, t = f[..., :self.dim_split_b], f[..., self.dim_split_b:]
        else:
            t = 0

        binAct = BinActive.apply
        f1 = binAct(f)
        f=(f1+1.)/2. # rescale -1,1 to 0,1

        if not self.grad_mode:
            f = f.detach()
            if t is not 0:
                t = t.detach()
            self.set_gradient_mode_transform(True)

        input_by  = 1. - input_b
        xb = th.cat([input_b.unsqueeze(-1), input_by.unsqueeze(-1)], dim = -1)
        fy = 1. - f
        mxa = th.cat([f.unsqueeze(-1), fy.unsqueeze(-1)], dim=-1)

        mix_output_b = disc_utils.one_hot_add(xb, mxa)
        mix_output_b = mix_output_b[:,:, 1]

        output = torch.cat([input_a, mix_output_b], dim=-1)

        output = output[..., self.inverse_idx]

        return output


    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        input_a = output[..., self.idx_a]
        if auxiliary_input is not None:
            ndimdiff = input_a.ndimension() - auxiliary_input.ndimension()
            if ndimdiff>0:
                auxiliary_input = auxiliary_input.expand(*input_a.shape[:-1], auxiliary_input.shape[-1])
            input_net = torch.cat([
                input_a,
                auxiliary_input
            ], dim=-1)
        else:
            input_net = input_a
        f = self.transform_ft(input_net)
        if self.biased:
            f, t = f[..., :self.dim_split_b], f[..., self.dim_split_b:]
        else:
            t = 0
        binAct = BinActive.apply
        f1 = binAct(f)
        f = (f1 + 1.) / 2.  # rescale -1,1 to 0,1

        output_b = output[..., self.idx_b]
        if not self.grad_mode:
            f = f.detach()
            if t is not 0:
                t = t.detach()
            self.set_gradient_mode_transform(True)

        output_by = 1. - output_b
        yb = th.cat([output_b.unsqueeze(-1), output_by.unsqueeze(-1)], dim=-1)
        fy = 1. - f
        mya = th.cat([f.unsqueeze(-1), fy.unsqueeze(-1)], dim=-1)

        mix_input_b = disc_utils.one_hot_minus(yb, mya)

        mix_input_b = mix_input_b[:,:,-1]

        input = torch.cat([input_a, mix_input_b], dim=-1)
        input = input[..., self.inverse_idx]

        return input #, ldj

    
    def reverse(self, output, mf):
        """Reverse pass for left-to-right autoregressive generation. Latent to data.
        Expects to recieve a onehot."""

        length = output.shape[-2]
        # Slowly go down the length of the sequence.
        # the batch is computed in parallel, dont get confused with it and the sequence components!
        # From initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
        # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
        outputs = self._initial_call(output[:, 0, :], length, mf)

        for t in range(1, length):
            outputs = self._per_timestep_call(outputs,
                                              output[..., t, :],
                                              length,
                                              t,
                                              **kwargs)

        return outputs

    
    def _initial_call(self, new_inputs, length, mf):
        """Returns Tensor of shape [..., 1, vocab_size].
        Args:
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output.
        length: Length of final desired sequence.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = new_inputs.unsqueeze(1)  # new_inputs[..., tf.newaxis, :] # batch x 1 x onehots

        padded_inputs = F.pad(
            inputs, (0, 0, 0, length - 1))

        """
        All this is doing is filling the input up to its length with 0s. 
        [[0, 0]] * 2 + [[0, 50 - 1], [0, 0]] -> [[0, 0], [0, 0], [0, 49], [0, 0]]
        what this means is, dont add any padding to the 0th dimension on the front or back. 
        same for the 2nd dimension (here we assume two tensors are for batches), for the length dimension, 
        add length -1 0s after. 

        """
        net = self.layer(padded_inputs, **kwargs)  # feeding this into the MADE network. store these as net.
        if net.shape[-1] == 2 * self.vocab_size:  # if the network outputted both a location and scale.
            loc, scale = torch.split(net, self.vocab_size,
                                     dim=-1)  # tf.split(net, 2, axis=-1) # split in two into these variables
            loc = loc[..., 0:1, :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., 0:1, :]
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = disc_utils.multiplicative_inverse(scale,
                                                              self.vocab_size)  # could be made more efficient by calculating the argmax once and passing it into both functions.
            shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
            outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., 0:1, :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            outputs = disc_utils.one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        pdb.set_trace()
        return outputs

    
    def _per_timestep_call(self,
                           current_outputs,
                           new_inputs,
                           length,
                           timestep,
                           **kwargs):
        """Returns Tensor of shape [..., timestep+1, vocab_size].
        Args:
        current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
            generated sequence Tensor.
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output given current_outputs.
        length: Length of final desired sequence.
        timestep: Current timestep.
        **kwargs: Optional keyword arguments to layer.
        """
        inputs = torch.cat([current_outputs,
                            new_inputs.unsqueeze(1)], dim=-2)
    
        padded_inputs = F.pad(
            inputs, (0, 0, 0, length - timestep - 1))  # only pad up to the current timestep

        net = self.layer(padded_inputs, **kwargs)
        if net.shape[-1] == 2 * self.vocab_size:
            loc, scale = torch.split(net, self.vocab_size, dim=-1)
            loc = loc[..., :(timestep + 1), :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            scale = scale[..., :(timestep + 1), :]
            scale = disc_utils.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
            inverse_scale = disc_utils.multiplicative_inverse(scale, self.vocab_size)
            shifted_inputs = disc_utils.one_hot_minus(inputs, loc)
            new_outputs = disc_utils.one_hot_multiply(shifted_inputs, inverse_scale)
        elif net.shape[-1] == self.vocab_size:
            loc = net
            loc = loc[..., :(timestep + 1), :]
            loc = disc_utils.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
            new_outputs = disc_utils.one_hot_minus(inputs, loc)
        else:
            raise ValueError('Output of layer does not have compatible dimensions.')
        outputs = torch.cat([current_outputs, new_outputs[..., -1:, :]], dim=-2)
        pdb.set_trace()

        return outputs

    
def test_auxiliary():
    from nfLayers.nf import NF
    n_aux = 14

    batch_size = 3

    N = 5
    out_size, vocab_size = N ** 2, 2

    # data process
    c1 = DiscreteCouplingFlow(n_aux, out_size, 3, 10, torch.nn.Tanh, False, True, scale_value=0.9)

    logits = torch.tensor(torch.rand(out_size, vocab_size), requires_grad=True)
    graph_flow = GraphFlows()
    x = graph_flow.sample_graphs_from_base(batch_size, logits).float()

    y = torch.randn(batch_size, n_aux) + 1
    c1.set_auxiliary(y)

    z = c1(x)
    c1.erase_cache()
    xx = c1._inverse(z)
    print('reverse xx', xx - x)

    grad2 = torch.autograd.grad(z.sum(), c1.parameters(), retain_graph=True, allow_unused=True)
    print('grad,',grad2)

    # --- flow model
    flow = NF(
        n_aux,
        latent_size=out_size,
        nlayers=3,
        layer_type=DiscreteCouplingFlow,
        sub_network_layers=3,
        sub_network_cells=10,
        sub_network_activation=th.nn.Tanh,
        sub_network_batch_norm=False,
        scaled=True,
        scale_value=0.9,
    )

    flow.set_auxiliary(y)

    output = flow(x)
    print('output,', output)

    xf = flow.backward(output)
    print('reverse xf', xf - x)

    zs = torch.cat([xf.unsqueeze(-1), (1-xf).unsqueeze(-1)],dim=-1).reshape(batch_size, N*N, 2)
    logprobs = zs * logits  # zs are onehot so zero out all other logprobs.
    logprobs = th.sum(logprobs, dim=(1, 2))

    grad2 = torch.autograd.grad(logprobs.sum(), flow.parameters(), retain_graph=True, allow_unused=True)

    print('grad,',grad2)


if __name__ == "__main__":
    test_auxiliary()
