import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample
import time


class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, messg_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()
        self.input_dim = input_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim + messg_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin


    def forward(self, X, message):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot

        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        observation_messg = torch.cat((h1, message), dim=1)
        h2 = self.nonlin(self.fc2(observation_messg))
        out = self.fc3(h2)
        return out, h2


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs_message, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False, return_hidden_state=True):

        obs     = obs_message[:,:self.input_dim]
        message = obs_message[:,self.input_dim:]
        out, hidden_state = super(DiscretePolicy, self).forward(obs, message)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if return_hidden_state:
            rets.append(hidden_state.detach().numpy())
        if len(rets) == 1:
            return rets[0]

        return rets
