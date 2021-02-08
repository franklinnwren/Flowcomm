from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions import Transform

def tanh5scale(m):
    return 5*(m/5).tanh()

def log1p_softplus(x):
    return torch.log1p(F.softplus(x))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)#, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ShiftLayer(nn.Module, Transform):
    event_dim=0
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('shift', nn.Parameter(torch.zeros(dim), requires_grad=True))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def set_mode(self, mode='forward'):
        self.mode = mode

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            return (self.forward(x[0]), *x[1:])

        if not self.initialized.item():
            self.shift.data = x.view(-1, x.shape[-1]).mean(0).detach()
            self.initialized.data = torch.tensor([True])
            print('shift layer initialized')

        if self.mode == 'backward':
            return x - self.shift.expand_as(x)
        elif self.mode == 'forward':
            return x + self.shift.expand_as(x)
        else:
            raise NotImplementedError(self.mode)

    def __call__(self, x):
        self.set_mode('forward')
        return self.forward(x)

    def inv(self, y):
        self.set_mode('backward')
        return self.forward(y)

    def log_abs_det_jacobian(self, x, y):
        return 0.0
