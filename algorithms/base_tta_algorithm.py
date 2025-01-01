import torch
import torch.nn as nn
from copy import deepcopy

class BaseTestTimeAlgorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """
    def __init__(self, configs, hparams, model, optimizer):
        super(BaseTestTimeAlgorithm, self).__init__()
        self.configs = configs
        self.hparams = hparams
        self.model = self.configure_model(model)
        params, param_names = self.collect_params(self.model)
        if len(param_names) == 0:
            self.optimizer = None
        else:
            self.optimizer = optimizer(params)

        self.steps = self.hparams['steps']
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names

    def configure_model(self, model):
        raise NotImplementedError

    def forward_and_adapt(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, trg_idx = None):
        for _ in range(self.steps):
            if trg_idx != None:
                outputs = self.forward_and_adapt(x, self.model, self.optimizer, trg_idx)
            else:
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model
# 
@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def symm_softmax_entropy(x, x_ema):
    alpha = 0.3
    return -(1-alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def im_softmax_entropy(x, x_ema):
    return - (x_ema.softmax(1).mean(0) * torch.log(x.softmax(1).mean(0)) + 1e-5).sum()