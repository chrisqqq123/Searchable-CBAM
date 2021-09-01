import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pdb import set_trace as bp
# from operations import *


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model,  distill=False):
        self.network_momentum = 0.9
        self.network_weight_decay = 5e-4
        self.model = model
        # self._args = args
        # self._distill = distill
        # self._kl = nn.KLDivLoss().cuda()
        self.optimizers = [
            torch.optim.Adam(arch_param, lr=0.1, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
            for arch_param in self.model._arch_parameters ]
        # self.latency_weight = args.latency_weight
        # assert len(self.latency_weight) == len(self.optimizers)
        # self.latency = 0

        print("architect initialized!")

    # def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    #     loss = self.model._loss(input, target)
    #     theta = _concat(self.model.parameters()).data
    #     try:
    #         moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    #     except:
    #         moment = torch.zeros_like(theta)
    #     dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    #     unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    #     return unrolled_model

    def step(self, input_valid, target_valid, eta=None, network_optimizer=None, unrolled=False, penalty = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        # if unrolled:
        #         loss = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        # else:
        loss = self._backward_step(input_valid, target_valid)
        if penalty:
            print('penalty loss:  ',self.model._arch_loss(ratio = [3,1,1,2,2], wgt = 0.1))
            loss += self.model._arch_loss(ratio = [3,1,1,2,2], wgt = 0.1)
        loss.backward()
        # if loss_latency != 0: loss_latency.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def _backward_step(self, input_valid, target_valid):
        loss_function = nn.CrossEntropyLoss()
        outputs = self.model(input_valid)
        loss = loss_function(outputs, target_valid)
        return loss


