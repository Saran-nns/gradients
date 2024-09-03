# MIT License

# Copyright (c) 2021 Saranraj Nambusubramaniyan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import logging
import copy
import math
from torch.nn import functional as F

log = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class Gradient:
    def __init__(self, model, x, y, criterion, eps):
        super(Gradient, self).__init__()
        self.model = model
        self.x = x
        self.y = y
        self.criterion = criterion
        self.eps = eps
        self.loss = self.param = None

    def __setattr__(self, name, value) -> None:
        self.__dict__[name] = value

    def __new__(cls, model, x, y, criterion, eps) -> object:
        gc = super(Gradient, cls).__new__(cls)
        gc.__init__(model, x, y, criterion, eps)
        return gc.check()

    def set_attr(self, param) -> None:
        self.__setattr__("param", param)
        try:
            self.__setattr__("theta", getattr(self.model, param.split(".")[0]).data)
        except Exception as _:
            if "weight" in self.param:
                self.__setattr__(
                    "theta", getattr(self.model, param.split(".")[0]).weight
                )
            else:
                self.__setattr__("theta", getattr(self.model, param.split(".")[0]).bias)

    @staticmethod
    def get_idx(theta):
        valid_idx = (theta.data != -1).nonzero()
        try:
            choice = torch.multinomial(torch.arange(valid_idx.size(0)).float(), 1)
        except Exception as _:
            try:
                choice = torch.multinomial(torch.arange(valid_idx.float(), 1))
            except Exception as _:
                choice = valid_idx

        if len(theta.size()) > 1:
            return valid_idx[choice].squeeze().chunk(2)
        else:
            return valid_idx[choice]

    def J(self, model, param, eps):
        sd = model.state_dict()
        w = sd[param].data.flatten()
        temp = w.clone()
        j = torch.zeros(w.size())
        for i in range(len(w)):
            w[i] += eps
            sd[param].data = w.reshape(sd[param].data.size())
            model.load_state_dict(sd)
            j[i] = self.numgrad(model)
            w[i] = temp[i]
        return j.reshape(sd[param].data.size())

    def anagrad(self):

        try:
            grad = getattr(self.model, self.param).grad
        except Exception as _:
            if len(self.theta.size()) > 1 and "weight" in self.param:
                grad = getattr(self.model, self.param.split(".")[0]).weight.grad
            elif "bias" in self.param:
                grad = getattr(self.model, self.param.split(".")[0]).bias.grad
        return grad

    def numgrad(self, model):
        model.zero_grad()
        y_ = model(self.x)
        loss = self.criterion(y_, self.y)
        return loss.item()

    def forward(self, model, param, eps):
        grad = self.J(model, param, eps)
        return grad

    def check_(self, anagrad, numgrad_plus, numgrad_minus):

        numgrad = (numgrad_plus - numgrad_minus) / (2.0 * self.eps)

        diff = torch.norm(anagrad - numgrad) / (
            torch.norm(anagrad) + torch.norm(numgrad)
        )

        if diff > 1e-6:
            print(
                f"Parameter {self.param} Relative difference between analytical and numerical gradient is {diff}>1e-6: Check Failed"
            )
        else:
            print(
                f"Parameter {self.param} Relative difference between analytical and numerical gradient is {diff}<1e-6: Check Passed"
            )

    def check(self):
        # Analytical gradient
        self.model.zero_grad()
        model = copy.deepcopy(self.model)
        y_ = self.model(self.x)
        self.loss = self.criterion(y_, self.y)
        self.loss.backward()

        for self.param, self.theta in self.model.named_parameters():
            if self.theta.requires_grad:
                self.set_attr(self.param)
                ana_grad = self.anagrad()
                grad_plus = self.forward(model, self.param, self.eps)
                grad_minus = self.forward(model, self.param, -self.eps)
                self.check_(ana_grad, grad_plus, grad_minus)
