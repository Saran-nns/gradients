# Gradients
## Build your deep learning models with confidence

Medium article is under work

[![Build Status](https://travis-ci.com/Saran-nns/gradients.svg?branch=main)](https://travis-ci.com/Saran-nns/gradients)
[![codecov](https://codecov.io/gh/Saran-nns/gradients/branch/main/graph/badge.svg)](https://codecov.io/gh/Saran-nns/gradients)
[![PyPI version](https://badge.fury.io/py/gradients.svg)](https://badge.fury.io/py/gradients)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gradients.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Gradients is a library for gradients checking using centered finite difference approximation. Currently gradients supports only PyTorch and its models built with custom layers, custom loss functions, activation functions or any neural network function subclassing AutoGrad.

Optimizing deep learning models is a two step process:
    1. Compute gradients with respect to parameters
    2. Update the parameters given the gradients

In pytorch, step 1 is done by the type-based automatic differentiation system `torch.nn.autograd` and step 2 by `torch.optim`. Using  them, we can build fully customized deep learning models with `torch.nn.Module` as follows;
Example:

### Activation function with backward

```python
class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = ctx.save_for_backward(output)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return input*(1-input)*grad_output
```

### Loss function with autograd backward

```python
class MSELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_pred, y):
        ctx.save_for_backward(y_pred, y)
        return ((y_pred-y)**2).sum()/y_pred.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        grad_input = 2 * (y_pred-y)/y_pred.shape[0]
        return grad_input, None
```
### Pytorch Model

```python
class MyModel(torch.nn.Module):
    def __init__(self,D_in, D_out):
        super(MyModel,self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out), requires_grad=True)
    def forward(self,x):
        y_pred = mysigmoid(x.mm(self.w1))
        return y_pred
```
### Optimizer
```python
class SGD(torch.optim.Optimizer):
    """Reference: http://pytorch.org/docs/master/_modules/torch/optim/sgd.html#SGD"""
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(SGD,self).__init__(params,defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
```
TODO
### Instantiate, gradcheck and train the model




