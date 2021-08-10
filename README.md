<a href="url"><img src="https://raw.githubusercontent.com/Saran-nns/gradients/master/imgs/LOGO.jpg"></a>
## Build your deep learning models with confidence


[![Build Status](https://travis-ci.com/Saran-nns/gradients.svg?branch=main)](https://travis-ci.com/Saran-nns/gradients)
[![codecov](https://codecov.io/gh/Saran-nns/gradients/branch/main/graph/badge.svg)](https://codecov.io/gh/Saran-nns/gradients)
[![PyPI version](https://badge.fury.io/py/gradients.svg)](https://badge.fury.io/py/gradients)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gradients.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/386222818.svg)](https://zenodo.org/badge/latestdoi/386222818)

Gradients provide a unit test function to perform gradient checking on your deep learning models. It uses centered finite difference approximation method to check the difference between analytical and numerical gradients and report if the check fails on any parameters of your model. Currently the library supports only PyTorch models built with custom layers, custom loss functions, activation functions and any neural network function subclassing `AutoGrad`.

### Installation

``` pip install gradients```

### Package Overview
Optimizing deep learning models is a two step process:

1. Compute gradients with respect to parameters

2. Update the parameters given the gradients

In PyTorch, step 1 is done by the type-based automatic differentiation system `torch.nn.autograd` and 2 by the package implementing optimization algorithms `torch.optim`. Using  them, we can develop fully customized deep learning models with `torch.nn.Module` and test them using `Gradient` as follows;

### Activation function with backward

```python
class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = 1/(1+torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*input*(1-input)
```

### Loss function with backward

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
        self.sigmoid = MySigmoid.apply
    def forward(self,x):
        y_pred = self.sigmoid(x.mm(self.w1))
        return y_pred
```
### Check your implementation using Gradient

```python
import torch
from gradients import Gradient

N, D_in, D_out = 10, 4, 3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct model by instantiating the class defined above
mymodel = MyModel(D_in, D_out)
criterion = MSELoss.apply

# Test custom build model
Gradient(mymodel,x,y,criterion,eps=1e-8)

```


Citing `Gradients`

```python
@software{nambusubramaniyan_saranraj_2021_5176243,
  author       = {Nambusubramaniyan, Saranraj},
  title        = {gradients},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.5176243},
  url          = {https://doi.org/10.5281/zenodo.5176243}
}```
