import torch
import logging
import copy
from torch.nn import functional as F
from gradients.gradients import Gradient

log = logging.getLogger(__name__)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
class Model(torch.nn.Module):
    def __init__(self,D_in, D_out):
        super(Model,self).__init__()
        # Create random Tensors for weights.
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out),
                                     requires_grad=True)
        # self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = self.sig(x.mm(self.w1))
        return y_pred
class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0.
        return grad_input

# To apply our Function, we use Function.apply method. We alias this as 'relu'.
myrelu = MyReLU.apply

class MySigmoid(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        output = ctx.save_for_backward(output)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return input*(1-input)*grad_output

# To apply our Function, we use Function.apply method. We alias this as 'relu'.
mysigmoid = MySigmoid.apply

# Activation function with learnable parameter
class LearnedSwish(torch.nn.Module):
    """This function use the pytorch autograd to compute gradient"""
    def __init__(self, slope = 1):
        super().__init__()
        self.slope = slope * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.slope * x * torch.sigmoid(x)


# Reference: http://pytorch.org/docs/master/_modules/torch/optim/sgd.html#SGD
class SGD(torch.optim.Optimizer):

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

mycriterion = MSELoss.apply

class MyModel(torch.nn.Module):
    def __init__(self,D_in, D_out):
        super(MyModel,self).__init__()
        # Create random Tensors for weights.
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out), requires_grad=True)
        # self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = self.sig(x.mm(self.w1))
        return y_pred

