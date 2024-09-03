import torch


class MySigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * input * (1 - input)


class MSELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_pred, y):
        ctx.save_for_backward(y_pred, y)
        return ((y_pred - y) ** 2).sum() / y_pred.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        grad_input = 2 * (y_pred - y) / y_pred.shape[0]
        return grad_input, None


class MyModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(MyModel, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out), requires_grad=True)
        self.sigmoid = MySigmoid.apply

    def forward(self, x):
        y_pred = self.sigmoid(x.mm(self.w1))
        return y_pred
