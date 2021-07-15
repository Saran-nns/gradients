import torch
import gradients


class MLPModel(torch.nn.Module):
    def __init__(self,D_in, D_out):
        super(MLPModel,self).__init__()
        # Create random Tensors for weights.
        self.w1 = torch.nn.Parameter(torch.randn(D_in, D_out), requires_grad=True)
        # self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
    def forward(self,x):
        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = self.sig(x.mm(self.w1))
        return y_pred


N, D_in, D_out = 10, 4, 3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = MLPModel(D_in, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='mean')

gradients.check(model,x,y,criterion,eps=1e-8)