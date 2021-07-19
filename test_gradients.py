import unittest
from gradients.gradients import Gradient
from example import *

N, D_in, D_out = 10, 4, 3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = Model(D_in, D_out)
mymodel = MyModel(D_in,D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='mean')
mycriterion = MSELoss.apply

class TestGradient(unittest.TestCase):
    def testGradient(self):
        self.assertRaises(Exception, Gradient().check(model,x,y,criterion,eps=1e-8))
        self.assertRaises(Exception, Gradient().check(mymodel,x,y,criterion,eps=1e-8))
        self.assertRaises(Exception, Gradient().check(model,x,y,mycriterion,eps=1e-8))
        self.assertRaises(Exception, Gradient().check(mymodel,x,y,mycriterion,eps=1e-8))
