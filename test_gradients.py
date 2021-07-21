import unittest
from gradients.gradients import Gradient
from example import *

N, D_in, D_out = 10, 4, 3
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = Model(D_in, D_out)
mymodel = MyModel(D_in,D_out)
criterion = torch.nn.MSELoss(reduction='mean')
mycriterion = MSELoss.apply
class TestGradient(unittest.TestCase):
    def testGradient(self):
        self.assertRaises(Exception, Gradient(model,x,y,criterion,eps=1e-8).check())
        self.assertRaises(Exception, Gradient(model,x,y,criterion,eps=1e-8).check())
        self.assertRaises(Exception, Gradient(model,x,y,criterion,eps=1e-8).check())
        self.assertRaises(Exception, Gradient(model,x,y,criterion,eps=1e-8).check())

if __name__ == "__main__":
    unittest.main()