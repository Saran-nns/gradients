import unittest
from gradients.gradients import Gradient
from examples import custom_module as cm
import torch

N, D_in, D_out = 10, 4, 3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct model by instantiating the class defined above
mymodel = cm.MyModel(D_in, D_out)
# criterion = cm.MSELoss.apply
criterion = torch.nn.MSELoss(reduction="mean")

# Test custom build model
Gradient(mymodel, x, y, criterion, eps=1e-8)


class TestGradient(unittest.TestCase):

    def testGradient(self):
        self.assertRaises(Exception, Gradient(mymodel, x, y, criterion, eps=1e-8))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
