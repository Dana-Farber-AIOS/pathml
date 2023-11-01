import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        y = self.linear(x)
        return y
