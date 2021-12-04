import torch
import torch.nn as nn


class basicModel(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2):
        super(basicModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hid_dim1
        self.hidden_dim2 = hid_dim2

        self.add_module('Linear1', nn.Linear(self.input_dim, self.hidden_dim1))
        self.add_module('Linear2', nn.Linear(self.hidden_dim1, self.hidden_dim2))
        self.add_module('Linear3', nn.Linear(self.hidden_dim2, 7))
        self.add_module('softmax', nn.Softmax(dim=0))

    def forward(self, x):
        # x = torch.from_numpy(x).to(torch.float32)
        x = x.to(torch.float32)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        # x = self.softmax(x)
        return x
