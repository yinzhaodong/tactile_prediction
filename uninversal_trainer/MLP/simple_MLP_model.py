# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn

seed = 42
epochs = 50
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

train_percentage = 0.9
validation_percentage = 0.1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class simple_MLP(nn.Module):
    def __init__(self):
        super(simple_MLP, self).__init__()
        self.fc1 = nn.Linear((48*10), (48*10)).to(device)
        self.tan_activation = nn.Tanh().to(device)

    def forward(self, tactiles, actions):
        tactile = tactiles[:context_frames].permute(1, 0, 2)
        tactile = tactile.flatten(start_dim=1)
        output = self.tan_activation(self.fc1(tactile))
        return output
