# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn

class ACTP(nn.Module):
    def __init__(self, device, context_frames):
        super(ACTP, self).__init__()
        self.device = device
        self.context_frames = context_frames
        self.lstm1 = nn.LSTM(48, 200).to(device)  # tactile
        self.lstm2 = nn.LSTM(200 + 48, 200).to(device)  # tactile
        self.fc1 = nn.Linear(200 + 48, 200).to(device)  # tactile + pos
        self.fc2 = nn.Linear(200, 48).to(device)  # tactile + pos
        self.tan_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)

    def forward(self, tactiles, actions):
        state = actions[0]
        state.to(self.device)
        batch_size__ = tactiles.shape[1]
        outputs = []
        hidden1 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
        hidden2 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))

        for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
            # 2. Run through lstm:
            if index > self.context_frames-1:
                out4 = out4.squeeze()
                out1, hidden1 = self.lstm1(out4.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                outputs.append(out4.squeeze())
            else:
                out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                last_output = out4

        outputs = [last_output] + outputs
        return torch.stack(outputs)
