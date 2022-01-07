# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(self.device))


class ACTVP(nn.Module):
    def __init__(self, device, context_frames):
        super(ACTVP, self).__init__()
        self.device = device
        self.context_frames = context_frames
        self.convlstm1 = ConvLSTMCell(input_dim=3, hidden_dim=12, kernel_size=(3, 3), bias=True, device=self.device).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=24, hidden_dim=24, kernel_size=(3, 3), bias=True, device=self.device).cuda()
        self.conv1 = nn.Conv2d(in_channels=27, out_channels=3, kernel_size=5, stride=1, padding=2).cuda()

    def forward(self, tactiles, actions):
        self.batch_size = actions.shape[1]
        state = actions[0]
        state.to(self.device)
        batch_size__ = tactiles.shape[1]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        outputs = []
        for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
            sample_tactile.to(self.device)
            sample_action.to(self.device)
            # 2. Run through lstm:
            if index > self.context_frames-1:
                hidden_1, cell_1 = self.convlstm1(input_tensor=output, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, output), 1)
                output = self.conv1(skip_connection_added)
                outputs.append(output)
            else:
                hidden_1, cell_1 = self.convlstm1(input_tensor=sample_tactile, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, sample_tactile), 1)
                output = self.conv1(skip_connection_added)
                last_output = output
        outputs = [last_output] + outputs
        return torch.stack(outputs)

