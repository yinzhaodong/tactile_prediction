import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
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
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class convlstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(convlstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1).cuda()  # embed
        self.convLSTMs = [ConvLSTMCell(self.hidden_size, self.hidden_size, (3, 3), True).cuda() for i in range(self.n_layers)]
        self.conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.output_size, kernel_size=3, stride=1, padding=1).cuda()  # embed
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size, 8, 8).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size, 8, 8).cuda()))
        return hidden

    def forward(self, input):
        h_in = self.conv1(input)
        for i in range(self.n_layers):
            self.hidden[i] = self.convLSTMs[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        h_out = self.conv2(h_in)
        return h_out


class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1).cuda()
        self.convLSTMs = [ConvLSTMCell(self.hidden_size, self.hidden_size, (3, 3), True).cuda() for i in range(self.n_layers)]
        self.conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=8, stride=1, padding=0).cuda()

        self.mu_net = nn.Linear(self.hidden_size, output_size)
        self.logvar_net = nn.Linear(self.hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size, 8, 8).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size, 8, 8).cuda()))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        h_in = self.conv1(input)
        for i in range(self.n_layers):
            self.hidden[i] = self.convLSTMs[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        linearised = self.conv2(h_in).squeeze(-1).squeeze(-1)
        mu = self.mu_net(linearised)
        logvar = self.logvar_net(linearised)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar  # z = [32, 10]
