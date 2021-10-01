import torch
import torch.nn as nn


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1, out_channels=3):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nf) x 32 x 32
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf*2) x 16 x 16
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*4) x 8 x 8
        self.c3 = nn.Sequential(
                nn.Conv2d(nf * 2, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3, [h1, h2]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, out_channels=3):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nf*2) x 8 x 8
        self.upc1 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf*2) x 16 x 16
        self.upc2 = dcgan_upconv(nf * 2, nf)
        # state size. (nf*2) x 32 x 32
        self.upc3 = nn.Conv2d(in_channels=nf, out_channels=out_channels, kernel_size=5, stride=1, padding=2).cuda()
        self.tanh = nn.Tanh()

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(torch.cat([vec, skip[1]], 1))
        d2 = self.upc2(torch.cat([d1, skip[0]], 1))
        output = self.tanh(self.upc3(d2))
        return output
