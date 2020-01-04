import math
import torch
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d
from torch.nn import LeakyReLU, Sigmoid, ReLU, Tanh


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class Generator(torch.nn.Module):
    def __init__(self, options):
        """
        Generator class, inspired by DCGAN.
        """
        super(Generator, self).__init__()
        self.opt = options
        num_levels = int(math.log2(self.opt.im_size) - 1)
        mf = self.opt.latent_size // 2
        nf = [min(self.opt.base_num_filters*2**i, mf) for i in range(num_levels)][::-1]

        # Layers
        layers = []
        layers.extend([
            ConvTranspose2d(self.opt.latent_size, nf[0], 4, 1, 0, bias=False),
            BatchNorm2d(nf[0])
        ])
        
        for i in range(1, num_levels):
            layers.extend([
                ConvTranspose2d(nf[i-1], nf[i], 4, 2, 1, bias=False),
                BatchNorm2d(nf[i]),
                ReLU(True)
            ])
        
        layers.extend([
            ConvTranspose2d(nf[-1], 1, 4, 2, 1, bias=False),
            Tanh()
        ])

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(torch.nn.Module):
    def __init__(self, options):
        """
        Discriminator class, inspired by DCGAN.
        """
        super(Discriminator, self).__init__()
        self.opt = options
        num_levels = int(math.log2(self.opt.im_size) - 1)
        mf = self.opt.latent_size // 2
        nf = [min(self.opt.base_num_filters*2**i, mf) for i in range(num_levels)]

        # Layers
        layers = []
        layers.extend([
            Conv2d(1, nf[0], 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
        ])

        for i in range(1, num_levels):
            layers.extend([
                ConvTranspose2d(nf[i-1], nf[i], 4, 2, 1, bias=False),
                LeakyReLU(0.2, inplace=True)
            ])

        layers.extend([
            Conv2d(nf[-1], 1, 4, 1, 0, bias=False),
            Sigmoid()
        ])

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)