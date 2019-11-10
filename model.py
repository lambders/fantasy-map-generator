import math

import torch
from torch.nn import ModuleList, Linear, Conv2d, ConvTranspose2d, AvgPool2d, LeakyReLU
from torch.nn.functional import interpolate 
# TODO: Equalized learning rate (skip), pixel_norm for latents (maybe)
# TODO: Progressive learning (maybe)

class PixelNorm(torch.nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt() 
        y = x / y  
        return y

class GeneratorBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GeneratorBlock, self).__init__()
        self.upsample = lambda x: interpolate(x, scale_factor=2)
        self.conv1 = Conv2d(in_channels, out_channels, (3,3), padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, (3,3), padding=1)
        self.lrelu = LeakyReLU(0.2)
        self.pixelnorm = PixelNorm()

    def forward(self, x):
        y = self.upsample(x)
        y = self.pixelnorm(self.lrelu(self.conv1(y)))
        y = self.pixelnorm(self.lrelu(self.conv2(y)))

class ProGANish(torch.nn.Module):
    def __init__(self, options):
        super(ProGANish, self).__init__()
        self.opt = options
    
    def build(self):
        return None
    
    def forward(self):
        return None

class Generator(torch.nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.opt = options

        self.init_size = 4
        self.depth = int(math.log2(self.opt.im_size/self.init_size))

        # Layer lists
        self.layers = ModuleList([])
        self.to_im_layers = ModuleList([])

        # Initial block
        self.linear = Linear(1, self.init_size**2)
        self.conv1 = Conv2d(self.opt.latent_size, self.opt.latent_size, (4,4))
        self.conv2 = Conv2d(self.opt.latent_size, self.opt.latent_size, (3,3), padding=1)
    
        # Intermediate blocks
        for i in range(self.depth - 1):
            if i <= 2:
                in_channels = self.opt.latent_size
                out_channels = self.opt.latent_size
            else: 
                in_channels = int(self.opt.latent_size // 2**(i-3))
                out_channels = int(self.opt.latent_size // 2**(i-2))
            layer = GeneratorBlock(in_channels, out_channels)
            self.layers.append(layer)
            self.to_im_layers = Conv2d(out_channels, 1, (1,1))


    def forward(self, x):
        y = self.linear(x)
        y = y.reshape([self.batch_size, self.opt.latent_size, 4, 4])
    
        for block in self.layers:
            y = block(y)

class Discriminator(torch.nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()
        self.opt = options