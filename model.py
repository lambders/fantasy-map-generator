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

class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
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
        return y

class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, in_channels, (3, 3), padding=1)
        self.conv2 = Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.downsample = AvgPool2d(2)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.conv2(y))
        y = self.downsample(y)
        return y

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
        self.lrelu = LeakyReLU(0.2)
        self.pixelnorm = PixelNorm()
        # self.conv1 = Conv2d(self.opt.latent_size, self.opt.latent_size, (3,3))
    
        # Intermediate blocks
        for i in range(self.depth):
            if i <= 2:
                in_channels = self.opt.latent_size
                out_channels = self.opt.latent_size
            else: 
                in_channels = int(self.opt.latent_size // 2**(i-3))
                out_channels = int(self.opt.latent_size // 2**(i-2))
            layer = GeneratorBlock(in_channels, out_channels)
            self.layers.append(layer)
            self.to_im_layers.append(Conv2d(out_channels, 1, (1,1)))


    def forward(self, x):
        # torch.Size([8, 128, 4, 4])
        # torch.Size([8, 128, 8, 8])
        # torch.Size([8, 128, 16, 16])
        # torch.Size([8, 128, 32, 32])
        # torch.Size([8, 64, 64, 64])
        # torch.Size([8, 32, 128, 128])
        # torch.Size([8, 16, 256, 256])
        # torch.Size([8, 1, 256, 256])

        y = self.pixelnorm(self.lrelu(self.linear(x)))
        y = y.reshape([self.opt.batch_size, self.opt.latent_size, 4, 4])
        # print(y.shape)

        for block in self.layers:
            y = block(y)
            # print(y.shape)
        
        y = self.to_im_layers[-1](y)
        # print(y.shape)
        return y



class Discriminator(torch.nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()
        self.opt = options

        self.init_size = 4
        self.depth = int(math.log2(self.opt.im_size/self.init_size))

        # Layer lists 
        self.layers = ModuleList([])
        self.from_im_layers = ModuleList([])

        # Intermediate blocks
        for i in range(self.depth):
            if i >= 1:
                in_channels = self.opt.im_size // 2**(i+1)
                out_channels = self.opt.im_size // 2**i
            else:
                in_channels = self.opt.latent_size
                out_channels = self.opt.latent_size
            layer = DiscriminatorBlock(in_channels, out_channels)
            self.layers.append(layer)
            self.from_im_layers.append(Conv2d(1, in_channels, (1,1)))
        
        # Final block
        self.batch_discriminator = MinibatchStdDev()
        self.conv1 = Conv2d(self.opt.latent_size + 1, self.opt.latent_size, (3, 3), padding=1)
        self.conv2 = Conv2d(self.opt.latent_size, self.opt.latent_size, (4, 4))
        self.conv3 = Conv2d(self.opt.latent_size, 1, (1, 1))
        self.lrelu = LeakyReLU(0.2)

    
    def forward(self, x):
        # torch.Size([8, 1, 256, 256])
        # torch.Size([8, 4, 256, 256])
        # torch.Size([8, 8, 128, 128])
        # torch.Size([8, 16, 64, 64])
        # torch.Size([8, 32, 32, 32])
        # torch.Size([8, 64, 16, 16])
        # torch.Size([8, 128, 8, 8])
        # torch.Size([8, 128, 4, 4])
        # torch.Size([8, 129, 4, 4])
        # torch.Size([8, 128, 4, 4])
        # torch.Size([8, 128, 1, 1])
        # torch.Size([8, 1, 1, 1])
        # torch.Size([8])

        # print(x.shape)
        y = self.from_im_layers[-1](x)
        # print(y.shape)

        # Intermediate blocks
        for block in reversed(self.layers):
            y = block(y)
            # print(y.shape)

        # Final block
        y = self.batch_discriminator(y)
        # print(y.shape)
        y = self.lrelu(self.conv1(y))
        # print(y.shape)
        y = self.lrelu(self.conv2(y))
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        # print(y.view(-1).shape)

        # Ends in (8, 256, 1, 1) --> (8, 1, 1, 1)
        return y.view(-1)