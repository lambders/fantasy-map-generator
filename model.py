import math
import torch
from torch.nn import Conv2d, ConvTranspose2d, GroupNorm
from torch.nn import LeakyReLU, Sigmoid, ReLU, Tanh


class Generator(torch.nn.Module):
    def __init__(self, im_size, latent_size, num_blocks):
        """
        Generator class.
        """
        super(Generator, self).__init__()

        def block(in_channels, out_channels, kernel_size, stride, padding):
            return torch.nn.Sequential(
                ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size, stride, padding, bias=False
                ),
                ReLU()
            )

        # Layers
        # ------
        layers = []
        # First block
        num_filters = min(im_size * 2**(num_blocks-1), im_size*8)
        layers.append(block(latent_size, num_filters, 4, 1, 0))

        # Middle blocks
        for i in range(num_blocks - 1):
            num_filters_ = num_filters
            num_filters = min(im_size * 2** (num_blocks-2-i), latent_size*8)
            layer = block(num_filters_, num_filters, 4, 2, 1)
            layers.append(layer)

        # End block
        layers.extend([
            ConvTranspose2d(num_filters, 1, 4, 2, 1),
            Tanh()
        ])
    
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Discriminator(torch.nn.Module):
    def __init__(self, im_size, latent_size, num_blocks):
        """
        Discriminator class, inspired by DCGAN.
        """
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels, kernel_size, stride, padding):
            return torch.nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                GroupNorm(1, out_channels),
                LeakyReLU(0.2)
            )

        # Layers
        # --------
        layers = []
        # First block
        num_filters = im_size
        layers.extend([
            Conv2d(1, num_filters, 4, 2, 1),
            LeakyReLU(0.2)
        ])

        # Middle blocks
        for i in range(num_blocks - 1):
            num_filters_ = num_filters
            num_filters = min(im_size * 2**(i+1), im_size*8)
            layers.append(block(num_filters_, num_filters, 4, 2, 1))
        
        layers.append(Conv2d(num_filters, 1, 4, 1, 0))

        self.layers = torch.nn.Sequential(*layers)
    

    def forward(self, x):
        return self.layers(x)