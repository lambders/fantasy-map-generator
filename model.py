'''
Networks for the map generator.
'''
import os
from glob import glob
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import spectral_norm



class FantasyMapGAN(nn.Module):

    def __init__(self, lr: float, batch_size: int, ngf: int = 64, ndf: int = 64, latent_dim: int = 100):
        '''
        Fantasy map generator. 
        A combination of a conditional variational autoencoder (for stability and attribute control) and a GAN (for generation power).

        Parameters
        ----------
        lr: float
            learning rate
        ngf: int
            number of base filters to be used in the generator (encoder and decoder networks)
        ndf: int
            number of base filters to be used in the discriminators (ImageDiscriminator and LatentDiscriminator)
        latent_dim: int
            size of latent dimension
        isTrain: bool
            set to True if we want to train the network
        '''
        super(FantasyMapGAN, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim 

        # Networks
        self.encoder = Encoder(ngf, latent_dim)
        self.decoder = Decoder(ngf, latent_dim)
        self.disc_image = DiscriminatorImage(ndf)
        self.disc_latent = DiscriminatorLatent(ndf, latent_dim)

        # Optimizers
        self.opt_generator = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr, betas=(0.5, 0.9))
        self.opt_disc_image = torch.optim.Adam(self.disc_image.parameters(), lr/2, betas=(0.5, 0.9))
        self.opt_disc_latent = torch.optim.Adam(self.disc_latent.parameters(), lr/2, betas=(0.5, 0.9))

        # Losses 
        self.real_label = torch.ones(batch_size).cuda()
        self.fake_label = torch.zeros(batch_size).cuda()
    
    def forward(self, x: torch.Tensor):

        # Discriminator loss
        self.opt_disc_latent.zero_grad()
        self.opt_disc_image.zero_grad()

        z = torch.randn((self.batch_size, self.latent_dim), requires_grad=True)
        z = z.cuda()

        with torch.no_grad():
            x_hat = self.decoder(z)
            z_hat = self.encoder(x)
            x_tilde = self.decoder(z_hat)
            z_tilde = self.encoder(x_hat)

        x_conf = self.disc_image(x)
        x_hat_conf = self.disc_image(x_hat)
        x_tilde_conf = self.disc_image(x_tilde)
        z_conf = self.disc_latent(z)
        z_hat_conf = self.disc_latent(z_hat)
        z_tilde_conf = self.disc_latent(z_tilde)

        x_loss = 2 * F.binary_cross_entropy_with_logits(x_conf, self.real_label)
        x_hat_loss = F.binary_cross_entropy_with_logits(x_hat_conf, self.fake_label)
        x_tilde_loss = F.binary_cross_entropy_with_logits(x_tilde_conf, self.fake_label)
        z_loss = 2 * F.binary_cross_entropy_with_logits(z_conf, self.real_label)
        z_hat_loss = F.binary_cross_entropy_with_logits(z_hat_conf, self.fake_label)
        z_tilde_loss = F.binary_cross_entropy_with_logits(z_tilde_conf, self.fake_label)

        disc_image_loss = (x_loss + x_hat_loss + x_tilde_loss) / 4
        disc_latent_loss = (z_loss + z_hat_loss + z_tilde_loss) / 4
        disc_loss = disc_image_loss + disc_latent_loss

        disc_loss.backward()
        self.opt_disc_latent.step()
        self.opt_disc_image.step()

        # Generator loss
        self.opt_generator.zero_grad()

        z2 = torch.randn((self.batch_size, self.latent_dim), requires_grad=True)
        z2 = z2.cuda()

        x_hat = self.decoder(z2)
        z_hat = self.encoder(x)
        x_tilde = self.decoder(z_hat)
        z_tilde = self.encoder(x_hat)

        x_hat_conf = self.disc_image(x_hat)
        z_hat_conf = self.disc_latent(z_hat)
        x_tilde_conf = self.disc_image(x_tilde)
        z_tilde_conf = self.disc_latent(z_tilde)

        x_hat_loss = F.binary_cross_entropy_with_logits(x_hat_conf, self.real_label)
        z_hat_loss = F.binary_cross_entropy_with_logits(z_hat_conf, self.real_label)
        x_tilde_loss = F.binary_cross_entropy_with_logits(x_tilde_conf, self.real_label)
        z_tilde_loss = F.binary_cross_entropy_with_logits(z_tilde_conf, self.real_label)

        x_recon_loss = F.l1_loss(x_tilde, x) 
        x_loss = (x_hat_loss + x_tilde_loss) / 2 * 0.005
        z_loss = (z_hat_loss + z_tilde_loss) / 2 * 0.1
        gen_loss = x_loss + z_loss + x_recon_loss 

        gen_loss.backward()
        self.opt_generator.step()

        # Return losses
        loss_dict = {
            'generator/im_recon_loss': x_recon_loss,
            'generator/gan_loss': x_loss + z_loss,
            'generator/total_loss': gen_loss,
            'discriminator/latent_loss': disc_latent_loss,
            'discriminator/image_loss': disc_image_loss
        }
        return loss_dict
    
    def sample(self, batch_size: int):
        '''
        Sample a map image.

        Parameters
        ----------
        batch_size: int
            batch size of generated sample
        '''
        with torch.no_grad():
            z = torch.randn((batch_size, self.latent_dim))
            z = z.cuda()
            x = self.decoder(z)
        return x 
    

    def reconstruct(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    
    def save(self, save_dir: str, epoch: int) -> None:
        '''
        Save network weights.

        Parameters
        ----------
        save_dir: str
            path to save network weights
        epoch: int
            current epoch
        '''
        # Save
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "%i_enc.pth" % epoch))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "%i_dec.pth" % epoch))
        torch.save(self.disc_image.state_dict(), os.path.join(save_dir, "%i_disc_image.pth" % epoch))
        torch.save(self.disc_latent.state_dict(), os.path.join(save_dir, "%i_disc_latent.pth" % epoch))

        # Only keep three most recent saves of our four models
        num_keep = 3 * 4
        fs = glob(os.path.join(save_dir, '*.pth'))
        fs.sort(key=os.path.getmtime)
        for f in fs[:-num_keep]:
            os.remove(f)

    
    def load(self, load_dir: str) -> None:
        '''
        Load network weights.

        Parameters
        ----------
        load_dir: str
            path to load network weights from
        '''
        if not load_dir:
            return

        # Find most recent epoch
        fs = glob(os.path.join(load_dir, '*.pth'))
        fs.sort(key=os.path.getmtime)
        epoch = int(fs[-1].split('_')[0])

        # Load
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir, "%i_enc.pth" % epoch)))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir, "%i_dec.pth" % epoch)))
        self.disc_image.load_state_dict(torch.load(os.path.join(load_dir, "%i_disc_image.pth" % epoch)))
        self.disc_latent.load_state_dict(torch.load(os.path.join(load_dir, "%i_disc_latent.pth" % epoch)))



class Encoder(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100):
        '''
        The encoder maps from the image space to the latent space.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(Encoder,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1,num_filters,5,2,2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters,2*num_filters,5,2,2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(2*num_filters,4*num_filters,5,2,2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(4*num_filters,8*num_filters,5,2,2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Conv2d(8*num_filters,8*num_filters,5,2,2),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(8192,latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of encoder.

        Parameters
        ----------
        x: torch.Tensor
            input image, a tensor of shape (?,3,h,w)
        
        Returns
        -------
        torch.Tensor: latent vector of shape (?, latent_dim)
        '''
        batch_size = x.shape[0]
        conv = self.layers(x)
        conv = conv.view(batch_size,-1)
        out = self.fc(conv)
        return out



class Decoder(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100, color_dim: int = 16):
        '''
        The decoder maps from the latent space to the image space.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(Decoder,self).__init__()
        self.num_filters = num_filters
        self.color_dim = color_dim 

        def color_picker(input_dim: int, output_dim: int):
            '''
            Create and choose from a color palette during map generation.
            Helps with overall perceptual quality of network. 
            '''
            colorspace = nn.Sequential(
                nn.Linear(input_dim,128,bias=True),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Linear(128,64,bias=True),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Linear(64,output_dim,bias=True),
                nn.Tanh(),
            )
            return colorspace

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16*num_filters),
            nn.BatchNorm1d(16*num_filters),
            nn.ReLU(True)
        )

        self.upconv= nn.Sequential( # 6 12 24 48 96
            nn.Conv2d(16*num_filters,8*num_filters,3,1,1),
            nn.BatchNorm2d(8*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8*num_filters,4*num_filters,3,1,1),
            nn.BatchNorm2d(4*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4*num_filters,2*num_filters,3,1,1),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*num_filters,num_filters,3,1,1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters,num_filters,3,1,1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters,color_dim,3,1,1), 
            nn.Softmax(),
        )

        self.color_picker = color_picker(16*num_filters, color_dim)
        

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of generator network.

        Parameters
        ----------
        z: torch.Tensor
            latent input tensor, random noise
        
        Returns
        -------
        torch.Tensor: generated images
        '''
        # Generate 16-channel intermediate image from latent vector
        x = self.fc(z)
        x_square = x.view(-1,16*self.num_filters,1,1)
        x_square = F.upsample(x_square, scale_factor=4)
        intermediate = self.upconv(x_square)

        # Pick from color palette
        r = self.color_picker(x)
        r = r.view((-1, 16, 1, 1))
        r = F.upsample(r, scale_factor=128)
        r = intermediate * r
        r = torch.sum(r, dim=1, keepdim=True) 

        return r


class DiscriminatorImage(nn.Module):

    def __init__(self, num_filters: int = 64):
        '''
        Discriminator for generated/real images.

        Parameters
        ----------
        num_filters: int 
            base number of filters used, by default 32
        '''
        super(DiscriminatorImage,self).__init__()
        self.num_filters = num_filters

        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(1,num_filters,4,2,1)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_filters,num_filters*2,4,2,1)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_filters*2,num_filters*4,4,2,1)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_filters*4,num_filters*8,4,2,1)),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(num_filters*8,num_filters*8,4,2,1)),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(8*4*4*num_filters,1024)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(1024,1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of image discriminator.

        Parameters
        ----------
        x: torch.Tensor
            input image, a tensor of shape (?, 3, h, w)
        
        Returns
        -------
        torch.Tensor: real/fake activations, a vector of shape (?,)
        '''
        y = self.layers(x)
        y = y.view(-1,8*4*4*self.num_filters)
        out = self.fc(y)
        return out.squeeze()


class DiscriminatorLatent(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100):
        '''
        Discriminator for latent vectors.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(DiscriminatorLatent,self).__init__()

        self.layers = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim,num_filters*4)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(num_filters*4,num_filters*2)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(num_filters*2,num_filters)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(num_filters,1))
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of latent discriminator.

        Parameters
        ----------
        z: torch.Tensor
            input latent, a tensor of shape (?, latent_dim)
        
        Returns
        -------
        torch.Tensor: real/fake activations, a vector of shape (?,)
        '''
        return self.layers(z).squeeze()