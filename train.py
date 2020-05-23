import os 
import json
import time 

import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from model import *


class Trainer():
    def __init__(self, options):
        """
        Initialize the high-level controller which controls model training.
        Args:
            options: Refer to main.py for a detailed description of the options you can use.
        """
        self.opt = options

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.step = 0

        # Results directory
        self.result_dir = os.path.join(self.opt.save_dir, self.opt.exp_name)

        if not os.path.exists(self.opt.save_dir):
            os.mkdir(self.opt.save_dir)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        
        # Save opts for current experiment
        with open(os.path.join(self.result_dir, 'opt.json'), 'w') as f:
            json.dump(self.opt.__dict__.copy(), f, indent=2)
        
        # Writer
        self.writer = SummaryWriter(self.result_dir)
        
        # Dataset 
        dataset = datasets.ImageFolder(
            root = self.opt.data_dir,
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(self.opt.im_size),
                transforms.ToTensor(),
        ]))

        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers)
        
        # Models
        self.gen = Generator(self.opt.im_size, self.opt.latent_size, self.opt.num_blocks).to(self.device)
        self.disc = Discriminator(self.opt.im_size, self.opt.latent_size, self.opt.num_blocks).to(self.device)
        if self.opt.weights_dir is not None:
            self.load_model()

        # Optimizer
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.opt.learning_rate, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.opt.learning_rate, betas=(0.5, 0.999))

        # Labels
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.real_labels = torch.ones(self.opt.batch_size)
        self.fake_labels = torch.zeros(self.opt.batch_size)
            
   
    def train(self):
        """
        Main training loop function for class.
        """
        for self.epoch in range(self.opt.num_epochs):
            for _, (inputs, _) in enumerate(self.dataloader):
                # Train
                gen_loss = self.train_generator()
                disc_loss = self.train_discriminator(inputs)

                # Log
                if (self.step + 1) % self.opt.log_frequency == 0:
                    self.writer.add_scalar("loss/gen", gen_loss.item(), self.step)
                    self.writer.add_scalar("loss/disc", disc_loss.item(), self.step)

                print(gen_loss.item(), disc_loss.item())
                self.step += 1

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                # Save weights
                save_epoch_dir = os.path.join(self.result_dir, "weights_" + str(self.epoch).zfill(3))
                if not os.path.exists(save_epoch_dir):
                    os.mkdir(save_epoch_dir)
                torch.save(self.gen.state_dict(), os.path.join(save_epoch_dir, "gen.pth"))
                torch.save(self.disc.state_dict(), os.path.join(save_epoch_dir, "disc.pth"))
                torch.save(self.gen_optimizer.state_dict(), os.path.join(save_epoch_dir, "adam_gen.pth"))
                torch.save(self.disc_optimizer.state_dict(), os.path.join(save_epoch_dir, "adam_disc.pth"))

                # Log sample image
                with torch.no_grad():
                    x = self.sample()
                    val_image = make_grid(x, nrow=1)
                    self.writer.add_image('generated image', val_image, self.step)
            
        print("Training complete.")

    
    def train_discriminator(self, x_real):

        # Forward
        z = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device) 
        x_fake = self.gen(z)
        logit_real = self.disc(x_real)
        logit_fake = self.disc(x_fake)

        # Calculate loss
        loss_real = self.bce_loss(logit_real, torch.ones_like(logit_real))
        loss_fake = self.bce_loss(logit_fake, torch.zeros_like(logit_fake))

        # Sample for gradient penalty
        x_real_perturb = x_real + 0.5 * x_real.std() * torch.rand_like(x_real)
        alpha = torch.rand([self.opt.batch_size, 1, 1, 1])
        interp = x_real + alpha * (x_real_perturb - x_real)
        interp = interp.detach()
        interp.requires_grad = True

        # Calculate gradient penalty
        logit_interp = self.disc(interp)
        grad = torch.autograd.grad(logit_interp, interp, grad_outputs=torch.ones_like(logit_interp), create_graph=True)[0]
        norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
        gp = ((norm - 1)**2).mean()

        # Optimize
        loss = (loss_real + loss_fake) + gp * self.opt.gradient_penalty_weight
        self.disc.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss

    
    def train_generator(self):
        # Forward
        z = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device) 
        x_fake = self.gen(z)
        logit_fake = self.disc(x_fake)

        # Calculate loss
        loss = self.bce_loss(logit_fake, torch.ones_like(logit_fake))

        # Optimize
        self.gen.zero_grad()
        loss.backward()
        self.gen_optimizer.step()

        return loss
    

    def sample():
        z = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device)
        with torch.no_grad():
            x_fake = self.gen(z)
        return x_fake


    def load_model(self):
        """
        Loading the weights for the network.
        """
        # loading main network
        load_path = os.path.join(self.opt.weights_dir, "gen.pth")
        model_dict = torch.load(load_path)
        self.gen.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "disc.pth")
        model_dict = torch.load(load_path)
        self.disc.load_state_dict(model_dict)

        # loading adam state
        load_path = os.path.join(self.opt.weights_dir, "adam_gen.pth")
        model_dict = torch.load(load_path)
        self.gen_optimizer.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "adam_disc.pth")
        model_dict = torch.load(load_path)
        self.disc_optimizer.load_state_dict(model_dict)

