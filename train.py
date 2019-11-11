import os 
import json
import time 

import torch 
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from model import Generator, Discriminator


class Trainer():
    def __init__(self, options):
        """
        Initialize the instance with models, optimizer, etc.
        Refer to main.py for a detailed description of the options you can use.
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
        # TODO: Find mean and standard deviation of dataset [0, 1]
        dataset = datasets.ImageFolder(
            root = self.opt.data_dir,
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(self.opt.im_size),
                transforms.ToTensor(),
                # transforms.Normalize(0.5, 0.5),
        ]))

        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers)
        
        # Models
        self.generator = Generator(self.opt).to(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)
        if self.opt.weights_dir is not None:
            self.load_model()

        # Optimizer
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), self.opt.learning_rate)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.opt.learning_rate)
            
   
    def train(self):
        """
        Main training loop function for class.
        """
        val_latent = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device)

        for self.epoch in range(self.opt.num_epochs):
            # Train on dataset 
            for _, (inputs, _) in enumerate(self.dataloader):
                self.process_batch(inputs)
                self.step += 1

            if (self.epoch + 1) % self.opt.save_frequency:
                # Save weights
                save_epoch_dir = os.path.join(self.results_dir, "weights_" + str(self.epoch).zfill(3))
                torch.save(self.gen.state_dict(), os.path.join(save_epoch_dir, "gen.pth"))
                torch.save(self.disc.state_dict(), os.path.join(save_epoch_dir, "disc.pth"))
                torch.save(self.optimizer.state_dict(), os.path.join(save_epoch_dir, "adam.pth"))

                # Log validation image
                with torch.no_grad():
                    val_image = make_grid(self.generator(val_latent), nrow=1)
                    self.writer.add_image('generated image', val_image, self.step)
            
            print("Training complete.")
    
    def process_batch(self, inputs):
        """
        Run one training epoch.
        """
        # Forward pass 
        real_images = inputs.to(self.device)
        fake_latent = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device)
        fake_images = self.generator(fake_latent)  
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images)       

        # Compute losses
        gen_loss = self.compute_generator_loss(fake_logits)
        disc_loss = self.compute_discriminator_loss(real_logits, fake_logits, real_images, fake_images)

        # Optimizer step
        self.disc_optimizer.zero_grad()
        disc_loss.backward(retain_graph=True)
        self.disc_optimizer.step()

        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        print(disc_loss, gen_loss)
        # Logging
        if (self.step + 1) % self.opt.log_frequency == 0:
            self.writer.add_scalar(gen_loss, "generator loss", self.step)
            self.writer.add_scalar(disc_loss, "discriminator loss", self.step)

    
    def compute_generator_loss(self, fake_logits):
        """
        Generator WGAN-GP Loss.
        """
        loss = -torch.mean(fake_logits)
        return loss 
    
    
    def compute_discriminator_loss(self, real_logits, fake_logits, real_images, fake_images, reg_lambda=10):
        """
        Discriminator WGAN-GP Loss.
        """
        # Wasserstein loss ---
        loss = (torch.mean(fake_logits) - torch.mean(real_logits))

        # Gradient penalty ----
        # create the merge of both real and fake samples
        epsilon = torch.rand((self.opt.batch_size, 1, 1, 1)).to(self.device)
        merged = epsilon * real_images + ((1 - epsilon) * fake_images)
        merged.requires_grad_(True)

        # forward pass
        op = self.discriminator(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(outputs=op, inputs=merged, grad_outputs=torch.ones_like(op), 
                                    create_graph=True, retain_graph=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()
        loss += penalty

        # Add small term to keep discriminator output from drifting too far away from zero ---
        loss += self.opt.disc_drift * torch.mean(real_logits ** 2)

        return loss


    def load_model(self):
        # loading main network
        load_path = os.path.join(self.opt.weights_dir, "gen.pth")
        model_dict = torch.load(load_path)
        self.generator.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "disc.pth")
        model_dict = torch.load(load_path)
        self.discriminator.load_state_dict(model_dict)

        # loading adam state
        load_path = os.path.join(self.opt.weights_dir, "gen_adam.pth")
        model_dict = torch.load(load_path)
        self.gen_optimizer.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "disc_adam.pth")
        model_dict = torch.load(load_path)
        self.disc_optimizer.load_state_dict(model_dict)

