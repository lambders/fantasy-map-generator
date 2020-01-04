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
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(self.opt).to(self.device)
        self.discriminator.apply(weights_init)
        if self.opt.weights_dir is not None:
            self.load_model()

        # Optimizer
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opt.learning_rate, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.learning_rate, betas=(0.5, 0.999))

        # Labels
        self.bce_loss = torch.nn.BCELoss()
        self.real_labels = torch.ones(self.opt.batch_size)
        self.fake_labels = torch.zeros(self.opt.batch_size)
            
   
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

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                # Save weights
                save_epoch_dir = os.path.join(self.result_dir, "weights_" + str(self.epoch).zfill(3))
                if not os.path.exists(save_epoch_dir):
                    os.mkdir(save_epoch_dir)
                torch.save(self.generator.state_dict(), os.path.join(save_epoch_dir, "gen.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(save_epoch_dir, "disc.pth"))
                torch.save(self.gen_optimizer.state_dict(), os.path.join(save_epoch_dir, "adam_gen.pth"))
                torch.save(self.disc_optimizer.state_dict(), os.path.join(save_epoch_dir, "adam_disc.pth"))

                # Log validation image
                with torch.no_grad():
                    val_image = make_grid(self.generator(val_latent), nrow=1)
                    self.writer.add_image('generated image', val_image, self.step)
            
        print("Training complete.")
    
    def process_batch(self, inputs):
        """
        Run one training batch.
        Args:
            inputs: batch of images 
        """      
        # Discriminator step - real
        self.disc_optimizer.zero_grad()
        real_images = inputs.to(self.device)
        real_logits = self.discriminator(real_images)
        disc_loss_real = self.bce_loss(real_logits, self.real_labels)
        disc_loss_real.backward()
        D_x = real_logits.mean().item()

        # Discriminator step - fake
        fake_latent = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device)
        fake_images = self.generator(fake_latent)  
        fake_logits = self.discriminator(fake_images.detach())
        disc_loss_fake = self.bce_loss(fake_logits, self.fake_labels)
        disc_loss_fake.backward()
        D_G_z1 = fake_logits.mean().item()
        disc_loss = disc_loss_real + disc_loss_fake
        self.disc_optimizer.step()

        # Generator step
        self.gen_optimizer.zero_grad()
        fake_logits = self.discriminator(fake_images)
        gen_loss = self.bce_loss(fake_logits, fake_labels)
        gen_loss.backward()
        D_G_z2 = fake_logits.mean().item()
        self.gen_optimizer.step()

        # Logging
        if (self.step + 1) % self.opt.log_frequency == 0:
            self.writer.add_scalar("generator loss", gen_loss.item(), self.step)
            self.writer.add_scalar("discriminator loss", disc_loss.item(), self.step)
            self.writer.add_scalar("D(x)", D_x, self.step)
            self.writer.add_scalar("D(G(z1))", D_G_z1, self.step)
            self.writer.add_scalar("D(G(z2))", D_G_z2, self.step)
        print(gen_loss.item(), disc_loss.item())


    def load_model(self):
        """
        Loading the weights for the network.
        """
        # loading main network
        load_path = os.path.join(self.opt.weights_dir, "gen.pth")
        model_dict = torch.load(load_path)
        self.generator.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "disc.pth")
        model_dict = torch.load(load_path)
        self.discriminator.load_state_dict(model_dict)

        # loading adam state
        load_path = os.path.join(self.opt.weights_dir, "adam_gen.pth")
        model_dict = torch.load(load_path)
        self.gen_optimizer.load_state_dict(model_dict)

        load_path = os.path.join(self.opt.weights_dir, "adam_disc.pth")
        model_dict = torch.load(load_path)
        self.disc_optimizer.load_state_dict(model_dict)

