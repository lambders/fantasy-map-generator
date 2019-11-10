import os 
import json
import time 

import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
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
        # TODO: Find mean and standard deviation of dataset
        dataset = datasets.ImageFolder(
            root = self.opt.data_dir,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.opt.im_size),
                transforms.Normalize(0.5, 0.5),
        ]))

        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers)
        
        # Models
        self.generator = Generator(self.opt).to(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)
        if self.opt.weights_folder is not None:
            self.load_model()

        # Optimizer
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), self.opt.learning_rate)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.opt.learning_rate)
            
   
    def train(self):
        """
        Main training loop function for class.
        """
        for self.epoch in range(self.opt.num_epochs):
            for batch_idx, inputs in enumerate(self.dataloader):
                self.process_batch(inputs)
                self.step += 1

            # Save model
            if (self.epoch + 1) % self.opt.save_frequency:
                save_epoch_dir = os.path.join(self.results_dir, "weights_" + str(self.epoch).zfill(3))
                torch.save(self.gen.state_dict(), os.path.join(save_epoch_dir, "gen.pth"))
                torch.save(self.disc.state_dict(), os.path.join(save_epoch_dir, "disc.pth"))
                torch.save(self.optimizer.state_dict(), os.path.join(save_epoch_dir, "adam.pth"))

    
    def process_batch(self, inputs):
        """
        Run one training epoch.
        """
        # Forward pass 
        real_images = inputs.to(self.device)
        fake_latent = torch.randn(self.opt.batch_size, self.opt.latent_size).to(self.device)
        fake_images = self.generator(fake_latent)         

        # Compute losses
        gen_loss = self.compute_generator_loss(real_images, fake_images)
        disc_loss = self.compute_discriminator_loss(real_logits, fake_logits)

        # Optimizer step
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Logging
        if (self.step + 1) % self.opt.log_frequency == 0:
            self.log(losses)

    
    def compute_generator_loss(self, fake_logits):
        """
        Generator WGAN-GP Loss.
        """
        loss = -th.mean(fake_logits)
        return loss 
    
    
    def compute_discriminator_loss(self, real_logits, fake_logits):
        """
        Discriminator WGAN-GP Loss.
        """
        # Wasserstein loss
        loss = (torch.mean(fake_logits) - torch.mean(real_logits)

        # Gradient penalty
        

        # Add small term to keep discriminator output from drifting too far away from zero
        loss += self.opt.disc_drift * th.mean(real_logits ** 2))

        return loss

    


    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        op = self.dis(merged, height, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha):


    def gen_loss(self, _, fake_samps, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, height, alpha))

        return loss

    def log(self, losses):
        """
        Write logging statements to the Tensorboard file.
        """
        for l, v in losses.items():
            self.writer.add_scalar(l, v, self.step)
            # TODO: Add sampled image, reconstructed image to writer


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

