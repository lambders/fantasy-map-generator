import os 
import json
import time 

import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model import StyleGAN


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
        self.writer = SummaryWriter(self.log_dir)
        
        # Dataset 
        dataset = datasets.ImageFolder(
            root = self.opt.data_dir,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
        ]))

        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers)
        
        # Model
        self.model = StyleGAN(self.opt)
        self.model.to(self.device)
        if self.opt.weights_folder is not None:
            self.load_model()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.learning_rate)
            
   
    def train(self):
        """
        Main training loop function for class.
        """
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            # Save model
            if (self.epoch + 1) % self.opt.save_frequency:
                save_epoch_dir = os.path.join(self.results_dir, "weights_" + str(self.epoch).zfill(3))
                torch.save(self.model.state_dict(), os.path.join(save_epoch_dir, "model.pth"))
                torch.save(self.optimizer.state_dict(), os.path.join(save_epoch_dir, "adam.pth"))

    
    def run_epoch(self):
        """
        Run one training epoch.
        """
        for batch_idx, inputs in enumerate(self.dataloader):

            # Forward pass 
            outputs, losses = self.process_batch(inputs)

            # Compute losses
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # Logging
            if (self.step + 1) % self.opt.log_frequency == 0:
                self.log("train", inputs, outputs, losses)

            self.step += 1

    
    def compute_losses(self):
        return None 

    
    def log(self):
        """
        Write logging statements to the Tensorboard file.
        """
        for l, v in losses.items():
            self.writer.add_scalar(l, v, self.step)
            # TODO: Add sampled image, reconstructed image to writer


    def load_model(self):
        # loading main network
        optimizer_load_path = os.path.join(self.opt.weights_dir, "model.pth")
        model_dict = torch.load(model_load_path)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.weights_dir, "adam.pth")
        optimizer_dict = torch.load(optimizer_load_path)
        self.optimizer.load_state_dict(optimizer_dict)
