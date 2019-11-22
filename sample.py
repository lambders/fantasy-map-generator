import os 
import torch 
from torchvision.utils import save_image
from model import Generator


class Sampler():
    def __init__(self, options):
        """
        Initialize the high-level controller which controls sampling.
        Args:
            options: Refer to main.py for a detailed description of the options you can use.
        """
        self.opt = options

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # Models
        self.generator = Generator(self.opt).to(self.device)
        if self.opt.weights_dir is not None:
            self.load_model()


    def sample(self):
        """
        Generates opt.batch_size images.
        """
        fake_latent = torch.randn(self.opt.batch_size, self.opt.latent_size, 1, 1).to(self.device)
        fake_images = self.generator(fake_latent) 
        save_image(fake_images, 'sample.png', nrow=1)

    def load_model(self):
        """
        Loading the weights for the network.
        """
        load_path = os.path.join(self.opt.weights_dir, "gen.pth")
        model_dict = torch.load(load_path)
        self.generator.load_state_dict(model_dict)
