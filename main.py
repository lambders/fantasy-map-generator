import os 
import argparse 

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tvt 
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FantasyMapGAN, Decoder



def train(data_dir: str, load_dir: str, save_dir: str, 
          num_epochs: int, batch_size: int, lr: float, 
          use_gpu: bool):
    '''
    Train the SpriteGAN.
    Saves checkpoints and training logs to `save_dir`.

    Parameters
    ----------
    data_dir: str
        folder path to training data
    load_dir: str
        folder to load network weights from
    save_dir: str
        folder to save network weights to
    num_epochs: int
        number of epochs to train the network
    batch_size: int
        batch size for each network update step
    lr: float
        learning rate for generator, discriminator will have learning rate lr/2
    use_gpu: bool
        Set to true to run training on GPU, otherwise run on CPU
    '''
    # Dataloader
    dataset = datasets.ImageFolder(
        root = data_dir,
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(128),
            transforms.ToTensor(),
        ]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Writer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    writer = SummaryWriter(save_dir)

    # Network
    map_gan = FantasyMapGAN(lr, batch_size, use_gpu)

    # Train
    step = 0
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(loader):
            if use_gpu:
                x = x.cuda()
            loss_dict = map_gan.forward(x)
            step += 1
            print("Step % i" % step, loss_dict)

            # Log
            if i % 100 == 0:
                for (k,v) in loss_dict.items():
                    writer.add_scalar(k, v.item(), step)
        
        # Save
        if epoch % 20 == 0:
            map_gan.save(save_dir, epoch)
            print("Save new model at epoch %i." % epoch)

            # Log sample image
            with torch.no_grad():
                n_sample = 3
                x_gen = map_gan.sample(n_sample)
                gen_image = make_grid(x_gen, nrow=n_sample, normalize=True)
                writer.add_image('image/gen', gen_image, epoch)

                x_recon = map_gan.reconstruct(x[:n_sample])
                recon_image = make_grid(x_recon, nrow=n_sample, normalize=True)
                writer.add_image('image/recon', recon_image, epoch)

                orig_image = make_grid(x[:n_sample], nrow=n_sample, normalize=True)
                writer.add_image('image/orig', orig_image, epoch)


def sample(load_dir: str, save_dir: str, use_gpu: bool) -> None:
    '''
    Sample the FantasyMapGAN for new maps.
    Saves the generated images to `save_dir`.

    Parameters
    ----------
    load_dir: str
        folder to load network weights from
    save_dir: str
        folder to save network weights to
    use_gpu: bool
        Set to true to run training on GPU, otherwise run on CPU
    '''
    # Network
    model = Decoder()
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    if load_dir:
        fs = glob(os.path.join(load_dir, '*_dec.pth'))
        fs.sort(key=os.path.getmtime)
        model.load_state_dict(torch.load(fs[-1]))
    
    # Generate
    z = torch.randn((1, model.latent_dim))
    x = model.forward(z)

    # Save
    save_path = os.path.join(save_dir, str(uuid.uuid1()) + '.png')
    save_image(x.squeeze(), save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fantasy-map-generator options")
    parser.add_argument("--mode",
                        type=str,
                        help="run the network in train or sample mode",
                        default="train",
                        choices=["train", "sample"])

    # DIRECTORY options
    parser.add_argument("--save_dir",
                        type=str,
                        help="path to save model and logs",
                        default="logs")
    parser.add_argument("--load_dir",
                        type=str,
                        help="name of model to load")

    # TRAINING options
    parser.add_argument("--data_dir",
                        type=str,
                        help="path to the training data",
                        default="./maps")
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=64)
    parser.add_argument("--learning_rate",
                        type=float,
                        help="learning rate",
                        default=2e-4)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="number of epochs",
                        default=50000)
    parser.add_argument("--use_gpu",
                        help="if set, train on gpu instead of cpu",
                        action="store_true")

    # Train or sample!
    options = parser.parse_args()
    if options.mode == 'train':
        train(options.data_dir, options.load_dir, options.save_dir, options.num_epochs, options.batch_size, options.learning_rate, options.use_gpu)
    elif options.mode == 'sample':
        sample(options.load_dir, options.save_dir, options.use_gpu)