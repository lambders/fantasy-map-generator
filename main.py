import argparse
from train import Trainer
from sample import Sampler 

# ARGPARSER 
parser = argparse.ArgumentParser(description="fantasy-map-generator options")
parser.add_argument("mode",
                    type=str,
                    help="run the network in train or sample mode",
                    default="train",
                    chiuces=["train", "sample"])

# DIRECTORY options
parser.add_argument("--data_dir",
                    type=str,
                    help="path to the training data",
                    default="dataset")
parser.add_argument("--save_dir",
                    type=str,
                    help="path to save model and logs",
                    default="logs")
parser.add_argument("--exp_name",
                    type=str,
                    help="name of experiment, will be a subfolder of save_dir",
                    default="exp1")
parser.add_argument("--weights_dir",
                    type=str,
                    help="name of model to load")

# TRAINING options
parser.add_argument("--batch_size",
                    type=int,
                    help="batch size",
                    default=12)
parser.add_argument("--learning_rate",
                    type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--num_epochs",
                    type=int,
                    help="number of epochs",
                    default=20)
parser.add_argument("--num_workers",
                    type=int,
                    help="number of workers for the data loaing",
                    default=4)

# LOGGING options
parser.add_argument("--log_frequency",
                    type=int,
                    help="number of batches between each tensorboard log",
                    default=250)
parser.add_argument("--save_frequency",
                    type=int,
                    help="number of epochs between each save",
                    default=1)



if __name__ == '__main__': 
    options = parser.parse_args()

    if options.mode == 'train':
        trainer = Trainer(options)
        trainer.train()

    elif options.mode == 'sample':
        sampler = Sampler(options)
        sampler.sample()
