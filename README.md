# fantasy-map-generator
Using Generative Adversarial Networks (GANs) to produce awesome looking fantasy maps

## Overview
The overall network architecture is inspired by the Progressive Growing of GANs paper, but modified and simplified. The main entrypoint for training is at `main.py`. Run with the `-h` flag if you'd like a detailed overview of the possible command-line options.

Some sample commands:
```
# To train
python main.py --mode=train

# To sample
python main.py --mode=sample --weights-dir=/path/to/pth_files

# View the options
python main.py -h
usage: main.py [-h] [--mode {train,sample}] [--data_dir DATA_DIR]
               [--save_dir SAVE_DIR] [--exp_name EXP_NAME]
               [--weights_dir WEIGHTS_DIR] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--disc_drift DISC_DRIFT]
               [--num_epochs NUM_EPOCHS] [--num_workers NUM_WORKERS]
               [--no_cuda] [--im_size IM_SIZE] [--latent_size LATENT_SIZE]
               [--log_frequency LOG_FREQUENCY]
               [--save_frequency SAVE_FREQUENCY]

fantasy-map-generator options

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,sample}
                        run the network in train or sample mode
  --data_dir DATA_DIR   path to the training data
  --save_dir SAVE_DIR   path to save model and logs
  --exp_name EXP_NAME   name of experiment, will be a subfolder of save_dir
  --weights_dir WEIGHTS_DIR
                        name of model to load
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --disc_drift DISC_DRIFT
                        small term added to discriminator loss
  --num_epochs NUM_EPOCHS
                        number of epochs
  --num_workers NUM_WORKERS
                        number of workers for the data loading
  --no_cuda             if set, train on cpu instead of gpu
  --im_size IM_SIZE     size of image in pixels
  --latent_size LATENT_SIZE
                        size of image in pixels
  --log_frequency LOG_FREQUENCY
                        number of batches between each tensorboard log
  --save_frequency SAVE_FREQUENCY
                        number of epochs between each save
```


## The dataset
I used the Selenium IDE to build the dataset of fantasy maps. Credits to @mewo2, who built the map generator I used to build the dataset. The dataset is available on Google Drive [link](https://drive.google.com/file/d/176W29rLNqD4a6xEs5UicnChoq49ufgM5/view?usp=sharing). The dataset contains 4000 images.