import os
import torch
import argparse
import pathlib
from lib import models
from torchvision.utils import save_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a artificial dataset')
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist'], default='mnist')
    parser.add_argument('--model', '-m', type=str, choices=['dcgan', 'xgan'], default='dcgan')
    args = parser.parse_args()

    device = torch.device('cpu')
    path = '/Users/guilherme/Downloads/artificial_datasets/' + args.model + '/' + args.dataset + '/'

    generator = models.Generator28(noise_dim=100, channels=1, feature_maps=64).to(device)
    generator.load_state_dict(torch.load('../weights/dcgan/mnist/gen_epoch_9.pth'))

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for label in labels:
        if not os.path.exists(path + label):
            folder = path + label
            folder = pathlib.Path(folder)
            folder.mkdir(parents=True)

        output_folder = path + label + '/'

        for i in range(1, 6000):
            noise = torch.randn(1, 100, 1, 1, device=device)
            fake = generator(noise)
            save_image(fake, output_folder + f'{i:04d}' + '.png', normalize=True)
