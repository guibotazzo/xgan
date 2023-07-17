import os
import torch
import argparse
import pathlib
import models
from torchvision.utils import save_image


def _create_mnist(args, device):
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


def _create_nhl256(args, device):
    num_imgs = 1024
    batch = 32
    labels = ['cll', 'fl', 'mcl']
    path = '/Users/guilherme/Downloads/artificial_datasets/' + args.model + '/' + args.dataset + '/'

    generator = models.Generator256(noise_dim=100, channels=3, feature_maps=16)

    for label in labels:
        if not os.path.exists(path + label):
            folder = path + label
            folder = pathlib.Path(folder)
            folder.mkdir(parents=True)

        generator.load_state_dict(torch.load(
            '/Users/guilherme/Documents/Doutorado/Sources/xgan/weights/xwgan-gp/nhl256/' + label + '_gen',
            map_location=torch.device('cpu')
        ))

        output_folder = path + label + '/'

        i = 1
        for _ in range(int(num_imgs/batch)):
            noise = torch.randn(batch, 100, 1, 1, device=device)
            fake = generator(noise)

            for j in range(batch):
                save_image(fake[j], output_folder + f'{i:04d}' + '.png', normalize=True)
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a artificial dataset')
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'nhl256'], default='mnist')
    parser.add_argument('--model', '-m', type=str, choices=['dcgan', 'xgan', 'xwgan-gp'], default='dcgan')
    arguments = parser.parse_args()

    dev = torch.device('cpu')

    if arguments.dataset == 'mnist':
        _create_mnist(arguments, dev)

    if arguments.dataset == 'nhl256':
        _create_nhl256(arguments, dev)
