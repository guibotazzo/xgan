import os
import torch
import argparse
import pathlib
import models
from torchvision.utils import save_image
import numpy as np
from torch.autograd import Variable


# def _create_mnist(args, device):
#     path = '/Users/guilherme/datasets/artificial/' + args.model + '/' + args.dataset + '/'
#
#     generator = models.GeneratorACGAN(n_classes=10, z_dim=100, img_size=32, channels=1).to(device)
#     # generator = models.Generator28(noise_dim=100, channels=1, feature_maps=64).to(device)
#     generator.load_state_dict(
#         torch.load('/Users/guilherme/Documents/Doutorado/Sources/xgan/weights/xacgan/mnist/saliency/gen_epoch_10.pth',
#                    map_location=torch.device('cpu')))
#
#     float_tensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
#     long_tensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor
#
#     labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#
#     for label in labels:
#         if not os.path.exists(path + label):
#             folder = path + label
#             folder = pathlib.Path(folder)
#             folder.mkdir(parents=True)
#
#         output_folder = path + label + '/'
#
#         for i in range(1, 6000):
#             # noise = torch.randn(1, 100, 1, 1, device=device)
#             # fake = generator(noise)
#             # save_image(fake, output_folder + f'{i:04d}' + '.png', normalize=True)
#
#             z = Variable(float_tensor(np.random.normal(0, 1, (64, args.noise_dim)))).to(device)
#             gen_labels = Variable(long_tensor(np.random.randint(0, args.n_classes, 64))).to(device)
#
#             # Generate a batch of images
#             gen_imgs = generator(z, gen_labels)
#
#             save_image(fake, output_folder + f'{i:04d}' + '.png', normalize=True)


def _create_mnist(args, device):
    num_imgs = 6000
    batch = 64
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    path = '/Users/guilherme/datasets/artificial/' + args.model + '/' + args.dataset + '/'

    generator = models.GeneratorACGAN(n_classes=10, z_dim=100, img_size=32, channels=1).to(device)
    weights_path = '/Users/guilherme/Documents/Doutorado/Sources/xgan/weights/xacgan/mnist/deeplift/gen_epoch_10.pth'
    generator.load_state_dict(torch.load(weights_path, map_location=device))

    float_tensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor

    for label in labels:
        if not os.path.exists(path + label):
            folder = path + label
            folder = pathlib.Path(folder)
            folder.mkdir(parents=True)

        output_folder = path + label + '/'

        i = 1
        for _ in range(int(num_imgs/batch)):
            # noise = torch.randn(batch, 100, 1, 1, device=device)
            # fake = generator(noise)

            z = Variable(float_tensor(np.random.normal(0, 1, (batch, 100)))).to(device)
            label_array = np.ones(batch, dtype=int) * int(label)
            gen_labels = Variable(long_tensor(label_array)).to(device)

            fake = generator(z, gen_labels)

            for j in range(batch):
                save_image(fake[j], output_folder + f'{i:04d}' + '.png', normalize=True)
                i += 1


def _create_nhl256(args, device):
    num_imgs = 1024
    batch = 32
    labels = ['cll', 'fl', 'mcl']
    path = '/Users/guilherme/datasets/artificial' + args.model + '/' + args.dataset + '/'

    generator = models.Generator256(noise_dim=100, channels=3, feature_maps=16)

    for label in labels:
        if not os.path.exists(path + label):
            folder = path + label
            folder = pathlib.Path(folder)
            folder.mkdir(parents=True)

        generator.load_state_dict(torch.load(
            '/Users/guilherme/Documents/Doutorado/Sources/xdcgan/weights/xwgan-gp/nhl256/' + label + '_gen',
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
    parser.add_argument('--model', '-m', type=str, choices=['dcgan', 'xdcgan', 'xwgan-gp', 'xacgan'], default='dcgan')
    arguments = parser.parse_args()

    dev = torch.device('cpu')

    if arguments.dataset == 'mnist':
        _create_mnist(arguments, dev)

    if arguments.dataset == 'nhl256':
        _create_nhl256(arguments, dev)
