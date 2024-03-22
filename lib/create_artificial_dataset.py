import os
import torch
import argparse
import pathlib
import models
from torchvision.utils import save_image


def _create_dataset(args):
    device = torch.device('cpu')

    pathh = f'./weights/{args.gan}/{args.dataset}/{args.xai}/'

    generator = models.Generator(args).apply(models.weights_init).to(device)

    if args.dataset == 'ucsb' or args.dataset == 'cr':
        labels = ['Benign', 'Malignant']
    elif args.dataset == 'la':
        labels = ['1', '2', '3', '4']
    elif args.dataset == 'lg':
        labels = ['Class 1', 'Class 2']
    else:
        labels = ['cll', 'fl', 'mcl']

    path = f'./datasets/artificial/{args.dataset.upper()}{args.img_size}/{args.gan}/{args.xai}/'

    for label in labels:
        if not os.path.exists(path + label):
            folder = path + label
            folder = pathlib.Path(folder)
            folder.mkdir(parents=True)

        weights_path = pathh + label + f'/gen_epoch_{args.epoch:03d}.pth'

        generator.load_state_dict(torch.load(weights_path, map_location=device))

        output_folder = path + label + '/'

        i = 1
        for _ in range(int(args.num_imgs/args.batch_size)):
            noise = torch.randn(args.batch_size, args.z_size, 1, 1, device=device)
            fake = generator(noise)

            for j in range(args.batch_size):
                save_image(fake[j], output_folder + f'{i:04d}' + '.png', normalize=True)
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a artificial dataset')
    parser.add_argument('--dataset', '-d', type=str, choices=['cr', 'ucsb', 'la', 'lg', 'nhl'])
    parser.add_argument('--img_size', '-s', type=int, default=64)
    parser.add_argument('--channels', '-c', type=int, default=3)
    parser.add_argument('--num_imgs', '-n', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--gan', type=str,
                        choices=['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN'],
                        default='DCGAN')
    parser.add_argument('--xai', '-x', type=str, choices=['none', 'saliency', 'deeplift', 'inputxgrad'], default='none')

    ######################
    # Generator parameters
    ######################
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--G_h_size', type=int, default=128, help='Number of feature maps')
    parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate')
    parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--spectral_G', type=bool, default=False,
                        help='Use spectral norm. to make the generator Lipschitz (Generally only D is spectral).')
    parser.add_argument('--no_batch_norm_G', type=bool, default=False, help='If True, no batch norm in G.')
    parser.add_argument('--Tanh_GD', type=bool, default=False, help='If True, tanh everywhere.')
    parser.add_argument('--SELU', type=bool, default=False,
                        help='Use SELU instead of ReLU with BatchNorm. This improves stability.')
    parser.add_argument("--NN_conv", type=bool, default=False,
                        help="Uses nearest-neighbor resized convolutions instead of strided convolutions")
    parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
    parser.add_argument('--grad_penalty', type=bool, default=False,
                        help='If True, use gradient penalty of WGAN-GP but with whichever gan chosen.')
    arguments = parser.parse_args()

    _create_dataset(arguments)
