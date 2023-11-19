import torch
import argparse
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from lib import models, datasets, utils


def _minmax_scaler(arr, *, vmin=0, vmax=255):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def _compute_fid(args, generator, dataset, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    end = 50000/32
    i = 1
    with tqdm(total=int(end), desc='Computing FID') as pbar:
        for reals, _ in dataset:
            reals = reals.to(device)
            reals = _minmax_scaler(reals)

            if args.channels == 1:
                reals = reals.repeat(1, 3, 1, 1)

            fid.update(reals.to(torch.uint8), real=True)

            noise = torch.randn(args.batch_size, args.z_size, 1, 1, device=device)
            fakes = generator(noise)
            fakes = _minmax_scaler(fakes)

            if args.channels == 1:
                fakes = fakes.repeat(1, 3, 1, 1)

            fid.update(fakes.to(torch.uint8), real=False)

            pbar.update(1)
            i = i + 1

            if i == end:
                break

    print(fid.compute())


def _compute_is(args, generator, dataset, device):
    inception_score = InceptionScore().to(device)

    with tqdm(total=len(dataset), desc='Computing IS') as pbar:
        for _, _ in dataset:
            noise = torch.randn(args.batch_size, args.z_size, 1, 1, device=device)
            fakes = generator(noise)
            fakes = _minmax_scaler(fakes)

            if args.channels == 1:
                fakes = fakes.repeat(1, 3, 1, 1)

            inception_score.update(fakes.to(torch.uint8))

            pbar.update(1)

    print(inception_score.compute())


def main(args):
    device = utils.select_device(args.cuda_device)

    if args.xai == 'none':
        weights_path = 'weights/' + args.gan + '/' + args.dataset + f'/gen_epoch_{args.epoch:d}.pth'
    else:
        weights_path = 'weights/' + args.gan + '/' + args.dataset + '/' + args.xai + f'/gen_epoch_{args.epoch:d}.pth'

    # Load generator
    generator = models.Generator(args).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))

    # Load the dataset (real images)
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.image_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)
    utils.print_style('Loaded dataset: ' + args.dataset.upper(), color='CYAN', formatting="ITALIC")

    # Compute metric
    if args.metric == 'fid':
        _compute_fid(args, generator, dataset, device)
    elif args.metric == 'is':
        _compute_is(args, generator, dataset, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation metrics')
    parser.add_argument('--metric', '-m', type=str, choices=['fid', 'is'], default='fid')
    parser.add_argument('--gan', type=str,
                        choices=['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN'],
                        default='DCGAN')
    parser.add_argument('--xai', '-x', type=str, choices=['none', 'saliency', 'deeplift', 'inputxgrad'], default='none')
    parser.add_argument('--dataset', '-d', type=str,
                        choices=['mnist', 'fmnist', 'cifar10', 'celeba', 'nhl'],
                        default='cifar10')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--image_size', '-s', type=int, default=32)
    parser.add_argument('--channels', '-c', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')

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

    main(arguments)
