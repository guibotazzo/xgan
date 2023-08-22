import torch
import argparse
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from lib import models, datasets, utils


def _load_models(dataset, noise_dim: int, channels: int, feature_maps: int, device):
    if dataset == 'mnist' or dataset == 'fmnist':
        return models.Generator28(noise_dim, channels, feature_maps).to(device).apply(models.weights_init), \
               models.Discriminator28(channels, feature_maps).to(device).apply(models.weights_init)
    elif dataset == 'cifar10':
        return models.Generator32(noise_dim, channels, feature_maps).to(device).apply(models.weights_init), \
            models.Discriminator32(channels, feature_maps).to(device).apply(models.weights_init)
    elif dataset == 'celeba':
        return models.Generator64(noise_dim, channels, feature_maps).to(device).apply(models.weights_init),\
               models.Discriminator64(channels, feature_maps).to(device).apply(models.weights_init)
    elif dataset == 'nhl':
        return models.Generator256(noise_dim, channels, feature_maps).to(device).apply(models.weights_init),\
               models.Discriminator256(channels, feature_maps).to(device).apply(models.weights_init)
    else:
        utils.print_style('ERROR: This dataset is not implemented.', color='RED', formatting="ITALIC")


def _compute_fid(args, generator, dataset, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    with tqdm(total=len(dataset), desc='Computing FID') as pbar:
        for reals, _ in dataset:
            reals = reals.to(device)
            if args.channels == 1:
                reals = reals.repeat(1, 3, 1, 1)

            fid.update(reals.to(torch.uint8), real=True)

            noise = torch.randn(args.batch_size, 100, 1, 1, device=device)
            fakes = generator(noise)
            if args.channels == 1:
                fakes = fakes.repeat(1, 3, 1, 1)

            fid.update(fakes.to(torch.uint8), real=False)

            pbar.update(1)

    print(fid.compute())


def _compute_is(args, generator, dataset, device):
    inception_score = InceptionScore().to(device)

    with tqdm(total=len(dataset), desc='Computing IS') as pbar:
        for _, _ in dataset:
            noise = torch.randn(args.batch_size, 100, 1, 1, device=device)
            fakes = generator(noise)
            if args.channels == 1:
                fakes = fakes.repeat(1, 3, 1, 1)

            inception_score.update(fakes.to(torch.uint8))

            pbar.update(1)

    print(inception_score.compute())


def main():
    parser = argparse.ArgumentParser(description='Evaluation metrics')
    parser.add_argument('--metric', '-m', type=str, choices=['fid', 'is'], default='fid')
    parser.add_argument('--gan', '-g', type=str, choices=['dcgan', 'xdcgan'], default='dcgan')
    parser.add_argument('--xai', '-x', type=str, choices=['saliency', 'deeplift', 'gradcam'], default='saliency')
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'fmnist', 'cifar10', 'celeba', 'nhl'],
                        default='mnist')
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--img_size', '-s', type=int, default=28)
    parser.add_argument('--channels', '-c', type=int, default=1)
    parser.add_argument('--noise_dim', '-z', type=int, default=100)
    parser.add_argument('--feature_maps', '-f', type=int, default=64)
    args = parser.parse_args()

    device = utils.select_device()
    if args.gan == 'xdcgan':
        weights_path = 'weights/xdcgan/' + args.dataset + '/' + args.xai + f'/gen_epoch_{args.epoch:02d}.pth'
    else:
        weights_path = 'weights/dcgan/' + args.dataset + f'/gen_epoch_{args.epoch:d}.pth'

    generator, _ = _load_models(dataset=args.dataset,
                                noise_dim=args.noise_dim,
                                channels=args.channels,
                                feature_maps=args.feature_maps,
                                device=device)

    generator.load_state_dict(torch.load(weights_path, map_location=device))

    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
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
    main()
