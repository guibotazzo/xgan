import torch
import argparse
from tqdm import tqdm
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.autograd import Variable
from lib import models, datasets, utils


def _load_models(args, device):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        generator = models.GeneratorACGAN(args.n_classes, args.noise_dim, args.img_size, args.channels).to(device)
        generator.apply(models.weights_init)

        discriminator = models.DiscriminatorACGAN(args.channels, args.img_size, args.n_classes).to(device)
        discriminator.apply(models.weights_init)

        return generator, discriminator


def _compute_fid(args, generator, dataset, device):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    float_tensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor

    with tqdm(total=len(dataset), desc='Computing FID') as pbar:
        for reals, _ in dataset:
            reals = reals.to(device)
            if args.channels == 1:
                reals = reals.repeat(1, 3, 1, 1)

            fid.update(reals.to(torch.uint8), real=True)

            z = Variable(float_tensor(np.random.normal(0, 1, (args.batch_size, args.noise_dim)))).to(device)
            gen_labels = Variable(long_tensor(np.random.randint(0, args.n_classes, args.batch_size))).to(device)

            fakes = generator(z, gen_labels)
            if args.channels == 1:
                fakes = fakes.repeat(1, 3, 1, 1)

            fid.update(fakes.to(torch.uint8), real=False)

            pbar.update(1)

    print(fid.compute())


def _compute_is(args, generator, dataset, device):
    inception_score = InceptionScore().to(device)
    float_tensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor

    with tqdm(total=len(dataset), desc='Computing IS') as pbar:
        for _, _ in dataset:
            z = Variable(float_tensor(np.random.normal(0, 1, (args.batch_size, args.noise_dim)))).to(device)
            gen_labels = Variable(long_tensor(np.random.randint(0, args.n_classes, args.batch_size))).to(device)

            fakes = generator(z, gen_labels)
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
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--img_size', '-s', type=int, default=32)
    parser.add_argument('--channels', '-c', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--noise_dim', '-z', type=int, default=100)
    parser.add_argument('--feature_maps', '-f', type=int, default=64)
    args = parser.parse_args()

    device = utils.select_device()
    weights_path = 'weights/acgan/' + args.dataset + f'/gen_epoch_{args.epoch:02d}.pth'

    generator, _ = _load_models(args, device)

    generator.load_state_dict(torch.load(weights_path, map_location=device))

    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)
    utils.print_style('Loaded dataset: ' + args.dataset.upper(), color='GREEN', formatting="ITALIC")

    # Compute metric
    if args.metric == 'fid':
        _compute_fid(args, generator, dataset, device)
    elif args.metric == 'is':
        _compute_is(args, generator, dataset, device)


if __name__ == '__main__':
    main()
