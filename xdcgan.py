import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torchvision.utils import make_grid
from lib import models, datasets
from captum.attr import Saliency
from torch.utils.tensorboard import SummaryWriter
import pathlib


def _select_device():
    if torch.backends.mps.is_available():
        print("MPS device selected.")
        return torch.device("mps")  # For M1 Macs
    elif torch.cuda.is_available():
        print("CUDA device selected.")
        return torch.device("cuda:0")
    else:
        print("CPU device selected.")
        return torch.device('cpu')


def _load_models(ds: str, im_size: int, noise_dim: int, channels: int, feature_maps: int):
    if ds == 'mnist' or ds == 'fmnist':
        return models.Generator28(noise_dim, channels, feature_maps).to(device).apply(models.weights_init), \
               models.Discriminator28(channels, feature_maps).to(device).apply(models.weights_init)
    elif ds == 'nhl':
        if im_size == 256:
            return models.Generator256(noise_dim, channels, feature_maps).to(device).apply(models.weights_init),\
                   models.Discriminator256(channels, feature_maps).to(device).apply(models.weights_init)
    else:
        return models.Generator64(noise_dim, channels, feature_maps).to(device).apply(models.weights_init),\
               models.Discriminator64(channels, feature_maps).to(device).apply(models.weights_init)


def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGAN')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'fmnist', 'celeba', 'nhl'], default='mnist')
    parser.add_argument('--img_size', '-s', type=int, default=28)
    parser.add_argument('--channels', '-c', type=int, default=1)
    parser.add_argument('--noise_size', '-z', type=int, default=100)
    parser.add_argument('--feature_maps', '-f', type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists('weights/xdcgan/' + args.dataset):
        path = 'weights/xdcgan/' + args.dataset
        path = pathlib.Path(path)
        path.mkdir(parents=True)

    # Set manual seed to a constant get a consistent output
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Parameters
    device = _select_device()

    # Load dataset
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Create models
    generator, discriminator = _load_models(ds=args.dataset,
                                            im_size=args.img_size,
                                            noise_dim=args.noise_size,
                                            channels=args.channels,
                                            feature_maps=args.feature_maps)

    ###############
    # Training Loop
    ###############
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    cross_entropy = nn.BCELoss()  # Binary cross entropy function

    real_label = 1.
    fake_label = 0.

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        running_loss_g = 0.0
        running_loss_d = 0.0

        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch+1)) as pbar:
            for i, data in enumerate(dataset):
                ############################
                # (1) Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # Train with real samples
                discriminator.zero_grad()

                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size, 1, 1, 1) if args.channels == 3 else (batch_size, ),
                                   real_label,
                                   dtype=torch.float,
                                   device=device)

                output = discriminator(real_cpu)
                errD_real = cross_entropy(output, label)
                errD_real.backward()

                # Train with fake samples
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake = generator(noise)
                label.fill_(fake_label)

                output = discriminator(fake.detach())
                errD_fake = cross_entropy(output, label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                discriminator_optimizer.step()

                ############################
                # (2) Update generator: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                label.fill_(real_label)

                output = discriminator(fake)

                if epoch > int(args.epochs/2):
                    # -------------------------------------
                    fooled = (output > 0.6).float().reshape((batch_size, 1, 1, 1))

                    saliency = Saliency(discriminator)
                    explanations = saliency.attribute(fake)
                    explanations = minmax_scaler(explanations)
                    explanations = fooled * explanations

                    errG = cross_entropy(output, label)
                    errX = errG * explanations

                    shape = torch.ones_like(input=explanations, dtype=torch.float32)
                    errX.backward(gradient=shape)
                    # -------------------------------------
                else:
                    errG = cross_entropy(output, label)
                    errG.backward()

                generator_optimizer.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                running_loss_g += errG.item()
                running_loss_d += errD.item()

                # pbar.set_postfix(g_loss=errG.item(), d_loss=errD.item())
                pbar.update(1)

            # Write epoch loss on Tensorboard
            writer.add_scalar('xdcgan/loss/generator', running_loss_g/len(dataset.dataset), epoch)
            writer.add_scalar('xdcgan/loss/discriminator', running_loss_d/len(dataset.dataset), epoch)

            # Save images of the epoch
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer.add_image("Fake images", img_grid_fake, global_step=epoch)

            saliency = Saliency(discriminator)
            explanations = saliency.attribute(fake)
            exp_grid = make_grid(explanations[:32], normalize=True)
            writer.add_image("Saliency", exp_grid, global_step=epoch)

            # Save models
            torch.save(generator.state_dict(), 'weights/xdcgan/' + args.dataset + '/gen_epoch_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'weights/xdcgan/' + args.dataset + '/disc_epoch_%d.pth' % epoch)

    writer.flush()
    writer.close()
