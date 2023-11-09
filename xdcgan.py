import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from torchvision.utils import make_grid
from lib import models, datasets, utils
from captum.attr import Saliency, DeepLift, InputXGradient
from torch.utils.tensorboard import SummaryWriter
import pathlib


def _minmax_scaler(arr, *, vmin=1, vmax=2):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def _xai_method(dataset, method: str, model):
    if method == 'saliency':
        return Saliency(model)
    elif method == 'deeplift':
        return DeepLift(model)
    elif method == 'gradcam':
        return InputXGradient(model)
    else:
        utils.print_style('ERROR: This XAI method is not implemented.', color='RED', formatting="ITALIC")


def main():
    parser = argparse.ArgumentParser(description='XDCGAN')
    parser.add_argument('--gan', '-g', type=str, choices=['xdcgan'], default='xdcgan')
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'fmnist', 'cifar10', 'celeba', 'nhl'],
                        default='mnist')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--img_size', '-s', type=int, default=28)
    parser.add_argument('--channels', '-c', type=int, default=1)
    parser.add_argument('--feature_maps', '-f', type=int, default=64)
    parser.add_argument('--noise_dim', '-z', type=int, default=100)
    parser.add_argument('--xai', '-x', type=str, choices=['saliency', 'deeplift', 'gradcam'], default='saliency')
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')
    args = parser.parse_args()

    weights_path = 'weights/xdcgan/' + args.dataset + '/' + args.xai

    if not os.path.exists(weights_path):
        path = pathlib.Path(weights_path)
        path.mkdir(parents=True)

    # Set manual seed to a constant get a consistent output
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Parameters
    device = utils.select_device(args.cuda_device)

    # Load dataset
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)
    utils.print_style('Loaded dataset: ' + args.dataset.upper(), color='CYAN', formatting="ITALIC")

    # Create models
    generator, discriminator = models.load_models(args, device)

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

                if epoch > -1:  # int(args.epochs/2):
                    # -------------------------------------
                    # fooled = (output > 0.6).float().reshape((batch_size, 1, 1, 1))

                    saliency = _xai_method(args.dataset, args.xai, discriminator)
                    explanations = saliency.attribute(fake)
                    explanations = _minmax_scaler(explanations)
                    explanations = explanations.clone().detach().requires_grad_(True)
                    # explanations = fooled * explanations

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
            writer.add_scalar('xdcgan/' + args.xai + '/loss/generator', running_loss_g/len(dataset.dataset), epoch)
            writer.add_scalar('xdcgan/' + args.xai + '/loss/discriminator', running_loss_d/len(dataset.dataset), epoch)

            # Save images of the epoch
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer.add_image("XDCGAN + " + args.xai, img_grid_fake, global_step=epoch)

                saliency = Saliency(discriminator)
                explanations = saliency.attribute(fake)
                exp_grid = make_grid(explanations[:32], normalize=True)
                writer.add_image(args.xai.upper(), exp_grid, global_step=epoch)

            # Save models
            torch.save(generator.state_dict(), weights_path + f'/gen_epoch_{epoch+1:02d}.pth')
            torch.save(discriminator.state_dict(), weights_path + f'/disc_epoch_{epoch+1:02d}.pth')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
