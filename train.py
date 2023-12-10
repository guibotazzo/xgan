# Based on https://github.com/AlexiaJM/RelativisticGAN

import os
import argparse
import numpy
import torch
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pathlib
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from prettytable import PrettyTable
from captum.attr import Saliency, DeepLift, InputXGradient
from lib import datasets, models, utils


def _minmax_scaler(arr, *, vmin=1, vmax=2):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def _xai_method(method: str, model):
    if method == 'saliency':
        return Saliency(model)
    elif method == 'deeplift':
        return DeepLift(model)
    elif method == 'inputxgrad':
        return InputXGradient(model)
    else:
        utils.print_style('ERROR: This XAI method is not implemented.', color='RED', formatting="ITALIC")


def main(args):
    device = utils.select_device(args.cuda_device)

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True

    torch.utils.backcompat.broadcast_warning.enabled = True

    # Create weights folder
    if args.xai == 'none':
        weights_path = 'weights/' + args.gan + '/' + args.dataset
    else:
        weights_path = 'weights/' + args.gan + '/' + args.dataset + '/' + args.xai

    if not os.path.exists(weights_path):
        path = pathlib.Path(weights_path)
        path.mkdir(parents=True)

    # Setting seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.image_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Load models
    # if args.dataset == 'mnist':
    #     generator = models.GeneratorMNIST(args.z_size, args.channels, args.G_h_size).apply(models.weights_init).to(device)
    #     discriminator = models.DiscriminatorMNIST(args.channels, args.D_h_size).apply(models.weights_init).to(device)
    # else:
    generator = models.Generator(args).apply(models.weights_init).to(device)
    discriminator = models.Discriminator(args).apply(models.weights_init).to(device)

    # Criterion
    criterion = torch.nn.BCELoss().to(device)
    bce_stable = torch.nn.BCEWithLogitsLoss().to(device)
    bce_stable_no_reduce = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    real_label = 1.
    fake_label = 0.

    # Optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=args.lr_D,
                                               betas=(args.beta1, args.beta2),
                                               weight_decay=args.weight_decay)
    generator_optimizer = torch.optim.Adam(generator.parameters(),
                                           lr=args.lr_G,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    # Exponential weight decay on lr
    discriminator_decay = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=1 - args.decay)
    generator_decay = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=1 - args.decay)

    # Load existing models
    # if args.load:
    #     checkpoint = torch.load(args.load)
    #     current_set_images = checkpoint['current_set_images']
    #     iter_offset = checkpoint['i']
    #     generator.load_state_dict(checkpoint['G_state'])
    #     discriminator.load_state_dict(checkpoint['D_state'])
    #     generator_optimizer.load_state_dict(checkpoint['G_optimizer'])
    #     discriminator_optimizer.load_state_dict(checkpoint['D_optimizer'])
    #     generator_decay.load_state_dict(checkpoint['G_scheduler'])
    #     discriminator_decay.load_state_dict(checkpoint['D_scheduler'])
    #     # z_test.copy_(checkpoint['z_test'])
    #     del checkpoint
    #     print(f'Resumed from iteration {current_set_images * args.gen_every}.')
    # else:
    #     current_set_images = 0
    #     iter_offset = 0

    iter_offset = 0
    fixed_noise = torch.randn(32, args.z_size, 1, 1, device=device)

    writer = SummaryWriter()
    print("Starting Training Loop...")
    for epoch in range(iter_offset, args.epochs):
        # Fake images saved
        running_loss_g = 0.0
        running_loss_d = 0.0

        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch + 1)) as pbar:
            for data in dataset:
                real = data[0].to(device)
                current_batch_size = real.size(0)

                for p in discriminator.parameters():
                    p.requires_grad = True

                for _ in range(args.Diters):

                    ##########################
                    # (1) Update discriminator
                    ##########################
                    discriminator.zero_grad()

                    y_pred = discriminator(real)

                    if args.gan in ['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN']:
                        # Train with real data
                        y = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)
                        if args.gan == 'DCGAN':
                            errD_real = criterion(y_pred, y)
                        if args.gan == 'LSGAN':
                            errD_real = torch.mean((y_pred - y) ** 2)
                        if args.gan == 'WGAN-GP':
                            errD_real = -torch.mean(y_pred)
                        if args.gan == 'HingeGAN':
                            errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))

                        errD_real.backward()

                        # Train with fake data
                        z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)
                        y = torch.full((current_batch_size,), fake_label, dtype=torch.float, device=device)

                        fake = generator(z)
                        y_pred_fake = discriminator(fake.detach())

                        if args.gan == 'DCGAN':
                            errD_fake = criterion(y_pred_fake, y)
                        if args.gan == 'LSGAN':
                            errD_fake = torch.mean(y_pred_fake ** 2)
                        if args.gan == 'WGAN-GP':
                            errD_fake = torch.mean(y_pred_fake)
                        if args.gan == 'HingeGAN':
                            errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))

                        errD_fake.backward()
                        errD = errD_real + errD_fake
                    else:
                        y = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)
                        y2 = torch.full((current_batch_size,), fake_label, dtype=torch.float, device=device)
                        z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)

                        fake = generator(z)
                        y_pred_fake = discriminator(fake.detach())

                        if args.gan == 'RSGAN':
                            errD = bce_stable(y_pred - y_pred_fake, y)
                        if args.gan == 'RaSGAN':
                            errD = (bce_stable(y_pred - torch.mean(y_pred_fake), y) + bce_stable(
                                y_pred_fake - torch.mean(y_pred),
                                y2)) / 2
                        if args.gan == 'RaLSGAN':  # (y_hat-1)^2 + (y_hat+1)^2
                            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean(
                                (y_pred_fake - torch.mean(y_pred) + y) ** 2)) / 2
                        if args.gan == 'RaHingeGAN':
                            errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                                torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))) / 2
                        errD_real = errD
                        errD_fake = errD
                        errD.backward()

                    if args.gan in ['WGAN-GP'] or args.grad_penalty:
                        # Gradient penalty
                        b, c, h, w = real.shape
                        alpha = torch.rand((b, 1, 1, 1)).repeat(1, c, h, w).to(device)
                        interpolated_images = real * alpha + fake * (1 - alpha)
                        mixed_scores = discriminator(interpolated_images)
                        gradient = torch.autograd.grad(
                            inputs=interpolated_images,
                            outputs=mixed_scores,
                            grad_outputs=torch.ones_like(mixed_scores),
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        gradient = gradient.view(gradient.shape[0], -1)
                        gradient_norm = gradient.norm(2, dim=1)
                        grad_penalty = torch.mean((gradient_norm - 1) ** 2)
                        grad_penalty.backward()

                    discriminator_optimizer.step()

                ########################
                # (2) Update generator network #
                ########################

                # Make it a tiny bit faster
                for p in discriminator.parameters():
                    p.requires_grad = False

                for t in range(args.Giters):
                    generator.zero_grad()
                    y = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)
                    z = torch.randn(current_batch_size, args.z_size, 1, 1, device=device)
                    fake = generator(z)
                    y_pred_fake = discriminator(fake)

                    if args.gan == 'DCGAN':
                        errG = criterion(y_pred_fake, y)
                    if args.gan == 'LSGAN':
                        errG = torch.mean((y_pred_fake - y) ** 2)
                    if args.gan == 'WGAN-GP':
                        errG = -torch.mean(y_pred_fake)
                    if args.gan == 'HingeGAN':
                        errG = -torch.mean(y_pred_fake)
                    if args.gan == 'RSGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        errG = bce_stable(y_pred_fake - y_pred, y)
                    if args.gan == 'RaSGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        y2 = torch.full((current_batch_size,), fake_label, dtype=torch.float, device=device)
                        errG = (bce_stable(y_pred - torch.mean(y_pred_fake), y2) + bce_stable(y_pred_fake - torch.mean(y_pred),
                                                                                              y)) / 2
                    if args.gan == 'RaLSGAN':
                        y_pred = discriminator(real)
                        errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean(
                            (y_pred_fake - torch.mean(y_pred) - y) ** 2)) / 2
                    if args.gan == 'RaHingeGAN':
                        y_pred = discriminator(real)
                        # Non-saturating
                        errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(
                            torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))) / 2
                    if args.xai != 'none':
                        saliency = _xai_method(args.xai, discriminator)
                        explanations = saliency.attribute(fake)
                        explanations = _minmax_scaler(explanations)
                        explanations = explanations.clone().detach().requires_grad_(True)

                        errX = errG * explanations

                        shape = torch.ones_like(input=explanations, dtype=torch.float32)
                        errX.backward(gradient=shape)
                    else:
                        errG.backward()

                    generator_optimizer.step()

                discriminator_decay.step()
                generator_decay.step()

                running_loss_g += errG.item()
                running_loss_d += errD.item()

                pbar.update(1)

            ###############
            # Saving the results
            ###############
            writer.add_scalar(args.gan + '/loss/generator', running_loss_g / len(dataset.dataset), epoch)
            writer.add_scalar(args.gan + '/loss/discriminator', running_loss_d / len(dataset.dataset), epoch)

            # Save images of the epoch
            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer.add_image(args.gan.upper(), img_grid_fake, global_step=epoch)

            # Save models
            torch.save(generator.state_dict(), weights_path + f'/gen_epoch_{epoch + 1:02d}.pth')
            torch.save(discriminator.state_dict(), weights_path + f'/disc_epoch_{epoch + 1:02d}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #####################
    # Training parameters
    #####################
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Helps convergence but leads to artifacts in images, not recommended.')
    parser.add_argument('--decay', type=float, default=0,
                        help='Decay to apply to lr each cycle.')
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')

    ####################
    # Dataset parameters
    ####################
    parser.add_argument('--dataset', '-d',
                        type=str,
                        choices=['mnist', 'fmnist', 'cifar10', 'celeba', 'nhl', 'caltech', 'cr'],
                        default='cifar10')
    parser.add_argument('--image_size', '-s', type=int, default=32)
    parser.add_argument('--channels', '-c', type=int, default=3)

    ################
    # GAN parameters
    ################
    parser.add_argument('--gan', type=str, default='DCGAN',
                        choices=['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN'])
    parser.add_argument('--xai', '-x', type=str, choices=['none', 'saliency', 'deeplift', 'inputxgrad'], default='none')
    parser.add_argument('--SELU', type=bool, default=False,
                        help='Use SELU which instead of ReLU with BatchNorm. This improves stability.')
    parser.add_argument("--NN_conv", type=bool, default=False,
                        help="Uses nearest-neighbor resized convolutions instead of strided convolutions.")
    parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
    parser.add_argument('--Tanh_GD', type=bool, default=False, help='If True, tanh everywhere.')
    parser.add_argument('--grad_penalty', type=bool, default=False,
                        help='Use gradient penalty of WGAN-GP but with whichever gan chosen.')

    ######################
    # Generator parameters
    ######################
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--G_h_size', type=int, default=128,
                        help='Number of hidden nodes in the Generator.')
    parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate')
    parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--spectral_G', type=bool, default=False,
                        help='Use spectral norm. to make the generator Lipschitz (Generally only D is spectral).')
    parser.add_argument('--no_batch_norm_G', type=bool, default=False, help='If True, no batch norm in G.')

    ##########################
    # Discriminator parameters
    ##########################
    parser.add_argument('--D_h_size', type=int, default=128,
                        help='Number of feature maps in the Discriminator.')
    parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate')
    parser.add_argument('--spectral', type=bool, default=False,
                        help='Use spectral norm. to make the discriminator Lipschitz.')
    parser.add_argument('--no_batch_norm_D', type=bool, default=False, help='If True, no batch norm in D.')
    parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')

    arguments = parser.parse_args()

    # Print parameters
    conf = PrettyTable()
    conf.field_names = ["Parameters", "Values"]
    conf.add_row(["Method", arguments.gan])
    conf.add_row(["XAI method", arguments.xai])
    conf.add_row(["Dataset", arguments.dataset.upper()])
    conf.add_row(["Image size", arguments.image_size])
    conf.add_row(["Channels", arguments.channels])
    conf.add_row(["Batch size", arguments.batch_size])
    conf.add_row(["Epochs", arguments.epochs])
    print(conf)

    main(arguments)
