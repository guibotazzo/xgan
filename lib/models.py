import torch
import torch.nn as nn


def weights_init(m):
    """
        Initial weights for DCGAN.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = args.img_size // 8

        # Start block
        # Z_size random numbers
        if args.spectral_G:
            main.add_module('Start-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(
                torch.nn.ConvTranspose2d(args.z_size, args.G_h_size * mult, kernel_size=4, stride=1, padding=0,
                                         bias=False)))
        else:
            main.add_module('Start-ConvTranspose2d',
                            torch.nn.ConvTranspose2d(args.z_size, args.G_h_size * mult, kernel_size=4, stride=1,
                                                     padding=0, bias=False))
        if args.SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            if not args.no_batch_norm_G and not args.spectral_G:
                main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(args.G_h_size * mult))
            if args.Tanh_GD:
                main.add_module('Start-Tanh', torch.nn.Tanh())
            else:
                main.add_module('Start-ReLU', torch.nn.ReLU())
        # Size = (G_h_size * mult) x 4 x 4

        # Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            if args.NN_conv:
                main.add_module('Middle-UpSample [%d]' % i, torch.nn.Upsample(scale_factor=2))
                if args.spectral_G:
                    main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(
                        torch.nn.Conv2d(args.G_h_size * mult, args.G_h_size * (mult // 2), kernel_size=3,
                                        stride=1, padding=1)))
                else:
                    main.add_module('Middle-Conv2d [%d]' % i,
                                    torch.nn.Conv2d(args.G_h_size * mult, args.G_h_size * (mult // 2),
                                                    kernel_size=3, stride=1, padding=1))
            else:
                if args.spectral_G:
                    main.add_module('Middle-SpectralConvTranspose2d [%d]' % i, torch.nn.utils.spectral_norm(
                        torch.nn.ConvTranspose2d(args.G_h_size * mult, args.G_h_size * (mult // 2), kernel_size=4,
                                                 stride=2, padding=1, bias=False)))
                else:
                    main.add_module('Middle-ConvTranspose2d [%d]' % i,
                                    torch.nn.ConvTranspose2d(args.G_h_size * mult, args.G_h_size * (mult // 2),
                                                             kernel_size=4, stride=2, padding=1, bias=False))
            if args.SELU:
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                if not args.no_batch_norm_G and not args.spectral_G:
                    main.add_module('Middle-BatchNorm2d [%d]' % i,
                                    torch.nn.BatchNorm2d(args.G_h_size * (mult // 2)))
                if args.Tanh_GD:
                    main.add_module('Middle-Tanh [%d]' % i, torch.nn.Tanh())
                else:
                    main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        # End block
        # Size = G_h_size x image_size/2 x image_size/2
        if args.NN_conv:
            main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
            if args.spectral_G:
                main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(args.G_h_size, args.channels, kernel_size=3, stride=1, padding=1)))
            else:
                main.add_module('End-Conv2d',
                                torch.nn.Conv2d(args.G_h_size, args.channels, kernel_size=3, stride=1, padding=1))
        else:
            if args.spectral_G:
                main.add_module('End-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(
                    torch.nn.ConvTranspose2d(args.G_h_size, args.channels, kernel_size=4, stride=2, padding=1,
                                             bias=False)))
            else:
                main.add_module('End-ConvTranspose2d',
                                torch.nn.ConvTranspose2d(args.G_h_size, args.channels, kernel_size=4, stride=2,
                                                         padding=1, bias=False))
        main.add_module('End-Tanh', torch.nn.Tanh())
        # Size = channels x image_size x image_size
        self.main = main

    def forward(self, images):
        return self.main(images)


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        main = torch.nn.Sequential()

        # Start block
        # Size = channels x image_size x image_size
        if args.spectral:
            main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(args.channels, args.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        else:
            main.add_module('Start-Conv2d',
                            torch.nn.Conv2d(args.channels, args.D_h_size, kernel_size=4, stride=2, padding=1,
                                            bias=False))
        if args.SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            if args.Tanh_GD:
                main.add_module('Start-Tanh', torch.nn.Tanh())
            else:
                main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = args.img_size // 2
        # Size = D_h_size x image_size/2 x image_size/2

        # Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            if args.spectral:
                main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(
                    torch.nn.Conv2d(args.D_h_size * mult, args.D_h_size * (2 * mult), kernel_size=4, stride=2,
                                    padding=1, bias=False)))
            else:
                main.add_module('Middle-Conv2d [%d]' % i,
                                torch.nn.Conv2d(args.D_h_size * mult, args.D_h_size * (2 * mult), kernel_size=4,
                                                stride=2, padding=1, bias=False))
            if args.SELU:
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                if not args.no_batch_norm_D and not args.spectral:
                    main.add_module('Middle-BatchNorm2d [%d]' % i,
                                    torch.nn.BatchNorm2d(args.D_h_size * (2 * mult)))
                if args.Tanh_GD:
                    main.add_module('Start-Tanh [%d]' % i, torch.nn.Tanh())
                else:
                    main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        # End block
        # Size = (D_h_size * mult) x 4 x 4
        if args.spectral:
            main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(args.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
        else:
            main.add_module('End-Conv2d',
                            torch.nn.Conv2d(args.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0,
                                            bias=False))
        if args.gan in ['DCGAN']:
            main.add_module('End-Sigmoid', torch.nn.Sigmoid())
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, images):
        return self.main(images).view(-1)
