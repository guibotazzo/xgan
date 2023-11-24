import torch
import torch.nn as nn


#######################
# Models for generation
#######################
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
        mult = args.image_size // 8

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
        image_size_new = args.image_size // 2
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


class GeneratorMNIST(nn.Module):
    """
        Generator model for 28x28-sized images.
    """
    def __init__(self, noise_dim, channels, feature_maps):
        super(GeneratorMNIST, self).__init__()
        self.zd = noise_dim  # Size of the input noise
        self.nf = feature_maps  # Number of feature maps
        self.nc = channels  # Number of channels in the training images
        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.zd, out_channels=self.nf * 8, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(self.nf * 8),
            nn.ReLU(True),
            # state size. (nf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.nf * 8, out_channels=self.nf * 4, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(True),
            # state size. (nf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=self.nf * 4, out_channels=self.nf * 2, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU(True),
            # state size. (nf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=self.nf * 2, out_channels=self.nf, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.nf, out_channels=self.nc, kernel_size=1, stride=1,
                               padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        output = self.network(noise)
        return output


class DiscriminatorMNIST(nn.Module):
    """
        DCGAN Discriminator model for 28x28-sized images.
    """

    def __init__(self, channels, feature_maps):
        super(DiscriminatorMNIST, self).__init__()
        self.nf = feature_maps  # Size of feature maps
        self.nc = channels  # Number of channels in the training images
        self.network = nn.Sequential(
            # input is (1) x 64 x 64
            nn.Conv2d(in_channels=self.nc, out_channels=self.nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf) x 32 x 32
            nn.Conv2d(in_channels=self.nf, out_channels=self.nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*2) x 16 x 16
            nn.Conv2d(in_channels=self.nf * 2, out_channels=self.nf * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*4) x 8 x 8
            nn.Conv2d(in_channels=self.nf * 4, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        output = self.network(img)
        return output.view(-1, 1).squeeze(1)


class GeneratorCIFAR(nn.Module):
    def __init__(self, noise_dim, channels, feature_maps):
        super(GeneratorCIFAR, self).__init__()
        self.nz = noise_dim  # Size of z latent vector
        self.ngf = feature_maps  # Size of feature maps in generator
        self.nc = channels  # Number of channels in the training images
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf*8, kernel_size=3, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*8, out_channels=self.ngf*4, kernel_size=3, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*4, out_channels=self.ngf*2, kernel_size=3, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.ngf, kernel_size=3, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=self.nc, kernel_size=2, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)


class DiscriminatorCIFAR(nn.Module):
    def __init__(self, channels, feature_maps):
        super(DiscriminatorCIFAR, self).__init__()
        self.ndf = feature_maps  # Size of feature maps in discriminator
        self.nc = channels  # Number of channels of the training images
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=2, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*8, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.network(img)


###########################
# Models for classification
###########################
def reset_weights(m):
    """
        Reset model weights to avoid weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class ConvNet(nn.Module):
    """
        Simple ConvNet for classification (MNIST and FMNIST datasets).
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 10, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.layers(x)