import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator28(nn.Module):
    """
        Generator model for 28x28-sized images.
    """

    def __init__(self, noise_dim, channels, feature_maps):
        super(Generator28, self).__init__()
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

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator28(nn.Module):
    """
        DCGAN Discriminator model for 28x28-sized images.
    """

    def __init__(self, channels, feature_maps):
        super(Discriminator28, self).__init__()
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

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)


class Generator64(nn.Module):
    """
        Generator model for 64x64-sized images.
    """

    def __init__(self, noise_dim, channels, feature_maps):
        super(Generator64, self).__init__()
        self.zd = noise_dim  # Size of z latent vector
        self.ngf = feature_maps  # Number of feature maps
        self.nc = channels  # Number of channels in the training images
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.zd, out_channels=self.ngf * 8, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=self.nc, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator64(nn.Module):
    """
        DCGAN Discriminator model for 64x64-sized images.
    """

    def __init__(self, channels, feature_maps):
        super(Discriminator64, self).__init__()
        self.ndf = feature_maps  # Size of feature maps in discriminator
        self.nc = channels  # Number of channels in the training images
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
