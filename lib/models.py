import torch
import torch.nn as nn

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

#####################################
# Models for generation (DCGAN-based)
#####################################


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

    def forward(self, noise):
        output = self.network(noise)
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

    def forward(self, img):
        output = self.network(img)
        return output.view(-1, 1).squeeze(1)


class Generator32(nn.Module):
    def __init__(self, noise_dim, channels, feature_maps):
        super(Generator32, self).__init__()
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


class Discriminator32(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator32, self).__init__()
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

    def forward(self, noise):
        return self.main(noise)


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

    def forward(self, img):
        return self.main(img)


class Generator96(nn.Module):
    def __init__(self, noise_dim, channels, feature_maps):
        super(Generator96, self).__init__()
        self.nz = noise_dim  # Size of z latent vector
        self.ngf = feature_maps  # Size of feature maps in generator
        self.nc = channels  # Number of channels in the training images
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf*16, kernel_size=3, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*16, out_channels=self.ngf*8, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*8, out_channels=self.ngf*4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*4, out_channels=self.ngf*2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=self.nc, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator96(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator96, self).__init__()
        self.ndf = feature_maps  # Size of feature maps in discriminator
        self.nc = channels  # Number of channels of the training images
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*8, out_channels=self.ndf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.ndf*16, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.network(img)


class Generator256(nn.Module):
    def __init__(self, noise_dim, channels, feature_maps):
        super(Generator256, self).__init__()

        self.nz = noise_dim  # Size of z latent vector
        self.ngf = feature_maps  # Size of feature maps in generator
        self.nc = channels  # Number of channels in the training images
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf*16, kernel_size=10, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*16, out_channels=self.ngf*8, kernel_size=10, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*8, out_channels=self.ngf*4, kernel_size=10, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*4, out_channels=self.ngf*2, kernel_size=10, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=self.ngf*2, out_channels=self.nc, kernel_size=14, stride=2, padding=0,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator256(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator256, self).__init__()
        self.ndf = feature_maps  # Size of feature maps in discriminator
        self.nc = channels  # Number of channels of the training images

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=10, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf*2, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.ndf*2, out_channels=self.ndf*4, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.ndf*4, out_channels=self.ndf*8, kernel_size=10, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self.ndf*8, out_channels=1, kernel_size=10, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.network(img)

#####################################
# Models for generation (ACGAN-based)
#####################################


class GeneratorACGAN(nn.Module):
    def __init__(self, n_classes, z_dim, img_size, channels):
        super(GeneratorACGAN, self).__init__()
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.img_size = img_size
        self.channels = channels
        self.label_emb = nn.Embedding(self.n_classes, self.z_dim)
        self.init_size = self.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.z_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DiscriminatorACGAN(nn.Module):
    def __init__(self, channels, img_size, n_classes):
        super(DiscriminatorACGAN, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.n_classes = n_classes

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


class DiscriminatorACGANaux(nn.Module):
    def __init__(self, channels, img_size, n_classes):
        super(DiscriminatorACGANaux, self).__init__()
        self.channels = channels
        self.img_size = img_size
        self.n_classes = n_classes

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        # label = self.aux_layer(out)

        return validity
