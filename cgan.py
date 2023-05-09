import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils import datasets, models
import argparse
from tqdm import tqdm


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def training():
    dataset = datasets.make_dataset(args.dataset, args.batch_size)
    generator = models.ConditionalGeneratorMNIST().to(device)
    discriminator = models.ConditionalDiscriminatorMNIST().to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    writer = SummaryWriter()

    n_critic = 5
    display_step = 50
    for epoch in range(args.epochs):
        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch + 1)) as pbar:
            for i, (images, labels) in enumerate(dataset):

                step = epoch * len(dataset) + i + 1
                real_images = Variable(images).to(device)
                labels = Variable(labels).to(device)
                generator.train()

                d_loss = 0
                for _ in range(n_critic):
                    d_loss = discriminator_train_step(len(real_images), discriminator,
                                                      generator, d_optimizer, criterion,
                                                      real_images, labels)

                g_loss = generator_train_step(args.batch_size, discriminator, generator, g_optimizer, criterion)

                writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': (d_loss / n_critic)}, step)

                if step % display_step == 0:
                    generator.eval()
                    z = Variable(torch.randn(9, 100)).to(device)
                    labels = Variable(torch.LongTensor(np.arange(9))).to(device)
                    sample_images = generator(z, labels).unsqueeze(1)
                    grid = make_grid(sample_images, nrow=3, normalize=True)
                    writer.add_image('sample_image', grid, step)

                pbar.update(1)

        # Save models
        torch.save(generator.state_dict(),
                   'results/cgan/' + args.dataset + '/weights/gen_epoch_%d.pth' % epoch)
        torch.save(discriminator.state_dict(),
                   'results/cgan/' + args.dataset + '/weights/disc_epoch_%d.pth' % epoch)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='ConditionalGAN')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist'], default='mnist')
    parser.add_argument('--cuda_device', '-c', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:1')
    args = parser.parse_args()

    # Parameters
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # For M1 Macs
        print("MPS device selected.")
    elif torch.cuda.is_available():
        device = torch.device(args.cuda_device)
        print("CUDA device selected:", args.cuda_device)
    else:
        device = torch.device('cpu')
        print("CPU device selected.")

    training()
