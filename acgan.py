import argparse
import torch
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import tqdm
from lib import models, datasets


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


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, args.z_dim)))).to(device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels)).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    args = parser.parse_args()

    device = _select_device()

    # Initialize generator and discriminator
    generator = models.GeneratorACGAN(args.n_classes, args.z_dim, args.img_size, args.channels).to(device)
    discriminator = models.DiscriminatorACGAN(args.channels, args.img_size, args.n_classes).to(device)

    generator.apply(models.weights_init)
    discriminator.apply(models.weights_init)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss().to(device)
    auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)

    # Configure data loader
    # os.makedirs("data/mnist", exist_ok=True)
    dataloader = datasets.make_dataset(dataset='mnist',
                                       batch_size=args.batch_size,
                                       img_size=args.img_size,
                                       classification=False,
                                       artificial=False,
                                       train=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(args.n_epochs):
        with tqdm(total=len(dataloader), desc="Epoch {}".format(epoch + 1)) as pbar:
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor)).to(device)
                labels = Variable(labels.type(LongTensor)).to(device)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.z_dim)))).to(device)
                gen_labels = Variable(LongTensor(np.random.randint(0, args.n_classes, batch_size))).to(device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)
                g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                real_pred, real_aux = discriminator(real_imgs)
                d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                optimizer_D.step()

                batches_done = epoch * len(dataloader) + i
                if batches_done % args.sample_interval == 0:
                    sample_image(n_row=10, batches_done=batches_done)

                pbar.update(1)

        # Save models
        # torch.save(generator.state_dict(), 'results/weights/gen_epoch_%d.pth' % epoch)
        # torch.save(discriminator.state_dict(), 'results/weights/disc_epoch_%d.pth' % epoch)
