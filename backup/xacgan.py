import os
import torch
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from captum.attr import Saliency, DeepLift, GuidedGradCam
from lib import models, datasets, utils


def _load_models(args, device):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        generator = models.GeneratorACGAN(args.n_classes, args.noise_dim, args.img_size, args.channels).to(device)
        generator.apply(models.weights_init)

        discriminator = models.DiscriminatorACGAN(args.channels, args.img_size, args.n_classes).to(device)
        discriminator.apply(models.weights_init)

        return generator, discriminator


def _minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def _xai_method(method: str, model):
    if method == 'saliency':
        return Saliency(model)
    elif method == 'deeplift':
        return DeepLift(model)
    elif method == 'gradcam':
        return GuidedGradCam(model, model.network[9])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'fmnist'],
                        default='mnist')
    parser.add_argument('--img_size', '-s', type=int, default=32, help="size of each image dimension")
    parser.add_argument('--channels', '-c', type=int, default=1, help="number of image channels")
    parser.add_argument('--epochs', '-e', type=int, default=10, help="number of epochs of training")
    parser.add_argument('--n_classes', type=int, default=10, help="number of classes for dataset")
    parser.add_argument('--batch_size', '-b', type=int, default=64, help="size of the batches")
    parser.add_argument('--noise_dim', '-z', type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--lr', type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument('--b1', type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--b2', type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--sample_interval', type=int, default=400, help="interval between image sampling")
    parser.add_argument('--xai', '-x', type=str, choices=['saliency', 'deeplift', 'gradcam'], default='saliency')
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')
    args = parser.parse_args()

    weights_path = 'weights/xacgan/' + args.dataset + '/' + args.xai

    if not os.path.exists(weights_path):
        path = pathlib.Path(weights_path)
        path.mkdir(parents=True)

    device = utils.select_device(args.cuda_device)

    # Load dataset
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Create models
    generator, discriminator = _load_models(args, device)
    discriminatoraux = models.DiscriminatorACGANaux(args.channels, args.img_size, args.n_classes).to(device)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss().to(device)
    auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    float_tensor = torch.cuda.FloatTensor if device == torch.device('cuda') else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if device == torch.device('cuda') else torch.LongTensor

    # Lists to keep track of progress
    g_losses = []
    d_losses = []

    writer = SummaryWriter()

    ###############
    # Training Loop
    ###############
    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        running_loss_g = 0.0
        running_loss_d = 0.0

        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch + 1)) as pbar:
            for i, (imgs, labels) in enumerate(dataset):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(float_tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(float_tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

                # Configure input
                real_imgs = Variable(imgs.type(float_tensor)).to(device)
                labels = Variable(labels.type(long_tensor)).to(device)

                # -----------------
                #  Train Generator
                # -----------------

                generator_optimizer.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(float_tensor(np.random.normal(0, 1, (batch_size, args.noise_dim)))).to(device)
                gen_labels = Variable(long_tensor(np.random.randint(0, args.n_classes, batch_size))).to(device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = discriminator(gen_imgs)

                if epoch > -1:  # int(args.epochs/2):
                    # -------------------------------------
                    errG = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

                    discriminatoraux.load_state_dict(discriminator.state_dict())
                    saliency = _xai_method(args.xai, discriminatoraux)
                    # explanations = saliency.attribute(inputs=gen_imgs, target=torch.argmax(pred_label, dim=1))
                    explanations = saliency.attribute(gen_imgs)
                    explanations = _minmax_scaler(explanations)

                    errX = errG * explanations

                    shape = torch.ones_like(input=explanations, dtype=torch.float32)
                    errX.backward(gradient=shape)
                    # -------------------------------------
                else:
                    errG = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
                    errG.backward()

                generator_optimizer.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                discriminator_optimizer.zero_grad()

                # Loss for real images
                real_pred, real_aux = discriminator(real_imgs)
                errD_real = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = discriminator(gen_imgs.detach())
                errD_fake = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                errD = (errD_real + errD_fake) / 2

                # Calculate discriminator accuracy
                # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                errD.backward()
                discriminator_optimizer.step()

                # Save Losses for plotting later
                g_losses.append(errG.item())
                d_losses.append(errD.item())
                running_loss_g += errG.item()
                running_loss_d += errD.item()

                pbar.update(1)

            ###############
            # Saving the results
            ###############
            writer.add_scalar('xacgan/loss/generator', running_loss_g / len(dataset.dataset), epoch)
            writer.add_scalar('xacgan/loss/discriminator', running_loss_d / len(dataset.dataset), epoch)

            # Save images of the epoch
            n_row = 10
            z = Variable(float_tensor(np.random.normal(0, 1, (n_row ** 2, args.noise_dim)))).to(device)
            # Get labels ranging from 0 to n_classes for n rows
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = Variable(long_tensor(labels)).to(device)
            gen_imgs = generator(z, labels)
            img_grid_fake = make_grid(gen_imgs, normalize=True, nrow=n_row)
            writer.add_image("XACGAN fake images " + args.xai, img_grid_fake, global_step=epoch)

            # Save models
            torch.save(generator.state_dict(), weights_path + f'/gen_epoch_{epoch + 1:02d}.pth')
            torch.save(discriminator.state_dict(), weights_path + f'/disc_epoch_{epoch + 1:02d}.pth')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
