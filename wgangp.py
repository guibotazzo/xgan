import os
import argparse
import pathlib
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from lib import utils, datasets, models
from tqdm import tqdm


def _load_models(args, device):
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        generator = models.Generator256(args.noise_dim, args.channels, args.feature_maps).to(device)
        generator.apply(models.weights_init)

        discriminator = models.Critic256(args.channels, args.feature_maps).to(device)
        discriminator.apply(models.weights_init)

        return generator, discriminator


def _gradient_penalty(critic, real, fake, device="cpu"):
    b, c, h, w = real.shape

    alpha = torch.rand((b, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)


def main():
    parser = argparse.ArgumentParser(description='WGAN-GP')
    parser.add_argument('--dataset', '-d', type=str, choices=['nhl'], default='nhl')
    parser.add_argument('--img_size', '-s', type=int, default=256, help="size of each image dimension")
    parser.add_argument('--channels', '-c', type=int, default=1, help="number of image channels")
    parser.add_argument('--epochs', '-e', type=int, default=50, help="number of epochs of training")
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="size of the batches")
    parser.add_argument('--feature_maps', '-f', type=int, default=64)
    parser.add_argument('--z_dim', '-z', type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--lr', type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument('--b1', type=float, default=0.0, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--b2', type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--ci', type=int, default=5, help="critic iterations")
    parser.add_argument('--lambda_gp', type=int, default=10)
    args = parser.parse_args()

    weights_path = 'weights/xgangp/' + args.dataset

    if not os.path.exists(weights_path):
        path = pathlib.Path(weights_path)
        path.mkdir(parents=True)

    device = utils.select_device()

    # Load dataset
    dataset = datasets.make_dataset(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    img_size=args.img_size,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Create models
    generator, critic = _load_models(args, device)

    opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, args.z_dim, 1, 1).to(device)
    writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
    writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
    step = 0

    generator.train()
    critic.train()

    for epoch in range(args.epochs):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(tqdm(dataset)):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(args.ci):
                noise = torch.randn(cur_batch_size, args.z_dim, 1, 1).to(device)
                fake = generator(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = _gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake)) + args.lamba_gp * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(dataset)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = make_grid(real[:32], normalize=True)
                    img_grid_fake = make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


if __name__ == '__main__':
    main()
