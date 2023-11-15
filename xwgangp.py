import os
import argparse
import pathlib
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from lib import utils, datasets, models
from tqdm import tqdm
from captum.attr import Saliency, DeepLift, InputXGradient


def _xai_method(method: str, model):
    if method == 'saliency':
        return Saliency(model)
    elif method == 'deeplift':
        return DeepLift(model)
    elif method == 'gradcam':
        return InputXGradient(model)
    else:
        utils.print_style('ERROR: This XAI method is not implemented.', color='RED', formatting="ITALIC")


def _minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


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
    parser = argparse.ArgumentParser(description='XWGAN-GP')
    parser.add_argument('--gan', '-g', type=str, choices=['xwgangp'], default='xwgangp')
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'cifar10', 'nhl', 'cr', 'ucsb'], default='nhl')
    parser.add_argument('--img_size', '-s', type=int, default=256, help="size of each image dimension")
    parser.add_argument('--channels', '-c', type=int, default=3, help="number of image channels")
    parser.add_argument('--noise_dim', '-z', type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--epochs', '-e', type=int, default=50, help="number of epochs of training")
    parser.add_argument('--batch_size', '-b', type=int, default=64, help="size of the batches")
    parser.add_argument('--feature_maps', '-f', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument('--b1', type=float, default=0.0, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--b2', type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--ci', type=int, default=5, help="critic iterations")
    parser.add_argument('--lambda_gp', type=int, default=10)
    parser.add_argument('--xai', '-x', type=str, choices=['saliency', 'deeplift', 'gradcam'], default='saliency')
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')
    args = parser.parse_args()

    weights_path = 'weights/xwgangp/' + args.dataset + '/' + args.xai

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
    generator, critic = models.load_models(args, device)

    opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # for tensorboard plotting
    fixed_noise = torch.randn(32, args.noise_dim, 1, 1).to(device)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    writer = SummaryWriter()

    generator.train()
    critic.train()

    for epoch in range(args.epochs):
        running_loss_g = 0.0
        running_loss_d = 0.0

        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch + 1)) as pbar:
            for data in dataset:
                real = data[0].to(device)
                cur_batch_size = real.shape[0]

                # Train Critic: max E[critic(real)] - E[critic(fake)]
                # equivalent to minimizing the negative of that
                for _ in range(args.ci):
                    noise = torch.randn(cur_batch_size, args.noise_dim, 1, 1).to(device)
                    fake = generator(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake).reshape(-1)
                    gp = _gradient_penalty(critic, real, fake, device=device)
                    loss_critic = (
                            -(torch.mean(critic_real) - torch.mean(critic_fake)) + args.lambda_gp * gp
                    )
                    critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    opt_critic.step()

                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                gen_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)

                # -------------------------------------
                saliency = _xai_method(args.xai, critic)
                explanations = saliency.attribute(fake)
                explanations = _minmax_scaler(explanations)

                errX = loss_gen * explanations

                shape = torch.ones_like(input=explanations, dtype=torch.float32)
                # -------------------------------------

                generator.zero_grad()
                # loss_gen.backward()
                errX.backward(gradient=shape)
                opt_gen.step()

                # Save Losses for plotting later
                G_losses.append(loss_gen.item())
                D_losses.append(loss_critic.item())
                running_loss_g += loss_gen.item()
                running_loss_d += loss_critic.item()

                pbar.update(1)

            ###############
            # Saving the results
            ###############
            writer.add_scalar('wgan' + args.xai + '/loss/generator', running_loss_g / len(dataset.dataset), epoch)
            writer.add_scalar('wgan' + args.xai + '/loss/discriminator', running_loss_d / len(dataset.dataset), epoch)

            with torch.no_grad():
                fake = generator(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer.add_image("XWGAN-GP + " + args.xai, img_grid_fake, global_step=epoch)

            # Save models
            torch.save(generator.state_dict(), weights_path + f'/gen_epoch_{epoch + 1:02d}.pth')
            torch.save(critic.state_dict(), weights_path + f'/disc_epoch_{epoch + 1:02d}.pth')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
