import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils import models, datasets


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Set manual seed to a constant get a consistent output
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Parameters
    num_epochs = args.epochs

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # For M1 Macs
        print("MPS device selected.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device selected.")
    else:
        device = torch.device('cpu')
        print("CPU device selected.")

    # Load dataset
    dataset = datasets.make_mnist_dataset()

    # Create models
    generator = models.GeneratorMNIST().to(device)
    generator.apply(models.weights_init)

    discriminator = models.DiscriminatorMNIST().to(device)
    discriminator.apply(models.weights_init)

    ###############
    # Training Loop
    ###############
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    cross_entropy = nn.BCELoss()  # Binary cross entropy function

    real_label = 1.
    fake_label = 0.

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        with tqdm(total=len(dataset), desc="Epoch {}".format(epoch+1)) as pbar:
            for data in dataset:
                ############################
                # (1) Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # Train with real samples
                discriminator.zero_grad()

                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

                output = discriminator(real_cpu)
                errD_real = cross_entropy(output, label)
                errD_real.backward()

                # Train with fake samples
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake = generator(noise)
                label.fill_(fake_label)

                output = discriminator(fake.detach())
                errD_fake = cross_entropy(output, label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                discriminator_optimizer.step()

                ############################
                # (2) Update generator: maximize log(D(G(z)))
                ###########################
                generator.zero_grad()
                label.fill_(real_label)

                output = discriminator(fake)
                errG = cross_entropy(output, label)
                errG.backward()
                generator_optimizer.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # pbar.set_postfix(g_loss=errG.item(), d_loss=errD.item())
                pbar.update(1)

            ###############
            # Saving the results
            ###############
            
            # Save images of the epoch
            fake = generator(fixed_noise)
            vutils.save_image(fake.detach(), 'results/fake_images/fake_samples_epoch_%03d.png' % epoch, normalize=True)

            # Save models
            torch.save(generator.state_dict(), 'results/weights/gen_epoch_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'results/weights/disc_epoch_%d.pth' % epoch)

    # Plot the error
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
