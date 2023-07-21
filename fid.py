import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from lib import models, datasets
from lib.utils import select_device
from tqdm import tqdm

if __name__ == '__main__':
    device = select_device()
    batch_size = 64

    discriminator = models.Discriminator64(channels=3, feature_maps=64).to(device)
    generator = models.Generator64(noise_dim=100, channels=3, feature_maps=64).to(device)

    discriminator.load_state_dict(torch.load('weights/xdcgan/celeba/disc_epoch_49.pth', map_location=device))
    generator.load_state_dict(torch.load('weights/xdcgan/celeba/gen_epoch_49.pth', map_location=device))

    dataset = datasets.make_dataset(dataset='celeba',
                                    batch_size=batch_size,
                                    img_size=64,
                                    classification=False,
                                    artificial=False,
                                    train=True)

    # Compute FID
    fid = FrechetInceptionDistance(feature=2048).to(device)

    with tqdm(total=len(dataset), desc='Computing FID:') as pbar:
        for reals, _ in dataset:
            reals = reals.to(device)
            fid.update(reals.to(torch.uint8), real=True)

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fakes = generator(noise)
            fid.update(fakes.to(torch.uint8), real=False)

            pbar.update(1)

    print(fid.compute())
