import os
from tqdm import tqdm
from matplotlib.pyplot import imread, imsave
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, ImageFolder
from gdown import download
from zipfile import ZipFile


def make_mnist_dataset(batch_size: int):
    # loading the dataset
    dataset = MNIST(root='./datasets',
                    download=True,
                    transform=Compose([
                        Resize(28),
                        ToTensor(),
                        Normalize((0.5,), (0.5,)),
                    ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader


def make_fmnist_dataset(batch_size: int):
    # loading the dataset
    dataset = FashionMNIST(root='./datasets',
                           download=True,
                           transform=Compose([
                               Resize(28),
                               ToTensor(),
                               Normalize((0.5,), (0.5,)),
                           ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader


def make_celebA_dataset(image_size: int, batch_size: int):
    path = './datasets/CelebA/data.zip'

    if not os.path.exists(path):
        url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'

        download(url, path, quiet=True)

        with ZipFile(path, 'r') as zipobj:
            zipobj.extractall('./datasets/CelebA/')

    dataset = ImageFolder(root='./datasets/CelebA/',
                          transform=Compose([
                              Resize(image_size),
                              CenterCrop(image_size),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2
                            )

    return dataloader


def make_nhl_dataset(batch_size: int):
    path = '/Users/guilherme/Downloads/Datasets/Patches/NHL/one_class'

    dataset = ImageFolder(root=path,
                          transform=Compose([
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2
                            )

    return dataloader


def make_nhl_patches():
    root = '/Users/guilherme/Downloads/Datasets/Datasets/NHL/'
    width = 1040
    height = 1388
    size_out = 64

    for label in ['CLL', 'FL', 'MCL']:
        if label == 'CLL':
            num_samples = 113
        elif label == 'FL':
            num_samples = 139
        else:
            num_samples = 122

        path = root + label + '/'
        dir_out = '/Users/guilherme/Downloads/Datasets/Patches/NHL/' + label + '/'

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + label + ' (' + str(i) + ').png')

                for w in range(0, width-size_out, size_out):
                    for h in range(0, height-size_out, size_out):
                        patch = img[w:w+size_out, h:h+size_out, :]
                        imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                        k = k + 1

                pbar.update(1)


def make_dataset(dataset: str, batch_size: int):
    if dataset == 'mnist':
        return make_mnist_dataset(batch_size)
    elif dataset == 'fminist':
        return make_fmnist_dataset(batch_size)
    elif dataset == 'celeba':
        return make_celebA_dataset(64, batch_size)
    elif dataset == 'nhl':
        return make_nhl_dataset(batch_size)
