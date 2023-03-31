import os
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


def make_dataset(dataset: str, batch_size: int, image_size: int):
    if dataset == 'mnist':
        return make_mnist_dataset(batch_size)
    elif dataset == 'fminist':
        return make_fmnist_dataset(batch_size)
    else:
        return make_celebA_dataset(image_size, batch_size)
