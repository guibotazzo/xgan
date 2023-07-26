import os
from gdown import download
from zipfile import ZipFile
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, Grayscale
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageFolder


def _make_mnist_dataset(batch_size: int, img_size: int, classification: bool, train: bool):
    dataset = MNIST(root='./datasets',
                    download=True,
                    train=train,
                    transform=Compose([
                        Resize(img_size),
                        ToTensor(),
                        Normalize((0.5,), (0.5,)),
                    ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if classification:
        return dataset
    else:
        return dataloader


def _make_artificial_mnist_dataset(img_size: int):
    path = '/Users/guilherme/Downloads/artificial_datasets/mnist'

    return ImageFolder(root=path,
                       transform=Compose([
                           Resize(img_size),
                           Grayscale(),
                           ToTensor(),
                           Normalize((0.5,), (0.5,)),
                       ]))


def _make_fmnist_dataset(batch_size: int, img_size: int, classification: bool, train: bool):
    # loading the dataset
    dataset = FashionMNIST(root='./datasets',
                           download=True,
                           train=train,
                           transform=Compose([
                               Resize(img_size),
                               ToTensor(),
                               Normalize((0.5,), (0.5,)),
                           ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if classification:
        return dataset
    else:
        return dataloader


def _make_cifar10_dataset(batch_size: int, img_size: int, classification: bool, train: bool):
    dataset = CIFAR10(root='./datasets',
                      download=True,
                      train=train,
                      transform=Compose([
                          Resize(img_size),
                          ToTensor(),
                          Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if classification:
        return dataset
    else:
        return dataloader


def _make_celeba_dataset(batch_size: int, img_size: int, classification: bool):
    path = 'datasets/celeba/data.zip'

    if not os.path.exists(path):
        url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'

        download(url, path, quiet=True)

        with ZipFile(path, 'r') as zipobj:
            zipobj.extractall('datasets/celeba/')

    dataset = ImageFolder(root='datasets/celeba/',
                          transform=Compose([
                              Resize(img_size),
                              CenterCrop(img_size),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if classification:
        return dataset
    else:
        return dataloader


def _make_nhl256_dataset(path, batch_size: int, img_size: int, classification: bool):
    dataset = ImageFolder(root=path,
                          transform=Compose([
                              Resize(img_size),
                              ToTensor(),
                              Normalize((0.5,), (0.5,)),
                          ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    if classification:
        return dataset
    else:
        return dataloader


def make_dataset(dataset: str, batch_size: int, img_size: int, classification: bool, artificial: bool, train: bool):
    if dataset == 'mnist':
        if artificial:
            return _make_artificial_mnist_dataset(img_size)
        else:
            return _make_mnist_dataset(batch_size, img_size, classification, train)

    elif dataset == 'fmnist':
        return _make_fmnist_dataset(batch_size, img_size, classification, train)

    elif dataset == 'cifar10':
        return _make_cifar10_dataset(batch_size, img_size, classification, train)

    elif dataset == 'celeba':
        return _make_celeba_dataset(batch_size, img_size, classification)

    elif dataset == 'nhl256':
        if artificial:
            path = '/Users/guilherme/Downloads/datasets/artificial/xwgan-gp/nhl256/'
            return _make_nhl256_dataset(path, batch_size, img_size, classification)
        else:
            path = '/Users/guilherme/Downloads/datasets/patches/NHL256/'
            return _make_nhl256_dataset(path, batch_size, img_size, classification)
