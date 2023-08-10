import os
import pathlib
from gdown import download
from zipfile import ZipFile
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, Grayscale
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageFolder, STL10
from lib.utils import print_style


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


def _make_stl10_dataset(batch_size: int, img_size: int, classification: bool, train: bool):
    dataset = STL10(root='./datasets',
                    download=True,
                    # train=train,
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
    zip_path = './datasets/CelebA/data.zip'

    if not os.path.exists('./datasets/CelebA'):
        path = pathlib.Path('./datasets/CelebA')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'

        download(url, zip_path, quiet=True)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/CelebA/')

    dataset = ImageFolder(root='./datasets/CelebA/',
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


def _make_nhl256_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = './datasets/NHL256/nhl256_original.zip'

    if not os.path.exists('./datasets/NHL256'):
        path = pathlib.Path('./datasets/NHL256')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=10iIJXAWjTWeMTmU8lRiWZ7YT5CSpbjE-'

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/NHL256/')

    dataset = ImageFolder(root='./datasets/NHL256/NHL256/',
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
    elif dataset == 'stl10':
        return _make_stl10_dataset(batch_size, img_size, classification, train)
    elif dataset == 'celeba':
        return _make_celeba_dataset(batch_size, img_size, classification)
    elif dataset == 'nhl':
        return _make_nhl256_dataset(batch_size, img_size, classification)
    else:
        print_style('ERROR: This dataset is not implemented.', color='RED', formatting="ITALIC")
