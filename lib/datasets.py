import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, Grayscale
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, ImageFolder
from gdown import download
from zipfile import ZipFile


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


def _make_fmnist_dataset(batch_size: int, img_size=28):
    # loading the dataset
    dataset = FashionMNIST(root='./datasets',
                           download=True,
                           train=True,
                           transform=Compose([
                               Resize(img_size),
                               ToTensor(),
                               Normalize((0.5,), (0.5,)),
                           ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def _make_celeba_dataset(batch_size: int, img_size: int):
    path = './datasets/CelebA/data.zip'

    if not os.path.exists(path):
        url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'

        download(url, path, quiet=True)

        with ZipFile(path, 'r') as zipobj:
            zipobj.extractall('./datasets/CelebA/')

    dataset = ImageFolder(root='./datasets/CelebA/',
                          transform=Compose([
                              Resize(img_size),
                              CenterCrop(img_size),
                              ToTensor(),
                              Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def _make_nhl_dataset(batch_size: int, img_size: int):
    path = '/Users/guilherme/Downloads/artificial_datasets/mnist'

    dataset = ImageFolder(root=path,
                          transform=Compose([
                              Resize(img_size),
                              Grayscale(),
                              ToTensor(),
                              Normalize((0.5,), (0.5,)),
                          ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def make_dataset(dataset: str, batch_size: int, img_size: int, classification: bool, artificial: bool, train: bool):
    if dataset == 'mnist':
        if artificial:
            _make_artificial_mnist_dataset(img_size)
        else:
            return _make_mnist_dataset(batch_size, img_size, classification, train)
    elif dataset == 'fminist':
        return _make_fmnist_dataset(batch_size, img_size)
    elif dataset == 'celeba':
        return _make_celeba_dataset(batch_size, img_size)
    elif dataset == 'nhl':
        return _make_nhl_dataset(batch_size, img_size)
