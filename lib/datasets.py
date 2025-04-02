import os
import numpy as np
import pandas as pd
import skimage.io as skio

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, Caltech101, ImageFolder

from lib.utils import print_style


def _make_mnist_dataset(batch_size: int, img_size: int, train: bool):
    dataset = MNIST(root='./data',
                    download=True,
                    train=train,
                    transform=Compose([
                        Resize(img_size),
                        ToTensor(),
                        Normalize((0.5,), (0.5,)),
                    ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _make_fmnist_dataset(batch_size: int, img_size: int, train: bool):
    # loading the dataset
    dataset = FashionMNIST(root='./data',
                           download=True,
                           train=train,
                           transform=Compose([
                               Resize(img_size),
                               ToTensor(),
                               Normalize((0.5,), (0.5,)),
                           ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def _make_cifar10_dataset(batch_size: int, img_size: int, train: bool):
    dataset = CIFAR10(root='./data',
                      download=True,
                      train=train,
                      transform=Compose([
                          Resize(img_size),
                          ToTensor(),
                          Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def _make_caltech_dataset(batch_size: int, img_size: int):
    dataset = Caltech101(root='./data',
                         download=True,
                         transform=Compose([
                             Resize((img_size, img_size)),
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CustomDataset(Dataset):
    """Custom dataset class"""

    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.samples.iloc[idx, 0])
        image = skio.imread(img_name)

        label = self.samples.iloc[idx, 1]
        label = np.array(label)

        if self.transform:
            image = self.transform(image.copy())

        label = torch.from_numpy(label)
        # sample = {'image': image, 'label': label}

        return image, label


def _make_custom_dataset(args, csv_path):
    dataset = CustomDataset(csv_path=csv_path,
                            root_dir=f'./data/{args.dataset.upper()}{args.img_size}/',
                            transform=Compose([
                                ToTensor(),
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)



def make_dataset(args, csv_path, train=True):
    if args.dataset == 'mnist':
        return _make_mnist_dataset(args.batch_size, args.img_size, train)
    elif args.dataset == 'fmnist':
        return _make_fmnist_dataset(args.batch_size, args.img_size, train)
    elif args.dataset == 'cifar10':
        return _make_cifar10_dataset(args.batch_size, args.img_size, train)
    elif args.dataset == 'caltech':
        return _make_caltech_dataset(args.batch_size, args.img_size)
    elif args.dataset == 'custom':
        return _make_custom_dataset(args, csv_path)
    else:
        print_style('LOAD DATASET ERROR: This dataset is not implemented.', color='RED', formatting='ITALIC')
