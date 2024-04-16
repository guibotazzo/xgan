import os
import skimage.io as skio
import pandas as pd
import numpy as np
import pathlib
from gdown import download
from zipfile import ZipFile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, RandomRotation, \
    RandomHorizontalFlip, ColorJitter
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, Caltech101, ImageFolder
from lib.utils import print_style


def _make_mnist_dataset(batch_size: int, img_size: int, classification: bool, train: bool):
    dataset = MNIST(root='./datasets',
                    download=True,
                    train=train,
                    transform=Compose([
                        Resize(img_size),
                        ToTensor(),
                        # Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Normalize((0.5,), (0.5,)),
                    ]))

    # subset = Subset(dataset, range(12000))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if classification:
        return dataset
    else:
        return dataloader


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


def _make_caltech_dataset(batch_size: int, img_size: int):
    dataset = Caltech101(root='./datasets',
                         download=True,
                         transform=Compose([
                             Resize((img_size, img_size)),
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def _make_celeba_dataset(batch_size: int, image_size: int):
    path = './datasets/CelebA/data.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

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


# # Load a dataset from a .csv file
# def _make_deepweeds_dataset(args, csv_path):
#     # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     dataset = CustomDataset(csv_path=csv_path,
#                             root_dir='./datasets/DeepWeeds/images/',
#                             transform=Compose([
#                                 Resize((args.img_size, args.img_size)),
#                                 ToTensor(),
#                                 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ]))
#
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#
#     if args.classification:
#         return dataset
#     else:
#         return dataloader


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


def _make_he_dataset(args, csv_path, classic_aug):
    if classic_aug:
        dataset = CustomDataset(csv_path=csv_path,
                                root_dir=f'./datasets/patches/{args.dataset.upper()}{args.img_size}/',
                                transform=Compose([
                                    ToTensor(),
                                    RandomRotation(degrees=(-360, 360)),
                                    RandomHorizontalFlip(),
                                    ColorJitter(hue=.05, saturation=.05),
                                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    else:
        dataset = CustomDataset(csv_path=csv_path,
                                root_dir=f'./datasets/patches/{args.dataset.upper()}{args.img_size}/',
                                transform=Compose([
                                    ToTensor(),
                                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.classification:
        return dataset
    else:
        return dataloader


def load_aug_dataset(args):
    csv_path = f'datasets/artificial/{args.dataset.upper()}{args.img_size}/{args.gan}/{args.xai}/labels.csv'
    return CustomDataset(csv_path=csv_path,
                         root_dir=f'datasets/artificial/{args.dataset.upper()}{args.img_size}/{args.gan}/{args.xai}/',
                         transform=Compose([
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))


def make_dataset(args, csv_path, classic_aug=False, train=True):
    if args.dataset == 'mnist':
        return _make_mnist_dataset(args.batch_size, args.img_size, args.classification, train)
    elif args.dataset == 'fmnist':
        return _make_fmnist_dataset(args.batch_size, args.img_size, args.classification, train)
    elif args.dataset == 'cifar10':
        return _make_cifar10_dataset(args.batch_size, args.img_size, args.classification, train)
    elif args.dataset == 'celeba':
        return _make_celeba_dataset(args.batch_size, args.img_size)
    elif args.dataset == 'caltech':
        return _make_caltech_dataset(args.batch_size, args.img_size)
    elif args.dataset == 'cr' or args.dataset == 'la' or args.dataset == 'lg' or args.dataset == 'ucsb' or \
            args.dataset == 'nhl':
        return _make_he_dataset(args, csv_path, classic_aug)
    else:
        print_style('LOAD DATASET ERROR: This dataset is not implemented.', color='RED', formatting='ITALIC')
