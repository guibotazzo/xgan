import os
import pathlib
from gdown import download
from zipfile import ZipFile
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, Grayscale, Lambda
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


def _make_nhl_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = f'./datasets/NHL{img_size}_original.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        if img_size == 64:
            url = 'https://drive.google.com/uc?id=1fad5RFsKIHwFaLeq4xANr9sWUD0D1_ht'
        else:
            url = ''

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root=f'./datasets/NHL{img_size}/',
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


def _make_cr_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = f'./datasets/CR{img_size}_original.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        if img_size == 64:
            url = 'https://drive.google.com/uc?id=1kt4HbpsaHGlHmf7btde6hlU2aGFJB5tP&confirm=t'
        else:
            url = ''

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root=f'./datasets/CR{img_size}/',
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


def _make_ucsb_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = f'./datasets/UCSB{img_size}_original.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        if img_size == 64:
            url = 'https://drive.google.com/uc?id=1tuoNpTYl5mRLDnk4_J27VYEJII5Z5Nm3&confirm=t'
        else:
            url = ''

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root=f'./datasets/UCSB{img_size}/',
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


def _make_la_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = f'./datasets/LA{img_size}_original.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=1kheHHPlTg60Lprlhq9v7p2InCj1lW4Lg&confirm=t'

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root=f'./datasets/LA{img_size}/',
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


def _make_lg_dataset(batch_size: int, img_size: int, classification: bool):
    zip_path = f'./datasets/LG{img_size}_original.zip'

    if not os.path.exists('./datasets/'):
        path = pathlib.Path('./datasets/')
        path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=1WH12pOvHYqA64DkhkS_B9LX-L_ptFCF1&confirm=t'

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root=f'./datasets/LG{img_size}/',
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


def load_aug_dataset(args):
    return ImageFolder(root=f'datasets/artificial/{args.dataset.upper()}{args.img_size}/{args.xai}/',
                       transform=Compose([
                           Resize(args.img_size),
                           ToTensor(),
                           Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))


def make_dataset(args, train: bool):
    if args.dataset == 'mnist':
        return _make_mnist_dataset(args.batch_size, args.img_size, args.classification, args.train)
    elif args.dataset == 'fmnist':
        return _make_fmnist_dataset(args.batch_size, args.img_size, args.classification, train)
    elif args.dataset == 'cifar10':
        return _make_cifar10_dataset(args.batch_size, args.img_size, args.classification, train)
    elif args.dataset == 'celeba':
        return _make_celeba_dataset(args.batch_size, args.img_size)
    elif args.dataset == 'nhl':
        return _make_nhl_dataset(args.batch_size, args.img_size, args.classification)
    elif args.dataset == 'cr':
        return _make_cr_dataset(args.batch_size, args.img_size, args.classification)
    elif args.dataset == 'ucsb':
        return _make_ucsb_dataset(args.batch_size, args.img_size, args.classification)
    elif args.dataset == 'la':
        return _make_la_dataset(args.batch_size, args.img_size, args.classification)
    elif args.dataset == 'lg':
        return _make_lg_dataset(args.batch_size, args.img_size, args.classification)
    elif args.dataset == 'caltech':
        return _make_caltech_dataset(args.batch_size, args.img_size)
    else:
        print_style('LOAD DATASET ERROR: This dataset is not implemented.', color='RED', formatting='ITALIC')
