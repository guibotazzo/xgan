import os
import pathlib
from gdown import download
from zipfile import ZipFile
from torch.utils.data import DataLoader, Subset
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


def _make_artificial_mnist_dataset(batch_size: int, img_size: int, classification: bool):
    # path = '/Users/guilherme/datasets/artificial/xacgan/mnist'
    path = '/Users/guilherme/datasets/artificial/acgan/mnist'

    dataset = ImageFolder(root=path,
                          transform=Compose([
                              Resize(img_size),
                              Grayscale(),
                              ToTensor(),
                              Normalize((0.5,), (0.5,)),
                          ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
                             Resize((img_size,img_size)),
                             ToTensor(),
                             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def _make_celeba_dataset(batch_size: int, image_size: int):
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


def _make_nhl_dataset(batch_size: int, img_size: int, classification: bool):
    # zip_path = './datasets/NHL256/nhl256_original.zip'
    zip_path = './datasets/NHL' + str(img_size) + '_original.zip'

    # if not os.path.exists('./datasets/NHL256'):
    #     path = pathlib.Path('./datasets/NHL256')
    #     path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        if img_size == 256:
            url = 'https://drive.google.com/uc?id=10iIJXAWjTWeMTmU8lRiWZ7YT5CSpbjE-'
        elif img_size == 64:
            url = 'https://drive.google.com/uc?id=1fad5RFsKIHwFaLeq4xANr9sWUD0D1_ht'
        else:
            url = ''

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root='./datasets/NHL' + str(img_size) + '/',  # Modificar apontamento para a pasta da classe desejada
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
    zip_path = './datasets/CR' + str(img_size) + '_original.zip'

    # if not os.path.exists('./datasets/'):
    #     path = pathlib.Path('./datasets/')
    #     path.mkdir(parents=True)

    # if not os.path.exists(zip_path):
    #     if img_size == 256:
    #         url = 'https://drive.google.com/uc?id=1cCahPLuY2__RJ2V_L-TVgvIGYa97hEXG&confirm=t'
    #     elif img_size == 128:
    #         url = 'https://drive.google.com/uc?id=1iN3U7CgXiS-KJWaE0Zr2uf13J3zjq5w0&confirm=t'
    #     elif img_size == 64:
    #         url = 'https://drive.google.com/uc?id=1kt4HbpsaHGlHmf7btde6hlU2aGFJB5tP&confirm=t'
    #     else:
    #         url = ''

        # download(url, zip_path, quiet=False)
        #
        # with ZipFile(zip_path, 'r') as zipobj:
        #     zipobj.extractall('./datasets/')

    dataset = ImageFolder(root='./datasets/CR' + str(img_size) + '/',  # Modificar apontamento para a pasta da classe desejada
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
    # zip_path = './datasets/UCSB256_original.zip'
    zip_path = './datasets/UCSB' + str(img_size) + '_original.zip'

    # if not os.path.exists('./datasets/'):
    #     path = pathlib.Path('./datasets/')
    #     path.mkdir(parents=True)

    # if not os.path.exists(zip_path):
    #     if img_size == 256:
    #         url = 'https://drive.google.com/uc?id=16gaFfP5GzfitpMNi-lF63grOOCxVOx3Q&confirm=t'
    #     elif img_size == 64:
    #         url = 'https://drive.google.com/uc?id=1tuoNpTYl5mRLDnk4_J27VYEJII5Z5Nm3&confirm=t'
    #     else:
    #         url = ''
    #
    #     download(url, zip_path, quiet=False)
    #
    #     with ZipFile(zip_path, 'r') as zipobj:
    #         zipobj.extractall('./datasets/')

    dataset = ImageFolder(root='./datasets/UCSB' + str(img_size) + '/',  # Modificar apontamento para a pasta da classe desejada
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
    zip_path = './datasets/LA' + str(img_size) + '_original.zip'

    # if not os.path.exists('./datasets/'):
    #     path = pathlib.Path('./datasets/')
    #     path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=1kheHHPlTg60Lprlhq9v7p2InCj1lW4Lg&confirm=t'

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root='./datasets/LA' + str(img_size) + '/',  # Modificar apontamento para a pasta da classe desejada
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
    zip_path = './datasets/LG' + str(img_size) + '_original.zip'

    # if not os.path.exists('./datasets/'):
    #     path = pathlib.Path('./datasets/')
    #     path.mkdir(parents=True)

    if not os.path.exists(zip_path):
        url = 'https://drive.google.com/uc?id=1WH12pOvHYqA64DkhkS_B9LX-L_ptFCF1&confirm=t'

        download(url, zip_path, quiet=False)

        with ZipFile(zip_path, 'r') as zipobj:
            zipobj.extractall('./datasets/')

    dataset = ImageFolder(root='./datasets/LG' + str(img_size) + '/',  # Modificar apontamento para a pasta da classe desejada
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
            return _make_artificial_mnist_dataset(batch_size, img_size, classification)
        else:
            return _make_mnist_dataset(batch_size, img_size, classification, train)
    elif dataset == 'fmnist':
        return _make_fmnist_dataset(batch_size, img_size, classification, train)
    elif dataset == 'cifar10':
        return _make_cifar10_dataset(batch_size, img_size, classification, train)
    elif dataset == 'celeba':
        return _make_celeba_dataset(batch_size, img_size)
    elif dataset == 'nhl':
        return _make_nhl_dataset(batch_size, img_size, classification)
    elif dataset == 'cr':
        return _make_cr_dataset(batch_size, img_size, classification)
    elif dataset == 'ucsb':
        return _make_ucsb_dataset(batch_size, img_size, classification)
    elif dataset == 'la':
        return _make_la_dataset(batch_size, img_size, classification)
    elif dataset == 'lg':
        return _make_lg_dataset(batch_size, img_size, classification)
    elif dataset == 'caltech':
        return _make_caltech_dataset(batch_size, img_size)
    else:
        print_style('LOAD DATASET ERROR: This dataset is not implemented.', color='RED', formatting="ITALIC")
