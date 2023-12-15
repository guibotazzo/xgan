import argparse
import os
import pathlib
from tqdm import tqdm
from matplotlib.pyplot import imread, imsave


def create_fake_dataset():
    pass


def make_ucsb_patches(patch_size):
    root = '/Users/guilherme/datasets/original/UCSB/'
    width = 896
    height = 768
    size_out = patch_size

    for label in ['Benign', 'Malignant']:
        if label == 'Benign':
            num_samples = 32
            class_name = 'UCSBB'
        else:
            num_samples = 26
            class_name = 'UCSBM'

        path = root + label + '/'
        dir_out = '/Users/guilherme/datasets/patches/UCSB' + str(size_out) + '/' + label + '/'
        if not os.path.exists(dir_out):
            pathh = pathlib.Path(dir_out)
            pathh.mkdir(parents=True)

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + class_name + ' (' + str(i) + ').png')

                for h in range(0, height-size_out, size_out):
                    for w in range(0, width - size_out, size_out):
                        patch = img[w:w + size_out, h:h + size_out, :]
                        if patch.shape[0] == size_out and patch.shape[1] == size_out:
                            imsave(dir_out + class_name + ' (' + str(k) + ').png', patch)
                            k = k + 1

                pbar.update(1)


def make_cr_patches(patch_size):
    root = '/Users/guilherme/datasets/original/CR/'
    width = 775
    height = 522
    size_out = patch_size

    for label in ['Benign', 'Malignant']:
        if label == 'Benign':
            num_samples = 74
        else:
            num_samples = 91

        path = root + label + '/'
        dir_out = '/Users/guilherme/datasets/patches/CR' + str(size_out) + '/' + label + '/'
        if not os.path.exists(dir_out):
            pathh = pathlib.Path(dir_out)
            pathh.mkdir(parents=True)

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + label + ' (' + str(i) + ').png')

                for h in range(0, height-size_out, size_out):
                    for w in range(0, width - size_out, size_out):
                        patch = img[w:w + size_out, h:h + size_out, :]
                        if patch.shape[0] == size_out and patch.shape[1] == size_out:
                            imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                            k = k + 1

                pbar.update(1)


def make_la_patches(patch_size):
    root = '/Users/guilherme/datasets/original/LA/'
    width = 417
    height = 312
    size_out = patch_size

    for label in ['1', '2', '3', '4']:
        if label == '1':
            num_samples = 100
        elif label == '2':
            num_samples = 115
        elif label == '3':
            num_samples = 162
        else:
            num_samples = 151

        path = root + label + '/'
        dir_out = '/Users/guilherme/datasets/patches/LA' + str(size_out) + '/' + label + '/'
        if not os.path.exists(dir_out):
            pathh = pathlib.Path(dir_out)
            pathh.mkdir(parents=True)

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + 'Class' + label + ' (' + str(i) + ').png')

                for h in range(0, height-size_out, size_out):
                    for w in range(0, width - size_out, size_out):
                        patch = img[w:w + size_out, h:h + size_out, :]
                        if patch.shape[0] == size_out and patch.shape[1] == size_out:
                            imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                            k = k + 1

                pbar.update(1)


def make_lg_patches(patch_size):
    root = '/Users/guilherme/datasets/original/LG/'
    width = 417
    height = 312
    size_out = patch_size

    for label in ['Class 1', 'Class 2']:
        if label == 'Class 1':
            num_samples = 150
        else:
            num_samples = 115

        path = root + label + '/'
        dir_out = '/Users/guilherme/datasets/patches/LG' + str(size_out) + '/' + label + '/'
        if not os.path.exists(dir_out):
            pathh = pathlib.Path(dir_out)
            pathh.mkdir(parents=True)

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + label + ' (' + str(i) + ').png')

                for h in range(0, height-size_out, size_out):
                    for w in range(0, width - size_out, size_out):
                        patch = img[w:w + size_out, h:h + size_out, :]
                        if patch.shape[0] == size_out and patch.shape[1] == size_out:
                            imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                            k = k + 1

                pbar.update(1)


def make_nhl_patches(patch_size):
    root = '/Users/guilherme/datasets/original/NHL/'
    width = 1388
    height = 1040
    size_out = patch_size

    for label in ['CLL', 'FL', 'MCL']:
        if label == 'CLL':
            num_samples = 113
        elif label == 'FL':
            num_samples = 139
        else:
            num_samples = 122

        path = root + label + '/'
        dir_out = '/Users/guilherme/datasets/patches/NHL' + str(size_out) + '/' + label + '/'
        if not os.path.exists(dir_out):
            pathh = pathlib.Path(dir_out)
            pathh.mkdir(parents=True)

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + label + ' (' + str(i) + ').png')

                for h in range(0, height-size_out, size_out):
                    for w in range(0, width - size_out, size_out):
                        patch = img[w:w + size_out, h:h + size_out, :]
                        if patch.shape[0] == size_out and patch.shape[1] == size_out:
                            imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                            k = k + 1

                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Create patches')
    parser.add_argument('--dataset', '-d', type=str, choices=['cr', 'ucsb', 'la', 'lg', 'nhl'])
    parser.add_argument('--patch_size', '-s', type=int)
    args = parser.parse_args()

    if args.dataset == 'cr':
        make_cr_patches(args.patch_size)
    elif args.dataset == 'la':
        make_la_patches(args.patch_size)
    elif args.dataset == 'lg':
        make_lg_patches(args.patch_size)
    elif args.dataset == 'ucsb':
        make_ucsb_patches(args.patch_size)
    elif args.dataset == 'nhl':
        make_nhl_patches(args.patch_size)
    else:
        print('Dataset not found.')


if __name__ == '__main__':
    main()
