import os
import pathlib
from tqdm import tqdm
from matplotlib.pyplot import imread, imsave
import models


def create_fake_dataset():
    pass


def make_nhl_patches():
    dataset = 'UCSB'
    root = '/Users/guilherme/datasets/original/UCSB/'
    width = 896
    height = 768
    size_out = 256

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
                        patch = img[w:w+size_out, h:h+size_out, :]
                        imsave(dir_out + class_name + ' (' + str(k) + ').png', patch)
                        k = k + 1

                pbar.update(1)


if __name__ == '__main__':
    make_nhl_patches()
