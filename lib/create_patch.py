from tqdm import tqdm
from matplotlib.pyplot import imread, imsave
import models


def create_fake_dataset():
    pass


def make_nhl_patches():
    root = '/Users/guilherme/Downloads/datasets/original/LG/'
    width = 417
    height = 312
    size_out = 256

    for label in ['Class 1', 'Class 2']:
        if label == 'Class 1':
            num_samples = 150
        else:
            num_samples = 115

        path = root + label + '/'
        dir_out = '/Users/guilherme/Downloads/datasets/patches/LG' + str(size_out) + '/' + label + '/'

        k = 1
        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples+1):
                img = imread(path + label + ' (' + str(i) + ').png')

                for w in range(0, width-size_out, size_out):
                    for h in range(0, height-size_out, size_out):
                        patch = img[w:w+size_out, h:h+size_out, :]
                        imsave(dir_out + label + ' (' + str(k) + ').png', patch)
                        k = k + 1

                pbar.update(1)


if __name__ == '__main__':
    make_nhl_patches()
