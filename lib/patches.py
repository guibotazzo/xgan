from tqdm import tqdm
from matplotlib.pyplot import imread, imsave
import models


def create_fake_dataset():
    pass


def make_nhl_patches():
    root = '/Users/guilherme/Downloads/Datasets/Datasets/NHL/'
    width = 1040
    height = 1388
    size_out = 512

    for label in ['CLL', 'FL', 'MCL']:
        if label == 'CLL':
            num_samples = 113
        elif label == 'FL':
            num_samples = 139
        else:
            num_samples = 122

        path = root + label + '/'
        dir_out = '/Users/guilherme/Downloads/Datasets/Patches/NHL' + str(size_out) + '/' + label + '/'

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
