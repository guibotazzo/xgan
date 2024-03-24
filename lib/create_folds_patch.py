import os
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from imageio import imwrite
from matplotlib.pyplot import imread
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def dataset_info(dataset_name: str) -> [dict, dict]:
    if dataset_name == 'cr':
        classes = {'Benign': 74, 'Malignant': 91}
        size = {'width': 775, 'height': 522}
    elif dataset_name == 'ucsb':
        classes = {'Benign': 32, 'Malignant': 26}
        size = {'width': 896, 'height': 768}
    elif dataset_name == 'la':
        classes = {'1': 100, '2': 115, '3': 162, '4': 151}
        size = {'width': 417, 'height': 312}
    elif dataset_name == 'lg':
        classes = {'1': 150, '2': 115}
        size = {'width': 417, 'height': 312}
    else:  # NHL
        classes = {'CLL': 113, 'FL': 139, 'MCL': 122}
        size = {'width': 1388, 'height': 1040}

    return classes, size


def create_csv(args):
    print('Creating the original dataset\'s csv file...')

    info, _ = dataset_info(args.dataset)  # Dataset info: ['Class name': Number of images]
    data = []

    int_label = 0
    for label, num_samples in info.items():
        path = f'{label}/'

        if args.dataset == 'la':
            img_name = f'Class{label}'
        elif args.dataset == 'lg':
            img_name = f'Class {label} -'
        elif args.dataset == 'ucsb':
            img_name = 'UCSBB' if label == 'Benign' else 'UCSBM'
        else:
            img_name = label

        for i in range(1, num_samples + 1):
            data.append([f'{path}{img_name} ({i}).png', int_label])

        int_label = int_label + 1

    patch_path = f'./datasets/original/{args.dataset.upper()}/'

    if not os.path.exists(patch_path):
        path = pathlib.Path(patch_path)
        path.mkdir(parents=True)

    dataframe = pd.DataFrame(data, columns=['File', 'Label'])
    dataframe.to_csv(path_or_buf=f'{patch_path}labels.csv', sep=',', index=False)

    print(f'Original dataset\'s csv file saved at: {patch_path}.')


def create_csv_patch(args):
    print('Creating the patch dataset\'s csv file...')

    phases = ['train_set_', 'val_set_']
    root = f'./datasets/original/{args.dataset.upper()}/'
    dir_out = f'./datasets/patches/{args.dataset.upper()}{args.patch_size}/'

    _, size = dataset_info(args.dataset)
    num_patches = int(((size['width'] - args.patch_size + 1) * (size['height'] - args.patch_size + 1)) / args.patch_size**2)

    # Generate for train and validation sets
    for phase in phases:
        for fold in range(args.num_folds):
            df = pd.read_csv(f'{root}{phase}{fold}.csv')

            data = []
            for idx in df.index:
                file_name = df['File'][idx]
                file_name = file_name[0:-4]  # Remove '.png'

                for jdx in range(1, num_patches + 2):
                    data.append([f'{file_name}({jdx}).png', df['Label'][idx]])

            dataframe = pd.DataFrame(data, columns=['File', 'Label'])
            dataframe.to_csv(path_or_buf=f'{dir_out}{phase}{fold}.csv', sep=',', index=False)

    # Generate for test set
    df = pd.read_csv(f'{root}test_set.csv')

    data = []
    for idx in df.index:
        file_name = df['File'][idx]
        file_name = file_name[0:-4]  # Remove '.png'

        for jdx in range(1, num_patches + 2):
            data.append([f'{file_name}({jdx}).png', df['Label'][idx]])

    dataframe = pd.DataFrame(data, columns=['File', 'Label'])
    dataframe.to_csv(path_or_buf=f'{dir_out}test_set.csv', sep=',', index=False)


def create_folds(args):
    print(f'Creating csv files for {args.num_folds} folds...')

    dataframe = pd.read_csv(f'./datasets/original/{args.dataset.upper()}/labels.csv')

    #############
    # Split dataset into train and test sets
    #############
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_set_size, random_state=42)

    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    for train_idx, test_idx in sss.split(dataframe, dataframe['Label'].to_list()):
        train_set = dataframe.loc[train_idx].reset_index().drop('index', axis=1)
        test_set = dataframe.loc[test_idx]

    test_set.to_csv(path_or_buf=f'./datasets/original/{args.dataset.upper()}/test_set.csv', sep=',', index=False)

    #############
    # Split the training dataset into k folds
    #############
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    for fold, (k_train_idx, k_val_idx) in enumerate(skf.split(train_set, train_set['Label'].to_list())):
        k_train_set = train_set.loc[k_train_idx].reset_index().drop('index', axis=1)
        k_val_set = train_set.loc[k_val_idx].reset_index().drop('index', axis=1)

        k_train_set.to_csv(path_or_buf=f'./datasets/original/{args.dataset.upper()}/train_set_{fold}.csv', sep=',', index=False)
        k_val_set.to_csv(path_or_buf=f'./datasets/original/{args.dataset.upper()}/val_set_{fold}.csv', sep=',', index=False)

    print(f'Fold\'s csv files saved at ./datasets/original/{args.dataset.upper()}.')


def make_patches(args):
    root = f'./datasets/original/{args.dataset.upper()}/'
    classes, size = dataset_info(args.dataset)

    for label, num_samples in classes.items():
        if args.dataset == 'la':
            img_name = f'Class{label}'
        elif args.dataset == 'lg':
            img_name = f'Class {label} -'
        elif args.dataset == 'ucsb':
            img_name = 'UCSBB' if label == 'Benign' else 'UCSBM'
        else:
            img_name = label

        dir_out = f'./datasets/patches/{args.dataset.upper()}{args.patch_size}/{label}/'
        if not os.path.exists(dir_out):
            path = pathlib.Path(dir_out)
            path.mkdir(parents=True)

        with tqdm(total=num_samples, desc='Generating patches for the ' + label + ' class') as pbar:
            for i in range(1, num_samples + 1):
                img = imread(f'{root}/{label}/{img_name} ({i}).png')

                k = 1
                for h in range(0, size['height'] - args.patch_size + 1, args.patch_size):
                    for w in range(0, size['width'] - args.patch_size + 1, args.patch_size):
                        patch = img[w:w + args.patch_size, h:h + args.patch_size, :]
                        if patch.shape[0] == args.patch_size and patch.shape[1] == args.patch_size:
                            patch = patch * 255
                            patch = patch.astype(np.uint8)
                            imwrite(f'{dir_out}{img_name} ({i})({k}).png', patch)
                            k = k + 1

                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Create folds for the k-fold cross validation patches classification')
    parser.add_argument('--dataset', type=str, choices=['cr', 'ucsb', 'la', 'lg', 'nhl'], default='cr')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--test_set_size', type=float, default=.2)
    args = parser.parse_args()

    # Print parameters
    settings = PrettyTable(['Parameters', 'Values'])
    settings.align['Parameters'] = 'l'
    settings.align['Values'] = 'l'
    settings.add_row(['Dataset', args.dataset.upper()])
    settings.add_row(['Patch size', f'{args.patch_size} x {args.patch_size}'])
    settings.add_row(['Number of folds', f'{args.num_folds}-folds'])
    settings.add_row(['Test set size', f'{args.test_set_size*100}%'])
    print(settings)

    create_csv(args)
    create_folds(args)
    make_patches(args)
    create_csv_patch(args)


if __name__ == '__main__':
    main()
