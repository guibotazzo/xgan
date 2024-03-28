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
            data.append([f'{path}{img_name} ({i:05d}).png', int_label])

        int_label = int_label + 1

    patch_path = f'./datasets/original/{args.dataset.upper()}/'

    if not os.path.exists(patch_path):
        path = pathlib.Path(patch_path)
        path.mkdir(parents=True)

    dataframe = pd.DataFrame(data, columns=['File', 'Label'])
    dataframe.to_csv(path_or_buf=f'{patch_path}labels.csv', sep=',', index=False)

    print(f'Original dataset\'s csv file saved at: {patch_path}.')


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
                            imwrite(f'{dir_out}{img_name} ({i:05d})({k:05d}).png', patch)
                            k = k + 1

                pbar.update(1)


def create_csv_patch(args):
    print('Creating the patch dataset\'s csv file...')

    dir_out = f'./datasets/patches/{args.dataset.upper()}{args.patch_size}/'

    # Create labels.csv
    labels, _ = dataset_info(args.dataset)

    k = 0
    data = []
    for label in labels:
        file_list = os.listdir(f'{dir_out}{label}')
        files = [f'{label}/{file}' for file in file_list]

        labels = [k] * len(files)
        data = data + [[file, label] for file, label in zip(files, labels)]
        k += 1

    df_labels_csv = pd.DataFrame(data, columns=['File', 'Label'])
    df_labels_csv = df_labels_csv.sort_values('File')
    df_labels_csv.to_csv(path_or_buf=f'{dir_out}labels.csv', sep=',', index=False)

    # Create folds' csv files
    phases = ['train_set_', 'val_set_']
    root = f'./datasets/original/{args.dataset.upper()}/'

    _, size = dataset_info(args.dataset)

    # Generate for train and validation sets
    for phase in phases:
        for fold in range(args.num_folds):
            df_fold = pd.read_csv(f'{root}{phase}{fold}.csv')

            temp = []
            for idx in df_fold.index:
                file_name = df_fold['File'][idx]
                file_name = file_name[0:-4]  # Remove '.png'

                # Get all elements with the file_name name
                elements = df_labels_csv['File'].str.contains(file_name, regex=False)
                temp.append(df_labels_csv.loc[elements])

            dataframe = pd.concat(temp)
            dataframe.to_csv(path_or_buf=f'{dir_out}{phase}{fold}.csv', sep=',', index=False)

    # Generate for test set
    df_test = pd.read_csv(f'{root}test_set.csv')

    temp = []
    for idx in df_test.index:
        file_name = df_test['File'][idx]
        file_name = file_name[0:-4]  # Remove '.png'

        elements = df_labels_csv['File'].str.contains(file_name, regex=False)
        temp.append(df_labels_csv.loc[elements])

    dataframe = pd.concat(temp)
    dataframe.to_csv(path_or_buf=f'{dir_out}test_set.csv', sep=',', index=False)


def create_csv_labels(args):
    path = f'./datasets/patches/{args.dataset}{args.patch_size}/'
    main_df = pd.read_csv(f'{path}labels.csv')

    labels, _ = dataset_info(args.dataset)

    k = 0
    for label in labels:
        elements = main_df['Label'].astype('str').str.contains(str(k))

        df = pd.DataFrame(main_df.loc[elements], columns=['File', 'Label'])
        df.to_csv(path_or_buf=f'{path}labels_{label}.csv', sep=',', index=False)

        k += 1


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
    create_csv_labels(args)


if __name__ == '__main__':
    main()
