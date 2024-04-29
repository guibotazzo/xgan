import timm
import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from lib import utils, datasets
from sklearn.metrics import classification_report, confusion_matrix


def _load_model(args, model_name, device):
    if model_name == 'vit':
        return timm.create_model('vit_base_patch16_224', img_size=args.img_size, pretrained=False,
                                 num_classes=args.num_classes).to(device)
    elif model_name == 'pvt':
        return timm.create_model('pvt_v2_b5', img_size=args.img_size, pretrained=False,
                                 num_classes=args.num_classes).to(device)
    elif model_name == 'deit':
        return timm.create_model('deit3_base_patch16_224', img_size=args.img_size, pretrained=False,
                                 num_classes=args.num_classes).to(device)
    elif model_name == 'coatnet':
        return timm.create_model('coatnet_2_rw_224.sw_in12k', img_size=args.img_size, pretrained=False,
                                 num_classes=args.num_classes).to(device)


def ensemble(args):
    device = utils.select_device(args.cuda_device)
    sigmoid = torch.nn.Sigmoid()

    test_labels_path = f'datasets/patches/{args.dataset.upper()}64/test_set.csv'
    test_ds = datasets.make_dataset(args, test_labels_path)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    true_labels = pd.read_csv(test_labels_path).to_numpy()
    true_labels = true_labels[:, 1]
    true_labels = true_labels.astype('int')

    scores = torch.zeros(true_labels.shape)

    for fold in range(args.folds):
        for model_name in ['vit', 'pvt', 'deit']:
            if args.gan_aug:
                fold_path = f'weights/classification/{args.dataset}/{model_name}/augmentation/{args.gan}/{args.xai}/fold_{fold + 1}'
            elif args.classic_aug:
                fold_path = f'weights/classification/{args.dataset}/{model_name}/augmentation/classic/fold_{fold + 1}'
            else:
                fold_path = f'weights/classification/{args.dataset}/{model_name}/no_augmentation/fold_{fold + 1}'

            model = _load_model(args, model_name, device)
            model.load_state_dict(torch.load(fold_path, map_location=device))

            actual = np.array([])

            for inputs, _ in test_dl:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                actual = np.concatenate([actual, predictions.cpu().numpy()])

            actual = sigmoid(actual)
            scores = torch.add(scores, actual)

        preds = torch.argmax(scores, dim=1)

        print(classification_report(preds, true_labels, digits=4))


def main():
    parser = argparse.ArgumentParser(description='Classification with Ensemble Learning')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--dataset', type=str, choices=['cr', 'la', 'lg', 'ucsb'], default='cr')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--img_size', '-s', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--gan_aug', action='store_true')
    parser.add_argument('--classic_aug', action='store_true')
    parser.add_argument('--gan', type=str, default='WGAN-GP', choices=['none', 'DCGAN', 'WGAN-GP', 'RaSGAN'])
    parser.add_argument('--xai', type=str, default='none', choices=['none', 'saliency', 'deeplift', 'inputxgrad'])
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')
    args = parser.parse_args()

    ensemble(args)


if __name__ == '__main__':
    main()
