import os
import copy
import torch
import argparse
import pathlib
import numpy as np
import timm
from tqdm import tqdm
from lib import datasets
from numpy import Infinity, zeros
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import densenet121, resnet50, efficientnet_b2
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from lib import utils


class AugDataset(Dataset):
    def __init__(self, base_dataset, transforms):
        super(AugDataset, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transforms(x), y


class PhaseHistory:
    def __init__(self, epochs: int) -> None:
        self.accuracy = zeros(epochs)
        self.loss = zeros(epochs)


class ModelLearningSummary:
    def __init__(self, epochs: int) -> None:
        self.best_accuracy = 0.0
        self.best_loss = Infinity
        self.training = PhaseHistory(epochs)
        self.eval = PhaseHistory(epochs)


def _load_model(args, device):
    if args.model == 'densenet121':
        model = densenet121(weights='DEFAULT') if args.transfer_learning else densenet121()
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=in_features, out_features=args.num_classes)
        # model.apply(models.reset_weights)
        return model.to(device)

    elif args.model == 'resnet50':
        model = resnet50(weights='DEFAULT') if args.transfer_learning else resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=in_features, out_features=args.num_classes)
        return model.to(device)

    elif args.model == 'efficientnet_b2':
        model = efficientnet_b2(weights='DEFAULT') if args.transfer_learning else efficientnet_b2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=args.num_classes)
        return model.to(device)

    elif args.model == 'vit':
        if args.transfer_learning:
            model = timm.create_model('vit_base_patch16_224', img_size=args.img_size, pretrained=True)
        else:
            model = timm.create_model('vit_base_patch16_224', img_size=args.img_size, pretrained=False)

        model.head = torch.nn.Linear(model.head.in_features, args.num_classes)
        return model.to(device)

    elif args.model == 'pvt':
        if args.transfer_learning:
            model = timm.create_model('pvt_v2_b5', img_size=args.img_size, pretrained=True)
        else:
            model = timm.create_model('pvt_v2_b5', img_size=args.img_size, pretrained=False)

        model.head = torch.nn.Linear(model.head.in_features, args.num_classes)
        return model.to(device)

    elif args.model == 'beit':
        if args.transfer_learning:
            model = timm.create_model('beit_base_patch16_224', img_size=args.img_size, pretrained=True)
        else:
            model = timm.create_model('beit_base_patch16_224', img_size=args.img_size, pretrained=False)

        model.head = torch.nn.Linear(model.head.in_features, args.num_classes)
        return model.to(device)


def train(args):
    if args.gan_aug:
        weights_path = f'weights/classification/{args.dataset}/{args.model}/augmentation/{args.gan}/{args.xai}/'
    elif args.classic_aug:
        weights_path = f'weights/classification/{args.dataset}/{args.model}/augmentation/classic/'
    else:
        weights_path = f'weights/classification/{args.dataset}/{args.model}/no_augmentation/'

    if not os.path.exists(weights_path):
        path = pathlib.Path(weights_path)
        path.mkdir(parents=True)

    # Print parameters
    settings = PrettyTable(['Parameters', 'Values'])
    settings.align['Parameters'] = 'l'
    settings.align['Values'] = 'l'

    settings.add_row(['Classifier', args.model.upper()])
    settings.add_row(['Transfer learning', 'Yes' if args.transfer_learning else 'No'])
    settings.add_row(['Dataset', args.dataset.upper()])
    settings.add_row(['Image size', str(args.img_size) + ' x ' + str(args.img_size)])
    settings.add_row(['Number of classes', str(args.num_classes) + ' classes'])
    settings.add_row(['Number of epochs', str(args.epochs) + ' epochs'])
    settings.add_row(['Batch size', args.batch_size])
    settings.add_row(['Number of folds', str(args.num_folds) + '-folds'])
    settings.add_row(['Test set size', str(args.test_set_size * 100) + '%'])

    if args.gan_aug:
        settings.add_row(['Data augmentation', 'GAN'])
        settings.add_row(['GAN', args.gan])
        settings.add_row(['XAI', args.xai.upper()])
    elif args.classic_aug:
        settings.add_row(['Data augmentation', 'Classic'])
    else:
        settings.add_row(['Data augmentation', 'No'])

    print(settings)

    with open(weights_path + 'results.txt', 'w') as file:
        file.write('#######################################\n')
        file.write('## CLASSIFICATION WITH DEEP LEARNING ##\n')
        file.write('#######################################\n\n')
        file.write(settings.get_string())
        file.close()

    device = utils.select_device(args.cuda_device)
    history = ModelLearningSummary(args.epochs)
    best_acc = 0.0
    avg_acc_val = []
    avg_acc_test = []

    results = PrettyTable(['Fold', 'Validation', 'Test'])
    results.align['Fold'] = 'l'
    results.align['Validation'] = 'l'
    results.align['Test'] = 'l'

    path = f'./datasets/patches/{args.dataset.upper()}{args.img_size}/'
    test_ds = datasets.make_dataset(args, f'{path}test_set.csv')
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for fold in range(args.num_folds):
        print(f"--- Fold {fold + 1} ---")

        val_ds = datasets.make_dataset(args, f'{path}val_set_{fold}.csv')
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

        if args.classic_aug:
            train_ds = datasets.make_dataset(args, f'{path}train_set_{fold}.csv', classic_aug=True)
        else:
            train_ds = datasets.make_dataset(args, f'{path}train_set_{fold}.csv')

        if args.gan_aug:
            aug_ds = datasets.load_aug_dataset(args)
            new_train_ds = ConcatDataset([train_ds, aug_ds])
            train_dl = DataLoader(new_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        else:
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

        ################
        # Begin training
        ################
        dataset_sizes = {'train': len(train_dl.dataset), 'valid': len(val_dl.dataset)}

        # Reset model
        model = _load_model(args, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = CrossEntropyLoss()

        # Training loop
        print("Starting Training Loop...")
        for epoch in range(args.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    ds = train_dl
                    description = "Epoch {} ".format(epoch + 1) + "(" + phase + ")"
                else:
                    model.eval()  # Set model to evaluate mode
                    ds = val_dl
                    description = '{}'.format(' ' * len("Epoch {} ".format(epoch + 1))) + "(" + phase + ")"

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                with tqdm(total=len(ds), desc=description) as pbar:
                    for inputs, labels in ds:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        pbar.update(1)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]

                # Save data to history
                if phase == 'train':
                    history.training.accuracy[epoch] = epoch_acc
                    history.training.loss[epoch] = epoch_loss
                else:
                    history.eval.accuracy[epoch] = epoch_acc
                    history.eval.loss[epoch] = epoch_loss

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    history.best_accuracy = best_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_model_wts)

        print(f'Validation Accuracy: {best_acc:.4f}\n')
        avg_acc_val.append(best_acc)

        #############
        # Begin testing
        #############
        actual = np.array([])
        expected = np.array([])

        print("Starting Testing Loop...")
        with tqdm(total=len(test_dl), desc="Processing images: ") as pbar:
            for inputs, true_labels in test_dl:
                inputs = inputs.to(device)
                true_labels = true_labels.to(device)

                outputs = model(inputs)

                _, predictions = torch.max(outputs, 1)

                actual = np.concatenate([actual, predictions.cpu().numpy()])
                expected = np.concatenate([expected, true_labels.cpu().numpy()])

                pbar.update(1)

        test_acc = (actual == expected).sum() / actual.size
        print(f'Test Accuracy: {test_acc:.4f}\n')
        avg_acc_test.append(test_acc)

        results.add_row([f'Fold {fold + 1}', f'{best_acc * 100:.2f}%', f'{test_acc * 100:.2f}%'])

        with open(weights_path + 'results.txt', 'a') as file:
            file.write('\n\n##################################\n')
            file.write(f'## Results on FOLD {fold + 1} (Test set) ##\n')
            file.write('##################################\n\n\n')
            file.write('--- Confusion matrix ---\n')
            file.write(np.array2string(confusion_matrix(expected, actual)))
            file.write('\n\n--- Classification report ---\n')
            file.write(classification_report(expected, actual, digits=4))
            file.close()

        # Plot graphs
        # graph_file_name = 'results/cotton/graphs/acc/' + file + '_fold_' + str(fold)
        # display.plot_acc(train_acc=history.training.accuracy, val_acc=history.eval.accuracy, file=graph_file_name)
        # display.print_style('Acc graph saved as: ' + graph_file_name, color='GREEN')

        # graph_file_name = 'results/cotton/graphs/loss/' + file + '_fold_' + str(fold)
        # display.plot_loss(train_loss=history.training.loss, val_loss=history.training.loss, file=graph_file_name)
        # display.print_style('Loss graph saved as: ' + graph_file_name, color='GREEN')

        # Save model weights
        trained_model_file = weights_path + 'fold_' + str(fold + 1)
        torch.save(model.state_dict(), trained_model_file)
        # display.print_style('Models weights saved as: ' + trained_model_file, color='GREEN')

    avg_acc_val = sum(avg_acc_val) / len(avg_acc_val)
    avg_acc_test = sum(avg_acc_test) / len(avg_acc_test)

    results.add_row(['Average', f'{avg_acc_val * 100:.2f}%', f'{avg_acc_test * 100:.2f}%'])

    with open(weights_path + 'results.txt', 'a') as file:
        file.write('\n\n######################################################\n')
        file.write('## SUMMARY OF THE CLASSIFICATION RESULTS (ACCURACY) ##\n')
        file.write('######################################################\n\n')
        file.write(results.get_string())
        file.close()

    print(results.get_string(title='Classification results (Accuracy)') + '\n')


def main():
    parser = argparse.ArgumentParser(description='Classification with Deep Learning')
    # Model settings
    parser.add_argument('--model', '-m', type=str, default='densenet121',
                        choices=['densenet121', 'resnet50', 'efficientnet_b2', 'vit', 'pvt', 'beit'])
    parser.add_argument('--transfer_learning', type=bool, default=True)

    # Dataset settings
    parser.add_argument('--dataset', '-d', type=str, choices=['cr', 'la', 'lg', 'ucsb', 'nhl'], default='ucsb')
    parser.add_argument('--img_size', '-s', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)

    # Training settings
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--num_folds', '-k', type=int, default=5)
    parser.add_argument('--test_set_size', type=float, default=.2)

    # Data augmentation settings
    parser.add_argument('--gan_aug', action='store_true')
    parser.add_argument('--classic_aug', action='store_true')
    parser.add_argument('--gan', type=str, default='WGAN-GP',
                        choices=['DCGAN', 'LSGAN', 'WGAN-GP', 'HingeGAN', 'RSGAN', 'RaSGAN', 'RaLSGAN', 'RaHingeGAN', 'StyleGAN2'])
    parser.add_argument('--xai', type=str, choices=['none', 'saliency', 'deeplift', 'inputxgrad'], default='none')

    # Other settings
    parser.add_argument('--artificial', '-a', type=bool, default=False)
    parser.add_argument('--classification', type=bool, default=True)
    parser.add_argument('--cuda_device', type=str, choices=['cuda:0', 'cuda:1'], default='cuda:0')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
