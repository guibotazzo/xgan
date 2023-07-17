import copy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from tqdm import tqdm
from numpy import Infinity, zeros
from sklearn.model_selection import StratifiedKFold
from lib import models, datasets  # our custom libs


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


def train_model(num_epochs, batch_size, k, device):
    history = ModelLearningSummary(num_epochs)
    best_acc = 0.0
    fold = 1

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    dataset = datasets.make_dataset(dataset='mnist', img_size=28, artificial=True, train=True)

    for train_idx, val_idx in skf.split(dataset, dataset.targets):
        print(f"--- Fold: {fold} ---")

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

        train_ds = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_ds = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        dataset_sizes = {'train': 48000, 'val': 12000}

        # Reset model
        model = models.ConvNet().to(device)
        model.apply(models.reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    ds = train_ds
                    description = "Epoch {} ".format(epoch + 1) + "(" + phase + ")"
                else:
                    model.eval()  # Set model to evaluate mode
                    ds = val_ds
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
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    history.best_accuracy = best_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_model_wts)

        print(f'Best val Acc: {best_acc:4f}\n')

        # Plot graphs
        # graph_file_name = 'results/cotton/graphs/acc/' + file + '_fold_' + str(fold)
        # display.plot_acc(train_acc=history.training.accuracy, val_acc=history.eval.accuracy, file=graph_file_name)
        # display.print_style('Acc graph saved as: ' + graph_file_name, color='GREEN')

        # graph_file_name = 'results/cotton/graphs/loss/' + file + '_fold_' + str(fold)
        # display.plot_loss(train_loss=history.training.loss, val_loss=history.training.loss, file=graph_file_name)
        # display.print_style('Loss graph saved as: ' + graph_file_name, color='GREEN')

        # Save model weights
        trained_model_file = 'weights/fold_' + str(fold)
        torch.save(model.state_dict(), trained_model_file)
        # display.print_style('Models weights saved as: ' + trained_model_file, color='GREEN')

        fold += 1


if __name__ == '__main__':
    train_model(num_epochs=10, batch_size=32, k=5, device=torch.device('cpu'))
