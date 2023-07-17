import torch
import numpy as np
from tqdm import tqdm
from lib import models, datasets
from sklearn.metrics import classification_report, confusion_matrix


def test(test_dataset, device):
    model = models.ConvNet().to(device)
    model.load_state_dict(torch.load('weights/fold_1'))

    actual = np.array([])
    expected = np.array([])

    with tqdm(total=len(test_dataset), desc="Processing images: ") as pbar:
        for inputs, true_labels in test_dataset:
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)

            outputs = model(inputs)

            _, predictions = torch.max(outputs, 1)

            actual = np.concatenate([actual, predictions.cpu().numpy()])
            expected = np.concatenate([expected, true_labels.cpu().numpy()])

            pbar.update(1)

    acc = (actual == expected).sum()/actual.size

    print('Confusion matrix:')
    print(confusion_matrix(expected, actual))

    print('\nClassification report:')
    print(classification_report(expected, actual, digits=4))


if __name__ == '__main__':
    ds = datasets.make_dataset(dataset='mnist', img_size=28, artificial=False, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)

    test(test_dataset=dl, device=torch.device('mps'))
