from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def make_mnist_dataset():
    # loading the dataset
    dataset = MNIST(root='./datasets', download=True,
                    transform=transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]))

    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=2)

    return dataloader
