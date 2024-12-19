import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_mnist(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.MNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train

def select_from_dataset(dataset, per_class_size, labels):
    indices_by_label = [[] for _ in range(10)]

    for i in range(len(dataset)):
        current_class = dataset[i][1]
        indices_by_label[current_class].append(i)
    indices_of_desired_labels = [indices_by_label[i] for i in labels]

    return Subset(dataset, [item for sublist in indices_of_desired_labels for item in sublist[:per_class_size]])

def load_fmnist(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.FashionMNIST(root=file_location, train=True, download=True, transform=transform)
    return mnist_train

def load_Emnist(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = torchvision.datasets.EMNIST(root=file_location, split='letters', train=True, download=True, transform=transform)
    return mnist_train

def load_celeba(file_location='./datasets', image_size=None):
    if not image_size is None:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(), transforms.Normalize((0.5,), (0.5,))])
    celeba_train = torchvision.datasets.CelebA(root=file_location, target_type="identity", download=False, transform=transform)
    return celeba_train

def select_from_celeba(dataset, size):
    return Subset(dataset, range(size))

class DigitsDataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, label=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(label)
        self.label = label

    def filter_by_label(self, label):
        # Use pandas to return a dataframe of only zeros
        df = pd.read_csv(self.csv_file)
        df = df.loc[df.iloc[:, -1] == label]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :-1] / 16
        image = np.array(image)
        image = image.astype(np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, self.label