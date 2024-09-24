from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset, random_split
import torch
from comData import CombinedDataset


class Data:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])]
        )
        self.dataLoad()
        self.testPrep()

    def dataLoad(self) -> None:
        self.train = datasets.MNIST(
            root="./data", train=True, transform=self.transform, download=True
        )

        mnist_test = datasets.MNIST(
            root="./data", train=False, transform=self.transform, download=True
        )
        self.val, self.mnist_test = random_split(
            mnist_test,
            [int(len(mnist_test) * 0.5), len(mnist_test) - int(len(mnist_test) * 0.5)],
        )
        self.fMnist_test = datasets.FashionMNIST(
            root="./data", train=False, transform=self.transform, download=True
        )

    def testPrep(self):
        mnist_images = torch.stack(
            [self.mnist_test[i][0] for i in range(len(self.mnist_test))]
        )
        mnist_labels = torch.tensor(
            [self.mnist_test[i][1] for i in range(len(self.mnist_test))]
        )
        np.random.seed(0)
        chosen_idx = np.random.choice(len(self.fMnist_test), replace=False, size=1000)
        fMnist_test = Subset(self.fMnist_test, chosen_idx)
        fashion_images = torch.stack(
            [fMnist_test[i][0] for i in range(len(fMnist_test))]
        )
        fashion_labels = torch.tensor(
            [fMnist_test[i][1] for i in range(len(fMnist_test))]
        )

        combined_images = torch.cat((mnist_images, fashion_images), dim=0)
        combined_labels = torch.cat((mnist_labels, fashion_labels), dim=0)
        self.test = CombinedDataset(combined_images, combined_labels)

    def returnTrain(self):
        return self.train

    def returnVal(self):
        return self.val

    def returnTest(self):
        self.testPrep()
        return self.test
