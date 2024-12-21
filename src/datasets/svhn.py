import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
import numpy as np


class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 credible_samples = None,
                 batch_size=128,
                 num_workers=0):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')

        self.train_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess
        )

        if credible_samples is not None:
            updated_data = [self.test_dataset.data[i] for i in credible_samples]
            updated_labels = [credible_samples[i] for i in credible_samples]
            self.test_dataset.data = np.array(updated_data)
            self.test_dataset.labels = np.array(updated_labels)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
