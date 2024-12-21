import os
import torch
import torchvision.datasets as datasets

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 credible_samples = None,
                 batch_size=128,
                 num_workers=0):


        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        if credible_samples is not None:
            updated_data = [self.test_dataset.data[i] for i in credible_samples]
            updated_targets = [credible_samples[i] for i in credible_samples]

            self.test_dataset.data = torch.stack(updated_data)
            self.test_dataset.targets = torch.tensor(updated_targets)

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