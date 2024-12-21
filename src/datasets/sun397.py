import os
import torch
import torchvision.datasets as datasets

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 credible_samples = None,
                 batch_size=32,
                 num_workers=0):
        # Data loading code
        traindir = os.path.join(location, 'sun397', 'train')
        valdir = os.path.join(location, 'sun397', 'test')


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)

        if credible_samples is not None:
            updated_samples = [(path, credible_samples[i]) for i, (path, label) in enumerate(self.test_dataset.samples) if i in credible_samples]
            self.test_dataset.samples = updated_samples
            self.test_dataset.imgs = updated_samples
            
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]
