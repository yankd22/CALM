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
            # 只保留 credible_samples 中的样本
            updated_data = [self.test_dataset.data[i] for i in credible_samples]  # 筛选符合条件的数据
            updated_targets = [credible_samples[i] for i in credible_samples]  # 使用 credible_samples 中的新标签

            # 更新 test_dataset 的 data 和 targets
            self.test_dataset.data = torch.stack(updated_data)  # 转换为张量
            self.test_dataset.targets = torch.tensor(updated_targets)  # 更新标签

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