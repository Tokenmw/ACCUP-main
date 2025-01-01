import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import os
import numpy as np

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(0, 2, 1)

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index

    def __len__(self):
        return self.len

class Load_ALL_Dataset(Dataset):
    def __init__(self, train_dataset, test_dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_train_data = train_dataset["samples"]
        x_test_data = test_dataset["samples"]
        # Load labels
        y_train_data = train_dataset.get("labels")
        y_test_data = test_dataset.get("labels")

        # combine train and test to x_data, y_data
        if isinstance(x_train_data, np.ndarray):
            x_data = np.concatenate([x_train_data, x_test_data], axis=0)
            y_data = np.concatenate([y_train_data, y_test_data], axis=0)
        else:
            x_data = torch.cat([x_train_data, x_test_data], dim=0)
            y_data = torch.cat([y_train_data, y_test_data], dim=0)

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(0, 2, 1)

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)


        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader


def whole_targe_data_generator(data_path, domain_id, dataset_configs, hparams):
    # loading dataset file from path
    train_dataset_file = torch.load(os.path.join(data_path, f"{'train'}_{domain_id}.pt"))
    test_dataset_file = torch.load(os.path.join(data_path, f"{'test'}_{domain_id}.pt"))

    # Loading datasets
    whole_dataset = Load_ALL_Dataset(train_dataset_file, test_dataset_file, dataset_configs)

    shuffle = False
    drop_last = False

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=whole_dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader


