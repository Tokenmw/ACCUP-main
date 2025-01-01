import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os, sys
import numpy as np
from .augmentations import DataTransform

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs, seed_id = 1):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        x_data = dataset["samples"]
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

        aug, aug2 = DataTransform(self.x_data, dataset_configs, seed_id = seed_id)
        if isinstance(aug, np.ndarray):
            aug = torch.from_numpy(aug)
        if isinstance(aug2, np.ndarray):
            aug2 = torch.from_numpy(aug2)
        if dataset_configs.normalize:
            aug_data_mean = torch.mean(aug, dim=(0, 2))
            aug_data_std = torch.std(aug, dim=(0, 2))
            self.aug_transform = transforms.Normalize(mean=aug_data_mean, std=aug_data_std)

            aug_data_mean2 = torch.mean(aug2, dim=(0, 2))
            aug_data_std2 = torch.std(aug2, dim=(0, 2))
            self.aug_transform2 = transforms.Normalize(mean=aug_data_mean2, std=aug_data_std2)
        self.aug = aug.float()
        self.aug2 = aug2.float()

    def __getitem__(self, index):
        x = self.x_data[index]
        aug = self.aug[index]
        aug2 = self.aug2[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
            aug = self.aug_transform(self.aug[index].reshape(self.num_channels, -1, 1)).reshape(self.aug[index].shape)
            aug2 = self.aug_transform2(self.aug2[index].reshape(self.num_channels, -1, 1)).reshape(self.aug2[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return (x, aug, aug2), y, index

    def __len__(self):
        return self.len

class Load_ALL_Dataset(Dataset):
    def __init__(self, train_dataset, test_dataset, dataset_configs, seed_id = 1):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        x_train_data = train_dataset["samples"]
        x_test_data = test_dataset["samples"]
        y_train_data = train_dataset.get("labels")
        y_test_data = test_dataset.get("labels")

        if isinstance(x_train_data, np.ndarray):
            x_data = np.concatenate([x_train_data, x_test_data], axis=0)
            y_data = np.concatenate([y_train_data, y_test_data], axis=0)
        else:
            x_data = torch.cat([x_train_data, x_test_data], dim=0)
            y_data = torch.cat([y_train_data, y_test_data], dim=0)

        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(1, 2)

        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

        aug, aug2 = DataTransform(self.x_data, dataset_configs, seed_id = seed_id)
        if isinstance(aug, np.ndarray):
            aug = torch.from_numpy(aug)
        if isinstance(aug2, np.ndarray):
            aug2 = torch.from_numpy(aug2)
        if dataset_configs.normalize:
            aug_data_mean = torch.mean(aug, dim=(0, 2))
            aug_data_std = torch.std(aug, dim=(0, 2))
            self.aug_transform = transforms.Normalize(mean=aug_data_mean, std=aug_data_std)

            aug_data_mean2 = torch.mean(aug2, dim=(0, 2))
            aug_data_std2 = torch.std(aug2, dim=(0, 2))
            self.aug_transform2 = transforms.Normalize(mean=aug_data_mean2, std=aug_data_std2)
        self.aug = aug.float()
        self.aug2 = aug2.float()

    def __getitem__(self, index):
        x = self.x_data[index]
        aug = self.aug[index]
        aug2 = self.aug2[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
            aug = self.aug_transform(self.aug[index].reshape(self.num_channels, -1, 1)).reshape(self.aug[index].shape)
            aug2 = self.aug_transform2(self.aug2[index].reshape(self.num_channels, -1, 1)).reshape(self.aug2[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return (x, aug, aug2), y, index

    def __len__(self):
        return self.len

def data_generator_demo(data_path, domain_id, dataset_configs, hparams, dtype):
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))
    dataset = Load_Dataset(dataset_file, dataset_configs)
    if dtype == "test":
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader


def data_generator_old(data_path, domain_id, dataset_configs, hparams):
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    train_dataset = Load_Dataset(train_dataset, dataset_configs)
    test_dataset = Load_Dataset(test_dataset, dataset_configs)

    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader


def few_shot_data_generator(data_loader, dataset_configs, num_samples=5):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    NUM_SAMPLES_PER_CLASS = num_samples
    NUM_CLASSES = len(torch.unique(y_data))

    counts = [y_data.eq(i).sum().item() for i in range(NUM_CLASSES)]
    samples_count_dict = {i: min(counts[i], NUM_SAMPLES_PER_CLASS) for i in range(NUM_CLASSES)}

    samples_ids = {i: torch.where(y_data == i)[0] for i in range(NUM_CLASSES)}
    selected_ids = {i: torch.randperm(samples_ids[i].size(0))[:samples_count_dict[i]] for i in range(NUM_CLASSES)}

    selected_x = torch.cat([x_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)
    selected_y = torch.cat([y_data[samples_ids[i][selected_ids[i]]] for i in range(NUM_CLASSES)], dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    few_shot_dataset = Load_Dataset(few_shot_dataset, dataset_configs)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)

    return few_shot_loader

def whole_targe_data_generator_demo(data_path, domain_id, dataset_configs, hparams, seed_id = 1):
    train_dataset_file = torch.load(os.path.join(data_path, f"{'train'}_{domain_id}.pt"))
    test_dataset_file = torch.load(os.path.join(data_path, f"{'test'}_{domain_id}.pt"))

    whole_dataset = Load_ALL_Dataset(train_dataset_file, test_dataset_file, dataset_configs, seed_id=seed_id)
    shuffle = False
    drop_last = False

    data_loader = torch.utils.data.DataLoader(dataset=whole_dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader