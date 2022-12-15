#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
#
"""
Utilities for computer vision experiments.
"""
import hashlib
import logging
import os
import tarfile

from datasets import load_dataset
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import tqdm


logger = logging.getLogger(__name__)


def create_model(dataset, model_name="resnet18", device=torch.device("cpu"), **kwargs):
    """
    Returns a model with a representation pre-trained on ImageNet, but an untrained final linear classification layer.
    """
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="DEFAULT", **kwargs)
        if not isinstance(dataset, ImageNet):
            model.fc = nn.Linear(512, dataset.n_class)

    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights="DEFAULT", **kwargs)
        if not isinstance(dataset, ImageNet):
            model.fc = nn.Linear(2048, dataset.n_class)

    elif model_name == "densenet121":
        model = torchvision.models.densenet121(weights="DEFAULT", **kwargs)
        if not isinstance(dataset, ImageNet):
            model.classifier = nn.Linear(1024, dataset.n_class)

    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(weights="DEFAULT", transform_input=False, **kwargs)
        if not isinstance(dataset, ImageNet):
            model.fc = nn.Linear(2048, dataset.n_class)

    elif model_name == "wide_resnet50":
        model = torchvision.models.wide_resnet50_2(weights="DEFAULT", **kwargs)
        if not isinstance(dataset, ImageNet):
            model.fc = nn.Linear(2048, dataset.n_class)

    else:
        raise NotImplementedError(f"Model {model_name} is not a supported pre-trained ImageNet model.")

    return model.to(device=device)


def data_loader(dataset, batch_size=256, epoch=0, pin_memory=True):
    shuffle = dataset.split == "train"
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        sampler.set_epoch(epoch)
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)


class DatasetMixIn(Dataset):
    def __init__(self, split):
        assert split in ["train", "valid", "test"]
        self.split = split
        if split == "train":
            resize = [transforms.Resize(256), transforms.RandomResizedCrop(224)]
        else:
            resize = [transforms.Resize(256), transforms.CenterCrop(224)]
        self.transform = transforms.Compose([*resize, transforms.ToTensor(), self.norm()])
        self.instance_key = "image"
        self.label_key = "label"

    def __len__(self):
        test_name = "test" if "test" in self.data else "valid"
        if self.split == "train":
            return len(self.train_idx)
        if self.split == "valid":
            return len(self.valid_idx)
        return len(self.data[test_name])

    def __getitem__(self, i):
        test_name = "test" if "test" in self.data else "valid"
        if self.split == "train":
            i = self.train_idx[i]
        elif self.split == "valid":
            i = self.valid_idx[i]

        # Get the data point
        instance = self.data[test_name if self.split == "test" else "train"][i]
        return self.transform(instance[self.instance_key].convert("RGB")), instance[self.label_key]


class ImageNet:
    def __init__(self, split, rootdir="/export/share/datasets/vision/imagenet"):
        if split == "train":
            self.data = ImageFolder(os.path.join(rootdir, "train"))
        else:
            self.data = ImageFolder(os.path.join(rootdir, "val"))
        resize = [transforms.Resize(256), transforms.CenterCrop(224)]
        self.split = split
        self.transform = transforms.Compose([*resize, transforms.ToTensor(), TinyImageNet.norm()])

    def __len__(self):
        return len(self.data) if self.split == "train" else len(self.data) // 2

    def __getitem__(self, i):
        img, label = self.data[i if self.split == "train" else 2 * i + (self.split == "test")]
        return self.transform(img), label

    @property
    def n_class(self):
        return 1000


class TinyImageNet(DatasetMixIn):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("Maysee/tiny-imagenet")
        self.valid_idx = list(range(0, 100000, 10))
        self.train_idx = [i for i in range(100000) if i not in self.valid_idx]

    @classmethod
    def norm(cls):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def n_class(self):
        return 200


class CIFAR10(DatasetMixIn):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("cifar10")
        self.train_idx = range(45000)
        self.valid_idx = range(45000, 50000)
        self.instance_key = "img"

    @classmethod
    def norm(cls):
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @property
    def n_class(self):
        return 10


class CIFAR100(DatasetMixIn):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("cifar100")
        self.train_idx = range(45000)
        self.valid_idx = range(45000, 50000)
        self.instance_key = "img"
        self.label_key = "fine_label"

    @classmethod
    def norm(cls):
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @property
    def n_class(self):
        return 100


class Downloader(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]
        if not isinstance(self.data, DatasetMixIn):
            image = self.transform(image)
        return image, label

    @staticmethod
    def extract_tar(tar_path, extract_dir=None, success_file=None):
        if extract_dir is None:
            extract_dir = os.path.join(os.path.dirname(tar_path), os.path.basename(tar_path)[: -len(".tar")])
        success_file = os.path.join(extract_dir, "_SUCCESS") if success_file is None else success_file
        if not os.path.isfile(success_file):
            logger.info(f"Extracting tarfile {os.path.basename(tar_path)}...")
            tarfile.open(tar_path).extractall(path=os.path.dirname(tar_path))
            with open(success_file, "w"):
                pass

    @staticmethod
    def download(url, file_name, expected_md5):
        if os.path.exists(file_name):
            logger.info("Checking MD5 checksum...")
            with open(file_name, "rb") as file_to_check:
                data = file_to_check.read()
                md5_returned = hashlib.md5(data).hexdigest()
            if md5_returned != expected_md5:
                logger.info("Invalid MD5 checksum. Restarting download.")
                os.remove(file_name)

        if os.path.exists(file_name):
            return

        logger.info(f"Downloading file {os.path.basename(file_name)}...")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            r = requests.get(url, stream=True)
            bar_format = "{l_bar}{bar}| {n:.1f}/{total:.1f}MB [{elapsed}<{remaining}, {rate_fmt}]"
            total_mb = int(r.headers["Content-Length"]) / (1024**2)
            with tqdm.tqdm(unit="MB", total=total_mb, bar_format=bar_format) as pbar:
                for chunk in r.iter_content(chunk_size=1 * (1024**2)):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk) / (1024**2))


class ImageNetC(Downloader):
    def __init__(self, corruption=None, severity=0):
        # Download dataset
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ImageNetC")
        if not os.path.exists(os.path.join(base_dir, "_SUCCESS")):
            base = "https://zenodo.org/record/2235448/files"
            suf = "?download=1"
            self.download(f"{base}/blur.tar{suf}", f"{base_dir}/blur.tar", "2d8e81fdd8e07fef67b9334fa635e45c")
            self.download(f"{base}/digital.tar{suf}", f"{base_dir}/digital.tar", "89157860d7b10d5797849337ca2e5c03")
            self.download(f"{base}/noise.tar{suf}", f"{base_dir}/noise.tar", "e80562d7f6c3f8834afb1ecf27252745")
            self.download(f"{base}/weather.tar{suf}", f"{base_dir}/weather.tar", "33ffea4db4d93fe4a428c40a6ce0c25d")
            with open(os.path.join(base_dir, "_SUCCESS"), "w"):
                pass

        # Extract tar files
        self.extract_tar(f"{base_dir}/blur.tar", success_file=f"{base_dir}/_SUCCESS_blur")
        self.extract_tar(f"{base_dir}/digital.tar", success_file=f"{base_dir}/_SUCCESS_digital")
        self.extract_tar(f"{base_dir}/noise.tar", success_file=f"{base_dir}/_SUCCESS_noise")
        self.extract_tar(f"{base_dir}/weather.tar", success_file=f"{base_dir}/_SUCCESS_weather")

        # Get the actual dataset
        if severity == 0 or corruption is None:
            self.data = ImageNet(split="test")
        else:
            valid_corruptions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            assert severity in range(1, 6), f"Got severity={severity}. Expected an int between 1 and 5."
            assert corruption in valid_corruptions, f"Got corruption={corruption}. Expected one of {valid_corruptions}"
            self.data = ImageFolder(os.path.join(base_dir, corruption, str(severity)))
        resize = [transforms.Resize(256), transforms.CenterCrop(224)]
        self.transform = transforms.Compose([*resize, transforms.ToTensor(), TinyImageNet.norm()])
        self.split = "test"

    def __getitem__(self, i):
        return self.data[i] if isinstance(self.data, ImageNet) else super().__getitem__(i)

    @property
    def n_class(self):
        return 1000


class TinyImageNetC(Downloader):
    def __init__(self, corruption=None, severity=0):
        # Download data & extract tar if needed
        url = "https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(base_dir, "data", "TinyImageNet-C.tar")
        data_dir = os.path.join(os.path.dirname(file_name), "TinyImageNet-C", "Tiny-ImageNet-C")
        if not os.path.exists(os.path.join(data_dir, "_SUCCESS")):
            self.download(url=url, file_name=file_name, expected_md5="3d9c6e89c2609aeb4198f84c8edd1ff0")
            self.extract_tar(file_name)
            self.extract_tar(os.path.join(os.path.dirname(file_name), "TinyImageNet-C", "Tiny-ImageNet-C.tar"))

        # Get the actual dataset
        if severity == 0 or corruption is None:
            self.data = TinyImageNet(split="test")
        else:
            valid_corruptions = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            assert severity in range(1, 6), f"Got severity={severity}. Expected an int between 1 and 5."
            assert corruption in valid_corruptions, f"Got corruption={corruption}. Expected one of {valid_corruptions}"
            self.data = ImageFolder(os.path.join(data_dir, corruption, str(severity)))
        resize = [transforms.Resize(256), transforms.CenterCrop(224)]
        self.transform = transforms.Compose([*resize, transforms.ToTensor(), TinyImageNet.norm()])
        self.split = "test"

    @property
    def n_class(self):
        return 200


class CIFAR10C(Downloader):
    def __init__(self, corruption=None, severity=0):
        # Download data & extract tar if needed
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(base_dir, "data", "CIFAR10-C.tar")
        data_dir = os.path.join(base_dir, "data", "CIFAR-10-C")
        if not os.path.exists(os.path.join(data_dir, "_SUCCESS")):
            self.download(url=url, file_name=file_name, expected_md5="56bf5dcef84df0e2308c6dcbcbbd8499")
            self.extract_tar(file_name, extract_dir=data_dir)

        # Get the actual dataset
        if severity == 0 or corruption is None:
            self.data = CIFAR10(split="test")
        else:
            valid_corruptions = [d[:-4] for d in os.listdir(data_dir) if d != "labels.npy" and d.endswith(".npy")]
            assert severity in range(1, 6), f"Got severity={severity}. Expected an int between 1 and 5."
            assert corruption in valid_corruptions, f"Got corruption={corruption}. Expected one of {valid_corruptions}"
            data = np.load(os.path.join(data_dir, corruption + ".npy"))
            labels = np.load(os.path.join(data_dir, "labels.npy"))
            self.data = [(data[i], labels[i]) for i in range((severity - 1) * 10000, severity * 10000)]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), CIFAR10.norm()])
        self.split = "test"

    @property
    def n_class(self):
        return 10


class CIFAR100C(Downloader):
    def __init__(self, corruption=None, severity=0):
        # Download data & extract tar if needed
        url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(base_dir, "data", "CIFAR100-C.tar")
        data_dir = os.path.join(base_dir, "data", "CIFAR-100-C")
        if not os.path.exists(os.path.join(data_dir, "_SUCCESS")):
            self.download(url=url, file_name=file_name, expected_md5="11f0ed0f1191edbf9fa23466ae6021d3")
            self.extract_tar(file_name, extract_dir=data_dir)

        # Get the actual dataset
        if severity == 0 or corruption is None:
            self.data = CIFAR100(split="test")
        else:
            valid_corruptions = [d[:-4] for d in os.listdir(data_dir) if d != "labels.npy" and d.endswith(".npy")]
            assert severity in range(1, 6), f"Got severity={severity}. Expected an int between 1 and 5."
            assert corruption in valid_corruptions, f"Got corruption={corruption}. Expected one of {valid_corruptions}"
            data = np.load(os.path.join(data_dir, corruption + ".npy"))
            labels = np.load(os.path.join(data_dir, "labels.npy"))
            self.data = [(data[i], labels[i]) for i in range((severity - 1) * 10000, severity * 10000)]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), CIFAR10.norm()])
        self.split = "test"

    @property
    def n_class(self):
        return 100
