"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-24 23:27:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-01 16:22:32
"""
import os
import numpy as np
import torch

from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.transforms import InterpolationMode

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["MMI", "MMIDataset"]


class MMI(VisionDataset):
    url = None
    filename_prefix = "MMI_scan_"
    train_filename = "training"
    test_filename = "test"
    file_type = ".csv"
    folder = "mmi"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        file_list: List[str] = ["port_5_res_4_range_003"],
        processed_dir: str = "processed",
        download: bool = False,
    ) -> None:
        self.processed_dir = processed_dir
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.file_list = sorted(file_list)
        self.filenames = [f"{self.filename_prefix}{f}{self.file_type}" for f in self.file_list]
        self.train_filename = self.train_filename
        self.test_filename = self.test_filename
        self.meta_data = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        self.data: Any = []
        self.targets = []
        self.eps_min = self.eps_max = None

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        processed_training_file = os.path.join(processed_dir, f"{self.train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.pt")
        if os.path.exists(processed_training_file) and os.path.exists(processed_test_file):
            with open(os.path.join(self.root, self.processed_dir, f"{self.test_filename}.pt"), "rb") as f:
                data_dict = torch.load(f)
                meta_data = data_dict["meta"]
                data, targets = data_dict["data"]
                if data.shape[1] == meta_data["n_pads"] and targets.shape[1:] == (
                    meta_data["n_ports"],
                    meta_data["n_ports"],
                ):
                    print("Data already processed")
                    return
                else:
                    print(
                        f'Pads {data.shape[1]} != {meta_data["n_pads"]} or {targets.shape[1:]} != {(meta_data["n_ports"],meta_data["n_ports"],)}'
                    )
        data, targets = self._load_dataset()
        (
            data_train,
            targets_train,
            data_test,
            targets_test,
        ) = self._split_dataset(data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(
            self.meta_data,
            data_train,
            targets_train,
            data_test,
            targets_test,
            processed_dir,
            self.train_filename,
            self.test_filename,
        )

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        _, n_ports, _, n_levels, _, pad_max = (
            self.filenames[0].rstrip(self.file_type).lstrip(self.filename_prefix).split("_")
        )
        n_ports, n_levels, pad_max = int(n_ports), int(n_levels), float(pad_max)
        self.meta_data["n_ports"] = n_ports
        self.meta_data["n_levels"] = n_levels
        self.meta_data["pad_max"] = pad_max
        raw_data_list = [
            np.loadtxt(os.path.join(self.root, "raw", filename), delimiter=",", dtype=str)
            for filename in self.filenames
        ]
        data_total = []
        targets_total = []
        for raw_data in raw_data_list:
            header = raw_data[0]
            raw_data = raw_data[1:, 1:]
            n_pads = sum("pad" in i for i in header)
            self.meta_data["n_pads"] = n_pads
            data = torch.from_numpy(raw_data[::n_ports, :n_pads].astype(np.complex64).real)
            data_total.append(data)
            targets = torch.from_numpy(raw_data[:, n_pads + 1 :].astype(np.complex64))
            targets = targets.reshape([targets.shape[0] // n_ports, n_ports, n_ports]).transpose(-1, -2)
            targets_total.append(targets)
        data = torch.cat(data_total, 0)
        data = data - data.min()
        # data /= pad_max  # index change is normalized from 0 to 1 for better learnability
        data /= data.max()  # index change is normalized from 0 to 1 for better learnability
        print(f"Min pad: ", data.min(), "Max pad: ", data.max(), np.unique(data.numpy()))
        targets = torch.cat(targets_total, 0)
        return data, targets

    def _split_dataset(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        (
            data_train,
            data_test,
            targets_train,
            targets_test,
        ) = train_test_split(data, targets, train_size=self.train_ratio, random_state=42)
        print(f"training: {data_train.shape[0]} examples, " f"test: {data_test.shape[0]} examples")
        return (
            data_train,
            targets_train,
            data_test,
            targets_test,
        )

    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        meta_data: Dict,
        data_train: Tensor,
        targets_train: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.pt")
        with open(processed_training_file, "wb") as f:
            torch.save(dict(meta=meta_data, data=(data_train, targets_train)), f)

        with open(processed_test_file, "wb") as f:
            torch.save(dict(meta=meta_data, data=(data_test, targets_test)), f)

        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = f"{self.train_filename}.pt" if train else f"{self.test_filename}.pt"
        with open(os.path.join(self.root, self.processed_dir, filename), "rb") as f:
            data_dict = torch.load(f)
            self.meta_data = data_dict["meta"]
            data, targets = data_dict["data"]
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
        return data, targets

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        for filename in self.filenames:
            file = os.path.join(self.root, "raw", filename)
            if not os.path.exists(file):
                print(f"{file} does not exists")
                return False
        return True
        # return all([os.path.exists(os.path.join(self.root, "raw", filename)) for filename in self.filenames])

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        # data [bs, n_pads] real
        # targets [bs, n_ports, n_ports] complex
        return self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class MMIDataset:
    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        file_list: List[str],
        processed_dir: str = "processed",
    ):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(f"Only support test_ratio from (0, 1), but got {test_ratio}")
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.meta_data = {}
        self.file_list = sorted(file_list)
        self.processed_dir = processed_dir

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = MMI(
                self.root,
                train=True,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                file_list=self.file_list,
                processed_dir=self.processed_dir,
            )
            self.meta_data = train_valid.meta_data

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1] > 0.99999:
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[: train_len + valid_len]
                train_valid.targets = train_valid.targets[: train_len + valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = MMI(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                file_list=self.file_list,
                processed_dir=self.processed_dir,
            )

            self.data = test
            self.meta_data = test.meta_data

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_mmi():
    import pdb

    # pdb.set_trace()
    mmi = MMI(root="../../data", download=True, processed_dir="port_5_res_2_range_006")
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMI(root="../../data", train=False, download=True, processed_dir="port_5_res_2_range_006")
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMIDataset(
        root="../../data",
        split="train",
        test_ratio=0.1,
        train_valid_split_ratio=[0.9, 0.1],
        file_list=["port_5_res_2_range_006"],
        processed_dir="port_5_res_2_range_006",
    )
    print(len(mmi))


if __name__ == "__main__":
    test_mmi()
