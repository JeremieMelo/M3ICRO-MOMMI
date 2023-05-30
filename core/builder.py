"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:51:50
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device

from core.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    FashionMNISTDataset,
    MMIDataset,
    MNISTDataset,
    StanfordDogsDataset,
    SVHNDataset,
)
from core.models import *

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(cfg: dict = None, splits=["train", "valid", "test"]) -> Tuple[DataLoader, DataLoader]:
    cfg = cfg or configs.dataset
    name = cfg.name.lower()
    if name == "fashionmnist":
        train_dataset, validation_dataset, test_dataset = (
            FashionMNISTDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar10":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR10Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "cifar100":
        train_dataset, validation_dataset, test_dataset = (
            CIFAR100Dataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "dogs":
        train_dataset, validation_dataset, test_dataset = (
            StanfordDogsDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "svhn":
        train_dataset, validation_dataset, test_dataset = (
            SVHNDataset(
                root=cfg.root,
                split=split,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                center_crop=cfg.center_crop,
                resize=cfg.img_height,
                resize_mode=cfg.resize_mode,
                binarize=False,
                grayscale=False,
                n_test_samples=cfg.n_test_samples,
                n_valid_samples=cfg.n_valid_samples,
                augment=cfg.transform != "basic",
            )
            for split in ["train", "valid", "test"]
        )
    elif name == "mmi":
        train_dataset, validation_dataset, test_dataset = (
            MMIDataset(
                root=cfg.root,
                split=split,
                test_ratio=cfg.test_ratio,
                train_valid_split_ratio=cfg.train_valid_split_ratio,
                file_list=cfg.file_list,
                processed_dir=cfg.processed_dir,
            )
            if split in splits
            else None
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            cfg.img_height,
            cfg.img_width,
            dataset_dir=cfg.root,
            transform=cfg.transform,
        )
        validation_dataset = None

    train_loader = (
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=configs.run.batch_size,
            shuffle=int(cfg.shuffle),
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if train_dataset is not None
        else None
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = (
        torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        if test_dataset is not None
        else None
    )

    return train_loader, validation_loader, test_loader


def make_model(device: Device, model_cfg: Optional[str] = None, random_state: int = None, **kwargs) -> nn.Module:
    model_cfg = model_cfg or configs.model
    name = model_cfg.name
    dpe = kwargs.get("dpe", None)
    if "mlp" in name.lower():
        model = eval(name)(
            n_feat=configs.dataset.img_height * configs.dataset.img_width,
            n_class=configs.dataset.n_class,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            mode=model_cfg.mode,
            v_max=configs.quantize.v_max,
            v_pi=configs.quantize.v_pi,
            act_thres=model_cfg.act_thres,
            photodetect=False,
            bias=False,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "dpe_cnn" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            kernel_list=model_cfg.kernel_list,
            kernel_size_list=model_cfg.kernel_size_list,
            pool_out_size=model_cfg.pool_out_size,
            stride_list=model_cfg.stride_list,
            padding_list=model_cfg.padding_list,
            hidden_list=model_cfg.hidden_list,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            pad_max=configs.quantize.pad_max,
            n_pads=model_cfg.n_pads,
            act_cfg=model_cfg.act_cfg,
            sigma_trainable=model_cfg.sigma_trainable,
            mode=model_cfg.mode,
            path_multiplier=model_cfg.path_multiplier,
            bias=True,
            dpe=dpe,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "dpe_vgg" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            pad_max=configs.quantize.pad_max,
            n_pads=model_cfg.n_pads,
            act_cfg=model_cfg.act_cfg,
            sigma_trainable=model_cfg.sigma_trainable,
            mode=model_cfg.mode,
            path_multiplier=model_cfg.path_multiplier,
            bias=True,
            dpe=dpe,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "dpe_resnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            pad_max=configs.quantize.pad_max,
            n_pads=model_cfg.n_pads,
            act_cfg=model_cfg.act_cfg,
            sigma_trainable=model_cfg.sigma_trainable,
            mode=model_cfg.mode,
            path_multiplier=model_cfg.path_multiplier,
            unfolding=model_cfg.unfolding,
            bias=True,
            dpe=dpe,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "dpe_efficientnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            pad_max=configs.quantize.pad_max,
            n_pads=model_cfg.n_pads,
            act_cfg=model_cfg.act_cfg,
            norm=model_cfg.norm,
            sigma_trainable=model_cfg.sigma_trainable,
            mode=model_cfg.mode,
            path_multiplier=model_cfg.path_multiplier,
            bias=True,
            dpe=dpe,
            device=device,
        ).to(device)
        model.reset_parameters(random_state)
    elif "dpe_mobilenet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            block_list=[int(i) for i in model_cfg.block_list],
            in_bit=configs.quantize.input_bit,
            w_bit=configs.quantize.weight_bit,
            pad_max=configs.quantize.pad_max,
            n_pads=model_cfg.n_pads,
            act_cfg=model_cfg.act_cfg,
            norm=model_cfg.norm,
            sigma_trainable=model_cfg.sigma_trainable,
            mode=model_cfg.mode,
            path_multiplier=model_cfg.path_multiplier,
            unfolding=model_cfg.unfolding,
            bias=True,
            dpe=dpe,
            device=device,
        ).to(device)
        model.reset_parameters()
    elif "vgg" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            act_cfg=model_cfg.act_cfg,
            bias=True,
            device=device,
        ).to(device)
    elif "resnet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            act_cfg=model_cfg.act_cfg,
            norm=model_cfg.norm,
            bias=False,
            device=device,
        ).to(device)
    elif "efficientnet" in name.lower():
        model = eval(name)(
            pretrained=True,
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            device=device,
        ).to(device)
    elif "mobilenet" in name.lower():
        model = eval(name)(
            img_height=configs.dataset.img_height,
            img_width=configs.dataset.img_width,
            num_classes=configs.dataset.num_classes,
            device=device,
        ).to(device)
        # model.reset_parameters()
    elif "unet" in name.lower():
        model = eval(name)(
            in_channels=configs.dataset.in_channels,
            out_channels=model_cfg.out_channels,
            dim=model_cfg.dim,
            act_func=model_cfg.act_func,
            domain_size=model_cfg.domain_size,
            grid_step=model_cfg.grid_step,
            buffer_width=model_cfg.buffer_width,
            dropout_rate=model_cfg.dropout_rate,
            drop_path_rate=model_cfg.drop_path_rate,
            aux_head=model_cfg.aux_head,
            aux_head_idx=model_cfg.aux_head_idx,
            pos_encoding=model_cfg.pos_encoding,
            device=device,
            **kwargs,
        ).to(device)
    elif "dpe" in name.lower():
        model = eval(name)(
            n_pads=model_cfg.n_pads,
            n_ports=model_cfg.n_ports,
            act_cfg=model_cfg.act_cfg,
            hidden_dims=model_cfg.hidden_dims,
            dropout=model_cfg.dropout_rate,
            device=device,
            **kwargs,
        ).to(device)

    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_cfg.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(cfg, "T", 3),
            alpha=getattr(cfg, "alpha", 0.9),
        )
    else:
        raise NotImplementedError(name)
    return criterion
