"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-26 00:11:01
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import torch.fft
from core import builder
from core.datasets.mixup import Mixup, MixupAll
import torch.cuda.amp as amp


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler: Optional[Callable] = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            regression_loss = criterion(output, target)
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        ## sam optimizer
        if "sam" in configs.optimizer.name.lower():
            optimizer.first_step(zero_grad=True)
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    m.eval()
            criterion(model(data), target).backward()
            optimizer.second_step()
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    m.train()
        else:
            grad_scaler.step(optimizer)

        grad_scaler.update()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics({"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)}, step=epoch)
    lg.info(f"{model.freq.data}")
    lg.info(f"{model.phase_bias.data}")
    return avg_regression_loss


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (data, target) in enumerate(validation_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i, vflip=False)

            output = model(data)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i + 10000, vflip=False)

            output = model(data)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())
    print(output[0])
    print(target[0])

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    train_loader, validation_loader, test_loader = builder.make_dataloader(splits=["train", "test"])
    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )

    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(device)
    aux_criterions = (
        {
            name: [builder.make_criterion(name, cfg=config), float(config.weight)]
            for name, config in configs.aux_criterion.items()
            if float(config.weight) > 0
        }
        if configs.aux_criterion is not None
        else {}
    )
    print(aux_criterions)
    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config) if mixup_config is not None else None
    test_mixup_fn = MixupAll(**configs.dataset.test_augment) if mixup_config is not None else None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=6,
        metric_name="err",
        format="{:.6f}",
    )
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "amp", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if int(configs.checkpoint.resume) and len(configs.checkpoint.restore_checkpoint) > 0:
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(model, validation_loader, 0, criterion, lossv, accv, device)

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train_loss = train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device,
                grad_scaler=grad_scaler,
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                )

            test(
                model,
                test_loader,
                epoch,
                criterion,
                lossv if validation_loader is None else [],
                accv if validation_loader is None else [],
                device,
                mixup_fn=test_mixup_fn,
            )
            saver.save_model(model, train_loss, epoch=epoch, path=checkpoint, save_model=False, print_msg=True)

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
