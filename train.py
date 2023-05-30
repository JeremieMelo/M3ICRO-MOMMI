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
import torch.cuda.amp as amp
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.datasets.mixup import Mixup, MixupAll
from core.utils import register_hidden_hooks


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
    teacher: Optional[nn.Module] = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    data_counter = 0
    correct = 0
    if hasattr(model, "requires_grad_dpe"):
        model.requires_grad_dpe(mode=True)
    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            class_loss = criterion(output, target)
            class_meter.update(class_loss.item())
            loss = class_loss

            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                if name == "kl_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data).data.detach()
                    T = 2
                    p = F.log_softmax(output / T, dim=1)
                    q = F.softmax(teacher_scores / T, dim=1)
                    aux_loss = F.kl_div(p, q, reduction="batchmean") * (T**2) * weight
                elif name == "mse_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher(data).data.detach()
                    teacher_hiddens = [m._recorded_hidden for m in teacher.modules() if hasattr(m, "_recorded_hidden")]
                    student_hiddens = [m._recorded_hidden for m in model.modules() if hasattr(m, "_recorded_hidden")]
                    
                    aux_loss = weight * sum(F.mse_loss(h1, h2) for h1, h2 in zip(teacher_hiddens, student_hiddens))
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        if configs.run.grad_clip:
            torch.nn.utils.clip_grad_value_(
                [p for p in model.parameters() if p.requires_grad], float(configs.run.max_grad_value)
            )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} class Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                class_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)
   
    scheduler.step()
    avg_class_loss = class_meter.avg
    accuracy = 100.0 * correct.to(torch.float32) / total_data
    lg.info(f"Train class Loss: {avg_class_loss:.4e}, Accuracy: {correct}/{total_data} ({accuracy:.2f}%)")
    mlflow.log_metrics(
        {
            "train_class": avg_class_loss,
            "train_acc": accuracy.item(),
            "lr": get_learning_rate(optimizer),
        },
        step=epoch,
    )


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target, random_state=i, vflip=False)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    lg.info(
        f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    mlflow.log_metrics({"val_loss": class_meter.avg, "val_acc": accuracy.item()}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("mse")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if mixup_fn is not None:
                    data, target = mixup_fn(data, target, random_state=i + 10000, vflip=False)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    mlflow.log_metrics({"test_loss": class_meter.avg, "test_acc": accuracy.item()}, step=epoch)


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
    has_lookup_table = False
    if configs.model.dpe is not None:
        dpe = builder.make_model(device, model_cfg=configs.model.dpe)
        load_model(dpe, path=configs.model.dpe.checkpoint)
        dpe.eval()
        dpe.requires_grad(False)
        dpe.set_dpe_noise_ratio(configs.model.dpe.dpe_noise_ratio)
        lg.info(f"Load DPE from {configs.model.dpe.checkpoint}")
        train_loader_mmi, _, test_loader_mmi = builder.make_dataloader(
            cfg=configs.dpe_dataset, splits=["train", "test"]
        )
        total_data = torch.cat([train_loader_mmi.dataset.data.dataset.data, test_loader_mmi.dataset.data.data], 0)
        total_targets = torch.cat(
            [train_loader_mmi.dataset.data.dataset.targets, test_loader_mmi.dataset.data.targets], 0
        )
        del train_loader_mmi
        del test_loader_mmi
        if float(configs.model.dpe.dpe_noise_ratio) > 1e-5:
            has_lookup_table = True
            dpe.build_lookup_table(
                total_data,
                total_targets,
                configs.quantize.weight_bit,  # how many levels, not typical bitwidth
                configs.quantize.pad_max,
            )
        lg.info(f"Build Predictor table")
        del total_data
        del total_targets
    else:
        dpe = None

    if configs.run.do_distill and configs.teacher is not None and os.path.exists(configs.teacher.checkpoint):
        teacher = builder.make_model(device, model_cfg=configs.teacher)
        load_model(teacher, path=configs.teacher.checkpoint)
        teacher.eval()
        lg.info(f"Load teacher model from {configs.teacher.checkpoint}")
    else:
        teacher = None

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state) if int(configs.run.deterministic) else None,
        dpe=dpe.forward if dpe is not None else None,  # only pass forward function, not the entire Module
    )

    optimizer = builder.make_optimizer(
        model.get_parameter_groups(weight_decay=float(configs.optimizer.weight_decay), lr=configs.optimizer.lr),
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(device)

    aux_criterions = dict()
    if configs.aux_criterion is not None:
        for name, config in configs.aux_criterion.items():
            if float(config.weight) > 0:
                try:
                    fn = builder.make_criterion(name, cfg=config)
                except NotImplementedError:
                    fn = name
                aux_criterions[name] = [fn, float(config.weight)]
    print(aux_criterions)
    if "mse_distill" in aux_criterions and teacher is not None:
        ## register hooks for teacher and student
        register_hidden_hooks(teacher)
        register_hidden_hooks(model)
        print(f"Register hidden state hooks for teacher and students")

    mixup_config = configs.dataset.augment
    mixup_fn = MixupAll(**mixup_config) if mixup_config is not None else None
    test_mixup_fn = MixupAll(**configs.dataset.test_augment) if mixup_config is not None else None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
    )
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
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
            test(
                model,
                validation_loader,
                0,
                criterion,
                lossv,
                accv,
                device,
                fp16=grad_scaler._enabled,
            )
        if teacher is not None:
            test(
                teacher,
                validation_loader,
                0,
                criterion,
                [],
                [],
                device,
                fp16=grad_scaler._enabled,
            )
            lg.info("Map teacher to student...")
            if hasattr(model, "load_from_teacher"):
                with amp.autocast(grad_scaler._enabled):
                    model.load_from_teacher(teacher)

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
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
                teacher=teacher,
            )

            if validation_loader is not None:
                if hasattr(model, "requires_grad_dpe") and has_lookup_table:
                    model.requires_grad_dpe(mode=False)
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    fp16=grad_scaler._enabled,
                )
            if hasattr(model, "requires_grad_dpe"):
                model.requires_grad_dpe(mode=True)
            test(
                model,
                test_loader,
                epoch,
                criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
            )
            if hasattr(model, "requires_grad_dpe") and has_lookup_table:
                model.requires_grad_dpe(mode=False)
            test(
                model,
                test_loader,
                epoch,
                criterion,
                lossv if validation_loader is None else [],
                accv if validation_loader is None else [],
                device,
                mixup_fn=test_mixup_fn,
                fp16=grad_scaler._enabled,
            )
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
