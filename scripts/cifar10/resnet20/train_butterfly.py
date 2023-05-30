import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar10"
model = "resnet20"
exp_name = "train_butterfly"
root = f"log/{dataset}/{model}/{exp_name}"
script = "train.py"
config_file = f"configs/{dataset}/{model}/train/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]

    w_bit, in_bit, lr, dpe_ratio, sigma_train, dpe_dataset, ckpt, mode, multiplier, n_ports, n_pads, dpe_act, dpe_dims, dpe_drop, id = args

    with open(
        os.path.join(
            root,
            f"FFT_ResNet20_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_dr-{dpe_ratio:.1f}_sig-{sigma_train}_{dpe_dataset}_{mode}_path-{multiplier}_port-{n_ports}_id-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--dataset.train_valid_split_ratio=[0.95, 0.05]",
            f"--dataset.test_ratio=0.1",
            f"--dataset.transform=augment",
            f"--dpe_dataset.file_list={str([dpe_dataset])}",
            f"--dpe_dataset.processed_dir={dpe_dataset}",
            f"--checkpoint.model_comment={model}_{exp_name}_{dataset}_lr-{lr:.4f}_dr-{dpe_ratio:.1f}_{dpe_dataset}_{mode}_path-{multiplier}_port-{n_ports}",
            f"--run.random_state={41+id}",
            f"--quantize.weight_bit={w_bit}",
            f"--quantize.input_bit={in_bit}",
            f"--model.dpe.dpe_noise_ratio={dpe_ratio}",
            f"--model.dpe.act_cfg.type={dpe_act}", # new 4x4 MMI simulation data use ReLU, otherwise GELU
            f"--model.dpe.hidden_dims={dpe_dims}",
            f"--model.dpe.checkpoint={ckpt}",
            f"--model.sigma_trainable={sigma_train}",
            f"--model.mode={mode}",
            f"--model.path_multiplier={multiplier}",
            f"--model.dpe.n_ports={n_ports}",
            f"--model.dpe.n_pads={n_pads}",
            f"--model.dpe.dropout_rate={dpe_drop}",
            f"--model.dpe=None",
            f"--model.n_pads={n_pads}",
            f"--model.block_list={[n_ports]*3}",
            f"--optimizer.lr={lr}",
            f"--optimizer.weight_decay=0.0001",
            f"--run.do_distill=True",
            f"--run.fp16=True",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        [
            1024, # 10 bit
            8,
            0.002,
            0,
            "row_col",
            "port_4_res_8_range_0.03",
            "./checkpoint/mmi/dpe/pretrain/MMI_port_4_pad_4_res_8_range_0.03.pt",
            "fsf",
            1,
            4,
            4,
            "GELU",
            [256,256,128,128,128],
            0,
            1,
        ], # 4x4 FFT
        [
            1024, # 10 bit
            8,
            0.002,
            0,
            "row_col",
            "port_4_res_8_range_0.03",
            "./checkpoint/mmi/dpe/pretrain/MMI_port_4_pad_4_res_8_range_0.03.pt",
            "fsf",
            1,
            8,
            8,
            "GELU",
            [256,256,128,128,128],
            0,
            1,
        ], # 8x8 FFT
        [
            1024, # 10 bit
            8,
            0.002,
            0,
            "row_col",
            "port_4_res_8_range_0.03",
            "./checkpoint/mmi/dpe/pretrain/MMI_port_4_pad_4_res_8_range_0.03.pt",
            "bsb",
            1,
            4,
            4,
            "GELU",
            [256,256,128,128,128],
            0,
            1,
        ], # 4x4 Butterfly
        [
            1024, # 10 bit
            8,
            0.002,
            0,
            "row_col",
            "port_4_res_8_range_0.03",
            "./checkpoint/mmi/dpe/pretrain/MMI_port_4_pad_4_res_8_range_0.03.pt",
            "bsb",
            1,
            8,
            8,
            "GELU",
            [256,256,128,128,128],
            0,
            1,
        ], # 8x8 Butterfly
        
    ]
    
    with Pool(1) as p:
        p.map(task_launcher, tasks[2:3])
    logger.info(f"Exp: {configs.run.experiment} Done.")
