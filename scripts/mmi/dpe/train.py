'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-30 01:56:48
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "dpe"
exp_name = "train"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train_dpe.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file,
            ]
    
    dims, dataset, dropout, n_ports, n_pads, bs, id = args
    
    with open(os.path.join(root, f"DPE_{'-'.join(map(str,dims))}_drop-{dropout:.2f}_{dataset}_id-{id}.log"), 'w') as wfid:
        exp = [
            f"--dataset.file_list={str([dataset])}",
            f"--dataset.processed_dir={dataset}",
            f"--dataset.train_valid_split_ratio=[1, 0]",
            f"--dataset.test_ratio=0.0001",
            f"--checkpoint.model_comment={model}_{exp_name}_{dataset}_{'-'.join(map(str,dims))}_drop-{dropout:.2f}",
            f"--run.log_interval=20",
            f"--run.n_epochs=4000",
            f"--run.random_state={41+id}",
            f"--run.batch_size={bs}",
            f"--model.hidden_dims={dims}",
            f"--model.dropout_rate={dropout}",
            f"--model.act_cfg.type=GELU",
            f"--model.n_ports={n_ports}",
            f"--model.n_pads={n_pads}",
            # f"--model.act_cfg.inplace=True",
            # f"--optimizer.name=sam_adam",
            f"--optimizer.name=adam",
            f"--optimizer.weight_decay=0",
            f"--criterion.name=cmse",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        [[256,256,128,128,128], "port_4_res_8_range_0.03", 0.0, 4, 4, 4095, 1],
        # [[256,256,128,128,128], "port_5_res_6_range_0.03", 0.0, 5, 5, 7775, 1], 
        # [[256,256,128,128,128], "port_10_res_6_range_0.03", 0.0, 10, 5, 7775, 1], 
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
