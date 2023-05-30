"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-29 22:43:12
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 19:57:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-15 15:49:13
"""

import torch.nn as nn


def hidden_register_hook(m, input, output):
    m._recorded_hidden = output


def register_hidden_hooks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Hardswish, nn.ReLU6)):
            m.register_forward_hook(hidden_register_hook)
