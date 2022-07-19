#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2022/02/23 21:33
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : fl_utils.py
# @Software  : PyCharm

import torch
from collections import OrderedDict

def weight_to_state_dict(keys, weights):
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, weights)})
    return state_dict


def state_dict_to_weight(state_dict):
    return [values.cpu().numpy() for _, values in state_dict.items()]