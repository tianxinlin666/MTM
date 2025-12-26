import os
import torch
import torch.nn as nn
import random
import numpy as np
import logging

def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:                 
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_log(file_path, file_name):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(os.path.join(file_path, file_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def save_ckpt(model, optimizer, config, ckpt_file, epoch, model_val, ema_model=None): 
    state = {
        'config': config,
        'epoch': epoch,
        'model_val': model_val,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if ema_model is not None: 
        state['ema_shadow'] = ema_model.shadow
    torch.save(state, ckpt_file)

def load_ckpt(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    config = ckpt['config']
    model_state_dict = ckpt['state_dict']
    optimizer_state_dict = ckpt['optimizer']
    current_epoch = ckpt['epoch']
    model_val = ckpt['model_val']
    ema_shadow = ckpt.get('ema_shadow', None) 

    return config, model_state_dict, optimizer_state_dict, current_epoch, model_val, ema_shadow

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data
