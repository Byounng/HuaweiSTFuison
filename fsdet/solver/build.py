
from typing import Any, Dict, List
import torch

from fsdet.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR
from .masked_sgd import MaskedSGD


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []

    
    masked_param = cfg.SOLVER.MASKED_PARAMS
    masked_param_inds = cfg.SOLVER.MASKED_PARAMS_INDS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if ("attn_weight_box" in key) or ("attn_weight_fused" in key):
        
            
            
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY            
        else: 
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            
            
            
            
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS


        if key in masked_param:
            filter_indices = masked_param_inds[masked_param.index(key)]
        else:
            filter_indices = []
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay,
                    "filter_indices": [filter_indices]}]

    if cfg.SOLVER.NAME == 'MaskedSGD':
        optimizer = MaskedSGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
        
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
