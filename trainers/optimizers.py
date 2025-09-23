"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import torch
import torch.nn as nn
import math 

AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw", "sghmc"]

class SGHMC(torch.optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, cycle_length=1, noise_last_epochs=-1):
        super(SGHMC, self).__init__(params, lr, momentum, dampening,
                                        weight_decay, nesterov)
        self.epoch = 0
        self.cycle_length = cycle_length
        self.noise_last_epochs = noise_last_epochs
        self.noise_temperature = 1.0  # can be tuned, usually set to 1.0
        self.dataset_size = 1  # to be set externally, used in noise calculation

    def set_epoch(self, epoch):
        self.epoch = epoch

    def step(self, closure=None):
        """Our handmade SGD step using current LR from the real optimizer"""
        current_lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        momentum = self.param_groups[0]['momentum']

        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                grad = param.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Apply momentum
                if 'momentum_buffer' not in self.state[param]:
                    buf = self.state[param]['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = self.state[param]['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                # sghmc noise scheduling
                if self.cycle_length > 0 and self.noise_last_epochs > 0:
                    # Include noise on the last N epochs of each cycle
                    cycle_epoch = (self.epoch + 1) % self.cycle_length
                    if cycle_epoch >= self.cycle_length - self.noise_last_epochs:
                        noise_std = math.sqrt(2 * weight_decay * current_lr)
                        # buf_new += (2.0*lr*args.alpha*args.temperature/datasize)**.5*eps
                        # noise_std = (2.0 * current_lr * momentum * self.noise_temperature / self.dataset_size)**0.5
                        noise = torch.randn_like(param) * noise_std * self.noise_temperature
                        buf.add_(noise)

                # Update parameters
                param.data.add_(buf, alpha=-current_lr)
    # def step(self, closure=None):
    #     """Our handmade SGD step using current LR from the real optimizer"""
    #     current_lr = self.param_groups[0]['lr']
    #     weight_decay = self.param_groups[0]['weight_decay']
    #     momentum = self.param_groups[0]['momentum']
    #     for group in self.param_groups:
    #         for i, param in enumerate(group['params']):
    #             if param.grad is None:
    #                 continue

    #             # Initialize momentum buffer if it doesn't exist (matching first implementation)
    #             if 'momentum_buffer' not in self.state[param]:
    #                 self.state[param]['momentum_buffer'] = torch.zeros_like(param.data)

    #             buf = self.state[param]['momentum_buffer']
                
    #             # Get gradient and apply weight decay (matching first implementation)
    #             d_p = param.grad.data
    #             d_p.add_(param.data, alpha=weight_decay)

    #             # Update momentum buffer using first implementation's formula
    #             buf_new = (1 - momentum) * buf - current_lr * d_p

    #             # Add noise if in the last N epochs of cycle
    #             if self.cycle_length > 0 and self.noise_last_epochs > 0:
    #                 cycle_epoch = (self.epoch + 1) % self.cycle_length
    #                 if cycle_epoch >= self.cycle_length - self.noise_last_epochs:
    #                     noise_std = (2.0 * current_lr * momentum * self.noise_temperature / self.dataset_size)**0.5
    #                     eps = torch.randn_like(param) * noise_std
    #                     buf_new.add_(eps)

    #             # Update parameters (matching first implementation)
    #             param.data.add_(buf_new)
                
    #             # Update buffer state
    #             self.state[param]['momentum_buffer'] = buf_new

def build_optimizer(model, optim_cfg, param_groups=None):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """
    optim = optim_cfg.NAME
    lr = optim_cfg.LR
    weight_decay = optim_cfg.WEIGHT_DECAY
    momentum = optim_cfg.MOMENTUM
    sgd_dampening = optim_cfg.SGD_DAMPNING
    sgd_nesterov = optim_cfg.SGD_NESTEROV
    rmsprop_alpha = optim_cfg.RMSPROP_ALPHA
    adam_beta1 = optim_cfg.ADAM_BETA1
    adam_beta2 = optim_cfg.ADAM_BETA2
    staged_lr = optim_cfg.STAGED_LR
    new_layers = optim_cfg.NEW_LAYERS
    base_lr_mult = optim_cfg.BASE_LR_MULT

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
        )

    if param_groups is not None and staged_lr:
        warnings.warn(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )

    if param_groups is None:
        if staged_lr:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )

            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn("new_layers is empty (staged_lr is useless)")
                new_layers = [new_layers]

            base_params = []
            base_layers = []
            new_params = []

            for name, module in model.named_children():
                if name in new_layers:
                    new_params += [p for p in module.parameters()]
                else:
                    base_params += [p for p in module.parameters()]
                    base_layers.append(name)

            param_groups = [
                {
                    "params": base_params,
                    "lr": lr * base_lr_mult
                },
                {
                    "params": new_params
                },
            ]

        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )
    elif optim == "sghmc": 
        optimizer = SGHMC(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=sgd_nesterov,
        )
    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )


    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer
