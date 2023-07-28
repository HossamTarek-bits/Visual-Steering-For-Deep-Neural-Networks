import torch.nn as nn


def get_model_last_conv(model: nn.Module):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def get_model_last_linear(model: nn.Module):
    last_linear = None
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_name = name
    return last_linear, last_name


def change_linear_layer(model: nn.Module, num_classes: int):
    last_linear, last_name = get_model_last_linear(model)
    if last_linear is not None:
        setattr(model, last_name, nn.Linear(last_linear.in_features, num_classes))
