import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

losses = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    "poisson": nn.PoissonNLLLoss,
}


optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamw": torch.optim.AdamW,
    "sparse_adam": torch.optim.SparseAdam,
    "adamax": torch.optim.Adamax,
}


models = {
    "resnet18": {
        "model": models.resnet18,
        "pretrained_weights": models.ResNet18_Weights.DEFAULT,
    },
    "resnet34": {
        "model": models.resnet34,
        "pretrained_weights": models.ResNet34_Weights.DEFAULT,
    },
    "resnet50": {
        "model": models.resnet50,
        "pretrained_weights": models.ResNet50_Weights.DEFAULT,
    },
    "resnet101": {
        "model": models.resnet101,
        "pretrained_weights": models.ResNet101_Weights.DEFAULT,
    },
    "resnet152": {
        "model": models.resnet152,
        "pretrained_weights": models.ResNet152_Weights.DEFAULT,
    },
    "vgg16": {
        "model": models.vgg16,
        "pretrained_weights": models.VGG16_Weights.DEFAULT,
    },
    "alexnet": {
        "model": models.alexnet,
        "pretrained_weights": models.AlexNet_Weights.DEFAULT,
    },
}


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name not in losses:
        raise ValueError(f"Loss {loss_name} not found")
    return losses[loss_name](**kwargs)


def get_optimizer(
    optimizer_name: str, model_parameters, **kwargs
) -> torch.optim.Optimizer:
    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not found")
    return optimizers[optimizer_name](model_parameters, **kwargs)


def get_model(
    model_name: str, **kwargs
) -> dict[str, nn.Module | torchvision.models.Weights]:
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")
    return models[model_name]
