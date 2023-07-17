from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor, hflip, vflip, rotate, normalize
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import PIL.Image
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomAutocontrast,
)
from torch.utils.data import Dataset
from pathlib import Path
from pytorch_msssim import SSIM
from scipy.spatial import distance
from typing import Optional, Union
import warnings
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import captum
import tqdm
import torch.optim as optim
from PIL import Image
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import random
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import multiclass_f1_score,multiclass_confusion_matrix,multiclass_auroc
from utils import PairedImageFolder, AverageMeter, seed_everything, EarlyStopping
import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='Training')
#parser.add_argument('-d', '--dataset', default='imagenette2', type=str)
parser.add_argument('-b', '--batch_size', default=8, type=int)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-m', '--model', default='resnet50', type=str)
parser.add_argument('-mp', '--model_path', default='models', type=str)
parser.add_argument('-s', '--seed', default=420, type=int)
parser.add_argument('-l', '--loss', default='cross_entropy', type=str)
parser.add_argumennt('-n', '--num_classes', default=10, type=int)
parser.add_argument('-gamma', default=0.1, type=float)
parser.add_argument('-mom', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-imf','--image_folder',default="imagenette2/Images/Train",type=str,metavar='PATH',help='path to train images folder')
parser.add_argument('-mf','--mask_folder',default="imagenette2/Masks/Train",type=str,metavar='PATH',help='path to train masks folder')
parser.add_argument('-test_imf','--test_image_folder',default="imagenette2All/val",type=str,metavar='PATH',help='path to test images folder')
parser.add_argument('--out_path', default='results', type=str,metavar='PATH',help='path to output model folder')
# parser.add_argument('--gpu-id', default='0', type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# use_cuda = torch.cuda.is_available()


def get_model_last_conv(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    
def training_loop(model):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, masks, target) in enumerate(train_loader):
            if batch_idx == 0:
                generate_heatmaps(data, masks, target)
                model.train()
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            masks = masks.mean(dim=1).unsqueeze(1).to(device)
            output = model(data)
            cam = captum.attr.LayerGradCam(model, get_model_last_conv())
            gradsB = captum.attr.LayerAttribution.interpolate(
                cam.attribute(data, target=target, relu_attributions=True),  # type: ignore
                interpolate_dims=(224, 224),
                interpolate_mode="bicubic",
            )
            grads = []
            for j in range(data.shape[0]):
                guided = torch.clamp(gradsB[j].mean(dim=0), 0)
                if guided.max() - guided.min() == 0:
                    guided = torch.clamp(guided, 0, 1)
                else:
                    guided = (guided - guided.min()) / (guided.max() - guided.min())
                masks[j] = torch.clamp(masks[j], 0, 1)
                grads.append(guided.squeeze())
            grads = torch.stack(grads).unsqueeze(1)

            attentionLoss = (
                F.mse_loss(grads[masks == 1], masks[masks == 1]) * 0.1
                + F.mse_loss(grads, masks) * 0.9
            )
            predictionLoss = criterion(output, target)
            loss = predictionLoss + attentionLoss * (10 if attentionLoss > 0.1 else 1)
            optimizer.step()
            loss_meter.update(loss.item(), len(data))
        return loss_meter.avg

def test_loop(model):
        model.eval()
        classesAccuracy = {k: 0 for k in range(num_classes)}
        for batch_idx, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                out = model(data)
                output = F.softmax(out, dim=1)
                for x in range(output.shape[0]):
                    classesAccuracy[target[x].item()] += (
                        1 if torch.argmax(output[x]) == target[x] else 0
                    )
        accuracy = sum(classesAccuracy.values()) / len(test_loader.dataset)

        return accuracy
def save_model(model, optimizer, loss, accuracy, model_output_path):
        torch.save(
            {
                "model": model,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "accuracy": accuracy,
            },
            model_output_path,
        )
def main():
    batch_size = args.batch_size
    criterion = args.Loss
    num_classes = args.num_classes

    train_loader = DataLoader(PairedImageFolder(args.image_folder,args.mask_folder,image_size=(224, 224),
                normalize=True,), batch_size=batch_size, shuffle=True)
    
    test_Loader = DataLoader(
            datasets.ImageFolder(
                args.test_image_folder,
                transform=transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    if args.model == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    else:
        model = torch.load(args.model_path)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loop = tqdm.tqdm(range(250), leave=True)
    counter = 0
    bestAccuracy = 0
    bestEpoch = 0
    custom_loss_all = WeightedCustomLoss(gamma=0.0, weight_factor=10).to(device)
    early_stopping = EarlyStopping(name="output", patience=30, delta=0)
