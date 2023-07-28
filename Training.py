from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import captum
import tqdm
from utils import (
    PairedImageFolder,
    AverageMeter,
    seed_everything,
    get_loss,
    get_optimizer,
    losses,
    optimizers,
    models,
    get_model_last_conv,
    change_linear_layer,
    get_model_last_linear,
)
import argparse
import os

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("-b", "--batch_size", default=32, type=int)
parser.add_argument("-e", "--epochs", default=100, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument(
    "-m", "--model", default="resnet50", type=str, choices=models.keys()
)
parser.add_argument("--pretrained", default=True, type=bool)
parser.add_argument("--pretrained_weights_path", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("-s", "--seed", default=42, type=int)
parser.add_argument(
    "-l", "--loss", default="cross_entropy", type=str, choices=losses.keys()
)
parser.add_argument(
    "-o", "--optimizer", default="sgd", type=str, choices=optimizers.keys()
)
parser.add_argument("--class_weights", nargs="+", type=float)
parser.add_argument("-n", "--num_classes", default=10, type=int)
parser.add_argument("--gamma", default=10, type=float)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 5e-4)",
)
parser.add_argument(
    "--image_folder",
    default="imagenette2/Images/Train",
    type=str,
    metavar="PATH",
    help="path to train images folder",
)
parser.add_argument(
    "--mask_folder",
    default="imagenette2/Masks/Train",
    type=str,
    metavar="PATH",
    help="path to train masks folder",
)
parser.add_argument(
    "--image_size",
    default=(224, 224),
    type=tuple,
    metavar="N",
    help="image size (default: (224, 224))",
)
parser.add_argument(
    "--test_image_folder",
    default="imagenette2All/val",
    type=str,
    metavar="PATH",
    help="path to test images folder",
)
parser.add_argument(
    "--out_path",
    default="results",
    type=str,
    metavar="PATH",
    help="path to output model folder",
)

parser.add_argument(
    "--gpu_id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

use_cuda = torch.cuda.is_available()

device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")

seed_everything(args.seed)


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    last_conv_layer: nn.Module,
    gamma: float,
) -> float:
    """
    Trains the model on the training dataset and returns the average loss.

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training dataset.
        optimizer: The optimizer to use for training.
        criterion: The loss function to use for training.
        last_conv_layer: The last convolutional layer of the model.
        gamma: The weight of the attention loss.

    Returns:
        The average loss of the model on the training dataset.
    """
    model.train()
    loss_meter = AverageMeter()
    # Create the Grad-CAM attribution object
    grad_cam = captum.attr.LayerGradCam(model, last_conv_layer)
    for batch_idx, (data, masks, target) in enumerate(
        tqdm.tqdm(train_loader, leave=False, desc="Training")
    ):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        masks = masks.mean(dim=1).unsqueeze(1).to(device)
        output = model(data)
        # Generate the Grad-CAM attribution
        grad_cam_attribution = captum.attr.LayerAttribution.interpolate(
            grad_cam.attribute(data, target=target, relu_attributions=True),  # type: ignore
            interpolate_dims=args.image_size,
            interpolate_mode="bicubic",
        )
        attribution = []
        for j in range(data.shape[0]):
            # average the attribution channel-wise
            guided = torch.clamp(grad_cam_attribution[j].mean(dim=0), 0)
            # if the attribution is 0 everywhere, just set it to 0
            if guided.max() - guided.min() == 0:
                guided = torch.clamp(guided, 0, 1)
            else:
                # normalize the attribution
                guided = (guided - guided.min()) / (guided.max() - guided.min())
            # clamp the mask to 0-1
            masks[j] = torch.clamp(masks[j], 0, 1)
            # add the attribution to the list
            attribution.append(guided.squeeze())
        # stack the attributions
        attribution = torch.stack(attribution).unsqueeze(1)

        # calculate the attention loss
        if torch.all(masks == 0):
            attentionLoss = F.mse_loss(attribution, masks)
        else:
            attentionLoss = (
                F.mse_loss(attribution[masks == 1], masks[masks == 1]) * 0.1
                + F.mse_loss(attribution, masks) * 0.9
            )
        # calculate the prediction loss
        predictionLoss = criterion(output, target)
        loss = predictionLoss + attentionLoss * gamma
        optimizer.step()
        loss_meter.update(loss.item(), len(data))
    return loss_meter.avg


def test_loop(model: nn.Module, test_loader: DataLoader, num_classes: int) -> float:
    """
    Tests the model on the test dataset and returns the accuracy.

    Args:
        model: The model to be tested.
        test_loader: The data loader for the test dataset.
        num_classes: The number of classes in the dataset.

    Returns:
        The accuracy of the model on the test dataset.
    """
    model.eval()
    classesAccuracy = {k: 0 for k in range(num_classes)}
    for batch_idx, (data, target) in enumerate(
        tqdm.tqdm(test_loader, leave=False, desc="Testing")
    ):
        with torch.no_grad():
            # move data to device
            data, target = data.to(device), target.to(device)
            out = model(data)
            output = F.softmax(out, dim=1)
            for x in range(output.shape[0]):
                classesAccuracy[target[x].item()] += (
                    1 if torch.argmax(output[x]) == target[x] else 0
                )
    accuracy = (
        sum(classesAccuracy.values()) / len(test_loader.dataset)  # type: ignore
        if len(test_loader.dataset) > 0  # type: ignore
        else 0
    )
    return accuracy


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    accuracy: float,
    model_output_path: str,
) -> None:
    """
    Saves the model, optimizer state dictionary, loss and accuracy to the specified path.

    Args:
        model: The model to be saved.
        optimizer: The optimizer to be saved.
        loss: The loss value to be saved.
        accuracy: The accuracy value to be saved.
        model_output_path: The path where the model will be saved.

    Returns:
        None
    """
    try:
        if os.path.exists(args.out_path) is False:
            os.makedirs(args.out_path)
    except Exception as e:
        print(f"Error creating directory: {e}")
    try:
        torch.save(
            {
                "model": model,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "accuracy": accuracy,
            },
            model_output_path,
        )
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    """
    Trains a deep neural network model using the specified arguments and saves the best model based on accuracy.

    Args:
        None

    Returns:
        None
    """
    batch_size = args.batch_size
    criterion = args.loss
    class_weights = args.class_weights
    if class_weights is not None:
        class_weights = torch.tensor(class_weights).to(device)
    criterion = get_loss(criterion, weight=class_weights)

    num_classes = args.num_classes
    img_size = args.image_size
    train_loader = DataLoader(
        PairedImageFolder(
            args.image_folder,
            args.mask_folder,
            image_size=img_size,
            normalize=True,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_Loader = DataLoader(
        datasets.ImageFolder(
            args.test_image_folder,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
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

    if args.model_path is not None:
        model = torch.load(args.model_path)
        if type(model) is dict:
            model = model["model"]
    elif args.pretrained_weights_path is not None:
        model_details = models[args.model]
        model = model_details["model"]()
        if num_classes != get_model_last_linear(model)[0].out_features:
            change_linear_layer(model, num_classes)
        model.load_state_dict(torch.load(args.pretrained_weights_path))
    elif args.pretrained:
        model_details = models[args.model]
        model = model_details["model"](weights=model_details["pretrained_weights"])
        change_linear_layer(model, num_classes)
    else:
        model_details = models[args.model]
        model = model_details["model"](num_classes=num_classes)
    model.to(device)
    last_conv_layer = get_model_last_conv(model)
    if last_conv_layer is None:
        raise ValueError("No conv layer found in the model")
    optimizer = args.optimizer
    kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.optimizer in ["sgd", "rmsprop"]:
        kwargs["momentum"] = args.momentum
    optimizer = get_optimizer(
        optimizer,
        model.parameters(),
        **kwargs,
    )
    loop = tqdm.tqdm(range(args.epochs), leave=True)
    best_accuracy = 0
    for i in loop:
        loss = training_loop(
            model,
            train_loader,
            optimizer,
            criterion,
            last_conv_layer,
            args.gamma,
        )
        accuracy = test_loop(model, test_Loader, num_classes)
        loop.set_description(
            f"Epoch [{i+1}/{args.epochs}] Loss: {loss:.4f} Accuracy: {accuracy:.4f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(
                model,
                optimizer,
                loss,
                accuracy,
                os.path.join(args.out_path, f"{args.model}_best.pth"),
            )


if __name__ == "__main__":
    main()
