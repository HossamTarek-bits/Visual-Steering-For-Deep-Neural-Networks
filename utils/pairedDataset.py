import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import (
    to_tensor,
    to_pil_image,
    hflip,
    vflip,
    rotate,
    normalize,
    to_grayscale,
    center_crop,
    gaussian_blur,
    solarize,
    posterize,
    perspective,
    adjust_hue,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_gamma,
    adjust_sharpness,
)


class PairedImageFolder(Dataset):
    def __init__(
        self,
        image_folder,
        mask_folder,
        rotation_degrees=30,
        random_rotation=False,
        random_rotation_p=0.5,
        random_horizontal_flip=False,
        random_horizontal_flip_p=0.5,
        random_vertical_flip=False,
        random_vertical_flip_p=0.5,
        image_size=(224, 224),
        crop_size=(224, 224),
        grayscale=False,
        normalize=False,
        num_output_channels=3,
        center_crop=False,
        gaussian_blur=False,
        solarize=False,
        posterize=False,
        perspective=False,
        adjust_hue=False,
        adjust_brightness=False,
        adjust_contrast=False,
        adjust_saturation=False,
        adjust_gamma=False,
        adjust_sharpness=False,
        random_perspective_p=0.5,
        noise_random_sigma=False,
        noise=False,
        noise_p=0.5,
        noise_sigma=25.0,
    ):
        self.images_dataset = ImageFolder(image_folder)
        self.masks_dataset = ImageFolder(mask_folder)
        self.rotation_degrees = rotation_degrees
        self.random_rotation = random_rotation
        self.random_rotation_p = random_rotation_p
        self.random_horizontal_flip = random_horizontal_flip
        self.random_horizontal_flip_p = random_horizontal_flip_p
        self.random_vertical_flip = random_vertical_flip
        self.random_vertical_flip_p = random_vertical_flip_p
        self.image_size = image_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.num_output_channels = num_output_channels
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.gaussian_blur = gaussian_blur
        self.solarize = solarize
        self.posterize = posterize
        self.perspective = perspective
        self.adjust_hue = adjust_hue
        self.adjust_brightness = adjust_brightness
        self.adjust_contrast = adjust_contrast
        self.adjust_saturation = adjust_saturation
        self.adjust_gamma = adjust_gamma
        self.adjust_sharpness = adjust_sharpness

        self.random_perspective_p = random_perspective_p
        self.noise = noise
        self.noise_p = noise_p
        self.noise_sigma = noise_sigma
        self.noise_random_sigma = noise_random_sigma
        assert len(self.images_dataset) == len(
            self.masks_dataset
        ), "Number of images and masks must be the same"

    def get_random_perspective(self, image, mask, p=0.5):
        width, height = image.size()[1:]
        startpoints, endpoints = [], []
        startpoints.append((0, 0))
        startpoints.append((width, 0))
        startpoints.append((width, height))
        startpoints.append((0, height))
        if random.random() < p:
            endpoints.append(
                (random.randint(0, width // 4), random.randint(0, height // 4))
            )
            endpoints.append(
                (
                    random.randint(3 * width // 4, width),
                    random.randint(0, height // 4),
                )
            )
            endpoints.append(
                (
                    random.randint(3 * width // 4, width),
                    random.randint(3 * height // 4, height),
                )
            )
            endpoints.append(
                (
                    random.randint(0, width // 4),
                    random.randint(3 * height // 4, height),
                )
            )
        else:
            endpoints.append((0, 0))
            endpoints.append((width, 0))
            endpoints.append((width, height))
            endpoints.append((0, height))
        startpoints = [list(map(int, p)) for p in startpoints]
        endpoints = [list(map(int, p)) for p in endpoints]
        image = perspective(image, startpoints, endpoints)
        mask = perspective(mask, startpoints, endpoints)
        return image, mask

    def gauss_noise_tensor(self, image, sigma=25.0):
        assert isinstance(image, torch.Tensor)
        dtype = image.dtype
        if not image.is_floating_point():
            image = image.to(torch.float32)

        out = image + sigma * torch.randn_like(image)

        if out.dtype != dtype:
            out = out.to(dtype)

        return out

    def __getitem__(self, index):
        # Load image, mask, and label
        image, label = self.images_dataset[index]
        mask, _ = self.masks_dataset[index]

        # Resize image and mask to the same size
        image = image.resize(self.image_size, resample=Image.BILINEAR)
        mask = mask.resize(self.image_size, resample=Image.BILINEAR)
        if self.grayscale:
            image = to_grayscale(image, num_output_channels=self.num_output_channels)
        image = to_tensor(image) if type(image) != torch.Tensor else image
        if self.center_crop:
            image = center_crop(image, self.crop_size)
            mask = center_crop(mask, self.crop_size)

        if self.gaussian_blur:
            image = gaussian_blur(
                image, kernel_size=random.randint(1, 5) // 2 * 2 + 1  # type: ignore
            )  # odd number between 1 and 5
        if self.solarize:
            image = solarize(image, random.randint(0, image.size()[0]))
        if self.posterize:
            image = posterize(image, random.randint(4, 8))
        # print(image.dtype, mask.dtype)

        if self.adjust_hue:
            image = adjust_hue(image, random.uniform(-0.5, 0.5))
        if self.adjust_brightness:
            image = adjust_brightness(image, random.uniform(0.8, 1.2))
        if self.adjust_contrast:
            image = adjust_contrast(image, random.uniform(0.8, 1.2))
        if self.adjust_saturation:
            image = adjust_saturation(image, random.uniform(0.8, 1.2))
        if self.adjust_gamma:
            image = adjust_gamma(image, random.uniform(0.8, 1.2))
        if self.adjust_sharpness:
            image = adjust_sharpness(image, random.uniform(0.8, 1.2))
        # Convert image and mask to tensors

        if self.noise and random.random() < self.noise_p:
            image = self.gauss_noise_tensor(
                to_tensor(image) if type(image) != torch.Tensor else image,
                sigma=random.random() * self.noise_sigma
                if self.noise_random_sigma
                else self.noise_sigma,
            )
        # Horizontal flip
        if (
            self.random_horizontal_flip
            and random.random() < self.random_horizontal_flip_p
        ):
            image = hflip(image)
            mask = hflip(mask)

        # Vertical flip
        if self.random_vertical_flip and random.random() < self.random_vertical_flip_p:
            image = vflip(image)
            mask = vflip(mask)

        # Rotation
        if self.random_rotation and random.random() < self.random_rotation_p:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image = rotate(image, angle)
            mask = rotate(mask, angle)
        if self.perspective:
            image, mask = self.get_random_perspective(
                image if type(image) == torch.Tensor else to_tensor(image),
                mask if type(mask) == torch.Tensor else to_tensor(mask),
                p=self.random_perspective_p if self.perspective else 0.0,
            )
        if self.normalize:
            image = normalize(
                to_tensor(image) if type(image) != torch.Tensor else image,
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010],
            )

        if type(mask) != torch.Tensor:
            mask = to_tensor(mask)
        if type(image) != torch.Tensor:
            image = to_tensor(image)
        return image, mask, label

    def __len__(self):
        return len(self.images_dataset)
