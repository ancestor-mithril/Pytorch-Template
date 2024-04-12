from __future__ import annotations

import logging
from typing import List, Tuple, Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import Tensor
from torchvision.transforms import v2
from .randaugment import RandAugment  # TODO: Use library


def init_transforms(transform: str) -> Tuple[List[TransformWrapper], List[TransformWrapper]]:
    transformations_config = OmegaConf.load(to_absolute_path(f'configs/transformations/{transform}.yaml'))

    cached_transforms, runtime_transforms = [], []
    for name, parameters in transformations_config.items():
        if name not in transformations:
            logging.error(f"Transformation {name} does not exist!")
            exit()

        apply_to = parameters.pop('apply_to') if 'apply_to' in parameters else None
        transform = transformations[name]["constructor"]
        transform = TransformWrapper(transform(**parameters), apply_to)

        if transformations[name]['cacheable']:
            cached_transforms.append(transform)
        else:
            runtime_transforms.append(transform)

    logging.info(f'Cached transforms: {cached_transforms}.')
    logging.info(f'Runtime transforms: {runtime_transforms}.')
    return cached_transforms, runtime_transforms


class TransformWrapper:
    def __init__(self, transform, apply_to: str | None = None):
        self.transform_name = f'{type(transform).__name__}({apply_to})'
        if apply_to is None:
            self.apply = lambda x: transform(x)
        elif apply_to == 'input':
            self.apply = lambda x: (transform(x[0]), x[1])
        elif apply_to == 'target':
            self.apply = lambda x: (x[0], transform(x[1]))
        else:
            raise ValueError("apply_to must be 'input', 'target' or None")

    def __call__(self, data):
        return self.apply(data)

    def __repr__(self):
        return self.transform_name


# TODO: Replace with library OneHot
class OneHot(object):
    def __init__(self, num_classes, on_value=1., off_value=0.):
        self.num_classes = num_classes
        self.on_value = on_value
        self.off_value = off_value

    def __call__(self, x):
        x = torch.LongTensor([x]).long().view(-1, 1)
        return torch.full((x.size(0), self.num_classes), self.off_value).scatter_(1, x, self.on_value).squeeze(0)


# TODO: Replace with torchvision.transforms.Resize or something else
class Resize:
    def __init__(self, size, interpolation):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, data: Tensor) -> Tensor:
        return torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, scale_factor=None,
                                               mode=self.interpolation, align_corners=None,
                                               recompute_scale_factor=None).squeeze(0)


class Unsqueeze:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.unsqueeze(self.dimension)


class Squeeze:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.squeeze(self.dimension)


class TensorType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.type(self.dtype)


class ToTensor:
    def __call__(self, tensor: Any) -> Tensor:
        return torch.tensor(tensor)


class MinMaxNormalization:
    def __call__(self, tensor: Tensor) -> Tensor:
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor


class MinMaxNormalizationCached:
    def __init__(self, maximum: float, minimum: float):
        self.minimum = minimum
        self.range = maximum - minimum

    def __call__(self, tensor: Tensor) -> Tensor:
        return (tensor - self.minimum) / self.range


class AutoAugment(v2.AutoAugment):
    def _get_policies(self, policy: str):
        if policy == 'cifar10':
            return super()._get_policies(v2.AutoAugmentPolicy.CIFAR10)
        if policy == 'imagenet':
            return super()._get_policies(v2.AutoAugmentPolicy.IMAGENET)
        raise super()._get_policies(policy)

# Removed:
# 1. LightingNoise
# 2. LambdaTransform => Not useful, create real transform from it
# 3. Identity => Not useful
# 4. RandomErasing => replaced with torchvision.v2.RandomErasing
# 5. ImageRandomResizedCrop => replaced with torchvision.v2.RandomResizedCrop
# 6. Half => we already have tensor type

# !!! cacheable augmentations are used first
transformations = {
    'ImageNormalize': {
        'constructor': v2.Normalize,
        'cacheable': False,
    },
    'ImageRandomCrop': {
        'constructor': v2.RandomCrop,
        'cacheable': False,
    },
    'ImageRandomResizedCrop': {
        'constructor': v2.RandomResizedCrop,
        'cacheable': False,
    },
    'RandomHorizontalFlip': {
        'constructor': v2.RandomHorizontalFlip,
        'cacheable': False,
    },
    'RandomVerticalFlip': {
        'constructor': v2.RandomVerticalFlip,
        'cacheable': False,
    },
    'ImageToTensor': {
        'constructor': v2.ToTensor,
        'cacheable': True,
    },
    'ToTensor': {
        'constructor': ToTensor,
        'cacheable': True,
    },
    'ImageRandomRotation': {
        'constructor': v2.RandomRotation,
        'cacheable': False,
    },
    'ImageColorJitter': {
        'constructor': v2.ColorJitter,
        'cacheable': False,
    },
    'ImageResize': {
        'constructor': v2.Resize,
        'cacheable': True,
    },
    'ImageCenterCrop': {
        'constructor': v2.CenterCrop,
        'cacheable': True,
    },
    'Resize': {
        'constructor': Resize,
        'cacheable': True,
    },
    'OneHot': {
        'constructor': OneHot,
        'cacheable': True,
    },
    'RandAugment': {
        'constructor': RandAugment,
        'cacheable': False,
    },
    'RandomErasing': {
        'constructor': v2.RandomErasing,
        'cacheable': False,
    },
    'Unsqueeze': {
        'constructor': Unsqueeze,
        'cacheable': True,
    },
    'Squeeze': {
        'constructor': Squeeze,
        'cacheable': True,
    },
    'TensorType': {
        'constructor': TensorType,
        'cacheable': True,
    },
    'MinMaxNormalization': {
        'constructor': MinMaxNormalization,
        'cacheable': True,
    },
    'MinMaxNormalizationCached': {
        'constructor': MinMaxNormalizationCached,
        'cacheable': True,
    },
    'AutoAugment': {
        'constructor': AutoAugment,
        'cacheable': False,
    },
}
