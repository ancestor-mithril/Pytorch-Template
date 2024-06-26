import logging
from collections import OrderedDict
import timm
import torch
from omegaconf import OmegaConf
from torch import nn

import models


def load_external_models(model_name, parameters):
    model = None
    if model_name == 'ResNest':
        from resnest.torch.models.resnet import ResNet
        from resnest.torch.models.resnet import Bottleneck

        num_classes = parameters['num_classes']
        layers = parameters['layers']
        stem_width = parameters.get('stem_width', 32)
        model = ResNet(Bottleneck, layers,
                       num_classes=num_classes,
                       radix=2, groups=1, bottleneck_width=64,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False)
    return model


def enable_bn(model):
    model.train()


def disable_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._NormBase, nn.LayerNorm)):
            module.eval()


def init_model(model_args):
    parameters = OmegaConf.to_container(model_args.parameters, resolve=True)
    parameters = {k: v for k, v in parameters.items() if v is not None}

    model = load_external_models(model_args.name, parameters)
    logging.info(f"Loading model {model_args.name}, {'internal' if model is None else 'external'} implementation.")
    if model is None:

        try:
            model = getattr(models, model_args.name)
        except NameError:
            logging.error(f"Model {model_args.name} does not exist!")
            exit()

        model = model(**parameters)

    return model


def init_weights(model, initialization_type):
    match initialization_type:
        case 1:
            xavier_init(model)
        case 2:
            he_init(model)
        case 3:
            selu_init(model)
        case 4:
            orthogonal_init(model)
        case _:
            print(f"Unknown initialization type {initialization_type}")
    if hasattr(model, "init_weights"):
        model.init_weights()


def init_batch_norm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def load_model(model_config, model_path: str, model, device):
    logging.info("Loading model from " + model_path)

    if model_config.name == "Recorder":
        loaded_model = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()

        for key, value in loaded_model.items():
            if key.startswith('net.'):
                new_state_dict[key[4:]] = value

        model.vit.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def xavier_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform(
                m.weight, gain=nn.init.calculate_gain('relu'))


def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in')


def selu_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))


def orthogonal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
