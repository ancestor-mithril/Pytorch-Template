import logging

from omegaconf import OmegaConf
from torch import optim
import bs_scheduler


def init_scheduler(scheduler_config, optimizer, dataloader):
    scheduler_config_list = list(scheduler_config.items())
    (name, parameters) = scheduler_config_list[1]
    if name not in schedulers:
        logging.error(f"Scheduler {name} does not exist!")
        exit()

    parameters = OmegaConf.to_container(parameters, resolve=True)
    parameters = {k: v for k, v in parameters.items() if v is not None}

    scheduler_type = scheduler_config_list[2][1]
    if scheduler_type == 'lr_scheduler':
        parameters["optimizer"] = optimizer
    elif scheduler_type == 'bs_scheduler':
        parameters["dataloader"] = dataloader
    else:
        raise NotImplementedError("not implemented yet")

    scheduler = schedulers[name](**parameters)
    return scheduler, name


class StaticScheduler(object):
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


schedulers = {
    'StepBS': bs_scheduler.StepBS,
    'MultiStepBS': bs_scheduler.MultiStepBS,
    'ConstantBS': bs_scheduler.ConstantBS,
    'LinearBS': bs_scheduler.LinearBS,
    'ExponentialBS': bs_scheduler.ExponentialBS,
    'PolynomialBS': bs_scheduler.PolynomialBS,
    'CosineAnnealingBS': bs_scheduler.CosineAnnealingBS,
    'IncreaseBSOnPlateau': bs_scheduler.IncreaseBSOnPlateau,
    'CyclicBS': bs_scheduler.CyclicBS,
    'CosineAnnealingBSWithWarmRestarts': bs_scheduler.CosineAnnealingBSWithWarmRestarts,
    'OneCycleBS': bs_scheduler.OneCycleBS,
    # From here on, learning rate
    'StepLR': optim.lr_scheduler.StepLR,
    'MultiStepLR': optim.lr_scheduler.MultiStepLR,
    'ConstantLR': optim.lr_scheduler.ConstantLR,
    'LinearLR': optim.lr_scheduler.LinearLR,
    'ExponentialLR': optim.lr_scheduler.ExponentialLR,
    'PolynomialLR': optim.lr_scheduler.PolynomialLR,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': optim.lr_scheduler.CyclicLR,
    'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'OneCycleLR': optim.lr_scheduler.OneCycleLR,
    # From here on, static
    'StaticScheduler': StaticScheduler
}
