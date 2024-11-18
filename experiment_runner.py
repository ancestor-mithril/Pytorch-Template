import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import freeze_support, current_process


processes_per_gpu = 1
gpu_count = 1

run_index = 0
last_index = -1
if len(sys.argv) >= 2:
    run_index = int(sys.argv[1])
if len(sys.argv) >= 3:
    last_index = int(sys.argv[2])


def run_command(command_idx):
    command, idx = command_idx
    gpu_index = current_process()._identity[0] % gpu_count
    command += f' device=cuda:{gpu_index}'
    print("Command:", idx, "on gpu", gpu_index, "on process", current_process()._identity[0])
    today = date.today()
    os.makedirs('./logs', exist_ok=True)
    try:
        start = time.time()
        with open(f"./logs/error_{idx}_{today}.txt", 'a+') as err:
            subprocess.run(command, shell=True, check=True, stderr=err)
        os.remove(f"./logs/error_{idx}_{today}.txt")
        elapsed = (time.time() - start)
        with open("./logs/finished_runs.txt", "a+") as fp:
            fp.write(f"{idx} -> {today} -> " + str(elapsed) + "s + " + command + "\n")
    except subprocess.CalledProcessError:
        with open(f"./logs/failed_runs_{today}.txt", "a+") as fp:
            fp.write(command + '\n')


def create_run(dataset, subset, model, optimizer, seed, epochs, es_patience, val_every, batch_size, scheduler_params):
    if scheduler_params[0] in ('IncreaseBSOnPlateau', 'ReduceLROnPlateau'):
        scheduler, factor = scheduler_params
        aux_save_dir = f"{optimizer}/{batch_size}/{scheduler}/{factor}"
        scheduler_args = f"scheduler.{scheduler}.factor={factor}"
    elif scheduler_params[0] in ('StepBS', 'StepLR'):
        scheduler, step_size, gamma = scheduler_params
        aux_save_dir = f"{optimizer}/{batch_size}/{scheduler}/{step_size}_{gamma}"
        scheduler_args = f"scheduler.{scheduler}.step_size={step_size} scheduler.{scheduler}.gamma={gamma}"
    else:
        raise "Not implemented"

    return f" aux_save_dir={aux_save_dir}" \
           f" dataset/train@train_dataset={dataset}" \
           f" dataset/val@val_dataset={dataset}" \
           f" train_dataset.subset={subset}" \
           f" model={model}" \
           f" model.parameters.dataset={dataset}" \
           f" optimizer={optimizer}" \
           f" scheduler={scheduler}" \
           f" {scheduler_args}" \
           f" seed={seed}" \
           f" epochs={epochs}" \
           f" es_patience={es_patience}" \
           f" val_every={val_every}" \
           f" train_dataset.batch_size={batch_size}" \
           f" progress_bar=False"


def generate_runs():
    datasets = [
        'cifar10', 'cifar100'
    ]
    subsets = [
        0.0
    ]
    models = [
        'PreResNet20'
    ]
    optimizers = [
        'sgd'
    ]
    seeds = [
        2525
    ]
    epochss = [
        250
    ]
    es_patiences = [
        20
    ]
    val_everys = [
        1
    ]
    batch_sizes = [
        10, # 30, # 50
    ]
    schedulers = [
        # ('IncreaseBSOnPlateau', 1.5), ('IncreaseBSOnPlateau', 2.0),
        # ('IncreaseBSOnPlateau', 3.0), ('IncreaseBSOnPlateau', 5.0),
        #
        # ('ReduceLROnPlateau', 0.2), ('ReduceLROnPlateau', 0.33),
        # ('ReduceLROnPlateau', 0.5), ('ReduceLROnPlateau', 0.66),
        ('StepBS', 30, 2.0), ('StepBS', 30, 4.0),
        ('StepBS', 50, 2.0), ('StepBS', 50, 4.0),

        ('StepLR', 30, 0.5), ('StepLR', 30, 0.25),
        ('StepLR', 50, 0.5), ('StepLR', 50, 0.25),
    ]

    runs = []
    for dataset, subset, model, optimizer, seed, epochs, es_patience, val_every, batch_size, scheduler_params in \
            itertools.product(
                datasets, subsets, models, optimizers, seeds, epochss, es_patiences, val_everys, batch_sizes,
                schedulers):
        run = create_run(dataset=dataset, subset=subset, model=model, optimizer=optimizer, seed=seed, epochs=epochs,
                         es_patience=es_patience,
                         val_every=val_every, batch_size=batch_size, scheduler_params=scheduler_params)
        runs.append(run)

    return [f"python main.py {i}" for i in runs]


if __name__ == "__main__":
    freeze_support()
    runs = generate_runs()
    print(len(runs))
    if last_index == -1 or last_index > len(runs):
        last_index = len(runs)

    with open("./logs/finished_runs.txt", "a+") as fp:
        fp.write("New experiment: ")
        fp.write("\n")
    with ProcessPoolExecutor(max_workers=gpu_count * processes_per_gpu) as executor:
        executor.map(run_command, [(runs[index], index) for index in range(run_index, last_index)])
    with open("./logs/finished_runs.txt", "a+") as fp:
        fp.write("\n")
