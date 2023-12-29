import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from multiprocessing import freeze_support, current_process

processes_per_gpu = 2
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
        os.remove(f"error_{idx}_{today}.txt")
        elapsed = (time.time() - start)
        with open("finished_runs.txt", "a+") as fp:
            fp.write(f"{idx} -> {today} -> " + str(elapsed) + "s + " + command + "\n")
    except subprocess.CalledProcessError:
        with open(f"failed_runs_{today}.txt", "a+") as fp:
            fp.write(command + '\n')


def generate_runs():
    with open("experiments_to_run.txt", 'r') as f:
        runs = [line.rstrip('\n') for line in f]

    return [f"python main.py {i}" for i in runs]
    aux_save_dir = f" aux_save_dir={aux_save_dir}"
    f" dataset/train@train_dataset={dataset}"
    f" dataset/val@val_dataset={dataset}"
    f" train_dataset.subset={subset}"
    f" model={model}"
    f" model.parameters.num_classes={10 if dataset == 'cifar10' else 100}"
    f" optimizer={optimizer}"
    f" seed={seed}"
    f" epochs={final_epochs}"
    f" es_patience={final_es_patience}"
    f" val_every={final_val_every}"
    f" optimizer={optimizer}"
    f" train_dataset.batch_size={batch_size}"
    f" progress_bar=False"


if __name__ == "__main__":
    freeze_support()
    runs = generate_runs()
    print(len(runs))
    if last_index == -1 or last_index > len(runs):
        last_index = len(runs)

    with ProcessPoolExecutor(max_workers=gpu_count * processes_per_gpu) as executor:
        executor.map(run_command, [(runs[index], index) for index in range(run_index, last_index)])
