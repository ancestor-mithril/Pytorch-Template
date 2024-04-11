import glob
import os
import zipfile

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def get_tensorboard_scalars(path):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _ = ea.Reload()
    return {scalar: tuple([x.value for x in ea.Scalars(scalar)]) for scalar in ea.Tags()["scalars"]}


def parse_dataset_name(path):
    dataset = "No dataset"
    if "CIFAR-10" in path:
        dataset = "CIFAR-10"
    if "CIFAR-100" in path:
        dataset = "CIFAR-100"
    return dataset


def parse_scheduler_type(path):
    scheduler = "No scheduler"
    if 'IncreaseBSOnPlateau' in path:
        scheduler = 'IncreaseBSOnPlateau'
    if 'ReduceLROnPlateau' in path:
        scheduler = 'ReduceLROnPlateau'
    if 'StepBS' in path:
        scheduler = 'StepBS'
    if 'StepLR' in path:
        scheduler = 'StepLR'

    split = path.split(scheduler)
    scheduler_param = split[1].lstrip(os.path.sep).split(os.path.sep)[0]
    initial_batch_size = split[0].rstrip(os.path.sep).split(os.path.sep)[-1]

    return scheduler, scheduler_param, initial_batch_size


def parse_tensorboard(path, epoch):
    scalars = get_tensorboard_scalars(path)
    dataset = parse_dataset_name(path)
    scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(path)

    epoch = min(epoch, len(scalars['Train/Time']))
    experiment_time = round(scalars['Train/Time'][epoch - 1] / 3600, 2)
    max_train_accuracy = round(max(scalars['Train/Accuracy'][:epoch]) * 100, 2)
    max_val_accuracy = round(max(scalars['Val/Accuracy'][:epoch]) * 100, 2)
    initial_learning_rate = round(scalars['Solver/Learning Rate'][0], 2)
    return {
        'dataset': dataset,
        'scheduler': scheduler,
        'scheduler_param': scheduler_param,
        'experiment_time': experiment_time,
        'experiment_times': scalars['Train/Time'][:epoch],
        'train_epochs': epoch,
        'max_train_accuracy': max_train_accuracy,
        'max_val_accuracy': max_val_accuracy,
        'initial_batch_size': initial_batch_size,
        'initial_learning_rate': initial_learning_rate,
        'train_accuracies': scalars['Train/Accuracy'][:epoch],
        'val_accuracies': scalars['Val/Accuracy'][:epoch],
        'batch_sizes': scalars['Solver/Batch Size'][:epoch],
        'learning_rates': scalars['Solver/Learning Rate'][:epoch],

    }


def get_tensorboard_paths(base_dir):
    return glob.glob(f"{base_dir}/**/events.out.tfevents*", recursive=True)


def match_paths_by_criteria(tb_paths):
    match_rules = {
        # Plateau
        'IncreaseBSOnPlateau': 'ReduceLROnPlateau',
        '1.5': '0.66',
        '2.0': '0.5',
        '3.0': '0.33',
        '5.0': '0.2',
        # Step
        'StepLR': 'StepBS',
        '30_2.0': '30_0.5',
        '50_2.0': '50_0.5',
        '30_4.0': '30_0.25',
        '50_4.0': '50_0.25',
    }
    match_rules.update({v: k for k, v in match_rules.items()})  # reversed rules

    groups = []
    while len(tb_paths):
        current_path = tb_paths.pop(0)
        dataset = parse_dataset_name(current_path)
        scheduler, scheduler_param, initial_batch_size = parse_scheduler_type(current_path)

        def is_ok(other_path):
            other_dataset = parse_dataset_name(other_path)
            other_scheduler, other_scheduler_param, other_initial_batch_size = parse_scheduler_type(other_path)
            if other_dataset != dataset or other_initial_batch_size != initial_batch_size:
                return False
            if other_scheduler != match_rules[scheduler] or other_scheduler_param != match_rules[scheduler_param]:
                return False
            return True

        matching = [x for x in tb_paths if is_ok(x)]

        if len(matching) == 0:
            print("No matching for", current_path)

        for x in matching:
            tb_paths.remove(x)

        groups.append((current_path, *matching))

    return groups


def create_tex(group_results, results_dir):
    scheduler_acronym = {
        'IncreaseBSOnPlateau': 'IBS',
        'ReduceLROnPlateau': 'RLR',
        'StepLR': 'StepLR',
        'StepBS': 'StepBS',
    }

    tex_file = os.path.join(results_dir, 'results_table.txt')
    if not os.path.exists(tex_file):
        open(tex_file, 'w').write(
            r'\begin{table}[]''\n'
            r'\resizebox{\textwidth}{!}{''\n'
            r'\begin{tabular}{|c|ccc|cc|c|}''\n'
            r'\hline''\n'
            r'Dataset & Scheduler & First LR & First BS & Train Acc. & Val Acc. & Time (h) \\ \hline''\n'
        )

    length = min([x['train_epochs'] for x in group_results])

    for result in group_results:
        scheduler_name = scheduler_acronym[result['scheduler']]
        experiment_time = round(result['experiment_times'][length - 1] / 3600, 2)
        open(tex_file, 'a').write(
            f"{result['dataset']} & "
            f"{scheduler_name}({result['scheduler_param'].replace('_', ',')}) & "
            f"{result['initial_learning_rate']} & "
            f"{result['initial_batch_size']} & "
            f"{result['max_train_accuracy']} & "
            f"{result['max_val_accuracy']} & "
            f"{experiment_time:.2f}"
            r'\\'
            '\n'
        )
    open(tex_file, 'a').write(r'\hline''\n')


def create_graphics(group_results, results_dir):
    exp_1, exp_2 = group_results

    colors = ['darkred', 'royalblue', 'orange']
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2))

        length = min(exp_1['train_epochs'], exp_2['train_epochs'])

        def create_df(exp):
            def pad_tuple(x):
                size = length - len(x)
                if size >= 0:
                    return x + (x[-1],) * size
                return x[:size]

            return pd.DataFrame.from_dict({
                'epoch': tuple(range(length)),
                'Train Acc.': pad_tuple(exp['train_accuracies']),
                'Val Acc.': pad_tuple(exp['val_accuracies']),
                'Batch Size': pad_tuple(exp['batch_sizes']),
                'Learning Rate': pad_tuple(exp['learning_rates']),
            })

        df_1 = create_df(exp_1)
        df_2 = create_df(exp_2)

        # Train
        sns.lineplot(x="epoch", y="Train Acc.", data=df_1, linewidth='1',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[0],
                     label=f"{exp_1['scheduler']}({exp_1['scheduler_param'].replace('_', ',')})")
        sns.lineplot(x="epoch", y="Train Acc.", data=df_2, linewidth='1',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[0],
                     label=f"{exp_2['scheduler']}({exp_2['scheduler_param'].replace('_', ',')})")
        axes[0].set_ylim(0.0, 1.1)
        sns.move_legend(axes[0], "upper left", bbox_to_anchor=(0.85, 1.2))

        # Val
        sns.lineplot(x="epoch", y="Val Acc.", data=df_1, linewidth='1',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[1])
        sns.lineplot(x="epoch", y="Val Acc.", data=df_2, linewidth='1',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[1])
        axes[1].set_ylim(0.0, 1.1)

        plt.savefig(os.path.join(results_dir, 'plots', f"{exp_1['scheduler']}_"
                                                       f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                                                       f"{exp_1['scheduler_param']}_"
                                                       f"{exp_1['initial_learning_rate']}_first.png"),
                    bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 / 2))
        # BS
        sns.lineplot(x="epoch", y="Batch Size", data=df_1, linewidth='1.5',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[0])
        sns.lineplot(x="epoch", y="Batch Size", data=df_2, linewidth='1.5',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[0])

        # LR
        sns.lineplot(x="epoch", y="Learning Rate", data=df_1, linewidth='1.5',
                     color=colors[0], linestyle='-', alpha=0.7, ax=axes[1])
        sns.lineplot(x="epoch", y="Learning Rate", data=df_2, linewidth='1.5',
                     color=colors[1], linestyle='-', alpha=0.7, ax=axes[1])

        plt.savefig(os.path.join(results_dir, 'plots', f"{exp_1['scheduler']}_"
                                                       f"{exp_1['dataset']}_{exp_1['initial_batch_size']}_"
                                                       f"{exp_1['scheduler_param']}_"
                                                       f"{exp_1['initial_learning_rate']}_second.png"),
                    bbox_inches='tight')
        plt.close()


def tensorboard_summary(base_dir, results_dir, epoch, time):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    tb_paths = get_tensorboard_paths(base_dir)

    tex_file = os.path.join(results_dir, 'results_table.txt')
    if os.path.exists(tex_file):
        os.remove(tex_file)
    for group in match_paths_by_criteria(tb_paths):
        group_results = [parse_tensorboard(x, epoch) for x in group]
        create_tex(group_results, results_dir)
        create_graphics(group_results, results_dir)
    open(tex_file, 'a').write(
        r'\end{tabular}''\n'
        r'}''\n'
        r'\end{table}''\n'
    )


if __name__ == '__main__':
    tensorboard_summary(r"C:\Users\GeorgeS\Documents\facultate\master\projects\Pytorch-Template\results_step",
                        'Graphics', epoch=250, time='epoch')
