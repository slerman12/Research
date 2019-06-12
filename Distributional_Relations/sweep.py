import argparse
import copy
import json
import os
import subprocess
import time
import numpy as np
import plotly.offline as plotly
import plotly.graph_objs as go


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default="sweep")
parser.add_argument('-program', type=str, default="Run.py")
parser.add_argument('-num_runs', type=int, default=10)
parser.add_argument('-num_epochs', type=int, default=100000)
parser.add_argument('-test_sweep', type=str2bool, default=False)
args = parser.parse_args()

sweep = []
# sweep.extend([{"distributional": True, "top_k": top_k, "sample": sam, "aggregate_method": agg, "slurm": True}
#               for top_k in range(1, 21) for sam in [True, False] for agg in ["max", "mean", "concat"]])
# sweep.extend([{"distributional": True, "top_k": top_k, "uniform_sample": True, "aggregate_method": agg, "slurm": True}
#               for top_k in range(1, 21) for agg in ["max", "mean", "concat"]])
# sweep.extend([{"distributional": False, "top_k": top_k, "aggregate_method": agg, "slurm": True}
#               for top_k in range(1, 21) for agg in ["max", "mean", "concat"]])

sweep.extend([{"top_k": top_k, "aggregate_method": "concat", "sample": sam, "slurm": True}
              for top_k in [1, 2, 3, 5, 7, 10, 15] for sam in [True, False]])
sweep.extend([{"top_k": top_k, "uniform_sample": True, "aggregate_method": "concat", "slurm": True}
              for top_k in [1, 2, 3, 5, 7, 10, 15]])
sweep.extend([{"distributional": False, "aggregate_method": agg, "slurm": True}
              for agg in ["max", "mean", "concat"]])
sweep.extend([{"top_k": top_k, "slurm": True}
              for top_k in [1, 2, 3, 5, 7, 10, 15]])
sweep.extend([{"top_k": top_k, "uniform_sample": True, "slurm": True}
              for top_k in [1, 2, 3, 5, 7, 10, 15]])  # might go up
sweep.extend([{"top_k": top_k, "sample": False, "slurm": True}
              for top_k in [1, 2, 3, 5, 7, 10, 15]])
for x in sweep:
    if args.test_sweep:
        x["epochs"] = 1
    else:
        x["epochs"] = args.num_epochs

log_name = "log"
stats_file_name = "stats"
path = os.getcwd()

# Whether to run or just evaluate
if args.mode == "sweep":
    slurm_script_name = "sweep"
    in_file_name = "in"

    open(in_file_name, 'w').close()
    open(slurm_script_name, 'w').close()

    for i, parameterization in enumerate(sweep):
        with open(in_file_name, "a") as file:
            params = ""
            for key in parameterization:
                params += '-{} {} '.format(key, parameterization[key])
            params += '-{} {} '.format("name", i)
            file.write(params + '\n')

    if not os.path.exists(path + "/eval"):
        os.makedirs(path + "/eval")


    def slurm_script(n):
        return r"""#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00 -o {}/{}.%a.{} -J run_{}
#SBATCH --mem=50gb 
#SBATCH --array=1-{}
module load anaconda3/5.2.0b
python {} -name_suffix {} `awk "NR==$SLURM_ARRAY_TASK_ID" {}`
""".format(path + "/eval", log_name, n + 1, n + 1, len(sweep), args.program, n + 1, in_file_name)


    # Create a job for each run, each consisting of all of the params (e.g. for mean and st.d)
    processes = []
    for run in range(args.num_runs):
        with open(slurm_script_name, "w") as file:
            file.write(slurm_script(run))
        # processes.append(subprocess.call(['sbatch', slurm_script_name], shell=True))
        subprocess.call(['sbatch {}'.format(slurm_script_name)], shell=True)
        time.sleep(1)
    # wait = [p.wait() for p in processes]


def evaluate_babi():
    stats = {}
    valids_stats = {}
    for param_set in range(len(sweep)):
        results = []
        valids = []
        for r in range(args.num_runs):
            with open("{}/{}.{}.{}".format(path + "/eval", log_name, param_set + 1, r + 1)) as f:
                liness = f.readlines()
                line = liness[-2]
                print(param_set + 1, r + 1)
                print([i for i in line.split(' ')])
                results.append([float(i) for i in line.split(' ')[:-1]])

                line = liness[-1]
                print(param_set + 1, r + 1)
                print([i for i in line.split(' ')])
                valids.append([float(i) for i in line.split(' ')[:-1]])

        stats["param_set_{}".format(param_set + 1)] = {}
        for task in range(20):
            stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)] = {}
            stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)]["mean"] = \
                np.mean([result[task] for result in results])
            stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)]["std"] = \
                np.std([result[task] for result in results])
            stats["param_set_{}".format(param_set + 1)]["all_tasks_mean"] = \
                np.mean([v for result in results for v in result])
            stats["param_set_{}".format(param_set + 1)]["all_tasks_std"] = \
                np.std([v for result in results for v in result])
            with open("in") as f:
                stats["param_set_{}".format(param_set + 1)]["params"] = f.readlines()[param_set]

        valids_stats["param_set_{}".format(param_set + 1)] = {}
        for task in range(20):
            valids_stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)] = {}
            valids_stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)]["mean"] = \
                np.mean([result[task] for result in valids])
            valids_stats["param_set_{}".format(param_set + 1)]["task_{}".format(task + 1)]["std"] = \
                np.std([result[task] for result in valids])
            valids_stats["param_set_{}".format(param_set + 1)]["all_tasks_mean"] = \
                np.mean([v for result in valids for v in result])
            valids_stats["param_set_{}".format(param_set + 1)]["all_tasks_std"] = \
                np.std([v for result in valids for v in result])
            with open("in") as f:
                valids_stats["param_set_{}".format(param_set + 1)]["params"] = f.readlines()[param_set]
    return stats, valids_stats


def graph_babi(data, data_valids):
    main_groups = {
        'Distributional Sampled - Salience': lambda param: "-distributional False" not in param and "-sample False"
                                                           not in param and "-uniform_sample True" not in param,
        'Distributional Deterministic - Salience': lambda param: "-sample False" in param,
        "Distributional Sampled - Uniform": lambda param: "-uniform_sample True" in param,
        "Standard, Not Distributional": lambda param: "-distributional False" in param and "-aggregate_method mean"
                                                      not in param}  # Can make bold with "<b></b>

    best_performing_per_group = {g: 0 for g in main_groups}
    best_performing_per_group_valid = {g: 0 for g in main_groups}

    for param_set in data:
        for group in main_groups:
            if main_groups[group](data[param_set]["params"]):
                if data_valids[param_set]["all_tasks_mean"] > best_performing_per_group_valid[group]:
                    best_performing_per_group[group] = data[param_set]["all_tasks_mean"]
                    best_performing_per_group_valid[group] = data_valids[param_set]["all_tasks_mean"]

    # TODO: label with +- std and maybe top k

    layout = go.Layout(
        xaxis=dict(
            tickfont=dict(
                size=15,
                color='rgb(0, 0, 0)'
            ),
            showticklabels=False
        ),
        yaxis=dict(range=[.85, .95]),
        # title='Best Performance Per Method',
        # legend=dict(orientation="h"),
        legend=dict(x=.6, y=1),
        font=dict(size=22)
    )
    # layout = go.Layout(
    #     # xaxis=dict(
    #     #     tickfont=dict(
    #     #         size=15,
    #     #         family='Arial',
    #     #         color='rgb(0, 0, 0)'
    #     #     )
    #     # ),
    #     yaxis=dict(range=[.85, .95]),
    #     title='Performance Per Method',
    #     font=dict(size=17, family="Arial")
    # )
    graph_data = [go.Bar(
        x=[key for key in best_performing_per_group],
        y=[best_performing_per_group[key] for key in best_performing_per_group],
        text=["{:.3%}".format(best_performing_per_group[key]) for key in best_performing_per_group],
        textposition='auto'
    )]

    graph_data = [go.Bar(
        name=key,
        x=[key],
        y=[best_performing_per_group[key]],
        text=["{:.3%}".format(best_performing_per_group[key])],
        textposition='auto'
    ) for key in best_performing_per_group]

    fig = go.Figure(data=graph_data, layout=layout)

    plotly.plot(fig, filename='bAbI_salience_sampling_bar_chart.html',
                image='png', image_filename='bAbI_salience_sampling_bar_chart',
                image_height=800, image_width=1300)
    return

    distributional_groups = {'Sampled<br>- Salience<br>- Concat': lambda param: "-distributional False" not in param and
                                                                                "-sample False" not in param and
                                                                                "-uniform_sample True" not in param and
                                                                                "-aggregate_method concat" in param,
                             'Deterministic<br>- Salience<br>- Concat': lambda param: "-sample False" in param and
                                                                                      "-aggregate_method concat" in param,
                             "Sampled<br>- Uniform<br>- Concat": lambda param: "-uniform_sample True" in param and
                                                                               "-aggregate_method concat" in param,
                             'Sampled<br>- Salience<br>- Max': lambda param: "-distributional False" not in param and
                                                                             "-sample False" not in param and
                                                                             "-uniform_sample True" not in param and
                                                                             ("-aggregate_method max" in param or
                                                                              "-aggregate_method" not in param),
                             'Deterministic<br>- Salience<br>- Max': lambda param: "-sample False" in param and
                                                                                   ("-aggregate_method max" in param or
                                                                                    "-aggregate_method" not in param),
                             "Sampled<br>- Uniform<br>- Max": lambda param: "-uniform_sample True" in param and
                                                                            ("-aggregate_method max" in param or
                                                                             "-aggregate_method" not in param)
                             }  # Em dash: &#8212; Bullet: &#8226;

    performance_for_each_k = {g: {} for g in distributional_groups}

    for param_set in data:
        for group in distributional_groups:
            if distributional_groups[group](data[param_set]["params"]):
                k = int(data[param_set]["params"].split("-top_k ")[1].split(' ')[0])
                performance_for_each_k[group][k] = data[param_set]["all_tasks_mean"]

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    num_sampled_k = [1, 2, 3, 5, 7, 10, 15]

    trace0 = go.Table(
        header=dict(
            values=[['MODEL']] + [[str(k)] for k in num_sampled_k],
            line=dict(color='#506784'),
            fill=dict(color=headerColor),
            align=['left', 'center'],
            font=dict(color='white', size=27),
            height=38
        ),
        columnwidth=[11, 7],
        cells=dict(
            values=[["<b>{}</b>".format(g) for g in performance_for_each_k] + ['<b>AVERAGE</b>']] +
                   [["{:.1%}".format(performance_for_each_k[g][k])  # Can add tsd but idk: <br>&plusmn; {:.1%}
                     if k != num_sampled_k[int(np.argmax([performance_for_each_k[g][kk] for kk in num_sampled_k]))]
                     else "<b>{:.1%}</b>".format(performance_for_each_k[g][k])
                     for g in performance_for_each_k] +
                    ['{:.1%}'.format(np.mean([performance_for_each_k[g][k] for g in performance_for_each_k]))
                     if k != num_sampled_k[int(np.argmax([np.mean([performance_for_each_k[gr][kk] for gr in
                                                                   performance_for_each_k]) for kk in num_sampled_k]))]
                     else '<b>{:.1%}</b>'.format(np.mean([performance_for_each_k[g][k]
                                                          for g in performance_for_each_k]))]
                    for k in num_sampled_k],
            line=dict(color='#506784'),
            fill=dict(color=[rowEvenColor, rowOddColor]),
            align=['left', 'center'],
            font=dict(color=['#506784'], size=[26, 30], ),
            # height=50
        ))

    table_data = [trace0]

    layout = go.Layout(
        # title='Model vs. Num Sampled',
        font=dict(size=22)
    )

    fig = go.Figure(data=table_data, layout=layout)

    plotly.plot(fig, filename='bAbI_salience_sampling_table.html',
                image='png', image_filename='bAbI_salience_sampling_table_chart',
                # image_height=700, image_width=1000
                image_height=1000, image_width=1355
                )

    tasks = range(1, 21)

    all_groups = copy.deepcopy(distributional_groups)
    all_groups.update({"Standard<br>- Max": lambda param: "-distributional False" in param and
                                                          ("-aggregate_method max" in param or
                                                           "-aggregate_method" not in param),
                       "Standard<br>- Concat": lambda param: "-distributional False" in param and
                                                             "-aggregate_method concat" in param})
    best_performance_for_each_task = {g: {task: 0 for task in tasks} for g in all_groups}
    best_performance_for_each_task_valid = {g: {task: 0 for task in tasks} for g in all_groups}

    for param_set in data:
        for group in all_groups:
            if all_groups[group](data[param_set]["params"]):
                for task in tasks:
                    if data_valids[param_set]["task_{}".format(task)]["mean"] > best_performance_for_each_task_valid[group][task]:
                        best_performance_for_each_task[group][task] = data[param_set]["task_{}".format(task)]["mean"]
                        best_performance_for_each_task_valid[group][task] = data_valids[param_set]["task_{}".format(task)]["mean"]

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    trace0 = go.Table(
        header=dict(
            values=[['MODEL']] + [[str(task)] for task in tasks]
                   + [["AVERAGE"]]
            ,
            line=dict(color='#506784'),
            fill=dict(color=headerColor),
            align=['left', 'center'],
            font=dict(color='white', size=27),
            height=38
        ),
        columnwidth=[11, 7],
        cells=dict(
            values=[["<b>{}</b>".format(g) for g in best_performance_for_each_task]] +
                   [["{:.1%}".format(best_performance_for_each_task[g][task])  # Can add sd but idk: <br>&plusmn; {:.1%}
                     if g != list(all_groups.keys())[int(np.argmax([best_performance_for_each_task[gr][task]
                                                                    for gr in list(all_groups.keys())]))]
                     else "<b>{:.1%}</b>".format(best_performance_for_each_task[g][task])
                     for g in best_performance_for_each_task]
                    for task in tasks]
                   + [["<b>{:.1%}</b>".format(np.mean([best_performance_for_each_task[g][task]
                                                                       for task in tasks])) if np.mean([best_performance_for_each_task[g][task]
                                                                                                        for task in tasks]) == np.max([np.mean([best_performance_for_each_task[gg][ttask]
                                                                                                                                        for ttask in tasks]) for gg in best_performance_for_each_task])
                                           else "{:.1%}".format(np.mean([best_performance_for_each_task[g][task]
                                                                            for task in tasks]))
                                          for g in best_performance_for_each_task]]
            ,
            line=dict(color='#506784'),
            fill=dict(color=[rowEvenColor, rowOddColor]),
            align=['left', 'center'],
            font=dict(color=['#506784'], size=[26, 30], ),
            # height=50
        ))

    table_data = [trace0]

    layout = go.Layout(
        # title='Model vs. Task',
        font=dict(size=22),
    )

    fig = go.Figure(data=table_data, layout=layout)

    plotly.plot(fig, filename='bAbI_salience_sampling_tasks_table.html',
                image='png', image_filename='bAbI_salience_sampling_tasks_table',
                image_height=1500, image_width=3000,
                # image_height=700, image_width=2500
                )

    stand_groups = {"Max": lambda param: "-distributional False" in param and
                                         ("-aggregate_method max" in param or
                                          "-aggregate_method" not in param),
                    "Mean": lambda param: "-distributional False" in param and
                                          "-aggregate_method mean" in param,
                    "Concat": lambda param: "-distributional False" in param and
                                            "-aggregate_method concat" in param}
    dist_groups_copy = {key.replace("<br>-", ""): {} for key in distributional_groups}
    for param_set in data:
        for group in distributional_groups:
            if distributional_groups[group](data[param_set]["params"]):
                k = int(data[param_set]["params"].split("-top_k ")[1].split(' ')[0])
                dist_groups_copy[group.replace("<br>-", "")][k] = data[param_set]

    for param_set in data:
        for g in stand_groups:
            if callable(stand_groups[g]):
                if stand_groups[g](data[param_set]["params"]):
                    stand_groups[g] = data[param_set]

    trace0 = go.Table(
        header=dict(
            values=[['Aggregation']] + [[str(task)] for task in tasks],
            line=dict(color='#506784'),
            fill=dict(color=headerColor),
            align=['left', 'center'],
            font=dict(color='white', size=17)
        ),
        columnwidth=[10, 8],
        cells=dict(
            values=[["<b>{}</b>".format(g) for g in stand_groups]] +
                   [["{:.1%}<br>&plusmn; {:.1%}".format(stand_groups[group]["task_{}".format(task)]["mean"],
                                                        stand_groups[group]["task_{}".format(task)]["std"])
                     if group != list(stand_groups.keys())[
                       int(np.argmax([stand_groups[g]["task_{}".format(task)]["mean"] for g in stand_groups]))]
                     else "<b>{:.1%}<br>&plusmn; {:.1%}</b>".format(stand_groups[group]["task_{}".format(task)]["mean"],
                                                                    stand_groups[group]["task_{}".format(task)]["std"])
                     for group in stand_groups]
                    for task in tasks],
            line=dict(color='#506784'),
            fill=dict(color=[rowEvenColor, rowOddColor]),
            align=['left', 'center'],
            font=dict(color=['#506784'], size=[17, 20], ),
            # height=50
        ))

    table_data = [trace0]

    layout = go.Layout(
        title="Standard, Not Distributional",
        font=dict(size=22),
    )

    fig = go.Figure(data=table_data, layout=layout)

    plotly.plot(fig, filename='bAbI_salience_sampling_tasks_table_standard.html',
                image='png', image_filename='bAbI_salience_sampling_tasks_table_standard',
                image_height=700, image_width=2300,
                # image_height=700, image_width=2500
                )

    for param_set in data:
        for group in all_groups:
            if all_groups[group](data[param_set]["params"]):
                if group.replace("<br>-", "") in dist_groups_copy:
                    print("blaaaa")
                    group = group.replace("<br>-", "")
                    trace0 = go.Table(
                        header=dict(
                            values=[['Num Sampled']] + [[str(task)] for task in tasks],
                            line=dict(color='#506784'),
                            fill=dict(color=headerColor),
                            align=['left', 'center'],
                            font=dict(color='white', size=17)
                        ),
                        columnwidth=[11, 7],
                        cells=dict(
                            values=[["<b>{}</b>".format(k) for k in num_sampled_k]] +
                                   [["{:.1%}<br>&plusmn; {:.1%}".format(
                                       dist_groups_copy[group][k]["task_{}".format(task)]["mean"],
                                       dist_groups_copy[group][k]["task_{}".format(task)]["std"])
                                     if k != num_sampled_k[int(np.argmax(
                                       [dist_groups_copy[group][kk]["task_{}".format(task)]["mean"] for kk in
                                        num_sampled_k]))]
                                     else "<b>{:.1%}<br>&plusmn; {:.1%}</b>".format(
                                       dist_groups_copy[group][k]["task_{}".format(task)]["mean"],
                                       dist_groups_copy[group][k]["task_{}".format(task)]["std"])
                                     for k in num_sampled_k]
                                    for task in tasks],
                            line=dict(color='#506784'),
                            fill=dict(color=[rowEvenColor, rowOddColor]),
                            align=['left', 'center'],
                            font=dict(color=['#506784'], size=[17, 20], ),
                            # height=50
                        ))

                    table_data = [trace0]

                    layout = go.Layout(
                        title=group.replace("<br>-", ""),
                        font=dict(size=22),
                    )

                    fig = go.Figure(data=table_data, layout=layout)

                    plotly.plot(fig, filename='bAbI_salience_sampling_tasks_table_{}.html'.format(
                        group.replace("<br>-", "")),
                                image='png', image_filename='bAbI_salience_sampling_tasks_table_{}'.format(
                            group.replace("<br>- ", "_")),
                                image_height=700, image_width=2200,
                                # image_height=700, image_width=2500
                                )
                    del dist_groups_copy[group]


if args.mode == "eval" or args.mode == "eval_and_graph":
    stats, valids_stats = evaluate_babi()
    with open(stats_file_name, "w") as file:
        file.write(json.dumps(stats))
    with open(stats_file_name + "_valids", "w") as file:
        file.write(json.dumps(valids_stats))

if args.mode == "graph" or args.mode == "eval_and_graph":
    with open(stats_file_name) as f:
        data = json.load(f)
    with open(stats_file_name + "_valids") as g:
        data_valids = json.load(g)
    graph_babi(data, data_valids)
