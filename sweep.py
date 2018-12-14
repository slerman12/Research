import argparse
import json
import os
import subprocess
import time
import numpy as np
import plotly
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
#SBATCH -t 2-00:00:00 -o {}/{}.%a.{} -J run_{}
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
    for param_set in range(len(sweep)):
        results = []
        for r in range(args.num_runs):
            with open("{}/{}.{}.{}".format(path + "/eval", log_name, param_set + 1, r + 1)) as f:
                line = f.readlines()
                line = line[-1]
                print(param_set + 1, r + 1)
                print([i for i in line.split(' ')])
                results.append([float(i) for i in line.split(' ')[:-1]])

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
    return stats


def graph_babi(data):
    main_groups = {'Distributional Sampled - Salience': lambda param: "-distributional False" not in param and "-sample False"
                                                           not in param and "-uniform_sample True" not in param,
                   'Distributional Deterministic - Salience': lambda param: "-sample False" in param,
                   "Distributional Sampled - Uniform": lambda param: "-uniform_sample True" in param,
                   "Standard, Not Distributional": lambda param: "-distributional False" in param}

    best_performing_per_group = {g: 0 for g in main_groups}

    for param_set in data:
        for group in main_groups:
            if main_groups[group](data[param_set]["params"]):
                if data[param_set]["all_tasks_mean"] > best_performing_per_group[group]:
                    best_performing_per_group[group] = data[param_set]["all_tasks_mean"]

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
        title='Performance Per Method',
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

    plotly.offline.plot(fig, filename='bAbI_salience_sampling_bar_chart.html',
                        image='jpeg', image_filename='bAbI_salience_sampling_bar_chart',
                        image_height=700, image_width=1300)


if args.mode == "eval" or args.mode == "eval_and_graph":
    with open(stats_file_name, "w") as file:
        file.write(json.dumps(evaluate_babi()))

if args.mode == "graph" or args.mode == "eval_and_graph":
    with open(stats_file_name) as f:
        data = json.load(f)
        graph_babi(data)
