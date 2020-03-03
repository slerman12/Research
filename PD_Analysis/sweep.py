import argparse
import json
import os
import subprocess
import time
import numpy as np
import plotly.offline as plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments TODO multiple runs but need to pass in run number so that saved models don't use each other's best
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default="sweep")
parser.add_argument('-program', type=str, default="runner.py")
parser.add_argument('-num_runs', type=int, default=3)
parser.add_argument('-num_epochs', type=int, default=5000) # TODO different num epochs for different models -- one to ones only need 40!
parser.add_argument('-test_sweep', type=str2bool, default=False)
args = parser.parse_args()

sweep = []

i = 0

for f_bl in [True, False]:
    for to_tspan in [None, [0, 0.5], [0, 1], [0, 1.5], [0, 2], [1.5, 2]]:
        for targ in ["UPDRS_III", "MSEADLG", "COGSTATE", "MCATOT"]:
            sweep.append(
                {"inference_type": "future_scores_one_to_one", "epochs": args.num_epochs if f_bl else 2000 if to_tspan is not None else 40, "learning_rate": 0.0001,
                 "name": i,
                 "data_file": "data/Processed/future_scores_{}_to_one_{}{}.csv".format("bl" if f_bl else "one",
                                                                                       "" if to_tspan is None else "timespan_" + str(
                                                                                           to_tspan[0]) + "_" + str(
                                                                                           to_tspan[1]) + "_", targ),
                 "classification_or_regression": "regression" if targ != "COGSTATE" else "classification", "slurm": True})
            i += 1

for f_bl in [True, False]:
    for c_or_r in [True, False]:
        for targ in ["UPDRS_III", "MSEADLG", "MCATOT"]:
            sweep.append(
                {"inference_type": "rates_one_to_one", "epochs": args.num_epochs if f_bl else 2000, "learning_rate": 0.0001,
                 "name": i,
                 "data_file": "data/Processed/rates_{}_to_one_{}_{}.csv".format("bl" if f_bl else "one",
                                                                                "classification" if c_or_r else "regression",
                                                                                targ),
                 "classification_or_regression": "classification" if c_or_r else "regression", "slurm": True})
            i += 1

for x in sweep:
    if args.test_sweep:
        x["epochs"] = 1
    else:
        x["epochs"] = args.num_epochs if "epochs" not in x else x["epochs"]

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


def evaluate():
    stats = {}
    for param_set in range(len(sweep)):
        results = []
        for r in range(args.num_runs):
            with open("{}/{}.{}.{}".format(path + "/eval", log_name, param_set + 1, r + 1)) as f:
                liness = f.readlines()
                line = liness[-1]
                print(param_set + 1, r + 1)
                print([i for i in line.split(' ')])
                results.append([float(i) for i in line.split(' ')])

        stats["param_set_{}".format(param_set + 1)] = {"loss": {}, "accuracy": {}}
        stats["param_set_{}".format(param_set + 1)]["loss"]["mean"] = \
            np.mean([result[0] for result in results])
        stats["param_set_{}".format(param_set + 1)]["loss"]["std"] = \
            np.std([result[0] for result in results])
        stats["param_set_{}".format(param_set + 1)]["accuracy"]["mean"] = \
            np.mean([result[1] for result in results])
        stats["param_set_{}".format(param_set + 1)]["accuracy"]["std"] = \
            np.std([result[1] for result in results])

        with open("in") as f:
            stats["param_set_{}".format(param_set + 1)]["params"] = f.readlines()[param_set]  # TODO why not just use sweep? Then no extracting anything later

    return stats


def graph(data):
    main_groups = {
        'UPDRS III Future Score': lambda param: "-inference_type future_scores_one_to_one" in param and
                                                "UPDRS_III" in param,
        'MSEADLG Future Score': lambda param: "-inference_type future_scores_one_to_one" in param and
                                              "MSEADLG" in param,
        'MCATOT Future Score': lambda param: "-inference_type future_scores_one_to_one" in param and
                                              "MCATOT" in param,
        'UPDRS III Rate Continuous Regression': lambda param: "-inference_type rates_one_to_one" in param and
                                                              "-classification_or_regression regression" in param and
                                                              "UPDRS_III" in param,
        'MSEADLG Rate Continuous Regression': lambda param: "-inference_type rates_one_to_one" in param and
                                                            "-classification_or_regression regression" in param and
                                                            "MSEADLG" in param,
        'MCATOT Rate Continuous Regression': lambda param: "-inference_type rates_one_to_one" in param and
                                                            "-classification_or_regression regression" in param and
                                                            "MCATOT" in param,
        "Categorical Classification": lambda param: "-classification_or_regression classification" in param or
                                                    "COGSTATE" in param}

    grouped_data = {g: {} for g in main_groups}

    for param_set in data:
        for group in main_groups:
            if main_groups[group](data[param_set]["params"]):
                grouped_data[group][param_set] = data[param_set]

                grouped_data[group][param_set]["generality"] = 0

                if "Classification" not in group or "COGSTATE" in grouped_data[group][param_set]["params"]:
                    if "bl" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 1
                    else:
                        grouped_data[group][param_set]["generality"] += 3
                    if "timespan" not in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 9
                    elif "1.5_2" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 8
                    elif "0_2" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 8
                    elif "0_1.75" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 7
                    elif "0_1.5" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 6
                    elif "0_1.25" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 5
                    elif "0_1" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 4
                    elif "0_0.75" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 3
                    elif "0_0.5" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 2
                    elif "0_0.25" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 1
                else:
                    if "bl" in grouped_data[group][param_set]["params"]:
                        grouped_data[group][param_set]["generality"] += 9
                    else:
                        grouped_data[group][param_set]["generality"] += 12

    # print(grouped_data[g].keys() for g in grouped_data)
    for group in grouped_data:
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{"type": "xy"}, {"type": "domain"}]],
                            column_widths=[6, 4])

        fig.add_trace(go.Scatter(
            x=[grouped_data[group][param_set]["generality"] for param_set in grouped_data[group]],
            y=[grouped_data[group][param_set]["loss" if "Classification" not in group else "accuracy"]["mean"] for
               param_set in grouped_data[group]],
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
            mode="markers+text",
            text=[ind for ind in range(len(grouped_data[group].keys()))],
            textposition="top center"
        ), row=1, col=1)

        fig.update_layout(
            xaxis_title="Generality (Lower - Higher)",
            yaxis_title="Performance {}".format("(MSE)" if "Classification" not in group else "(% Accuracy)"),
            yaxis_tickformat='%' if "Classification" in group else None
        )
        # TODO for classification, might be good to graph precisions in another color
        fig.add_trace(go.Table(header=dict(
            values=['ID', 'Method', "Performance {}".format("(MSE)" if "Classification" not in group else "(% Accuracy)"),
                    # "Generality"
                    ]),
            columnwidth=[5, 30, 11],
            cells=dict(values=[[ind for ind in range(len(grouped_data[group].keys()))],
                               ["Predicting {}from {} {}".format("" if "Classification" not in group else "UPDRS III Rate 'Slow' Vs. 'Fast' <br>" if "UPDRS_III" in grouped_data[group][param_set]["params"] else "MSEADLG Rate 'Slow' Vs. 'Fast' <br>" if "MSEADLG" in grouped_data[group][param_set]["params"] else "COGSTATE '1S', '2S', or '3S' <br>" if "COGSTATE" in grouped_data[group][param_set]["params"] else "MCATOT Rate 'Slow' Vs. 'Fast' <br>", "baseline" if "bl" in grouped_data[group][param_set]["params"] else "any time step", "to any future time step" if "timespan" not in grouped_data[group][param_set]["params"] and "rates" not in grouped_data[group][param_set]["params"] else "thereafter" if "rates" in grouped_data[group][param_set]["params"] else "to a future <br>timespan of up to 2 years" if "0_2" in grouped_data[group][param_set]["params"] else "to a future <br>timespan of up to 1.5 years" if "0_1.5" in grouped_data[group][param_set]["params"] else "to a future <br>timespan of up to 1 year" if "0_1" in grouped_data[group][param_set]["params"] else "to a future <br>timespan of up to 6 months" if "0_0.5" in grouped_data[group][param_set]["params"] else "to a future <br>timespan of between 1.5 and 2 years") for param_set in grouped_data[group]],
                               ["{0:.1%}<br>&plusmn; {1:.1%}".format(grouped_data[group][param_set]["loss" if "Classification" not in group else "accuracy"]["mean"], grouped_data[group][param_set]["loss" if "Classification" not in group else "accuracy"]["std"]) for param_set in grouped_data[group]] if "Classification" in group else ["{0:.2f}<br>&plusmn; {1:.2f}".format(grouped_data[group][param_set]["loss" if "Classification" not in group else "accuracy"]["mean"], grouped_data[group][param_set]["loss" if "Classification" not in group else "accuracy"]["std"]) for param_set in grouped_data[group]],
                               # [grouped_data[group][param_set]["generality"] for param_set in grouped_data[group]]
                               ],
                       height=70,
                       align='left')),
            row=1, col=2
        )

        fig.update_layout(
            title="Generality Vs. Performance Of {} Methods".format(group)
        )
        fig.update_xaxes(showticklabels=False)

        plotly.plot(fig, filename=group + '.html',
                    image='png', image_filename=group,
                    image_height=800 if len(list(grouped_data[group].keys())) < 10 else 1500, image_width=1400)


if args.mode == "eval" or args.mode == "eval_and_graph":
    stats = evaluate()
    with open(stats_file_name, "w") as file:
        file.write(json.dumps(stats))

if args.mode == "graph" or args.mode == "eval_and_graph":
    with open(stats_file_name) as f:
        data = json.load(f)
    graph(data)
