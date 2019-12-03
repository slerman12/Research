import argparse
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
parser.add_argument('-program', type=str, default="runner.py")
parser.add_argument('-num_runs', type=int, default=3)
parser.add_argument('-num_epochs', type=int, default=2000)
parser.add_argument('-test_sweep', type=str2bool, default=False)
args = parser.parse_args()

sweep = []

i = 0

for f_bl in [True, False]:
    for to_tspan in [None, [0, .25], [0, 0.5], [0, 0.75]]:
        for targ in ["UPDRS_III", "MSEADLG"]:
            sweep.append(
                {"inference_type": "future_scores_one_to_one", "epochs": args.num_epochs, "learning_rate": 0.0001,
                 "name_suffix": i,
                 "data_file": "data/Processed/future_scores_{}_to_one_{}{}.csv".format("bl" if f_bl else "one", "" if to_tspan is None else "timespan_" + str(to_tspan) + "_", targ),
                 "classification_or_regression": "regression", "slurm": True})
            i += 1

for f_bl in [True, False]:
    for c_or_r in [True, False]:
        for targ in ["UPDRS_III", "MSEADLG"]:
            sweep.append(
                {"inference_type": "rates_one_to_one", "epochs": args.num_epochs, "learning_rate": 0.0001,
                 "name_suffix": i,
                 "data_file": "data/Processed/rates_{}_to_one_{}_{}.csv".format("bl" if f_bl else "one", "classification" if c_or_r else "regression", targ),
                 "classification_or_regression": c_or_r, "slurm": True})
            i += 1

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
                results.append([float(i) for i in line.split(' ')[:-1]])

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
            stats["param_set_{}".format(param_set + 1)]["params"] = f.readlines()[param_set]

    return stats


def graph(data):
    main_groups = {
        'UPDRS III Future Score': lambda param: "-inference_type future_scores_one_to_one" in param and
                                                "UPDRS_III" in param,
        'MSEADLG Future Score': lambda param: "-inference_type future_scores_one_to_one" in param and
                                              "MSEADLG" in param,
        'UPDRS III Rate': lambda param: "-inference_type rates_one_to_one" in param and
                                        "-classification_or_regression regression" in param and
                                        "UPDRS_III" in param,
        'MSEADLG Rate': lambda param: "-inference_type rates_one_to_one" in param and
                                      "-classification_or_regression regression" in param and
                                      "MSEADLG" in param,
        "Classification Of Rate": lambda param: "-classification_or_regression classification" in param}

    grouped_data = {g: {} for g in main_groups}

    for param_set in data:
        for group in main_groups:
            if main_groups[group](data[param_set]["params"]):
                grouped_data[group][param_set] = data[param_set]

    for group in grouped_data:
        graph_data = [go.Scatter(
            x=[72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112],
            y=schools,
            marker=dict(color="crimson", size=12),
            mode="markers",
            name="Women",
            xaxis="Annual Salary (in thousands)",
            yaxis="School"
        ) for param_set in grouped_data[group]]

        layout = go.Layout(title="Gender Earnings Disparity")

        fig = go.Figure(data=graph_data, layout=layout)

        plotly.plot(fig, filename='bAbI_salience_sampling_bar_chart.html',
                    image='png', image_filename='bAbI_salience_sampling_bar_chart',
                    image_height=800, image_width=1300)


if args.mode == "eval" or args.mode == "eval_and_graph":
    stats = evaluate()
    with open(stats_file_name, "w") as file:
        file.write(json.dumps(stats))

if args.mode == "graph" or args.mode == "eval_and_graph":
    with open(stats_file_name) as f:
        data = json.load(f)
    graph(data)
