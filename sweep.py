import argparse
import json
import os
import subprocess
import time

import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-call_sweep', type=str2bool, default=True)
parser.add_argument('-program', type=str, default="Main.py")
parser.add_argument('-num_runs', type=int, default=5)
parser.add_argument('-test_sweep', type=str2bool, default=False)
args = parser.parse_args()

sweep = []
sweep.extend([{"distributional": True, "top_k": top_k, "sample": sam, "aggregate_method": agg, "slurm": True}
              for top_k in range(1, 21) for sam in [True, False] for agg in ["max", "mean", "concat"]])
sweep.extend([{"distributional": True, "top_k": top_k, "uniform_sample": True, "aggregate_method": agg, "slurm": True}
              for top_k in range(1, 21) for agg in ["max", "mean", "concat"]])
sweep.extend([{"distributional": False, "top_k": top_k, "aggregate_method": agg, "slurm": True}
              for top_k in range(1, 21) for agg in ["max", "mean", "concat"]])

# TODO delete, testing
if args.test_sweep:
    for x in sweep:
        x["epochs"] = 1

log_name = "log"
stats_file_name = "stats.txt"
path = os.getcwd()

# Whether to run or just evaluate
if args.call_sweep:
    slurm_script_name = "sweep"
    in_file_name = "in"

    open(in_file_name, 'w').close()
    open(slurm_script_name, 'w').close()

    for parameterization in sweep:
        with open(in_file_name, "a") as file:
            params = ""
            for key in parameterization:
                params += '-{} {} '.format(key, parameterization[key])
            file.write(params + '\n')

    if not os.path.exists(path + "/eval"):
        os.makedirs(path + "/eval")


    def slurm_script(n):
        return r"""#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00 -o {}/{}.%a.{}
#SBATCH --array=0-{}
module load anaconda3/5.2.0b
python {} `awk "NR==$SLURM_ARRAY_TASK_ID" {}`
""".format(path + "/eval", log_name, n, len(sweep) - 1, args.program, in_file_name)


    # Create a job for each run, each consisting of all of the params (e.g. for mean and st.d)
    processes = []
    for run in range(args.num_runs):
        with open(slurm_script_name, "w") as file:
            file.write(slurm_script(run))
        # processes.append(subprocess.call(['sbatch', slurm_script_name], shell=True))
        subprocess.call(['sbatch {}'.format(slurm_script_name)], shell=True)
        time.sleep(5)
    # wait = [p.wait() for p in processes]


def evaluate_babi():
    stats = {}
    for param_set in range(len(sweep)):
        results = []
        for r in range(args.num_runs):
            with open("{}/{}.{}.{}".format(path + "/eval", log_name, param_set, r)) as f:
                line = f.readlines()
                line = line[-1]
                results += [float(i) for i in line.split(' ')]
        for task in range(20):
            stats["param_set_{}".format(param_set)]["task_{}".format(task)]["mean"] = \
                np.mean([result[task] for result in results])
            stats["param_set_{}".format(param_set)]["task_{}".format(task)]["std"] = \
                np.std([result[task] for result in results])
            stats["param_set_{}".format(param_set)]["all_tasks_mean"] = \
                np.mean([v for v in result for result in results])
            stats["param_set_{}".format(param_set)]["all_tasks_std"] = \
                np.std([v for v in result for result in results])
            with open("in", "w") as f:
                stats["param_set_{}".format(param_set)]["params"] = f.readlines()[param_set]
    return stats


# Note: this won't give result since processes are merely queued; have to re-run this script after with call_sweep=False
with open(stats_file_name, "w") as file:
    file.write(json.dumps(evaluate_babi()))
