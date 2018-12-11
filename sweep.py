import argparse
import json
import subprocess
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-call_sweep', type=bool, default=True)
parser.add_argument('-program', type=str, default="Main.py")
parser.add_argument('-num_runs', type=int, default=10)
args = parser.parse_args()

sweep = []
sweep.extend([{"distributional": True, "top_k": top_k, "sample": sam, "aggregate_method": agg, "slurm": True}
              for top_k in range(1, 21) for sam in [True, False] for agg in ["max", "mean", "concat"]])
sweep.extend([{"distributional": True, "top_k": top_k, "uniform_sample": True, "aggregate_method": agg, "slurm": True}
              for top_k in range(1, 21) for agg in ["max", "mean", "concat"]])
sweep.extend([{"distributional": False, "aggregate_method": agg, "slurm": True} for agg in ["max", "mean", "concat"]])

# TODO delete, testing
for x in sweep:
    x["epochs"] = 1

log_name = "log"
stats_file_name = "stats.txt"

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


    def slurm_script(n):
        return r"""#!/bin/bash
    #SBATCH -p gpu
    #SBATCH --gres=gpu:1
    #SBATCH -t 5-00:00:00 -o {}.%a.{}
    #SBATCH --array=0-{}
    module load anaconda3/5.2.0b
    {} `awk "NR==$SLURM_ARRAY_TASK_ID" {}`
    """.format(log_name, n, len(sweep) - 1, args.program, in_file_name)


    # Create a job for each run, each consisting of all of the params (e.g. for mean and st.d)
    processes = []
    for run in range(args.num_runs):
        with open(slurm_script_name, "w") as file:
            file.write(slurm_script(run))
        processes.append(subprocess.call(['sbatch', slurm_script_name], shell=True))
    # wait = [p.wait() for p in processes]


def evaluate_babi():
    stats = {}
    for param_set in range(len(sweep)):
        results = []
        for r in range(args.num_runs):
            with open("{}.{}.{}".format(log_name, param_set, r)) as f:
                line = f.readlines()
                assert len(line) == 1
                line = line[0]
                results += [line.split(' ')]
        for task in range(20):
            stats["param_set_{}_task_{}_mean".format(param_set, task)] = np.mean([result[task] for result in results])
            stats["param_set_{}_task_{}_std".format(param_set, task)] = np.std([result[task] for result in results])
            stats["param_set_{}_all_tasks_mean".format(param_set)] = np.mean([v for v in result for result in results])
            stats["param_set_{}_all_tasks_std".format(param_set)] = np.std([v for v in result for result in results])
    return stats


# Note: this won't give result since processes are merely queued; have to re-run this script after with call_sweep=False
with open(stats_file_name, "w") as file:
    file.write(json.dumps(evaluate_babi()))
