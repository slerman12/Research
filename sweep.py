"""
Generate file 'in' that looks like as follows:

-learning_rate 0.01 -episodes 10000
-learning_rate 0.01 -episodes 20000
-learning_rate 0.01 -episodes 30000
-learning_rate 0.02 -episodes 10000
-learning_rate 0.02 -episodes 20000
-learning_rate 0.02 -episodes 30000

based on list of parameter key-values to sweep.

Then generate 'sweep' file such as:

#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 5-00:00:00 -o log.%a
#SBATCH --array=1-100
module load anaconda3/5.2.0b
Main.py `awk "NR==$SLURM_ARRAY_TASK_ID" in`
"""
import subprocess

sweep = [{"bla": 10, "boo": 9}, {"bo": 5}]

num_runs = 10

program_name = "Main.py"

log_name = "log"

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
""".format(log_name, n, len(sweep) - 1, program_name, in_file_name)


# Create a job for each run, each consisting of all of the params (e.g. for mean and st.d)
processes = []
for run in range(num_runs):
    with open(slurm_script_name, "w") as file:
        file.write(slurm_script(run))
    processes.append(subprocess.Popen(['sbatch', slurm_script_name], shell=True))
[p.wait() for p in processes]

print("done")


# TODO: get stats
def evaluate_babi():
    pass
