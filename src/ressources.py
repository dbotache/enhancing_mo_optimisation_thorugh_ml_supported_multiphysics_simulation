import os
import torch
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")

if SLURM_CPUS_PER_TASK is None:
    n_cpu = multiprocessing.cpu_count()
    SLURM_CPUS_PER_TASK = f'{n_cpu}'
    SERVER = False
else:
    SERVER = True
    n_cpu = int(SLURM_CPUS_PER_TASK)

os.environ["OMP_NUM_THREADS"] = SLURM_CPUS_PER_TASK
os.environ["MKL_NUM_THREADS"] = SLURM_CPUS_PER_TASK
os.environ["NUMEXPR_NUM_THREADS"] = SLURM_CPUS_PER_TASK

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

