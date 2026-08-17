# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend

slurm_options = ["-J optog", "--exclude=w[1-4,9,11-17]", "--mem=490gb", "--hint=nomultithread"]

CombinationParameterSearch(
    SlurmSequentialBackend(
        num_threads=1,
        num_mpi=32,
        path_to_mozaik_env=# CHANGE TO YOUR OWN VIRTUALENV! "/home/rozsa/virt_env/mozaik/bin/activate",
        slurm_options=slurm_options,
    ),
    {
        "trial": [0],
    },
).run_parameter_search()
