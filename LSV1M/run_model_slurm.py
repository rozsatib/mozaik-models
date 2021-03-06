# -*- coding: utf-8 -*-
import sys 
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch,SlurmSequentialBackend
import numpy
import time
slurm_options = ['-J model']

if True:
    CombinationParameterSearch(SlurmSequentialBackend(
            num_threads=16,
            num_mpi=1,
            path_to_mozaik_env='/home/rozsa/virt_env/mozaik/bin/activate',
            slurm_options=slurm_options),{
                                         'trial': [1]
                                         }).run_parameter_search()
