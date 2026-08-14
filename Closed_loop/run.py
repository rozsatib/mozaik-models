import os
import sys

import matplotlib

matplotlib.use("Agg")

import nest
from mpi4py import MPI

from model import SelfSustainedPushPull
from mozaik.controller import run_workflow

parameter_entry_file = os.path.basename(sys.argv[3])
if "closed_loop" in parameter_entry_file:
    from experiments_closed_loop import create_experiments
else:
    from experiments_basic import create_experiments

nest.Install("stepcurrentmodule")

data_store, model = run_workflow(
    "SelfSustainedPushPull", SelfSustainedPushPull, create_experiments
)
if MPI.COMM_WORLD.rank == 0:
    data_store.save()
