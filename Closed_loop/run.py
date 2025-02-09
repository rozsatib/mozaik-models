# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
from model import SelfSustainedPushPull
from experiments import *
import mozaik
from mozaik.controller import run_workflow
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD

import nest
nest.Install("stepcurrentmodule")
data_store, model = run_workflow("LSV1M", SelfSustainedPushPull, closed_loop_experiment)

if mpi_comm.rank == 0:
    data_store.save()
