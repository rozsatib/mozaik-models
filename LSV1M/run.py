# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from model import SelfSustainedPushPull
from experiments import create_experiments
import mozaik
from mozaik.controller import run_workflow, setup_logging
import mozaik.controller
import sys
from pyNN import nest

data_store, model = run_workflow(
    'SelfSustainedPushPull', SelfSustainedPushPull, create_experiments)
data_store.save()
