# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from model import SelfSustainedPushPull
import mozaik
from experiments import *
from mozaik.controller import run_workflow, setup_logging
import mozaik.controller
from mozaik.controller import *
import sys
from pyNN import nest

def run_workflow_with_exp_params(simulation_name, model_class, create_experiments, exp_params):
    # Prepare workflow - read parameters, setup logging, etc.
    sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
    if "baseline" in parameters.keys():
        del parameters["baseline"]
    # Prepare model to run experiments on
    model = model_class(sim,num_threads,parameters)
    # Run experiments with previously read parameters on the prepared model
    data_store = run_experiments(model,create_experiments(model, **exp_params),parameters)

    if mozaik.mpi_comm.rank == 0:
        data_store.save()
    import resource
    print("Final memory usage: %iMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024)))
    return (data_store, model)


num_images = 5
num_trials = 5
(
    simulation_run_name,
    simulator_name,
    num_threads,
    parameters_url,
    modified_parameters,
) = parse_workflow_args()
baseline = modified_parameters["baseline"]
num_skipped_images = baseline
exp_params = {
    "num_skipped_images": num_skipped_images,
    "num_images": num_images,
    "num_trials": num_trials,
}
data_store, model = run_workflow_with_exp_params(
    f"Images_from_{num_skipped_images}_to_{num_skipped_images + num_images}",
    SelfSustainedPushPull,
    create_experiment_of_NaturalImages,
    exp_params,
)
data_store.save()
