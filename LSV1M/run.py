# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from analysis_and_visualization import perform_analysis_and_visualization
from model import SelfSustainedPushPull
from experiments import create_experiments
import mozaik
from mozaik.controller import prepare_workflow, run_experiments, setup_logging
from tools import dummy_experiment, rf_params_from_annotations
import mozaik.controller
import sys
from pyNN import nest

logger = mozaik.getMozaikLogger()

model_class = SelfSustainedPushPull
simulation_name = "SelfSustainedPushPull"

if True:
    sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
    model = model_class(sim,num_threads,parameters)
    dummy_data_store = run_experiments(model, dummy_experiment(model), parameters)
    rf_params = rf_params_from_annotations(dummy_data_store)
    data_store = run_experiments(
        model, create_experiments(model, rf_params), parameters
    )

    data_store.save()
    import resource
    print "Final memory usage: %iMB" % (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024)
    )

else:
    setup_logging()
    data_store = PickledDataStore(load=True, parameters=ParameterSet(
        {'root_directory': 'SelfSustainedPushPull_test____', 'store_stimuli': False}), replace=True)

print "Starting visualization"
perform_analysis_and_visualization(data_store)
data_store.save()
