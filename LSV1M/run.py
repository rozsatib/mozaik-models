# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from analysis_and_visualization import find_RF_params, load_RF_params_from_datastore
from model import SelfSustainedPushPull
from experiments import sparse_noise_experiments, ideal_gabor_experiments
import mozaik
from mozaik.controller import prepare_workflow, run_experiments, setup_logging
import mozaik.controller
import sys 
from pyNN import nest

logger = mozaik.getMozaikLogger()

model_class = SelfSustainedPushPull
simulation_name = "SelfSustainedPushPull"

if True:
    sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
    model = model_class(sim,num_threads,parameters)
    sparse_noise_data_store = run_experiments(model, sparse_noise_experiments(model), parameters)
    sparse_noise_data_store.save()
    find_RF_params(sparse_noise_data_store,["V1_Exc_L4",'V1_Exc_L2/3'])
    rf_params = load_RF_params_from_datastore(sparse_noise_data_store, ["V1_Exc_L4",'V1_Exc_L2/3'])
    ideal_gabor_data_store = run_experiments(
        model, ideal_gabor_experiments(model, rf_params), parameters
    )   
    ideal_gabor_data_store.save()

    import resource
    print "Final memory usage: %iMB" % ( 
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024)
    )   

else:
    setup_logging()
    data_store = PickledDataStore(load=True, parameters=ParameterSet(
        {'root_directory': 'SelfSustainedPushPull_test____', 'store_stimuli': False}), replace=True)
