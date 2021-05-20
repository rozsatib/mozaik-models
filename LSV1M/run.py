# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
from model import SelfSustainedPushPull
from experiments import *
import mozaik
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import prepare_workflow, run_experiments, setup_logging
from analysis_and_visualization import load_RF_params_from_datastore
from tools import rf_params_from_annotations
import mozaik.controller
import sys
from pyNN import nest

logger = mozaik.getMozaikLogger()

model_class = SelfSustainedPushPull
simulation_name = "SelfSustainedPushPull"

rf_data_store_path = "/home/rozsa/dev/sparse_noise/VF15_42/measure_rf_analysis_result"

sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
model = model_class(sim,num_threads,parameters)
logger.info("Bla")
rf_data_store = PickledDataStore(load=True,parameters=ParameterSet(
    {'root_directory': rf_data_store_path, 'store_stimuli': False}), replace=False)

sheets = ["V1_Exc_L4"]#,'V1_Exc_L2/3']
rf_param_names = [
    "Receptive Field",
    "Masked Receptive Field",
    "Receptive Field x",
    "Receptive Field y",
    "Receptive Field diameter",
    "LGNAfferentOrientation",
    "LGNAfferentAspectRatio",
    "LGNAfferentFrequency",
    "LGNAfferentSize",
    "LGNAfferentPhase",
    "LGNAfferentX",
    "LGNAfferentY",
]

rf_params = {}
for sheet in sheets:
    rf_params.update(load_RF_params_from_datastore(rf_data_store,sheet,rf_param_names))

data_store = run_experiments(model, continuous_am_comparison_experiments(model, rf_params), parameters)
data_store.save()
