# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")

from analysis_and_visualization import (
    find_RF_params,
    load_RF_params_from_datastore,
    save_RF_params_to_datastore,
    save_rf_plots,
)
from tools import rf_params_from_annotations, save_rf_params
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from parameters import ParameterSet
from mozaik.controller import setup_logging
import sys
import os
import re

setup_logging()

sheets = ["V1_Exc_L4"]  # ,'V1_Exc_L2/3'])

# Load input datastore
data_store = PickledDataStore(
    load=True,
    parameters=ParameterSet({"root_directory": sys.argv[1], "store_stimuli": False}),
    replace=False,
)

# Create output datastore
output_data_store_path = sys.argv[2]
os.makedirs(output_data_store_path)
output_data_store = PickledDataStore(
    load=False,
    parameters=MozaikExtendedParameterSet(
        {"root_directory": output_data_store_path, "store_stimuli": False}
    ),
)

# Find RF params, save in separate datastore
find_RF_params(data_store, sheets, output_data_store)

# Plot RFs into images
rf_px_size = output_data_store.full_datastore.get_analysis_result(
    identifier="SingleValue", value_name="Receptive Field pixel size"
)[0].value


rf_param_names = [
    "Receptive Field",
    "Masked Receptive Field",
    "Receptive Field x",
    "Receptive Field y",
    "Receptive Field diameter",
]
for sheet in sheets:
    rf_params = load_RF_params_from_datastore(output_data_store, sheet, rf_param_names)
    rf_img_dir = output_data_store_path + "/" + sheet
    os.makedirs(rf_img_dir)
    save_rf_plots(rf_params, rf_img_dir, rf_px_size)

rf_annotations = [
    "LGNAfferentOrientation",
    "LGNAfferentAspectRatio",
    "LGNAfferentFrequency",
    "LGNAfferentSize",
    "LGNAfferentPhase",
    "LGNAfferentX",
    "LGNAfferentY",
]

for sheet in sheets:
    NeuronAnnotationsToPerNeuronValues(data_store, ParameterSet({})).analyse()
    rf_params = load_RF_params_from_datastore(data_store, sheet, rf_annotations)
    save_rf_params(output_data_store, rf_params, sheet, True)

output_data_store.save()
