#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.apparent_motion import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np
import mozaik
logger = mozaik.getMozaikLogger()

def measure_flash_duration_experiments(model, rf_params):

    experiments = [NoStimulation(model, ParameterSet({'duration': 100}))]
    params = {
        "num_trials": 30,
        "x": 1,
        "y": 1,
        "orientation": 0,
        "phase": 1,
        "spatial_frequency": 0.8,
        "sigma": 3.0 / 6.0,
        "n_sigmas": 3.0,
        "relative_luminance": 1.0,
        "min_duration": 14,
        "max_duration": 42,
        "step": 7,
        "blank_duration": 100,
    }

    for neuron_id in rf_params:
        p = rf_params[neuron_id]
        neuron_params = {
            "x": p["Receptive Field x"],
            "y": p["Receptive Field y"],
            "orientation": p["LGNAfferentOrientation"],
            "phase": p["LGNAfferentPhase"],
            "spatial_frequency": p["LGNAfferentFrequency"],
            "sigma": p["Receptive Field diameter"] / 6.0,
            "neuron_id" : neuron_id,
        }
        params.update(neuron_params)
        experiments.append(MeasureGaborFlashDuration(model,ParameterSet(params)))
    return experiments
