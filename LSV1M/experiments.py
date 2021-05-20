#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.apparent_motion import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np
import mozaik
logger = mozaik.getMozaikLogger()

def continuous_am_comparison_experiments(model, rf_params):

    experiments = [NoStimulation(model, ParameterSet({'duration': 100}))]
    params = {
        "num_trials": 30,
        "x": 0,
        "y": 0,
        "orientation": 0,
        "phase": 1,
        "spatial_frequency": 2,
        "sigma": 2.5 / 6.0,
        "n_sigmas": 3.0,
        "center_relative_luminance": 0.5,
        "surround_relative_luminance": 0.7,
        "movement_speeds": [5.0, 180.0],
        "angles": list(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
        "moving_gabor_orientation_radial": True,
        "n_circles": 2,
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
        experiments.append(CompareSlowVersusFastGaborMotion(model,ParameterSet(params)))
        params2 = params.copy()
        params2["moving_gabor_orientation_radial"]=False
        experiments.append(CompareSlowVersusFastGaborMotion(model,ParameterSet(params2)))
    return experiments
