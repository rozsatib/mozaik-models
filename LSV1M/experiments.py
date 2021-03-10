#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.apparent_motion import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet

CompareSlowVersusFastGaborMotion_default_parameters =  {
    "num_trials": 1,
    "x": 1,
    "y": 1,
    "orientation": 0,
    "phase": 1,
    "spatial_frequency": 2,
    "sigma": 0.17,
    "n_sigmas": 3.0,
    "center_relative_luminance": 0.5,
    "surround_relative_luminance": 0.7,
    "movement_speeds": [5.0, 180.0],
#"angles": list(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
    "angles": [0],
    "moving_gabor_orientation_radial": True,
    "radius": 5,
}

def create_experiments(model, rf_params):
    experiments = []
    for neuron_id in rf_params:
        params = CompareSlowVersusFastGaborMotion_default_parameters.copy()
        print(params)
        print(rf_params[neuron_id])
        params.update(rf_params[neuron_id])
        params.pop("aspect_ratio", None)
        print(params)
        experiments.append(CompareSlowVersusFastGaborMotion(model, ParameterSet(params)))
    return experiments
