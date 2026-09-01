#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np
import sys

from mozaik.tools.distribution_parametrization import load_parameters
import json

def select_experiment_func(e, fast_run):

    if fast_run:
        n_trials = 4
        sf_steps = 10
        c_steps = 5
        lum_steps = 16
        lum_duration = 1000
        tf_duration = 700
        tf_max_steps = 4
    else:
        n_trials = 120
        sf_steps = 30
        c_steps = 41
        lum_steps = 51
        lum_duration = 10000
        tf_duration = 10003
        tf_max_steps = sys.maxsize


    if e["type"] == "contrast":
        experiment = MeasureFrequencySensitivity
        params = {
            "orientation": 0.0,
            "temporal_frequencies": [e["temporal_frequency"]],
            "contrasts": list(np.linspace(0, 100, c_steps, endpoint=True)),
            "grating_duration": 143 * 7 * 2,
            "spatial_frequencies": [e["spatial_frequency"]],
            "num_trials": n_trials,
            "square": False,
            "shuffle_stimuli": False,
        }
    elif e["type"] == "spatial_frequency":
        experiment = MeasureFrequencySensitivity
        params = {
            "orientation": 0.0,
            "temporal_frequencies": [e["temporal_frequency"]],
            "contrasts": [e["contrast"]],
            "grating_duration": 143 * 7 * 2,
            "spatial_frequencies": list(
                np.linspace(0.1, 2.0, sf_steps, endpoint=True)
            ),
            "num_trials": n_trials,
            "square": False,
            "shuffle_stimuli": False,
        }
    elif e["type"] == "temporal_frequency":
        duration = tf_duration
        range2 = np.logspace(0, 8, 9, base=2)
        f0s = np.array(np.arange(1, 11) * 0.1)

        tfs = np.unique(np.outer(f0s, range2).flatten())
        tfs.sort()
        tfs = tfs[:tf_max_steps]

        experiment = MeasureFrequencySensitivity
        params = {
            "orientation": 0.0,
            "temporal_frequencies": list(tfs),
            "contrasts": [e["contrast"]],
            "grating_duration": duration,
            "spatial_frequencies": [e["spatial_frequency"]],
            "num_trials": n_trials,
            "square": False,
            "shuffle_stimuli": False,
        }
    elif e["type"] == "luminance":
        experiment = MeasureFlatLuminanceSensitivity
        params = {
            "luminances": list(
                np.logspace(-2, 2, lum_steps, endpoint=True)
            ),
            "step_duration": lum_duration,
            "num_trials": n_trials,
            "shuffle_stimuli": False,
        }
    elif e["type"] == "trial-to-trial-variance":
        experiment = MeasureFrequencySensitivity
        params = {
            "orientation": 0.0,
            "temporal_frequencies": [e["temporal_frequency"]],
            "contrasts": [e["contrast"]],
            "grating_duration": 36 * 7,  # close to 500 ms experiment value
            # TODO: rewrite spatial frequency to be more flexible!
            "spatial_frequencies": [0.8],
            "num_trials": 200,
            "square": False,
            "shuffle_stimuli": False,
        }
    elif e["type"] == "sparse_noise":
        experiment = MeasureSparseBar
        params = {
            "time_per_image": 28,
            "blank_time": 245,
            "total_number_of_images": 20 * 2 * (2 if fast_run else 750),
            "num_trials": 1,
            "orientation": 0,
            "bar_length": 5,
            "bar_width": 0.1,
            "x": 0,
            "y": 0,
            "n_positions": 20,
            "experiment_seed": 17,
            "shuffle_stimuli": False,
        }

    return lambda model: [experiment(model, ParameterSet(params))]
