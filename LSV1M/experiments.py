#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np


def sparse_noise_experiments(model):
    fast = True
    cell_size = 0.3 
    grid_size = 2.0 if fast else 12.0
    im_per_px = 1 if fast else 30
    nostim_duration = 10 if fast else 100
        
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
        {'duration': nostim_duration})),

        #RF estimation
        MeasureSparse(model,ParameterSet({
           'time_per_image': 70,
           'blank_time' : 0,
           'stim_size' : grid_size * cell_size,
           'total_number_of_images' : int(grid_size * grid_size * im_per_px),
           'num_trials' : 1,
           'experiment_seed' : 17,
           'grid_size' : int(grid_size),
           'grid' : True
        })),

    ]


def ideal_gabor_experiments(model, rf_params):
    stims = []
    for neuron_id in rf_params:
        stims.append(
            FindIdealGabor(model,ParameterSet({
                "relative_luminance": 1.0,
                "orientations": list(np.linspace(0,np.pi,6,endpoint=False)),
                "phases": list(np.linspace(0,np.pi*2,4,endpoint=False)),
                "spatial_frequencies": list(np.linspace(0.9,4.9,5)),
                "diameter": rf_params[neuron_id]["Receptive Field diameter"],
                "flash_duration": 40,
                "x": rf_params[neuron_id]["Receptive Field x"],
                "y": rf_params[neuron_id]["Receptive Field y"],
                "num_trials": 1,
                "duration" : 40,
            }
        ))
        )
