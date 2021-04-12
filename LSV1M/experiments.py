#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet


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
    return [
        #MeasureIdealGabor(model,ParameterSet({})) 
        NoStimulation(model, ParameterSet(
        {'duration': 10})),
    ]
