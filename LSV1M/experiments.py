#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet


def create_experiments(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
        {'duration': 2*5*3*8*7})),

        #RF estimation
        MeasureSparse(model,ParameterSet({
           'time_per_image': 70,
           'blank_time' : 0,
           'stim_size' : 2.4,
           'total_number_of_images' : 3000,
           'num_trials' : 1,
           'experiment_seed' : 17,
           'grid_size' : 8,
           'grid' : True
        })),

    ]
