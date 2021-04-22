#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np
import mozaik
logger = mozaik.getMozaikLogger()

def sparse_noise_experiments(model):
    fast = False
    cell_size = 0.3 
    grid_size = 14.0
    img_per_px = 50 if not fast else 1
    img_total = grid_size * grid_size * img_per_px
    max_im = 2500 if not fast else 10
    n = int(np.ceil(img_total/max_im))
    initial_experiment_seed = 17

    experiments = [NoStimulation(model, ParameterSet({'duration': 100}))]

    for i in range(n):
        n_imgs = max_im if i < n-1 else img_total % max_im
        logger.info("Number of images: %d" % n_imgs)
        #RF estimation
        experiments.append(MeasureSparse(model,ParameterSet({
           'time_per_image': 70,
           'blank_time' : 0,
           'stim_size' : grid_size * cell_size,
           'total_number_of_images' : int(n_imgs),
           'num_trials' : 1,
           'experiment_seed' : initial_experiment_seed + i,
           'grid_size' : int(grid_size),
           'grid' : True
        })))
    return experiments
