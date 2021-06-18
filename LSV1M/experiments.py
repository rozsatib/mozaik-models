#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.apparent_motion import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np
import mozaik

logger = mozaik.getMozaikLogger()


def am_configuration_experiments(model):

    experiments = [NoStimulation(model, ParameterSet({"duration": 143 * 7}))]
    configurations = [
        "SECTOR_ISO",
        "SECTOR_CROSS",
        "SECTOR_CF",
        "SECTOR_RND",
        "FULL_ISO",
        "FULL_CROSS",
        "FULL_RND",
        "CENTER_ONLY",
    ]
    params = {
        "num_trials": 15,
        "x": 0,
        "y": 0,
        "orientation": 0,
        "phase": 0,
        "spatial_frequency": 0.8,
        "sigma": 2.5 / 6.0,
        "n_sigmas": 3.0,
        "center_relative_luminance": 0.5,
        "surround_relative_luminance": 1.0,
        "configurations": configurations,
        "random_order": True,
        "n_circles": 2,
        "flash_center": True,
        "flash_duration": 28,
        "blank_duration" : 105,
        "neuron_id": 0,
    }
    experiments.append(RunApparentMotionConfigurations(model, ParameterSet(params)))
    return experiments
