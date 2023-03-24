#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np



def create_experiments(model):
    return [
        MeasureNaturalImagesWithEyeMovement(
            model, ParameterSet({"stimulus_duration": 2 * 143 * 7, "num_trials": 10})
        ),
    ]
