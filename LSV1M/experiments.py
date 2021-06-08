#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet


def create_experiments(model):
    return [
        MeasureSpontaneousActivity(
            model, ParameterSet({"duration": 143 * 7, "num_trials": 1})
        ),
    ]
