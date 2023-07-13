#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np

def create_experiment_of_NaturalImages(model, num_skipped_images, num_images, num_trials, **args):
    return [
        NoStimulation(model, ParameterSet({"duration": 270})),
        # Measure response to sequence of static natural images
        MeasureNaturalImages(
            model,
            ParameterSet(
                {
                    "duration": 147,
                    "num_images": num_images,
                    "images_dir": "/projects/imagenet/val_grayscale110x110_resize110_cropped/",
                    "image_display_duration": 147,
                    "num_trials": num_trials,
                    "num_skipped_images": num_skipped_images,
                    "size": 11,
                }
            ),
        ),
    ]
