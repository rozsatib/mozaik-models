#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import *
from mozaik.experiments.optogenetic import OptogeneticArrayImageStimulus
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import numpy as np



def create_experiments(model):
    experiments = []
    p = {
        "sheet_list": ["V1_Exc_L2/3", "V1_Inh_L2/3"],
        'sheet_intensity_scaler': [1.0, 0.0],
        'sheet_transfection_proportion': [1.0, 1.0],
        "num_trials": 1,
        "stimulator_array_parameters": MozaikExtendedParameterSet(
            {
                "size": 3000, # micrometers
                "spacing": 10.0, # micrometers
                "depth_sampling_step": 10, # micrometers
                "light_source_light_propagation_data": "light_scattering_radial_profiles_lsd10.pickle",
                "update_interval": 1, # ms
            }
        ),
        "intensities": [1],
        "duration": 150, # ms
        "onset_time": 50, # ms
        "offset_time": 100, # ms
    }
    p_ = MozaikExtendedParameterSet(deepcopy(p))

    import os
    # Show a single image, or all images in a folder
    single_image = False
    if single_image:
        p_.images_path = "optogenetic_stimuli/sheep.npy"
    else:
        p_.images_path = "optogenetic_stimuli"

    experiments.append(OptogeneticArrayImageStimulus(model, p_))
    return experiments
