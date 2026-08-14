from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from parameters import ParameterSet


def create_experiments(model):
    return [
        SingleOptogeneticArrayStimulus(
            model,
            MozaikExtendedParameterSet(
                {
                    "stimulator_array_list": [
                        {
                            "sheet": "V1_Exc_L2/3",
                            "name": "stimulator_array",
                            "intensity_scaler": 1.0,
                        }
                    ],
                    "num_trials": 1,
                    "stimulating_signal_function": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
                    "stimulating_signal_function_parameters": ParameterSet(
                        {
                            "shape": "circle",
                            "coords": [0, 0],
                            "radius": 300,
                            "intensity": 0.5,
                            "duration": 1000,
                            "onset_time": 0,
                            "offset_time": 1000,
                        }
                    ),
                }
            ),
        )
    ]
