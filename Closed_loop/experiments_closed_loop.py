from types import SimpleNamespace

import numpy as np
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation
from mozaik.sheets.direct_stimulator import simple_shapes_binary_mask
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from parameters import ParameterSet


class _NamedStaticMethod:
    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def named_static(name):
    def decorator(function):
        return staticmethod(_NamedStaticMethod(function, name))

    return decorator


class RegulatorSetup:
    STIMULATION_RADIUS = 300
    TARGET_RATE = 35.0

    @named_static("'RegulatorSetup.calculate_input'")
    def calculate_input(regulator):
        if regulator.state is None:
            regulator.state = SimpleNamespace(
                kp=0.05,
                ki=0.000005,
                kd=0.005,
                integral=0.0,
                previous_error=0.0,
                error=0.0,
                control_signal=0.0,
            )
        else:
            state = regulator.state
            state.integral += state.error * regulator.parameters.state_update_interval
            integral = np.clip(state.ki * state.integral, -0.2, 0.2)
            derivative = (
                state.kd
                * (state.error - state.previous_error)
                / regulator.parameters.state_update_interval
            )
            state.previous_error = state.error
            state.control_signal = np.clip(
                state.kp * state.error + integral + derivative, 0.0, 1.0
            )

        mask = simple_shapes_binary_mask(
            regulator.stimulator_coords_x,
            regulator.stimulator_coords_y,
            "circle",
            ParameterSet(
                {"coords": [0, 0], "radius": RegulatorSetup.STIMULATION_RADIUS}
            ),
        )
        samples = int(
            regulator.parameters.state_update_interval
            / regulator.parameters.update_interval
        )
        return np.repeat(
            (mask * regulator.state.control_signal)[:, :, np.newaxis],
            samples,
            axis=2,
        )

    @named_static("'RegulatorSetup.update_state'")
    def update_state(regulator):
        positions = regulator.recorded_neuron_positions("spikes")
        in_stimulation_circle = (
            np.hypot(positions[0], positions[1]) < RegulatorSetup.STIMULATION_RADIUS
        )
        assert np.any(
            in_stimulation_circle
        ), "No recorded neurons lie inside the closed-loop stimulation circle!"

        interval_seconds = regulator.parameters.state_update_interval / 1000.0
        current_rate = np.mean(
            np.asarray(regulator.last_spike_counts)[in_stimulation_circle]
            / interval_seconds
        )

        state = regulator.state
        if not hasattr(state, "smoothed_rate"):
            state.smoothed_rate = current_rate
        else:
            state.smoothed_rate = 0.1 * current_rate + 0.9 * state.smoothed_rate
        state.error = RegulatorSetup.TARGET_RATE - state.smoothed_rate


def create_experiments(model):
    return [
        ClosedLoopOptogeneticStimulation(
            model,
            MozaikExtendedParameterSet(
                {
                    "num_trials": 1,
                    "duration": 1000,
                    "stimulator_array_list": [
                        {
                            "sheet": "V1_Exc_L2/3",
                            "name": "closed_loop_array",
                            "input_calculation_function": RegulatorSetup.calculate_input,
                            "state_update_function": RegulatorSetup.update_state,
                        }
                    ],
                }
            ),
        )
    ]
