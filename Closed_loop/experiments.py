from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation

from mozaik.sheets.direct_stimulator import simple_shapes_binary_mask

from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet

from types import SimpleNamespace

import numpy as np
import pylab
import math

class _NamedStaticMethod:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

def named_static(name):
    def decorator(func):
        return staticmethod(_NamedStaticMethod(func, name))
    return decorator

class RegulatorSetup:
    stim_circle_radius = 300
    a = -0.483

    @staticmethod
    def append_history(state, update_dict):
        if state.history is None:
            names = ["LMS", "error", "Firing rate", "LFP", "Current rate"]
            state.history = {n: [] for n in names}
        for k in update_dict:
            state.history[k].append(update_dict[k])

    @staticmethod
    def calculate_lfp(regulator,electrode_x=0,electrode_y=0):
        e_rev_E, e_rev_I = regulator.sheet.parameters.cell.params['e_rev_E'], regulator.sheet.parameters.cell.params['e_rev_I']
        curr_time, prev_time = regulator.current_time(), regulator.current_time() - regulator.parameters.state_update_interval
        e_syn_all = regulator.get_recording("gsyn_exc",t_start=prev_time,t_stop=curr_time)
        i_syn_all = regulator.get_recording("gsyn_inh",t_start=prev_time,t_stop=curr_time)
        vm_all = regulator.get_recording("v",t_start=prev_time,t_stop=curr_time)

        x, y = regulator.recorded_neuron_positions()
        lfp = ((e_rev_E - vm_all) * e_syn_all) - ((e_rev_I - vm_all) * i_syn_all) # RWS proxy as Davis et al., 2021

        gauss = lambda r,sigma : (1 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))
        dists = np.sqrt((x-electrode_x)**2 + (y-electrode_y)**2)
        lfp = (lfp * gauss(dists,250)).sum(axis=1)
        return lfp

    @named_static("'RegulatorSetup.calculate_input'")
    def calculate_input(regulator):
        if regulator.state is None:
            duration = regulator.stimulation_duration
            t = np.arange(duration) / 1000.0

            target_signal = np.full_like(t, 50.0)
            #target_signal[t%4000 > 1000] = 10

            regulator.state = SimpleNamespace(
                mu=0.0001,
                lms_window=3,
                lms_weights=np.ones(3) / 3,
                lms_error_history=[],
                error=0,
                history=None,
                control_signal=0.5,
                target_signal=target_signal,
                lfp=0
            )
        s = regulator.state
        current_error = s.error

        sec = int(regulator.current_time() / 1000)
        stage = (sec // 2) % 10 + 1
        if sec % 2 == 0:
            s.control_signal = stage * 0.1
        else:
            s.control_signal = 0.0

        circular_mask = simple_shapes_binary_mask(
            regulator.stimulator_coords_x,
            regulator.stimulator_coords_y,
            'circle',
            ParameterSet({'coords': [0, 0], 'radius': RegulatorSetup.stim_circle_radius})
        )
        print("Current time: ",regulator.current_time())
        s.control_signal = 0.2 if regulator.current_time() % 400 < 200 else 0

        input_signal = circular_mask * s.control_signal * np.ones((
            regulator.parameters.state_update_interval // regulator.parameters.update_interval,
            len(regulator.stimulator_coords_x),
            len(regulator.stimulator_coords_y)
        ))
        print(f"LMS control signal: {s.control_signal:.4f}")
        RegulatorSetup.append_history(s, {"LMS": s.control_signal, "error": current_error})
        return input_signal.transpose((1, 2, 0))

    @named_static("'RegulatorSetup.update_state'")
    def update_state(regulator):

        curr_time, prev_time = regulator.current_time(), regulator.current_time() - regulator.parameters.state_update_interval
        last_spiketrains = regulator.get_recording("spikes",t_start=prev_time,t_stop=curr_time)

        in_circle_mask = np.sqrt(np.sum(np.array(regulator.recorded_neuron_positions()) ** 2, axis=0)) < RegulatorSetup.stim_circle_radius

        instantaneous_rates = []
        for i in range(len(last_spiketrains)):
            if in_circle_mask[i]:
                rate = len(last_spiketrains[i]) / (regulator.parameters.state_update_interval / 1000.0)
                instantaneous_rates.append(rate)
        current_rate = np.mean(instantaneous_rates) if instantaneous_rates else 0.0

        RegulatorSetup.append_history(regulator.state, {"Current rate": current_rate})

        alpha = 0.1
        if not hasattr(regulator.state, 'smoothed_rate'):
            regulator.state.smoothed_rate = current_rate
        else:
            regulator.state.smoothed_rate = alpha * current_rate + (1 - alpha) * regulator.state.smoothed_rate

        index = int(regulator.current_time() // regulator.parameters.update_interval) - 1
        if index < len(regulator.state.target_signal):
            target = regulator.state.target_signal[index]
        else:
            target = regulator.state.target_signal[-1]
        regulator.state.error = target - regulator.state.smoothed_rate

        print("Current instantaneous rate: %.3f, EWMA smoothed rate: %.3f, error: %.3f" %
              (current_rate, regulator.state.smoothed_rate, regulator.state.error))

        regulator.state.lfp = RegulatorSetup.calculate_lfp(regulator)
        RegulatorSetup.append_history(regulator.state, {
            "Firing rate": regulator.state.smoothed_rate,
            "LFP": regulator.state.lfp,
            "error": regulator.state.error
        })

        RegulatorSetup.plot_history(regulator)
        RegulatorSetup.plot_orientations_with_stimulus(regulator)

    @staticmethod
    def plot_orientations_with_stimulus(regulator):
        recorded_x, recorded_y = regulator.recorded_neuron_positions()
        ors = regulator.recorded_neuron_orientations()
        stim_x, stim_y = regulator.stimulator_coords_x, regulator.stimulator_coords_y
        input_signal_nonzeros = (regulator.input_signal.mean(axis=-1) > 0).astype(float).flatten()
        fig = pylab.figure(figsize=(20, 20))
        fontsize = 50
        pylab.axis('equal')
        im = pylab.scatter(recorded_x, recorded_y, c=ors, cmap='hsv', vmin=0, vmax=np.pi, s=400)
        led_size = (regulator.parameters.spacing * 72 / fig.dpi) ** 2
        pylab.scatter(stim_x.flatten(), stim_y.flatten(), alpha=0.4 * input_signal_nonzeros, color='k', marker='s', s=led_size)
        pylab.xlabel("x (um)", fontsize=fontsize)
        pylab.ylabel("y (um)", fontsize=fontsize)
        pylab.xticks(fontsize=fontsize)
        pylab.yticks(fontsize=fontsize)
        cbar = pylab.colorbar(im, aspect=17, ax=pylab.gca(), fraction=0.0527)
        cbar.set_label(label='Orientation', labelpad=-10, fontsize=fontsize)
        cbar.set_ticks([0, np.pi], labels=["0", "$\pi$"], fontsize=fontsize)
        fig.savefig("or_map.png")
        pylab.close()

    @staticmethod
    def plot_history(regulator):
        state = regulator.state
        len_error = len(state.history["error"])
        len_firing = len(state.history["Firing rate"])
        len_lms = len(state.history["LMS"]) if "LMS" in state.history else 0
        min_length = min(len_error, len_firing, len_lms) if len_lms > 0 else min(len_error, len_firing)
        t = np.arange(min_length) * regulator.parameters.state_update_interval / 1000
        fig, ax = pylab.subplots(2, 1, figsize=(10, 8))
        #ax[0].plot(t, np.array(state.history["error"])[:min_length], c='r', label="Error")
        #ax[0].plot(t, np.array(state.history["Firing rate"])[:min_length], c='k', label="Firing rate (sp/s)")
        lfp_plot = np.hstack(state.history["LFP"])
        ax[0].plot(np.linspace(0,t[-1],len(lfp_plot),endpoint=True), lfp_plot, c='b', label="LFP")
        ax2 = ax[0].twinx()
        ax2.plot(t, np.array(state.history["Firing rate"])[:min_length], c='k', label="Firing rate (sp/s)")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("LFP")
        ax2.set_ylabel("Firing rate (sp/s)")
        ax[0].legend()
        if "LMS" in state.history:
            ax[1].plot(t, np.array(state.history["LMS"])[:min_length], '--', label="LMS Control Signal")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("Control Signal")
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("autocontrol.png")
        pylab.close(fig)

        cols = [
            state.history["Current rate"][:min_length],
            state.history["Firing rate"][:min_length],
            state.history["LFP"][:min_length],
            state.history["error"][:min_length]
        ]
        header = "Time (s), Instantaneous Rate (sp/s), Firing Rate (sp/s), LFP, Error"

        if len_lms > 0:
            cols.append(state.history["LMS"][:min_length])
            header += ", Control Signal"

        data = np.column_stack((t, *cols))
        np.savetxt("history_data.csv", data, delimiter=",", header=header, comments="")

def closed_loop_experiment(model):
    return [
        ClosedLoopOptogeneticStimulation(
            model,
            MozaikExtendedParameterSet({
                "num_trials": 1,
                "duration": 4000,
                "stimulator_array_list": [
                    {
                        "sheet": "V1_Exc_L2/3",
                        "name": "closed_loop_array",
                        "input_calculation_function": RegulatorSetup.calculate_input,
                        "state_update_function": RegulatorSetup.update_state,
                    }
                ],
            }),
        ),
    ]
