#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import SingleOptogeneticArrayStimulus
from mozaik.experiments.closed_loop import ClosedLoopOptogeneticStimulation
from mozaik.sheets.direct_stimulator import simple_shapes_binary_mask
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from types import SimpleNamespace
import numpy as np

class RegulatorSetup:
    stim_circle_radius = 150 # um

    @staticmethod
    def append_history(state,update_dict):
        if state.history == None:
            names = ["P","I","D","PID","error","Firing rate"]
            state.history = {n : [] for n in names}
        for k in update_dict:
            state.history[k].append(update_dict[k])

    @staticmethod
    def calculate_input(regulator):
        if regulator.state is None: # Regulator state is not initialised
            regulator.state = SimpleNamespace(kp=0.000001, ki=0.000005, kd=0.001, integral=0, error=0, previous_error=0,
                                target_signal = np.ones((regulator.stimulation_duration)) * 30, history=None)
        s = regulator.state
        proportional = s.kp * s.error
        s.integral += s.error * regulator.parameters.state_update_interval
        integral = s.ki * s.integral
        derivative = s.kd * (s.error - s.previous_error) / regulator.parameters.state_update_interval
        s.previous_error = s.error
        control_signal = proportional + integral + derivative
        RegulatorSetup.append_history(s,{"P":proportional,"I":integral,"D":derivative,"PID":control_signal})
        # Clamp the output - it cannot be negative, and inputs above 1 drive the network into unreasonable firing rates
        control_signal = min(1,max(control_signal,0))
        # Set the input for the next state update interval
        circular_mask = simple_shapes_binary_mask(regulator.stimulator_coords_x, regulator.stimulator_coords_y,'circle', ParameterSet({'coords':[0,0],'radius':RegulatorSetup.stim_circle_radius}))
        input_signal = circular_mask * np.ones((regulator.parameters.state_update_interval // regulator.parameters.update_interval,len(regulator.stimulator_coords_x),len(regulator.stimulator_coords_y)))
        return input_signal.transpose((1,2,0))

    @staticmethod    
    def update_state(regulator):
        recorded_metric_names = ["spikes","v","gsyn_exc","gsyn_inh"]
        # Set t_start to 0 to retrieve the recording for the entire experiment
        last_spiketrains = regulator.get_recording("spikes",t_start=regulator.current_time()-regulator.parameters.state_update_interval,t_stop=regulator.current_time())
        #last_vm = regulator.get_recording("v",t_start=regulator.current_time()-regulator.parameters.state_update_interval,t_stop=regulator.current_time())
        # The stimulation circle is centered on (0,0)
        in_circle_mask = np.sqrt(np.sum(np.array(regulator.recorded_neuron_positions()) ** 2,axis=0)) < RegulatorSetup.stim_circle_radius
        fr_last = np.mean([len(last_spiketrains[i]) / (regulator.parameters.state_update_interval / 1000)  for i in range(len(last_spiketrains)) if in_circle_mask[i]])
        regulator.state.error = regulator.state.target_signal[int(regulator.current_time() // regulator.parameters.update_interval) - 1] - fr_last
        print("New FR last: %.3f, error: %.3f" % (fr_last, regulator.state.error))
        RegulatorSetup.append_history(regulator.state,{"Firing rate":fr_last,"error":regulator.state.error})
        RegulatorSetup.plot_history(regulator)
        RegulatorSetup.plot_orientations_with_stimulus(regulator)

    @staticmethod 
    def plot_orientations_with_stimulus(regulator):
        # Retrieve neuron orientations and positions
        recorded_x,recorded_y = regulator.recorded_neuron_positions()
        ors = regulator.recorded_neuron_orientations()
        # Retrieve optogenetic stimulator LED coordinates
        stim_x,stim_y = regulator.stimulator_coords_x, regulator.stimulator_coords_y
        # Retrieve mean LED intensity across current update interval
        input_signal_nonzeros = (regulator.input_signal.mean(axis=-1) > 0).astype(float).flatten()
        import pylab
        fig = pylab.figure(figsize=(20,20))
        fontsize=50
        pylab.axis('equal')
        im = pylab.scatter(recorded_x, recorded_y, c=ors, cmap='hsv', vmin=0, vmax=np.pi,s=400)
        # Calculate LED size in image coordinates
        led_size = (regulator.parameters.spacing * 72 / fig.dpi) ** 2  
        # Plot LED array, only show active LEDs
        pylab.scatter(stim_x.flatten(),stim_y.flatten(),alpha=0.4 * input_signal_nonzeros,color='k',marker='s',s=led_size)
        pylab.xlabel("x (um)",fontsize=fontsize)
        pylab.ylabel("y (um)",fontsize=fontsize)
        pylab.xticks(fontsize=fontsize)
        pylab.yticks(fontsize=fontsize)
        cbar = pylab.colorbar(im,aspect=17,ax=pylab.gca(),fraction=0.0527)
        cbar.set_label(label='Orientation', labelpad=-10,fontsize=fontsize)
        cbar.set_ticks([0,np.pi],labels=["0","$\pi$"],fontsize=fontsize)
        fig.savefig("or_map.png")
        pylab.close()

    @staticmethod 
    def plot_history(regulator):
        state = regulator.state
        import pylab
        fig,ax = pylab.subplots(2,1)
        t = np.arange(len(state.history["error"])) * regulator.parameters.state_update_interval / 1000
        ax[0].plot(t,state.history["error"],c='r')
        ax[0].plot(t,state.history["Firing rate"],c='k')
        ax[0].legend(["Error","Firing rate (sp/s)"])
        ax[1].plot(t,state.history["P"])
        ax[1].plot(t,state.history["I"])
        ax[1].plot(t,state.history["D"])
        ax[1].plot(t,state.history["PID"],'--')
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Firing rate (sp/s)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend(["P","I","D","PID"])
        fig.savefig("autocontrol.png")
        pylab.close()

def closed_loop_experiment(model):
    return [
        ClosedLoopOptogeneticStimulation(
        model,
        MozaikExtendedParameterSet(
            {
                "num_trials": 1,
                "duration": 5000,
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
        ),
    ]
