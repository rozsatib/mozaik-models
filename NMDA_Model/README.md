# NMDA_MODEL 

Incorporates multisynapse neuron models to implement multiple types of excitatory or inhibitory synapses based on different temporal dynamics and reversal potential. This version implments both AMPA, NMDA and GABA-A but can be easily reparametrized to also include others like GABA-B

To run the models present in this repository one must first install the Mozaik package and its dependencies (see [here](https://github.com/CSNG-MFF/mozaik) for more information and detailed installation instructions).

## Instructions

    1 Running the different experiments:

        First, in run_parameter_search.py, replace the field [PATH_TO_ENV] by the path of the activate executable of your virtual environment (for example, $HOME/virt_env/mozaik/bin/activate)

            - Drifting grating and natural images protocol::

                python run_parameter_search.py run.py nest param/defaults (~10h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n 16 python run.py nest 1 param/defaults SelfSustainedPushPull

            - Size tuning protocol::

                python run_parameter_search.py run_stc.py nest param/defaults (~10h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n 16 python run_stc.py nest 1 param/defaults SelfSustainedPushPull

            - Spontaneous activity protocol::

                     python run_parameter_search.py run_spont.py nest param_spont/defaults (~1h30 of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     mpirun -n 16 python run_spont.py nest 1 param_spont/defaults SelfSustainedPushPull



    2 Description of the files:
    
        - analysis_and_visualization.py: Contains the code for the different analysis and plotting. The function `perform_analysis_and_visualization` runs the analysis and the plots corresponding to the fullfield drifting grating and natural images protocol. The functions `perform_analysis_and_visualization_stc` and `perform_analysis_and_visualization_spont` do the same respectively for the size tuning and spontaneous activity protocol.
        - experiments.py: Defines for each protocol which experiments will be run as well as their parameters. `create_experiments` corresponds to the fullfield drifting grating protocol, `create_experiments_stc` corresponds to the size tuning protocol, and `create_experiments_spont` corresponds to the spontaneous activity protocol.
        - eye_path.pickle: Contains the coordinates of the eye path that is used by default in the natural image protocol. 
        - image_naturelle_HIGHC.bmp: The natural image that is used by default in the natural image protocol.
        - model.py: Contains the code which creates each layers of the model and build the connections based on the parameters used to run the model.
        - or_map_new_16x16: Contains the precomputed orientation map. A specific central portion of it can be cropped based on the or_map_stretch parameters.
        - param: Defines the parameters used in the fullfield drifting gratings and natural images protocol as well as for the size tuning protocol. The 'default' file contains the basic parameters of the model as well as the path of the parameters corresponding to each sheets of the model. 'SpatioTemporalFilterRetinaLGN_defaults' contains the parameters of the input model as well as the parameters of the LGN neurons. `l4_cortex_exc', `l4_cortex_inh`, `l23_cortex_exc`, `l23_cortex_inh` contains the parameters of each cortical population as well as the parameters of the connections of the model. Each '_rec' file contains the recording parameters for each population of the model.
        - param_spont: Contains the parameters for the spontaneous activity protocol of the model. The only difference with the `param` directory resides in the recording parameters.
        - parameter_search_analysis.py: Runs the analysis one multiple simulation directories belonging to the same parameter search, distributed on multiple computational nodes and using by default Slurm for scheduling.
        - run.py: Runs the model. Defines which experimental protocol (as defined in experiments.py) and which analysis (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol.
        - run_analysis.py: Runs only the analysis on the model on a mozaik datastore. Defines which analysis  (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol analysis.
        - run_parameter_search.py: Defines the parameters that will be used when running a search across multiple parameters. The parameter search will be distributed on different computational nodes, using Slurm as the scheduler by default. 
        - run_spont.py: Same as `run.py`, but runs the spontaneous activity protocol by default. 
        - run_stc.py: Same as `run.py`, but runs the size tuning protocol by default. 
        - visualization_functions.py: Contains the code specific to each figure. 

    3 Description of new parameters added in param files:

        Each neuron model has now a parameter 'receptors'. This should be set to None if the model is not a multisynapse model. Otherwise it should contain a dictionnary with the following structure:
            'receptors:{
                'name_receptor1': {
                  'name': synapse_model_type,
                  'params': synapse_model_params,
                },
                'name_receptor2': {
                  'name': synapse_model_type,
                  'params': synapse_model_params,
                },
                ...
                'name_receptorN': {
                  'name': synapse_model_type,
                  'params': synapse_model_params,
                },                
            }
        An arbitrary number of receptors can be defined this way.
        The names of the receptors are arbitrarily defined by the user. However, their corresponding conductance wil be included in the computation of excitatory (or inhibitory) conductance during analysis only if their name is included in the list passed as argument `excitatory_receptors` (or inhibitory_receptors`) in the function ExcitatoryConductanceGenerator (or InhibitoryConductanceGenerator) in analysis_and_visualization.py.
        `synapse_model_type` is a string which can take the following values 'ExpPSR', 'AlphaPSR' and 'BetaPSR'. These correspond to synapse models defined in PyNN, with ExpPSR correcponding to normal exponential decay, AlphaPSR corresponding to alpha synapses with rise and decay controlled by a single time constrant, and BetaPSR corresponding to a difference of two exponentials each controlling rise and decay with separate time constants (for reference see Roth, A., & van Rossum, M. C. (2009). Modeling synapses. Computational modeling methods for neuroscientists, 6(139), 700.).
        `params` is a dictionary with the following structure:
            'params': {
                "tau_syn": float,
                "e_syn": float,
            },        
        or for BetaPSR synapses:
            'params': {
                "tau_rise": float,
                "tau_decay": float,
                "e_syn": float,
            },        
        `e_syn` defines the reversal potential of the synapse. Crucially, it is the parameter which defines whether the synapse is excitatory or inhibitory (depending if it's greater or smaller than the resting potential of the neuron). Other parameters are the time constants of the synapse.

        The model presented here implements 4 receptors (AMPA, NMDA, GABAA and GABAB). However, no GABAB synpases are created as the corresponding connections are defined with 'num_samples':0 in the parameters. The following parameters are used defined to compute the number of samples of each receptor type for each connections
            - l4_cortex_exc/L4_NMDA_factor: The proportion of recurrent excitatory-to-excitatory connections in layer 4 which target NMDA receptors. The proportion of AMPA synapses is equal to 1 - L4_NMDA_factor
            - l4_cortex_exc/L4_GABAB_factor: The proportion of inhibitory connections in layer 4 which target GABA-B receptors. The proportion of GABA-A synapses is equal to 1 - L4_GABAB_factor
            - l23_cortex_exc/L23_NMDA_factor: The proportion of recurrent excitatory-to-excitatory connections in layer 2/3 which target NMDA receptors. The proportion of AMPA synapses is equal to 1 - L23_NMDA_factor
            - l23_cortex_exc/L23_GABAB_factor: The proportion of inhibitory connections in layer 2/3 which target GABA-B receptors. The proportion of GABA-A synapses is equal to 1 - L23_GABAB_factor
            - l4_cortex_exc/inh_NMDA_factor: Modulates the *_NMDA_factor parameters for connections targeting inhibitory neurons. The proportion of recurrent excitatory-to-inhibitory connections in layer 4 which target NMDA receptors is equal to L4_NMDA_factor * inh_NMDA_factor
            - l4_cortex_exc/inh_GABAB_factor: Modulates the *_GABAB_factor parameters for connections targeting inhibitory neurons. The proportion of recurrent inhibitory-to-inhibitory connections in layer 4 which target NMDA receptors is equal to L4_GABAB_factor * inh_GABAB_factor
            - l4_cortex_exc/afferent_NMDA_factor: Modulates the *_NMDA_factor parameters for afferent connections. The proportion of excitatory-to-excitatory afferent connections in layer 4 which target NMDA receptors is equal to L4_NMDA_factor * afferent_NMDA_factor.
            - l4_cortex_exc/feedback_NMDA_factor: Modulates the *_NMDA_factor parameters for feedback connections. The proportion of excitatory-to-excitatory feedback connections in layer 4 which target NMDA receptors is equal to L4_NMDA_factor * feedback_NMDA_factor.
            - l4_cortex_exc/NMDA_weight_scaler: Scaler for the weights of NMDA synapses relatively to those of corresponding AMPA synapses.

