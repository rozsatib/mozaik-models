from mozaik.storage.queries import *
from mozaik.analysis.data_structures import PerNeuronValue
import sys

def rf_params_from_annotations(data_store, sheet="V1_Exc_L4"):
    """
    Retrieves receptive field parameters of V1 neurons from their annotations saved in a
    datastore. These are the parameters used to create the LGN afferent connectivity
    to V1 layer 4. They may slightly differ from empirically measured parameters
    due to sampling during connectivity creation.


    Parameters
    ----------
    data_store : DataStore
                 Datastore to retrieve the annotations from
    sheet : string
            Code of the neuron layer/cell type to retrieve annotations for.
            Can be "V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"
            Defaults to "V1_Exc_L4", which are excitatory V1 Layer 4 neurons

    Returns
    -------
    rf_params : dict(neuron_id : dict(rf_parameter_name : parameter_value))
                Dictionary, where the keys are neuron ids for the given neuron sheet,
                and values are dictionaries of parameters, which are in the form of
                rf_parameter_name = parameter_value.
                Example: { 1 : { "LGNAfferentOrientation" : 1.57079632679, "size" : 2 } }
    """

def save_rf_params(data_store, rf_params, sheet, only_existing=True):
    if only_existing:
        results = data_store.get_analysis_result(
            identifier="PerNeuronValue",
            sheet_name=sheet,
        )
        if len(results) == 0:
            print("Error: No overlap between data store neurons and RF param neurons", file=sys.stderr)
            exit(1)
        neurons = list(set(results[0].ids) & set(rf_params.keys()))
    else:
        neurons = list(rf_params.keys())

    annotations = list(rf_params[neurons[0]].keys())
    data = [[] for a in annotations]

    for i in range(len(annotations)):
        for n_id in neurons:
            data[i].append(rf_params[n_id][annotations[i]])

    for i in range(len(annotations)):
        data_store.full_datastore.add_analysis_result(
            PerNeuronValue(
                data[i], neurons, None, value_name=annotations[i], sheet_name=sheet, period=None
            )
        )


def dummy_experiment(model):
    from mozaik.experiments import NoStimulation

    return [NoStimulation(model, ParameterSet({"duration": 10}))]
