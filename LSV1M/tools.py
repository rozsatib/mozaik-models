from mozaik.storage.queries import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues

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
                Example: { 1 : { "orientation" : 1.57079632679, "size" : 2 } }
    """

    assert sheet in data_store.sheets(), (
        sheet
        + " is not one of the sheets of the model, which are: "
        + str(data_store.sheets())
    )

    annotations = [
        "LGNAfferentOrientation",
        "LGNAfferentAspectRatio",
        "LGNAfferentFrequency",
        "LGNAfferentSize",
        "LGNAfferentPhase",
        "LGNAfferentX",
        "LGNAfferentY",
    ]

    keys = [
        "orientation",
        "aspect_ratio",
        "spatial_frequency",
        "size",
        "phase",
        "x",
        "y",
    ]

    NeuronAnnotationsToPerNeuronValues(data_store, ParameterSet({})).analyse()

    analog_ids = (
        param_filter_query(data_store, sheet_name=sheet)
        .get_segments()[0]
        .get_stored_esyn_ids()
    )
    spike_ids = (
        param_filter_query(data_store, sheet_name=sheet)
        .get_segments()[0]
        .get_stored_spike_train_ids()
    )

    ids = list(set(spike_ids) & set(analog_ids))

    rf_params = {id_: {} for id_ in ids}
    for i in range(len(annotations)):
        res = data_store.get_analysis_result(
            identifier="PerNeuronValue",
            value_name=annotations[i],
            sheet_name=sheet,
        )

        if len(res) == 0:
            continue

        v = res[0].get_value_by_id(ids)
        for j in range(len(v)):
            rf_params[ids[j]][keys[i]] = v[j]

    return rf_params
