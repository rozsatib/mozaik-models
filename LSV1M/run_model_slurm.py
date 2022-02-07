# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import (
    CombinationParameterSearch,
    SlurmSequentialBackend,
)
import numpy
import time

slurm_options = ["-J default", "--exclude=w[1-7,12]", "--mem=64gb", "--hint=nomultithread"]

CombinationParameterSearch(
    SlurmSequentialBackend(
        num_threads=16,
        num_mpi=1,
        path_to_mozaik_env="/home/rozsa/virt_env/mozaik/bin/activate",
        slurm_options=slurm_options,
    ),
    {
        #'sheets.l4_cortex_exc.params.density' : [1500,2000,2500,3000,4000,5000,6000,8000,10000,15000,20000,25000],
        #'sheets.l23_cortex_exc.L23ExcL23ExcConnection.weight_functions.f1.params.sigma':[0.7,0.6,0.5],
        #'sheets.l4_cortex_exc.L4ExcL4ExcConnection.weight_functions.f1.params.sigma':[0.75,0.7,0.65,0.6,0.55],
        #'sheets.l23_cortex_exc.L23ExcL23ExcConnection.weight_functions.f1.params.sigma':[0.75,0.7,0.65,0.6,0.55],
        #'sheets.l23_cortex_exc.L23ExcL23ExcConnection.weight_functions.f3.params.arborization_scaler':[4,8,16,32,64,128],

        'sheets.l4_cortex_exc.params.cell.params.tau_m': [8],
        'sheets.l4_cortex_inh.params.cell.params.tau_m': [9],
        'sheets.l4_cortex_inh.params.cell.params.v_rest': [-78],
        'sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight' : [0.00016],
        'sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight' : [0.00018],
        'sheets.l4_cortex_exc.AfferentConnection.base_weight' : [0.00115],
        'sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight' : [0.0015],
        'sheets.l23_cortex_inh.L23InhL23ExcConnection.short_term_plasticity.tau_rec' : [30],
        'sheets.l23_cortex_exc.feedback_arborization_ratio': [3],
        'sheets.l23_cortex_exc.K' : [2300],

        'sheets.l4_cortex_exc.params.cell.params.tau_syn_I' : [3.2],
    },
).run_parameter_search()
