# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend
import numpy
import time

slurm_options = [
        '-J V1Model',
        '--hint=nomultithread',
        '--mem-per-cpu=3999',
        ]

if True:
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=16, num_mpi=1, path_to_mozaik_env= '/home/cagnol/virt_env/mozaik_tibor/bin/activate', slurm_options=slurm_options ), {



      'sheets.l4_cortex_exc.L4ExcL4ExcConnection.base_weight' : [0.00015],
      'sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight' : [0.00021],
      'sheets.l4_cortex_exc.AfferentConnection.base_weight' : [0.00115],
      'sheets.l4_cortex_inh.L4InhL4ExcConnection.base_weight' : [0.0016],
      'sheets.l4_cortex_inh.L4InhL4ExcConnection.short_term_plasticity.tau_rec' : [70],
      'sheets.l23_cortex_exc.L23ExcL4ExcConnection.base_weight' : [0.00015],
      'sheets.l23_cortex_exc.L23ExcL4ExcConnection.short_term_plasticity.tau_rec' : [30],
      'sheets.l23_cortex_exc.K' : [2650],
      'sheets.l23_cortex_exc.layer23_aff_ratio' : [0.22],
      'sheets.l23_cortex_exc.p_inh' : [2],
    }).run_parameter_search()

