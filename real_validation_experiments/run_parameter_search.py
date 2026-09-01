# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import (
    CombinationParameterSearch,
    SlurmSequentialBackend,
)
import numpy
import numpy as np
import time

slurm_options = ["-J LGN", "--exclude=w[13-17]", "--hint=nomultithread"]#,"--begin=19:00"]
#slurm_options = ["-J LGN", "--hint=nomultithread"]

CombinationParameterSearch(
    SlurmSequentialBackend(
        num_threads=8,
        num_mpi=1,
        path_to_mozaik_env="/home/rozsa/virt_env/mozaik_lgn/bin/activate",
        slurm_options=slurm_options,
    ),
    {
        #"trial": list(range(32)),
        "trial": [0],
        # Before temporal adjustment because of neuron dynamics
        #"sheets.retina_lgn.params.receptive_field.func_params.K1": [1.37564875],
        #"sheets.retina_lgn.params.receptive_field.func_params.K2": [0.31283973],
        #"sheets.retina_lgn.params.receptive_field.func_params.c1": [0.61471462],
        #"sheets.retina_lgn.params.receptive_field.func_params.c2": [0.22824534],
        #"sheets.retina_lgn.params.receptive_field.func_params.t1": [-44.87613346],
        #"sheets.retina_lgn.params.receptive_field.func_params.t2": [-93.08789616],

        # After temporal adjustment because of neuron dynamics
        #"sheets.retina_lgn.params.receptive_field.func_params.K1": [3],
        #"sheets.retina_lgn.params.receptive_field.func_params.K2": [0.37],
        #"sheets.retina_lgn.params.receptive_field.func_params.c1": [0.983543392],
        #"sheets.retina_lgn.params.receptive_field.func_params.c2": [0.365192544],
        #"sheets.retina_lgn.params.receptive_field.func_params.t1": [-71.801813536],
        #"sheets.retina_lgn.params.receptive_field.func_params.t2": [-148.940633856],

        #"sheets.retina_lgn.params.receptive_field.func_params.As": [0.00002,0.00002336,0.000025,0.00003],
        #"sheets.retina_lgn.params.receptive_field.func_params.K1": [2.5,3,3.5,4],

        # sf: 0.52, 0.5, 0.94, 0.65, 0.4, 0.8, 1.0, 0.75, 0.45, 1.4, 0.43, 0.24,
        # tf: 4, 6.3, 8, 7.8,
        # sf:(0.24,1.4)
        # tf:(4,8)

        #"sheets.retina_lgn.params.receptive_field.func_params.As": [0.00002336],
        #"sheets.retina_lgn.params.receptive_field.func_params.K1": [3],
        #"sheets.retina_lgn.params.receptive_field.func_params.K2": [0.37],

        # 2 x 1D scaling change search
        #"sheets.retina_lgn.params.receptive_field.func_params.s_sc": list(np.linspace(0.5,4.5,64,endpoint=False)),

        #"sheets.retina_lgn.params.receptive_field.func_params.t_lsc": [0.4,0.6,0.8,1.0,1.4,1.8],
        #"sheets.retina_lgn.params.receptive_field.func_params.t_lsc": [0.6,0.6000001],
        #"sheets.retina_lgn.params.receptive_field.func_params.t_lsc": [0.6],
        # t_lsc_2 = 2/3 t_lsc

        #"sheets.retina_lgn.params.receptive_field.func_params.c2": [0.22824534 * 2 / 3],
        #"sheets.retina_lgn.params.receptive_field.func_params.t2": [-93.08789616 / (2 / 3)],

        #"sheets.retina_lgn.params.receptive_field.func_params.t_msc": list(np.linspace(0.015625,1,64,endpoint=True)),
        # 2D grid search
        #"sheets.retina_lgn.params.receptive_field.func_params.s_sc": list(np.linspace(0.6,3,16,endpoint=True)),
        #"sheets.retina_lgn.params.receptive_field.func_params.t_lsc": list(np.linspace(0.2,6.4,32,endpoint=True)),
        #"sheets.retina_lgn.params.receptive_field.temporal_resolution": [4],#list(np.linspace(1,3,11,endpoint=True)),

        #"input_space.background_luminance": [14,20,21,25,30,32,45,100,120,200],

        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_scaler": [2e-10],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_gain": [0.0115],

        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_scaler_OFF": [1e-10],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_gain_OFF": [0.0115],

        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_scaler_ON": [2e-10],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_gain_ON": [0.018],

        #"sheets.retina_lgn.params.gain_control.non_linear_gain.contrast_gain": [0.11],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.contrast_scaler": [0.0002], # final
        #"sheets.retina_lgn.params.noise.stdev": [1.25],
        #"sheets.retina_lgn.params.noise.mean": [0.66],
        #"sheets.retina_lgn.params.noise.X_ON.stdev": [0.975],
        #"sheets.retina_lgn.params.noise.X_ON.mean": [0.66],

        #"sheets.retina_lgn.params.noise.X_OFF.stdev": [1.25],
        #"sheets.retina_lgn.params.noise.X_OFF.mean": [0.66],


        # Old "working" parameterization
        #"sheets.retina_lgn.params.receptive_field.func_params.As": [0.00002336],
        #"sheets.retina_lgn.params.receptive_field.func_params.K1": [3],
        #"sheets.retina_lgn.params.receptive_field.func_params.K2": [0.37],
        #"sheets.retina_lgn.params.receptive_field.func_params.t_lsc": [2],
        #"sheets.retina_lgn.params.receptive_field.temporal_resolution": [1],
        #"input_space.background_luminance": [14,20,21,25,30,32,45,100,120,200],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.contrast_gain": [0.3],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.contrast_scaler": [0.0005],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_gain": [0.005],
        #"sheets.retina_lgn.params.gain_control.non_linear_gain.luminance_scaler": [5e-9],
        #"sheets.retina_lgn.params.noise.stdev": [1.3],
    },
).run_parameter_search()

