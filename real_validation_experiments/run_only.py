#!/usr/local/bin/ipython -i
from parameters import ParameterSet
import numpy as np
import json
from mozaik.tools.distribution_parametrization import load_parameters
from mozaik.controller import run_workflow
from model import SelfSustainedPushPull
from mozaik.experiments.vision import MeasureFrequencySensitivity, MeasureFlatLuminanceSensitivity, MeasureSparseBar
import os
import sys
from copy import deepcopy
import shutil
from pathlib import Path
from experiments import select_experiment_func

import nest
nest.Install("stepcurrentmodule")

def close_all_log_handlers():
    import logging
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
            logging.root.removeHandler(handler)
        except Exception as e:
            print(f"Error closing log handler: {e}")

def move_simulation_result(data_store, dir_name):
    new_dir = Path(data_store.parameters.root_directory).parent / dir_name
    os.makedirs(new_dir)
    for f in os.listdir(data_store.parameters.root_directory):
        if f.endswith("pickle") or f.endswith("json") or f.endswith("png") or f.endswith("csv"):
            shutil.move(os.path.join(data_store.parameters.root_directory,f),new_dir)
    close_all_log_handlers()
    shutil.rmtree(data_store.parameters.root_directory)

def run_simulation(e, paper, fig_name, fast_run):
    data_store, _ = run_workflow(
        "LGN_validation_run", SelfSustainedPushPull, select_experiment_func(e, fast_run)
    )
    data_store.save()

    move_simulation_result(data_store, paper + "_" + fig_name)

def find_experiment(ref_data, experiment_id):
    for paper in ref_data:
        for fig_name in ref_data[paper]["data"]:
            e = ref_data[paper]["data"][fig_name]
            if e["id"] == experiment_id:
                return paper, fig_name, e
    raise ValueError("Experiment with id: %d not found!" % experiment_id)

def get_s_sc(freq):
    a,b = 0.032790027697085944, 0.7938362678043797
    return a + 1/freq * b

def get_t_lsc(f):
    a,b = 2.5396758618754514, 1.9677575087817103 
    #return a+b/f TODO: switch to this implementation after t_lsc is inverted!!
    return (f - a) / b

all_p = load_parameters("param/defaults")
RF_p = all_p.sheets.retina_lgn.params.receptive_field

n_pixels_time = RF_p.duration / RF_p.temporal_resolution
n_pixels_space = RF_p.width / RF_p.spatial_resolution

assert RF_p.width == RF_p.height
assert all_p.visual_field.size[0] == all_p.visual_field.size[1]

with open("reference_data.json") as f:
    ref_data = json.load(f)

fast_run = False
run_parallel_slurm = False
sf = 0#1
tf = 0#1
c = 0#1
lum = 0#1
bars = 1
variance = 0#1

from mozaik.cli import parse_workflow_args
def fetch_selected_experiment_id():
    modified_params = parse_workflow_args()[4]
    assert "trial" in modified_params.keys(), "trial needs to be in modified params to run experiments in parallel!"
    return modified_params["trial"]

# Make a list of unique experiment parameters
for paper in ref_data:
    for fig_name in ref_data[paper]["data"]:
        e = ref_data[paper]["data"][fig_name]
        if run_parallel_slurm:
            selected_experiment_id = fetch_selected_experiment_id()
            if ref_data[paper]["data"][fig_name]["id"] != selected_experiment_id:
                continue

        # Fetch trial somehow
        if e["type"] == "contrast" and not c:
            continue
        if e["type"] == "temporal_frequency" and not tf:
            continue
        if e["type"] == "spatial_frequency" and not sf:
            continue
        if e["type"] == "luminance" and not lum:
            continue
        if e["type"] == "trial-to-trial-variance" and not variance:
            continue
        if e["skip"]:
            continue
        print(paper)
        print(fig_name)
        RF_p_modified = RF_p.func_params.copy()
        all_p_c = ParameterSet(deepcopy(all_p))
        all_p_c.sheets.retina_lgn.params.receptive_field.func_params = RF_p_modified

        if e["opt_sf"]:
            RF_p_modified["s_sc"] = get_s_sc(e["opt_sf"])
            rf_w = np.ceil(5 * RF_p.func_params["sigma_s"] * RF_p_modified["s_sc"] / 2) * 2
            ds = rf_w/n_pixels_space
            all_p_c.sheets.retina_lgn.params.receptive_field.width = rf_w
            all_p_c.sheets.retina_lgn.params.receptive_field.height = rf_w
            all_p_c.sheets.retina_lgn.params.receptive_field.spatial_resolution = ds

            n_pixels_visual_field = all_p.visual_field.size[0] / ds
            np.testing.assert_almost_equal(n_pixels_visual_field,int(n_pixels_visual_field))

        if e["opt_tf"]:
            tf = e["opt_tf"]
        else:
            tf = 5.5333
        RF_p_modified["t_lsc"] = get_t_lsc(tf)

        rf_dur = 4 * np.abs(RF_p.func_params["t1"]) / RF_p_modified["t_lsc"]
        dt = rf_dur / n_pixels_time
        n_dt_in_ui = np.ceil(all_p.input_space.update_interval / dt)
        dt = np.round((all_p.input_space.update_interval / n_dt_in_ui) * 10) // 10
        dt = max(dt, 1)
        print(dt)
        n_pixels = (int(rf_dur / dt) // 7) * 7
        rf_dur = n_pixels * dt
        all_p_c.sheets.retina_lgn.params.receptive_field.duration = rf_dur
        print(rf_dur)
        all_p_c.sheets.retina_lgn.params.receptive_field.temporal_resolution = dt
        if ref_data[paper]["background_luminance"] != None:
            all_p_c.input_space.background_luminance = ref_data[paper][
                "background_luminance"
            ]

        sys.argv[3] = str(all_p_c) + "\n"
        run_simulation(e, paper, fig_name, fast_run)

if bars:
    if run_parallel_slurm:
        selected_experiment_id = fetch_selected_experiment_id()
        ref_data_ids = [ref_data[paper]["data"][fig_name]["id"] for paper in ref_data for fig_name in ref_data[paper]["data"]]
        if selected_experiment_id in ref_data_ids:
            exit()
    print("Running sparse noise")
    sys.argv[3] = str(all_p)

    data_store, model = run_workflow(
        "LGN_validation_run",
        SelfSustainedPushPull,
        select_experiment_func({"type": "sparse_noise"}, fast_run)
    )
    data_store.save()

    move_simulation_result(data_store, "sparse_noise")
