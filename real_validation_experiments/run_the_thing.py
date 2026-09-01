#!/usr/local/bin/ipython -i
import matplotlib

matplotlib.use("Agg")

from parameters import ParameterSet
import numpy as np
import json
from mozaik.tools.distribution_parametrization import load_parameters
import os
import sys
import random

from mozaik.controller import run_workflow
from model import SelfSustainedPushPull
from mozaik.experiments.vision import *
import os
import sys
from copy import deepcopy
from aggr_imp_resp import *
from matplotlib.ticker import ScalarFormatter
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.analysis.vision import *
from mozaik.storage.queries import *
from mozaik.controller import Global
import shutil
from pathlib import Path

import nest
nest.Install("stepcurrentmodule")


class TrialAveragedNthHarmonicResponse(Analysis):
    required_parameters = ParameterSet(
        {
            "bin_length": float,  # the bin length of the PSTH
            "n_harmonic_freq": int,
        }
    )

    def first_n_harmonic_freq(self, s, n, carrier_f, sampling_f):
        f = np.fft.rfft(s)
        f[0] = 0

        ampl = 2 * np.abs(f) / len(s)
        carrier_f_pos = carrier_f * len(s) / sampling_f

        return [
            np.interp((i + 1) * carrier_f_pos, range(len(s) // 2 + 1), ampl)
            for i in range(n)
        ]

    def test_first_n_harmonic_freq(self):
        L = 3
        sf = 1000
        t = np.linspace(0, L, L * sf)
        A1, A2, A3 = 5, 3, 10
        f1, f2, f3 = 1, 12, 4
        p1, p2, p3 = 0, np.pi / 2, np.pi / 3
        s1 = A1 * np.sin(2 * pi * f1 * t + p1)
        s2 = A2 * np.sin(2 * pi * f2 * t + p2)
        s3 = A3 * np.sin(2 * pi * f3 * t + p3)
        s = s1 + s2 + s3
        s *= qt.ms

        binsize = 5
        ss = []
        for i in range((L * sf) // binsize):
            ss.append(np.mean(s[i * binsize : (i + 1) * binsize]))
        sf /= binsize

        carrier_f = f3
        AA3 = self.first_n_harmonic_freq(ss, 3, carrier_f, sf)
        print(AA3, A3)

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = param_filter_query(self.datastore, st_name="FullfieldDriftingSinusoidalGrating", sheet_name=sheet)
            PSTH(
                dsv, ParameterSet({"bin_length": self.parameters.bin_length})
            ).analyse()
            dsv1 = param_filter_query(
                dsv.full_datastore, analysis_algorithm="PSTH", st_name="FullfieldDriftingSinusoidalGrating",sheet_name=sheet
            )
            TrialMean(
                dsv1,
                ParameterSet({"vm": False, "cond_inh": False, "cond_exc": False}),
            ).analyse()

            results = dsv1.full_datastore.get_analysis_result(
                analysis_algorithm="TrialMean", sheet_name=sheet
            )

            for result in results:
                fm = []
                for n_id in result.ids:
                    tf = (
                        MozaikParametrized.idd(result.stimulus_id).temporal_frequency
                        * MozaikParametrized.idd(result.stimulus_id)
                        .getParams()["temporal_frequency"]
                        .units
                    )
                    fm.append(
                        self.first_n_harmonic_freq(
                            result.get_asl_by_id(n_id).flatten(),
                            self.parameters.n_harmonic_freq,
                            tf,
                            qt.s.rescale(result.x_axis_units) / self.parameters.bin_length,
                        )
                    )

                fm = np.array(fm)
                for i in range(self.parameters.n_harmonic_freq):
                    self.datastore.full_datastore.add_analysis_result(
                        PerNeuronValue(
                            fm[:, i],
                            result.ids,
                            result.y_axis_units,
                            stimulus_id=str(result.stimulus_id),
                            value_name="%d. Harmonic Response" % (i + 1),
                            sheet_name=sheet,
                            tags=self.tags,
                            analysis_algorithm=self.__class__.__name__,
                            period=None,
                        )
                    )

def select_experiment_func(e):

    fast = True
    if fast:
        n_trials = 4
        sf_steps = 10
        #tf_steps = 30#15
        c_steps = 3
        lum_steps = 16
    else:
        n_trials = 120
        sf_steps = 30
        #tf_steps = 30
        c_steps = 41 # 21
        lum_steps = 51

    if e["type"] == "contrast":
        experiment_func = lambda model: [
            MeasureFrequencySensitivity(
                model,
                ParameterSet(
                    {
                        "orientation": 0.0,
                        "temporal_frequencies": [e["temporal_frequency"]],
                        "contrasts": list(np.linspace(0, 100, c_steps, endpoint=True)),
                        "grating_duration": 143 * 7 * 2,
                        "spatial_frequencies": [e["spatial_frequency"]],
                        "num_trials": n_trials,
                        "square": False,
                        "shuffle_stimuli": False,
                    }
                ),
            )
        ]
    elif e["type"] == "spatial_frequency":
        experiment_func = lambda model: [
            MeasureFrequencySensitivity(
                model,
                ParameterSet(
                    {
                        "orientation": 0.0,
                        "temporal_frequencies": [e["temporal_frequency"]],
                        "contrasts": [e["contrast"]],
                        "grating_duration": 143 * 7 * 2,
                        "spatial_frequencies": list(
                            np.linspace(0.1, 2.0, sf_steps, endpoint=True)
                        ),
                        "num_trials": n_trials,
                        "square": False,
                        "shuffle_stimuli": False,
                    }
                ),
            )
        ]
    elif e["type"] == "temporal_frequency":
        duration = 10003
        range2 = np.logspace(0,8,9,base=2)
        f0s = np.array(np.arange(1,11)*0.1)
        #duration = 2499
        #range2 = np.logspace(2,5,4,base=2)
        #f0s = np.array(np.arange(1,7)*0.1)

        tfs = np.unique(np.outer(f0s,range2).flatten())
        #tfs = tfs[tfs>=1.0]
        #tfs = tfs[tfs <=20.0]
        tfs.sort()

        experiment_func = lambda model: [
            MeasureFrequencySensitivity(
                model,
                ParameterSet(
                    {
                        "orientation": 0.0,
                        "temporal_frequencies":list(tfs),
                        "contrasts": [e["contrast"]],
                        "grating_duration": duration,
                        "spatial_frequencies": [e["spatial_frequency"]],
                        "num_trials": n_trials,
                        "square": False,
                        "shuffle_stimuli": False,
                    }
                ),
            )
        ]
    elif e["type"] == "luminance":
        experiment_func = lambda model: [
            MeasureFlatLuminanceSensitivity(
                model,
                ParameterSet(
                    {
                        "luminances": list(
                            np.logspace(-2, 2, lum_steps, endpoint=True)
                        ),
                        "step_duration": 10000,
                        "num_trials": n_trials,
                        "shuffle_stimuli": False,
                    }
                ),
            )
        ]
    elif e["type"] == "trial-to-trial-variance":
        experiment_func = lambda model: [
            MeasureFrequencySensitivity(
                model,
                ParameterSet(
                    {
                        "orientation": 0.0,
                        "temporal_frequencies": [e["temporal_frequency"]],
                        "contrasts": [e["contrast"]],
                        "grating_duration": 36 * 7, # close to 500 ms experiment value
                        # TODO: rewrite spatial frequency to be more flexible!
                        "spatial_frequencies": [0.8],
                        "num_trials": 200,
                        "square": False,
                        "shuffle_stimuli": False,
                    }
                ),
            )
        ]
    return experiment_func


def calc_fr(data_store, sheets, ep):
    if ep["type"] == "spatial_frequency":
        dsv = param_filter_query(
            data_store,
            st_name="FullfieldDriftingSinusoidalGrating",
            st_temporal_frequency=ep["temporal_frequency"],
            st_contrast=ep["contrast"],
            sheet_name=sheets,
        )
    elif ep["type"] == "temporal_frequency":
        dsv = param_filter_query(
            data_store,
            st_name="FullfieldDriftingSinusoidalGrating",
            st_spatial_frequency=ep["spatial_frequency"],
            st_contrast=ep["contrast"],
            sheet_name=sheets,
        )
    elif ep["type"] == "contrast":
        dsv = param_filter_query(
            data_store,
            st_name="FullfieldDriftingSinusoidalGrating",
            st_spatial_frequency=ep["spatial_frequency"],
            st_temporal_frequency=ep["temporal_frequency"],
            sheet_name=sheets,
        )
    elif ep["type"] == "luminance":
        dsv = param_filter_query(
            data_store,
            st_name="Null",
            sheet_name=sheets,
        )

    if ep["type"] == "luminance":
        key = "background_luminance"
        TrialAveragedFiringRate(dsv, ParameterSet({})).analyse()
        results = dsv.full_datastore.get_analysis_result(value_name="Firing rate")
    else:
        key = ep["type"]
        TrialAveragedNthHarmonicResponse(
            dsv, ParameterSet({"bin_length": 5, "n_harmonic_freq": 1})
        ).analyse()
        results = dsv.full_datastore.get_analysis_result(
            value_name="1. Harmonic Response"
        )

    fr = {sheet: {} for sheet in sheets}
    for result in results:
        p = load_parameters(str(result), ParameterSet({}))
        sheet = p["sheet_name"]
        p = load_parameters(p["stimulus_id"], ParameterSet({}))
        fr[sheet][p[key]] = np.mean(result.get_value_by_id(result.ids))

    data_store.remove_ads_from_datastore()
    return fr


def plot_fr(fr, title, ep, path, x_log=False, y_log=True):
    xlabel = ep["type"]

    vals = sorted(fr["X_ON_model"].keys())
    on_frs = [fr["X_ON_model"][v] for v in vals]
    off_frs = [fr["X_OFF_model"][v] for v in vals]
    legend = ["LGN ON - model", "LGN OFF - model"]
    ax = pylab.subplot()
    lw = 2.5
    ax.plot(vals, on_frs, linewidth=lw)
    ax.plot(vals, off_frs, linewidth=lw)
    for key in sorted(fr.keys()):
        if "_model" in key:
            continue
        ax.plot(fr[key]["x"], fr[key]["y"], marker="o", linestyle="None")
        legend.append(key)

    if y_log:
        pylab.yscale("log")
    if x_log:
        pylab.xscale("log")
    ax.legend(legend)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    figtitle = "contrast: %d, sf: %.2f cyc/deg, tf: %.2f Hz" % (int(ep["contrast"] or -1), float(ep["spatial_frequency"] or -1), float(ep["temporal_frequency"] or -1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Firing rate (spikes/s)")
    if xlabel == "temporal_frequency" or xlabel == "spatial_frequency":
        y_frs = (np.array(on_frs) + np.array(off_frs))/2
        x_interp = np.logspace(np.log10(vals[0]),np.log10(vals[-1]),100,endpoint=False)
        x_interp[x_interp < min(vals)] = min(vals)
        y_interp = scipy.interpolate.interp1d(vals, y_frs, kind='cubic')(x_interp)
        y_interp = scipy.ndimage.gaussian_filter(y_interp,5)
        max_idx = np.argmax(y_interp)
        xmax, ymax = x_interp[max_idx],y_interp[max_idx]
        plt.plot(x_interp,y_interp,color='k',alpha=0.7)
        plt.plot(xmax,ymax,'or',ms=3)
        figtitle += "\nMaximum frequency: %.3f" % x_interp[max_idx]
        #pylab.show()

    print("%s%s_%s.png" % (path, xlabel, title))
    print("vals: ", vals)
    print("ON_frs: ", on_frs)
    print("OFF_frs: ", off_frs)
    plt.title(figtitle)
    pylab.savefig("%s%s_%s.png" % (path, xlabel, title))
    np.savetxt("%s%s_%s.csv" % (path, xlabel, title), np.vstack([vals,on_frs,off_frs]), delimiter=",")
    pylab.clf()

def plot_the_plot(data_store, ep, paper, fig_name):
    sheets = ["X_ON", "X_OFF"]
    out_dir = data_store.parameters.root_directory
    fr = calc_fr(data_store, sheets, ep)

    title = paper + " " + fig_name
    print("Title: %s" % title)
    data = fr

    if ep["type"] == "luminance":
        for pp in ref_data:
            for fn in ref_data[pp]["data"]:
                ee = ref_data[pp]["data"][fn]
                if ee["id"] == 28:
                    data[pp + " " + fn] = {"x": ee["x"], "y": ee["y"]}
                    print(data)
    else:
        data[title] = {"x": ep["x"], "y": ep["y"]}

    data["X_ON_model"] = data.pop("X_ON")
    data["X_OFF_model"] = data.pop("X_OFF")
    if ep["type"] == "contrast":
        plot_fr(data, title, ep, out_dir, x_log=False, y_log=True)
    elif ep["type"] == "luminance":
        plot_fr(data, title, ep, out_dir, x_log=True, y_log=True)
    else:
        plot_fr(data, title, ep, out_dir, x_log=True, y_log=True)

def spike_count_mean_variance(data_store, sheet):
    dsv = param_filter_query(data_store, st_name="FullfieldDriftingSinusoidalGrating", sheet_name=sheet)
    stim_len = ParameterSet(dsv.get_stimuli()[0]).duration
    sit_neurons_trials = []
    for seg in dsv.get_segments():
        sts = seg.get_spiketrains()
        sit_neurons = []
        for i in range(len(sts)):
            spike_in_time = np.zeros((stim_len))
            for t in sts[i]:
                if int(t) >= stim_len:
                    continue
                spike_in_time[int(t)] += 1

            spike_in_time = np.convolve(spike_in_time,np.ones(50),mode='valid')
            sit_neurons.append(spike_in_time)
        sit_neurons_trials.append(sit_neurons)
    sit_neurons_trials = np.array(sit_neurons_trials)
    spike_count_mean = sit_neurons_trials.mean(axis=0).flatten()
    spike_count_variance = sit_neurons_trials.var(axis=0).flatten()
    return spike_count_mean, spike_count_variance

def trial_to_trial_variance_plot(data_store, e, paper, fig_name):
    out_dir = data_store.parameters.root_directory

    scm_on, scv_on = spike_count_mean_variance(data_store, "X_ON")
    scm_off, scv_off = spike_count_mean_variance(data_store, "X_OFF")

    ax = pylab.subplot()
    ax.plot(scm_on,scv_on,'o',ms=1)
    ax.plot(scm_off,scv_off,'o',ms=1)
    ax.plot(e["x"],e["y"],'o',ms=1)
    ax.legend(['LGN ON - model', 'LGN OFF -model', paper])
    ax.set_xlabel("Spike count mean")
    ax.set_ylabel("Spike count variance")

    np.savetxt("%sspike_count_mean_variability_on.csv" % out_dir, np.vstack([scm_on,scv_on]), delimiter=",")
    np.savetxt("%sspike_count_mean_variability_off.csv" % out_dir, np.vstack([scm_off,scv_off]), delimiter=",")

    pylab.savefig("%sspike_count_variability.png" % out_dir)
    pylab.clf()

def close_all_log_handlers():
    import logging
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
            logging.root.removeHandler(handler)
        except Exception as e:
            print(f"Error closing log handler: {e}")

def move_simulation_result(data_store, new_dir):
    os.makedirs(new_dir)
    for f in os.listdir(data_store.parameters.root_directory):
        if f.endswith("pickle") or f.endswith("json") or f.endswith("png") or f.endswith("csv"):
            shutil.move(os.path.join(data_store.parameters.root_directory,f),new_dir)
    close_all_log_handlers()
    shutil.rmtree(data_store.parameters.root_directory)


def run_simulation(e, paper, fig_name):
    data_store, model = run_workflow(
        "LGN_validation_run", SelfSustainedPushPull, select_experiment_func(e)
    )
    if e["type"] == "trial-to-trial-variance":
        trial_to_trial_variance_plot(data_store, e, paper, fig_name)
    else:
        plot_the_plot(data_store, e, paper, fig_name)
    data_store.save()

    result_dir = Path(data_store.parameters.root_directory).parent / (paper + "_" + fig_name)
    move_simulation_result(data_store, result_dir)

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
orig_K_opt_sf, orig_K_opt_tf = 0.79525, 5.34925

n_pixels_time = RF_p.duration / RF_p.temporal_resolution
n_pixels_space = RF_p.width / RF_p.spatial_resolution

assert RF_p.width == RF_p.height
assert all_p.visual_field.size[0] == all_p.visual_field.size[1]

with open("reference_data.json") as f:
    ref_data = json.load(f)

sf = 1
tf = 1
c = 1
lum = 1
bars = 1
variance = 1

# Make a list of unique experiment parameters
for paper in ref_data:
    #if paper != "Bonin et al. 2005":
    #    continue
    #if paper != "Hamamoto et al. 1994" and paper != "Papaioannou et al. 1972":
    #if paper != "Hamamoto et al. 1994" and paper != "Bonin et al. 2005" and paper != "Derrington et al. 1980" and paper!= "Papaioannou et al. 1972":
    #    continue
    #if paper != "Cudeiro et al. 1996":
    #    continue
    #if paper != "Bonin et al. 2005" and paper != "Kara et al. 2000":
    #    continue
    #if paper != "Mante et al. 2008":
    #    continue
    #if paper != "Sclar 1987":
    #    continue

    for fig_name in ref_data[paper]["data"]:
        e = ref_data[paper]["data"][fig_name]
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
        print("RF_p: ", RF_p_modified)

        rf_dur = 4 * np.abs(RF_p.func_params["t1"]) / RF_p_modified["t_lsc"]
        #print("rf_dur:",rf_dur)
        dt = rf_dur / n_pixels_time
        print("dt:",dt)
        # visual space update interval has to be integer multiple of dt
        n_dt_in_ui = np.ceil(all_p.input_space.update_interval / dt)
        dt = np.round((all_p.input_space.update_interval / n_dt_in_ui) * 10) // 10
        dt = max(dt, 1)
        #print("dt:",dt)
        n_pixels = int(rf_dur / dt)
        #print("n_pixels:",n_pixels)
        rf_dur = n_pixels * dt
        #print("rf_dur:",rf_dur)
        all_p_c.sheets.retina_lgn.params.receptive_field.duration = rf_dur
        all_p_c.sheets.retina_lgn.params.receptive_field.temporal_resolution = dt
        #print("sf: ",e["opt_sf"],", tf: %.2f",e["opt_tf"])
        #print("rf_w:",all_p_c.sheets.retina_lgn.params.receptive_field.width)
        #print("rf_t:",all_p_c.sheets.retina_lgn.params.receptive_field.duration)
        #print("ds: %.2f, dt: %.2f" % (all_p_c.sheets.retina_lgn.params.receptive_field.spatial_resolution,all_p_c.sheets.retina_lgn.params.receptive_field.temporal_resolution))
        if ref_data[paper]["background_luminance"] != None:
            all_p_c.input_space.background_luminance = ref_data[paper][
                "background_luminance"
            ]

        sys.argv[3] = str(all_p_c) + "\n"
        run_simulation(e, paper, fig_name)

if bars:
    sbp = {
        "time_per_image": 28,
        "blank_time": 245,
        "total_number_of_images": 20 * 2 * 750,  # 20 * 2 * 200,
        "num_trials": 1,
        "orientation": 0,
        "bar_length": 5,
        "bar_width": 0.1,
        "x": 0,
        "y": 0,
        "n_positions": 20,
        "experiment_seed": 17,
        "shuffle_stimuli":False,
    }

    sys.argv[3] = str(all_p)

    data_store, model = run_workflow(
        "LGN_validation_run",
        SelfSustainedPushPull,
        lambda model: [MeasureSparseBar(model, ParameterSet(sbp))],
    )
    data_store.save()

    sparsenoise_bar_validation(data_store)
    move_simulation_result(data_store, "sparse_noise")

# TODO: join all of these together in a single results directory and jsons

