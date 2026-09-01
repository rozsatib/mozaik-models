from parameters import ParameterSet
import numpy as np
from pathlib import Path
import json
from mozaik.storage.queries import param_filter_query
from mozaik.tools.mozaik_parametrized import  MozaikParametrized
from mozaik.analysis.analysis import Analysis, PSTH, TrialMean, TrialAveragedFiringRate
from mozaik.analysis.data_structures import PerNeuronValue
import quantities as qt
from mozaik.tools.distribution_parametrization import load_parameters
from mozaik.visualization.plotting import Plotting
import matplotlib.gridspec as gridspec
from mozaik.storage.datastore import DataStoreView
import pylab
from matplotlib.ticker import ScalarFormatter
import scipy
from mozaik.tools.units import spike_per_sec
from mozaik.models.vision.cai97 import stRF_kernel_2d
from mozaik.controller import Global
import os
from pathlib import Path
import shutil
from numpyencoder import NumpyEncoder
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
import fcntl

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

class FlashedBarReceptiveField(Analysis):
    required_parameters = ParameterSet(
        {
            "vm": bool,
            "spikes": bool,
        }
    )
    
    def get_rf(self,dsv):
        segs = dsv.get_segments()
        stims = dsv.get_stimuli()
        vm_ids = segs[0].get_stored_vm_ids()
    
        stim_dur = ParameterSet(stims[0]).duration
        unique_stims = list(set(stims))
        y = sorted(set([ParameterSet(s)["y"] for s in unique_stims]))
        ydiff = y[1]-y[0]
        get_y_idx = lambda y_in: int(np.round((y_in - y[0]) / ydiff))
        all_y_ids = [get_y_idx(ParameterSet(s)["y"]) for s in stims]
        all_rel_lum = [int(ParameterSet(s)["relative_luminance"]) for s in stims]
    
        # n_neurons x n_positions x type_of_stim (positive/negative) x length of stimulus
        rf = np.zeros((len(ids),len(y),2,stim_dur))
        # Sometimes spikes fall on the very last ms, so we add one time step
        rf_spike = np.zeros((len(ids),len(y),2,stim_dur+1))
        rf_counts = np.zeros((len(y)))
        
        for j in range(len(segs)):
            for i in range(len(ids)):
                v = segs[j].get_vm(ids[i]).flatten()
                st = np.array(segs[j].get_spiketrain(ids[i])).astype(int)
                sp = np.zeros(stim_dur+1)
                sp[st] = 1
                rf[i,all_y_ids[j],all_rel_lum[j],:] += v
                rf_spike[i,all_y_ids[j],all_rel_lum[j],:] += sp
            rf_counts[all_y_ids[j]] += 1
    
        rf_mean = np.mean(rf[:,:,1,:] - rf[:,:,0,:],axis=0)
        rf_spike_mean = np.mean(rf_spike[:,:,1,:] - rf_spike[:,:,0,:],axis=0)
        
        rf_mean = (rf_mean.T/rf_counts).T
        rf_spike_mean = (rf_spike_mean.T/rf_counts).T
        rf = (rf.transpose(0,2,3,1)/rf_counts).transpose(0,3,1,2)
        
        return rf_mean, rf_spike_mean, rf, y, ids

    def fetch_rf_metadata(self, dsv):
        stims = dsv.get_stimuli()
        stim_psets = [ParameterSet(s) for s in stims]
        unique_stims = list(set(stims))
        y = sorted(set([ParameterSet(s)["y"] for s in unique_stims]))
        ydiff = y[1]-y[0]
        y_ids = np.array([int(np.round((p["y"] - y[0]) / ydiff)) for p in stim_psets])
        rel_lum = np.array([int(p["relative_luminance"]) for p in stim_psets])
        stim_dur = ParameterSet(stims[0]).duration
        return y_ids, rel_lum, stim_dur, stims[0]
        
    def calc_rf(self, rf_type, segs, y_ids, rel_lum, stim_dur):
        data, ids = self.get_data_ids(segs, rf_type, stim_dur)
        y, rf_counts = np.unique(y_ids,return_counts=True)
        
        # n_neurons x n_positions x type_of_stim (positive/negative) x length of stimulus
        rf = np.zeros((len(ids),len(y),2, data.shape[-1]))
        
        for j in range(len(segs)):
            rf[:,y_ids[j],rel_lum[j],:] += data[j,...]
        
        rf = rf[:,:,1,:] - rf[:,:,0,:]
        rf = (rf.transpose(0,2,1)/rf_counts).transpose(0,2,1)
        return rf, ids

    def calc_rf_spikes(self, segs, y_ids, rel_lum, stim_dur):
        # n_neurons x n_positions x type_of_stim (positive/negative) x length of stimulus
        y, rf_counts = np.unique(y_ids,return_counts=True)
        rf = np.zeros((len(st_ids), len(y), 2, stim_dur + 1))

        vms = np.empty((len(segs), len(vm_ids), stim_dur))
        for j in range(len(segs)):
            for i in range(len(ids)):
                st = np.array(segs[j].get_spiketrain(ids[i])).astype(int)
                sp = np.zeros(stim_dur+1)
                sp[st] = 1
                rf[i,all_y_ids[j],all_rel_lum[j],:] += v
                rf_spike[i,all_y_ids[j],all_rel_lum[j],:] += sp
            rf_counts[all_y_ids[j]] += 1
    
        rf_mean = np.mean(rf[:,:,1,:] - rf[:,:,0,:],axis=0)
        rf_spike_mean = np.mean(rf_spike[:,:,1,:] - rf_spike[:,:,0,:],axis=0)
        
        rf_mean = (rf_mean.T/rf_counts).T
        rf_spike_mean = (rf_spike_mean.T/rf_counts).T
        rf = (rf.transpose(0,2,3,1)/rf_counts).transpose(0,3,1,2)

    def store_analysis_result(self, rf, ids, rf_type, stimulus_id, sheet):
        self.datastore.full_datastore.add_analysis_result(
            PerNeuronValue(
                rf,
                ids,
                qt.mV if rf_type == "Voltage" else spike_per_sec,
                stimulus_id=stimulus_id,
                value_name=f"Receptive Field ({rf_type})",
                sheet_name=sheet,
                tags=self.tags,
                analysis_algorithm=self.__class__.__name__,
                period=None,
            )
        )
        
    def get_data_ids(self, segs, rf_type, stim_dur):
        if rf_type == "Voltage":
            ids = segs[0].get_stored_vm_ids()
            data = np.empty((len(segs), len(ids), stim_dur))
            for i in range(len(segs)):
                for j in range(len(ids)):
                    data[i,j,:] = segs[i].get_vm(ids[j]).squeeze()
        elif rf_type == "Spikes":
            ids = segs[0].get_stored_spike_train_ids()
            data = np.empty((len(segs), len(ids), stim_dur+1))
            for i in range(len(segs)):
                for j in range(len(ids)):
                    st = np.array(segs[i].get_spiketrain(ids[j])).astype(int)
                    data[i,j,st] = 1
            data /= 0.001 # 1 ms time bin for firing rate
        else:
            raise NotImplementedError(f"Unknown receptive field type {rf_type}!")
        
        return data, ids
        
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = param_filter_query(self.datastore,st_name='FlashedBar',sheet_name=sheet)
            y_ids, rel_lum, stim_dur, stimulus_id = self.fetch_rf_metadata(dsv)
            segs = dsv.get_segments()
            if self.parameters.vm:
                rf, ids = self.calc_rf("Voltage", segs, y_ids, rel_lum, stim_dur)
                self.store_analysis_result(rf, ids, "Voltage", stimulus_id, sheet)
            if self.parameters.spikes:
                rf, ids = self.calc_rf("Spikes", segs, y_ids, rel_lum, stim_dur)
                self.store_analysis_result(rf, ids, "Spikes", stimulus_id, sheet)

class LGNPlot(Plotting):
    
    required_parameters = ParameterSet({
        "src_data_store": DataStoreView,
        "reference_data_path": str,
    })

    def experiment_params_from_path(self,path,ref_data_path="reference_data.json"):
        paper, figure = Path(path).name.split("_")
        with open(ref_data_path) as f:
            ref_data = json.load(f)
        return paper, figure, ref_data[paper]['data'][figure], ref_data

class SpikeCountMeanVariancePlot(LGNPlot):
    
    def spike_count_mean_variance(self, data_store, sheet):
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
    
    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0,centering_pnv=None,spont_level_pnv=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        self.caption = "Plotting of trial-to-trial spike-count mean and variance in response to fullfield sinusoidal gratings."
    
    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            10, 10, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )

        paper, _, e, _ = self.experiment_params_from_path(self.parameters.src_data_store.parameters.root_directory,self.parameters.reference_data_path)

        ax = pylab.subplot(gs[:,:])
        scm_on, scv_on = self.spike_count_mean_variance(self.parameters.src_data_store, "X_ON")
        scm_off, scv_off = self.spike_count_mean_variance(self.parameters.src_data_store, "X_OFF")
    
        ax.plot(scm_on,scv_on,'o',ms=1)
        ax.plot(scm_off,scv_off,'o',ms=1)
        ax.plot(e["x"],e["y"],'o',ms=1)
        ax.legend(['LGN ON - model', 'LGN OFF -model', paper])
        ax.set_xlabel("Spike count mean")
        ax.set_ylabel("Spike count variance")
        
        self.parameters.src_data_store = "DataStoreView"
        return plots

class TuningPlotLGN(LGNPlot):
    
    def calc_fr(self, data_store, ep):
        if ep["type"] == "luminance":
            key = "background_luminance"
            results = data_store.full_datastore.get_analysis_result(value_name="Firing rate")
        else:
            key = ep["type"]
            results = data_store.full_datastore.get_analysis_result(
                value_name="1. Harmonic Response"
            )

        fr = {sheet: {} for sheet in data_store.sheets()}
        for result in results:
            p = load_parameters(str(result), ParameterSet({}))
            sheet = p["sheet_name"]
            p = load_parameters(p["stimulus_id"], ParameterSet({}))
            fr[sheet][p[key]] = np.mean(result.get_value_by_id(result.ids))
        return fr
    
    def __init__(self, datastore, parameters, plot_file_name=None,fig_param=None,frame_duration=0,centering_pnv=None,spont_level_pnv=None):
        Plotting.__init__(self, datastore, parameters, plot_file_name, fig_param,frame_duration)
        self.caption = "Plotting of measured LGN contrast, luminance and spatial/temporal frequency functions."
    
    def subplot(self, subplotspec):
        
        paper, figure, experiment_parameters, ref_data = self.experiment_params_from_path(self.parameters.src_data_store.parameters.root_directory,self.parameters.reference_data_path)
        data = self.calc_fr(self.parameters.src_data_store, experiment_parameters)
    
        title = paper + " " + figure
    
        if experiment_parameters["type"] == "luminance":
            for pp in ref_data:
                for fn in ref_data[pp]["data"]:
                    ee = ref_data[pp]["data"][fn]
                    if ee["id"] == 28:
                        data[pp + " " + fn] = {"x": ee["x"], "y": ee["y"]}
        else:
            data[title] = {"x": experiment_parameters["x"], "y": experiment_parameters["y"]}
    
        data["X_ON_model"] = data.pop("X_ON")
        data["X_OFF_model"] = data.pop("X_OFF")
    
        if experiment_parameters["type"] == "contrast":
            x_log, y_log = False, True
        else:
            x_log, y_log = True, True
        
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            10, 10, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )

        ep = experiment_parameters
        xlabel = ep["type"]

        fr = data
        vals = sorted(fr["X_ON_model"].keys())
        on_frs = [fr["X_ON_model"][v] for v in vals]
        off_frs = [fr["X_OFF_model"][v] for v in vals]
        legend = ["LGN ON - model", "LGN OFF - model"]
        
        ax = pylab.subplot(gs[:,:])
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
            ax.plot(x_interp,y_interp,color='k',alpha=0.7)
            ax.plot(xmax,ymax,'or',ms=3)
            figtitle += "\nMaximum frequency: %.3f" % x_interp[max_idx]
    
        ax.set_title(figtitle)
        self.parameters.src_data_store = "DataStoreView"
        return plots

class IndividualBarReceptiveFieldPlot(Plotting):
    
    required_parameters = ParameterSet({
        "src_data_store": DataStoreView,
        "rf_type": str,
        "neuron_id": int,
    })

    def subplot(self, subplotspec):
        data_store = self.parameters.src_data_store
        results = data_store.get_analysis_result(value_name=f"Receptive Field ({self.parameters.rf_type})")
        assert len(results) == 1, "DataStoreView must contain exactly 1 Receptive field analysis result, contains %d!" % len(results)
        rfs = results[0]
        assert self.parameters.neuron_id in rfs.ids, "Supplied neuron_id %d not in recorded receptive field ids: %s" % (self.parameters.neuron_id, rfs.ids)
        rf = rfs.values[rfs.ids.index(self.parameters.neuron_id),...]
        p = load_parameters(rfs.stimulus_id)

        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            5, 5, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )
        
        ax = pylab.subplot(gs[:,:])
        
        idx = data_store.get_sheet_indexes(rfs.sheet_name, [self.parameters.neuron_id])[0]
        cy = data_store.get_neuron_positions()[rfs.sheet_name][1][idx]
        cx = data_store.get_neuron_positions()[rfs.sheet_name][0][idx]
        
        ax.set_title(
            "LGN %s mean neuron response with center (%.2f,%.2f)\nto horizontal bars"
            % (rfs.sheet_name, cx, cy)
        )

        im = ax.imshow(rf, aspect="auto", interpolation='None',
                  extent=[0, rf.shape[-1] - 1, p["location_y"] - p["width"] * rf.shape[0] / 2, p["location_y"] + p["width"] * rf.shape[0] / 2],
                 )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Visual angle (°)")
        cbar = pylab.colorbar(im, ax=ax)
        
        if self.parameters.rf_type == "Voltage":
            cbar.set_label("Voltage (mV)")
        else:
            cbar.set_label("Firing rate (sp/s)")
        self.parameters.src_data_store = "DataStoreView"
        return plots

class BarReceptiveFieldValidationPlot(Plotting):
    
    required_parameters = ParameterSet({
        "src_data_store": DataStoreView,
        "rf_type": str,
    })

    def reference_rf(self, spatial_size):
        p = ParameterSet({
            'Ac': 0.00052060,
            'As': 0.00002536,
            'sigma_c': 0.10780073,
            'sigma_s': 0.48911512,
            'K1': 1.37564875,
            'K2': 0.31283973,
            'c1': 0.61471462,
            'c2': 0.22824534,
            't1': -44.87613346,
            't2': -93.08789616,
            'n1': 46.84962684,
            'n2': 44.70041180,
            'td': 8.86454539,
            'subtract_mean': False,
        })
        rf_ref = stRF_kernel_2d(duration=200.0, dt=1, size=spatial_size, scale_factor=100, p=p).sum(axis=0)
        return rf_ref

    def normalize_median(self, x):
        return (x-np.median(x))/(x.max()-x.min())
        
    def mse(self, x):
        return (x**2).mean()
        
    def align_rf_with_reference(self, rf, rf_ref):
        max_time_ref = np.unravel_index(np.argmax(rf_ref),rf_ref.shape)[1]
        max_time_measure = np.unravel_index(np.argmax(rf),rf.shape)[1]
        mtdiff = max_time_measure - max_time_ref
        if mtdiff > 0:
            return rf[:,mtdiff:]
        else:
            mtdiff = np.abs(mtdiff)
            return rf[:,:-mtdiff]

    def interp(self, base, to_interp):
        # Original grid
        x = np.arange(to_interp.shape[0])
        y = np.arange(to_interp.shape[1])
    
        # Interpolator expects (x, y) as separate dimensions
        f = scipy.interpolate.RegularGridInterpolator((x, y), to_interp)
    
        # New grid
        xnew = np.linspace(0, to_interp.shape[0] - 1, base.shape[0])
        ynew = np.linspace(0, to_interp.shape[1] - 1, base.shape[1])
        xnew_grid, ynew_grid = np.meshgrid(xnew, ynew, indexing='ij')
    
        # Interpolate at new coordinates
        points = np.stack((xnew_grid, ynew_grid), axis=-1)
        return f(points)
    
    def center_rfs(self, data_store, rfs, radius_mm=0.05):
        idx = data_store.get_sheet_indexes(rfs.sheet_name, rfs.ids)
        pos = data_store.get_neuron_positions()[rfs.sheet_name][:2,idx]
        return rfs.values[np.sum(pos**2,axis=1) < radius_mm**2,...]
    
    def subplot(self, subplotspec):
        data_store = self.parameters.src_data_store
        results = data_store.get_analysis_result(value_name=f"Receptive Field ({self.parameters.rf_type})")
        assert len(results) == 1, "DataStoreView must contain exactly 1 Receptive field analysis result, contains %d!" % len(results)
        rfs = results[0]

        rf = self.center_rfs(data_store, rfs).mean(axis=0)
        rf = scipy.ndimage.convolve1d(rf, np.ones(10)/10)
        
        rf_diameter = load_parameters(rfs.stimulus_id)["width"] * rf.shape[0]
        rf_ref = self.reference_rf(rf_diameter)
        
        rf = self.interp(rf_ref,rf)
        rf = self.normalize_median(rf)
        rf = scipy.ndimage.gaussian_filter1d(rf, sigma=3, axis=1, mode='constant')
        
        rf_ref = self.normalize_median(rf_ref)

        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=subplotspec, hspace=0.3, wspace=0.5
        )
        
        yticks = np.array([-0.5,0,0.5,1])

        if rfs.sheet_name == "X_OFF":
            rf *= -1
        
        rf = self.align_rf_with_reference(rf, rf_ref)
        
        if rfs.sheet_name == "X_OFF":
            yticks = np.sort(yticks * (-1))
            rf_ref *= -1
            rf *= -1
            
        
        ax = [pylab.subplot(gs[:,i]) for i in range(5)]
        ax[0].imshow(rf_ref, aspect="auto", interpolation='None', extent=[0, rf.shape[-1] - 1, - rf_diameter / 2, rf_diameter / 2])
        ax[0].set_title("Reference Receptive Field")
        ax[1].set_yticks([-1,0,1])
        ax[1].set_xticks([0,50,100,150])
        ax[1].set_xlim(0,150)
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel("Visual angle (°)")
        ax[1].set_title("%s receptive field\nDiff. from ref. MSE: %.3f" % (self.parameters.rf_type,self.mse(rf_ref[:,:rf.shape[1]]-rf)))
        ax[1].imshow(rf, aspect="auto", interpolation='None', extent=[0, rf.shape[-1] - 1, - rf_diameter / 2, rf_diameter / 2])
        ax[1].set_yticks([-1,0,1])
        ax[1].set_xticks([0,50,100,150])
        ax[1].set_xlim(0,150)
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylabel("Visual angle (°)")

        for i in range(2,5):
            ax[i].tick_params(width=1.25, length=6)
            ax[i].tick_params(width=0.75, length=3.5,which='minor')
            ax[i].spines[['left','bottom']].set_linewidth(1.25)
            ax[i].spines[['right', 'top']].set_visible(False)
        
        max_space, max_time = np.unravel_index(np.argmax(np.abs(rf_ref)),rf_ref.shape)
        ax[2].set_title("Middle (0°) timecourse")
        ax[2].plot(rf_ref[max_space,:])
        ax[2].plot(rf[max_space,:])
        ax[2].set_yticks(yticks)
        ax[2].set_xticks([0,50,100,150])
        ax[2].set_xlim(0,150)
        ax[2].set_ylim(yticks[0],yticks[-1])
        ax[2].set_xlabel("Time (ms)")
        ax[2].set_ylabel("Normalized amplitude")
        ax[2].legend(["Reference", f"{self.parameters.rf_type}"])

        ax[3].set_title("Sides ($\\pm$0.5°) timecourse")
        ax[3].plot(rf_ref[max_space//2,:])
        ax[3].plot(rf[max_space//2,:])
        ax[3].plot(rf[-max_space//2,:])
        ax[3].set_yticks(yticks)
        ax[3].set_xticks([0,50,100,150])
        ax[3].set_xlim(0,150)
        ax[3].set_ylim(yticks[0],yticks[-1])
        ax[3].set_xlabel("Time (ms)")
        ax[3].set_ylabel("Normalized amplitude")
        ax[3].legend(["Reference", f"{self.parameters.rf_type} 0.5°", f"{self.parameters.rf_type} -0.5°"])

        ax[4].set_title("Temporal max (%d ms)\nspatial profile" % max_time)
        ax[4].plot(np.linspace(-1,1,rf_ref.shape[0]),rf_ref[:,max_time])
        ax[4].plot(np.linspace(-1,1,rf.shape[0]),rf[:,max_time])
        ax[4].legend(["Reference", f"{self.parameters.rf_type}"])
        ax[4].set_yticks(yticks)
        ax[4].set_xticks([-1,0,1])
        ax[4].set_xlim(-1,1)
        ax[4].set_ylim(yticks[0],yticks[-1])
        ax[4].set_xlabel("Visual angle (°)")
        ax[4].set_ylabel("Normalized amplitude")
        self.parameters.src_data_store = "DataStoreView"
        return plots

def retrieve_result_dirs(base_dir_path):
    return [ f.path for f in os.scandir(base_dir_path) if f.is_dir() and "combined" not in f.path and "LGN_validation" not in f.path]

def run_analysis_plotting_loop(base_dir_path, analysis=True, plotting=True):
    result_dirs = retrieve_result_dirs()
    for path in result_dirs:
        if analysis:
            run_analysis(path)
        if plotting:
            run_plotting(path)

def run_analysis(path):
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    )
    data_store.remove_ads_from_datastore()
    
    if "sparse_noise" in path:
        FlashedBarReceptiveField(
            data_store, ParameterSet({"vm": True, "spikes": True})
        ).analyse()
    elif "FullfieldDriftingSinusoidalGrating" in str(data_store.get_stimuli()):
        dsv = param_filter_query(
            data_store,
            st_name="FullfieldDriftingSinusoidalGrating",
            sheet_name=["X_ON", "X_OFF"],
        )
        TrialAveragedNthHarmonicResponse(
            dsv, ParameterSet({"bin_length": 5, "n_harmonic_freq": 1})
        ).analyse()
    else:
        dsv = param_filter_query(
            data_store,
            st_name="Null",
            sheet_name=["X_ON", "X_OFF"],
        )
        TrialAveragedFiringRate(dsv, ParameterSet({})).analyse()

    data_store.save()

def run_plotting(path):
    combined_dir = str(Path(path).parent / "combined") + "/"
    os.makedirs(combined_dir, exist_ok=True)

    data_store_export = PickledDataStore(load=False, parameters=MozaikExtendedParameterSet({'root_directory': combined_dir,'store_stimuli' : False}))
    Global.root_directory = data_store_export.parameters.root_directory
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    )
    results_file = Path(combined_dir) / "results.json"
    if not os.path.isfile(results_file):
        with open(results_file, 'w') as f:
            json.dump([], f)
    with open(results_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX) # ensure only 1 process writes to results.json at a time
        if 'Kara et al. 2000' in path:
            SpikeCountMeanVariancePlot(
                data_store_export,
                ParameterSet(
                    {
                        "src_data_store": data_store,
                        "reference_data_path": "reference_data.json",
                    }
                ),
                fig_param={"dpi": 100, "figsize": (6, 4)},
                plot_file_name=f"{Path(path).name}.png",
            ).plot()
        elif "sparse_noise" in path:
            for sheet in ["X_ON", "X_OFF"]:
                dsv_on = param_filter_query(data_store,sheet_name=sheet)
                for rf_type in ["Voltage","Spikes"]:
                    for neuron_id in dsv_on.get_analysis_result(analysis_algorithm="FlashedBarReceptiveField")[0].ids:
                        IndividualBarReceptiveFieldPlot(
                            data_store_export,
                            ParameterSet(
                                {
                                    "src_data_store": dsv_on, 
                                    "rf_type": rf_type,
                                    "neuron_id": neuron_id,
                                }
                            ),
                            fig_param={"dpi": 100, "figsize": (6, 4)},
                            plot_file_name=f"RF_{rf_type}_{sheet}_{neuron_id}.png",
                        ).plot()
                    BarReceptiveFieldValidationPlot(
                        data_store_export,
                        ParameterSet(
                            {
                                "src_data_store": dsv_on, 
                                "rf_type": rf_type,
                            }
                        ),
                        fig_param={"dpi": 100, "figsize": (15,3)},
                        plot_file_name=f"RF_validation_{rf_type}_{sheet}.png",
                    ).plot()
        else:
            TuningPlotLGN(
                data_store_export,
                ParameterSet(
                    {
                        "src_data_store": data_store,
                        "reference_data_path": "reference_data.json",
                    }
                ),
                fig_param={"dpi": 100, "figsize": (6, 4)},
                plot_file_name=f"{Path(path).name}.png",
            ).plot()

def merge_results_jsons(base_dir_path):
    result_dirs = retrieve_result_dirs(base_dir_path)
    combined_result_dir = Path(result_dirs[0]).parent / "combined"
    path = Path(result_dirs[0])
    with open(path / "parameters.json",'r') as f:
        p = json.load(f)
    p['sheets']['retina_lgn']["params"]["receptive_field"]["func_params"]["t_lsc"] = 1 # Make sure t_lsc is indeed 1 by default!!
    p['sheets']['retina_lgn']["params"]["receptive_field"]["func_params"]["s_sc"] = 1
    shutil.copy(path / "parameters.json", combined_result_dir) # move this inside above once the todo change is done
    shutil.copy(path / "recorders.json", combined_result_dir)
    shutil.copy(path / "sim_info.json", combined_result_dir)
    
    for name in ["stimuli", "experimental_protocols"]:
        combine = {}
        for d in result_dirs:
            print(d)
            with open(Path(d) / f"{name}.json",'r') as f:
                p = json.load(f)
            combine[json.dumps(p)] = p
        combine_list = [v for k, v in sorted(combine.items())]
        with open(combined_result_dir / f"{name}.json", 'w', encoding='utf-8') as f:
            json.dump(combine_list, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
