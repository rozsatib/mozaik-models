from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
from mozaik.storage.queries import *
import sys
import numpy as np
import scipy
import pickle
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
import time
from scipy import signal
from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue
from som import SOM
import quantities as pq
import copy

from mozaik.visualization.plotting import Plotting
from mozaik.visualization.simple_plot import PixelMovie
from mozaik.visualization.plot_constructors import LinePlot
import matplotlib.gridspec as gridspec
import pylab
from mozaik.tools.mozaik_parametrized import varying_parameters
from mozaik.tools.distribution_parametrization import load_parameters

class KaschubePlot(Plotting):
    def fetch_postfixes(self, sheet_name, prefix, tags):
        res = tag_based_query(self.datastore.full_datastore, tags).get_analysis_result(
            identifier="SingleValue",
            analysis_algorithm="KaschubeAnalysis",
            sheet_name=sheet_name,
        )
        postfixes = sorted([load_parameters(str(r))["value_name"] for r in res])
        postfixes = list(set([p.split(";")[-1] for p in postfixes if "orientation map" not in p and prefix in p and ";" in p]))
        return postfixes if len(postfixes) > 0 else [""]
    
    
    def read_results(self, names, sheet, prefix, tags=[]):
        v = {}
        postfixes = self.fetch_postfixes(sheet, prefix, tags)
        for postfix in postfixes:
            v[postfix] = {}
            for name in names:
                value_name = (
                    sheet + "-" + name if name == "orientation map" else prefix + "-" + name
                )
                if postfix != '':
                    value_name += ";" + postfix
                res = tag_based_query(self.datastore.full_datastore, tags).get_analysis_result(
                    identifier="SingleValue",
                    value_name=value_name,
                    sheet_name=sheet,
                )
                if len(res) == 0:
                    mozaik.getMozaikLogger().warning(
                        "Analysis result " + value_name + " not found!!"
                    )
                    return None
                v[postfix][name] = res[0].value
        return v

class KaschubeMetricsPlot(KaschubePlot):

    required_parameters = ParameterSet({"prefix": str, "sheet_name": str, "tags": list})

    result_names = [
        "dimensionality",
        "local correlation eccentricities",
        "similarity map",
        "kohonen map",
        "orientation map",
    ]

    def local_correlation_eccentricity_data(self):
        # Data from Kaschube paper for local correlation eccentricity
        # Normalized so that the sum of the histogram == 1
        lce_x = np.linspace(0,1,21,endpoint=False)
        lce_y = np.array([0,0,0,0.05,0.075,0.14,0.26,0.3,0.47,0.77,1.02,1.35,1.52,1.94,
        2.09,2.10,2.36,1.75,1.27,0.85,0.35])
        lce_y /= lce_y.sum()
        return lce_x, lce_y

    def dimensionality_data(self):
        return np.array([4,7,15.7,15.9,21.4])

    def subplot(self, subplotspec):
        plots = {}
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )
        v = self.read_results(self.result_names,self.parameters.sheet_name, self.parameters.prefix,self.parameters.tags)
        if v is None:
            return plots
        v = v['']
        
        # Similarity map
        pylab.subplot(gs[0, 0])
        smap = pylab.imshow(v["similarity map"], vmin=0, vmax=1)
        pylab.axis("equal")
        pylab.title("Similarity map; mean=%.2f" % np.mean(v["similarity map"]))
        pylab.colorbar(smap)

        # Orientation map
        pylab.subplot(gs[0,1])
        ormap = pylab.imshow(v["orientation map"],'hsv')
        pylab.title("Orientation map")
        pylab.axis('equal')
        pylab.colorbar(ormap)

        # Kohonen map
        pylab.subplot(gs[0,2])
        kmap = pylab.imshow(v["kohonen map"],'hsv')
        pylab.title("Kohonen map")
        pylab.axis('equal')
        pylab.colorbar(kmap)

        # Local correlation eccentricity
        pylab.subplot(gs[1,0])
        pylab.title("Local correlation eccentricity")
        lce_x,lce_y = self.local_correlation_eccentricity_data()
        pylab.bar(lce_x,lce_y,width=1/len(lce_x),align="edge",color="tab:green",edgecolor="black", alpha=0.8)
        h, bins = np.histogram(v["local correlation eccentricities"],lce_x)
        h = h / h.sum()
        pylab.bar(bins[:-1],h,width=1/len(h),align="edge",color="tab:orange",edgecolor="black", alpha=0.8)

        # Dimensionality
        pylab.subplot(gs[1,1])
        pylab.title("Dimensionality")
        dim = self.dimensionality_data()
        pylab.errorbar(np.ones((len(dim)))*0.3,dim,marker='s',ls='',alpha=0.5)
        pylab.errorbar(0.3,dim.mean(),dim.std(),marker='s',ls='',markersize=12)
        pylab.plot(0.5,v["dimensionality"],'o',color='black')
        pylab.legend(['Model','Experimental data','Experimental data mean'])
        pylab.xlim(0,1)
        pylab.ylim(0)
        return plots


class KaschubeCorrelationPlot(KaschubePlot):

    required_parameters = ParameterSet({"prefix": str, "sheet_name": str, "tags":list})

    result_names = [
        "or_map 0-peak coords",
        "or_map 0-peak Cmaps",
        "orientation map"
    ]

    def subplot(self, subplotspec):
        plots = {}
        rows,cols=2,4
        gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=subplotspec, hspace=0.1, wspace=0.2
        )
        v = self.read_results(self.result_names, self.parameters.sheet_name, self.parameters.prefix,self.parameters.tags)
        if v is None:
            return plots
        v = v['']
        
        x = [c[0] for c in v["or_map 0-peak coords"][:(rows*cols-1)]]
        y = [c[1] for c in v["or_map 0-peak coords"][:(rows*cols-1)]]
        pylab.subplot(gs[0, 0])
        pylab.imshow(v["orientation map"].T,cmap='hsv')
        pylab.plot(x,y,'o',color='lime',markersize=9,mec='black',mew=2.5)
        print(v["or_map 0-peak coords"])
        for i in range(1,len(x)):
            xx,yy = v["or_map 0-peak coords"][i]
            pylab.subplot(gs[i//cols, i%cols])
            pylab.imshow(v["or_map 0-peak Cmaps"][i].T,cmap="bwr")
            pylab.plot(x,y,'o',color='lime')
            pylab.plot(xx,yy,'o',color='lime',markersize=9,mec='black',mew=2.5)
        return plots

    
class KaschubeEventPlot(KaschubePlot):

    required_parameters = ParameterSet({"prefix": str, "sheet_name": str, "tags": list})

    result_names = ["N_active_pixels", "N_active_pixels_thresh"]

    def plot_N_of_active_pixels(p, t_res):
        pylab.plot(
            np.linspace(
                0, len(p["N_active_pixels"]) * t_res, len(p["N_active_pixels"])
            ),
            p["N_active_pixels"],
        )
        if p["N_active_pixels_thresh"] != None:
            pylab.plot(
                [0, len(p["N_active_pixels"]) * t_res],
                [p["N_active_pixels_thresh"], p["N_active_pixels_thresh"]],
                "r",
            )
        pylab.xlabel("Time/ms")
        pylab.ylabel("Number of active pixels")

    def subplot(self, subplotspec):
        plots = {}
        rows, cols = 1, 1
        gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=subplotspec, hspace=0.1, wspace=0.2
        )
        v = self.read_results(
            self.result_names,
            self.parameters.sheet_name,
            self.parameters.prefix + "-event_detection",
            self.parameters.tags,
        )

        if v is None:
            return plots
        v = v['']
        
        t_res = [int(t.split(":")[-1]) for t in self.parameters.tags if "t_res" in t][0]
        KaschubeEventPlot.plot_N_of_active_pixels(v, t_res)

        return plots

class KaschubeActivityMovie(KaschubePlot):

    required_parameters = ParameterSet({"prefix": str, "sheet_name": str, "tags": list})

    postfixes = []
    result_names = ["smoothed", "spikes_only"]

    def subplot(self, subplotspec):
        self.postfixes = self.fetch_postfixes(**self.parameters)
        return LinePlot(function=self._ploter,length=len(self.postfixes),shared_axis=False).make_line_plot(subplotspec)
    
    def _ploter(self, idx,gs):
        
        v = self.read_results(
            self.result_names,
            self.parameters.sheet_name,
            self.parameters.prefix + "-Activity",
            self.parameters.tags,
        )

        if v is None:
            return []
        
        smoothed = v[self.postfixes[idx]]['smoothed']
        spikes_only = v[self.postfixes[idx]]['spikes_only']
        spikes_only /= spikes_only.max()
        spikes_only *= smoothed.max()
        
        movie = np.vstack([smoothed,spikes_only])
        movie = movie.transpose((2,0,1))

        return [("PixelMovie",PixelMovie(movie,movie.max()/2),gs,{'x_axis':False, 'y_axis':False})]   
    
class KaschubeAnalysis(Analysis):

    required_parameters = ParameterSet(
        {
            "t_res": int,  # Time resolution (bin size in ms) of activity maps
            "s_res": int,  # Space resolution (bin size in um) of activity maps
            "prefix": str,  # Prefix to all names
            "dimensionality_s_res": int,  # Space resolution (bin size in um) of activity maps for calculating dimensionality
            "event_detection_px_active_p": float, # Pixels become active only if their activity is above this p-value compared to their entire activity
            "event_detection_event_activity_p": float, # Frames are classified as events, only if number of active pixels is above this p-value compared to the entire run 
        }
    )
    
    def perform_analysis(self):
        assert len(self.datastore.sheets()) == 1, (
            "Can only do this analysis with a single sheet, number of sheets is: %d"
            % len(self.datastore.sheets())
        )
        self.sheet = self.datastore.sheets()[0]
        self.tags.extend(
            ["s_res: %d" % self.parameters.s_res, "t_res: %d" % self.parameters.t_res]
        )
        self.common_params = {
            "sheet_name": self.sheet,
            "analysis_algorithm": self.__class__.__name__,
            "stimulus_id": str(self.datastore.get_stimuli()[0]),
        }
        
        or_map = self.fetch_datastore_result("orientation map")
        if or_map is None:
            or_map = gen_or_map(self.datastore, self.sheet, self.parameters.s_res)
            self.save_or_map(or_map)   
        
        dsv_dicts = self.dsv_dicts()
        A = None
        for label in dsv_dicts:
            A_, A_sp_only = gen_st_array(
                dsv_dicts[label], s_res=self.parameters.s_res, t_res=self.parameters.t_res
            )
            self.save_results({"smoothed":A_,"spikes_only":A_sp_only}, "Activity",label)
            A = np.dstack([A,A_]) if A is not None else A_

        results = self.generate_results(A, or_map)
        self.save_results(results, "activity_based")
        
        E, event_plot_params = extract_events(A, self.parameters.s_res)
        self.save_results(event_plot_params, "event_detection")         

        if E is not None:
            results = self.generate_results(E, or_map)
            self.save_results(results, "event_based")

    def dsv_dicts(self):
        partitioned_dsvs = partition_by_stimulus_paramter_query(self.datastore,['trial'])
        varied = sorted([x for x in varying_parameters([MozaikParametrized.idd(s) for s in self.datastore.get_stimuli()]) if x != 'trial'])
        dsvs = {}
        for d in partitioned_dsvs:
            p = load_parameters(d.get_stimuli()[0])
            l = []
            for param in varied:
                l.append(param + ": %.3f" % p[param])
            dsvs[", ".join(l)] = d
        return dsvs
        
    def fetch_datastore_result(self, result_name, prefix="", postfix=""):
        tags = self.tags.copy()
        if result_name == "orientation map":
            tags.remove("t_res: %d" % self.parameters.t_res)
            value_name = self.sheet + "-" + result_name
        else:
            value_name = prefix + "-" + result_name
        if postfix != "":
            value_name += ";" + postfix
        res = tag_based_query(self.datastore.full_datastore, tags).get_analysis_result(
            identifier="SingleValue", value_name=value_name
        )
        return res[0].value if len(res) > 0 else None

    def generate_results(self, timecourse, or_map):
        or_map_column_width = 600
        result = {}
        s_res_factor = self.parameters.s_res / self.parameters.dimensionality_s_res
        dszx, dszy = int(max(timecourse.shape[0] * s_res_factor, 1)), int(
            max(timecourse.shape[0] * s_res_factor, 1)
        )
        result["dimensionality"] = dimensionality(resize_arr(timecourse, dszx, dszy))
        coords = [
            (x, y)
            for x in range(timecourse.shape[0])
            for y in range(timecourse.shape[1])
        ]
        Cmaps = correlation_maps(timecourse, coords)
        result["local correlation eccentricities"] = local_correlation_eccentricity(
            Cmaps, coords
        )
        result["similarity map"] = correlation_or_map_similarity(Cmaps, coords, or_map)
        result["kohonen map"] = kohonen_map(Cmaps)
        or_map_maxima = find_local_maxima(
            dist_or_map(rotate_or_map(or_map, np.pi / 2)),
            min_dist=int(or_map_column_width / self.parameters.s_res),
        )
        or_map_max_coords = [(m[0], m[1]) for m in or_map_maxima]
        result["or_map 0-peak coords"] = or_map_max_coords
        result["or_map 0-peak Cmaps"] = correlation_maps(timecourse, or_map_max_coords)
        return result

    def save_results(self, results, prefix="",postfix=""):
        for name in results.keys():
            value_name = self.parameters.prefix + "-" + prefix + "-" + name
            if postfix != "":
                value_name += ";" + postfix
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=results[name],
                    value_units=pq.dimensionless,
                    value_name=value_name,
                    tags=self.tags,
                    **self.common_params,
                )
            )

    def save_or_map(self, or_map):
        tags = self.tags.copy()
        tags.remove("t_res: %d" % self.parameters.t_res)
        self.datastore.full_datastore.add_analysis_result(
            SingleValue(
                value=or_map,
                value_units=pq.dimensionless,
                value_name=self.sheet + "-" + "orientation map",
                tags=tags,
                **self.common_params,
            )
        )

def t_kernel(t_res,length_ms=5000):
    # Based on https://doi.org/10.3389/fncir.2013.00201
    tau_on = 10 # ms rise time of calcium response
    tau_off = 1000 # ms fall time of calcium response

    # We ignore the rise time for the moment
    return np.exp(-np.linspace(0,length_ms,length_ms//t_res)/tau_off)

# size: https://www.sciencedirect.com/science/article/pii/S0165027018301274?via%3Dihub
def s_kernel(sp_res):
    neuron_diameter = 20 # um
    neuron_diameter = max(1,neuron_diameter // sp_res)
    
    if neuron_diameter < 4:
        s_kernel = np.ones((neuron_diameter,neuron_diameter))
    else:  
        s_kernel_1d = signal.tukey(int(neuron_diameter*1.1),0.3)
        klen = len(s_kernel_1d)//2
        s_kernel_1d = s_kernel_1d[klen:]
        s_kernel = np.zeros((2*klen,2*klen))
        for x in range(2*klen):
            for y in range(2*klen):
                r = int(round(np.sqrt((x-klen)**2 + (y-klen)**2)))
                if r < klen:
                    s_kernel[x,y] = s_kernel_1d[r]
                else:
                    s_kernel[x,y] = 0
    return s_kernel

def s_kernel_smoothed(sp_res,sigma_cort=100):
    # sigma_cort = sigma in cortical coordinates (micrometres)
    sigma = sigma_cort / sp_res
    n_sigmas = 3 # To how many sigmas we sample
    sm_ker = s_kernel(sp_res)
    sm_ker = np.pad(sm_ker,int(sigma*n_sigmas))
    return scipy.ndimage.gaussian_filter(sm_ker, sigma,mode='constant')

def get_st_ids(dsv):
    assert len(dsv.sheets()) == 1
    return [s for s in dsv.get_segments() if len(s.spiketrains) > 0][0].get_stored_spike_train_ids()

def get_s(dsv,s_res=None):
    if s_res == None:
        s_res = 1
    st_ids = get_st_ids(dsv)
    sheet = dsv.sheets()[0]
    pos = dsv.get_neuron_positions()[sheet]
    posx = (pos[0, dsv.get_sheet_indexes(sheet, st_ids)] / s_res * 1000).astype(int)
    posy = (pos[1, dsv.get_sheet_indexes(sheet, st_ids)] / s_res * 1000).astype(int)
    posx -= min(posx)
    posy -= min(posy)
    return posx, posy

def get_t(dsv,t_res=None):
    if t_res == None:
        t_res = 1
    st_ids = get_st_ids(dsv)
    segs = [s for s in dsv.get_segments() if len(s.spiketrains) > 0]
    t = [[] for i in range(len(st_ids))]
    time_passed = 0
    for i in range(len(segs)):
        if len(segs[i].spiketrains) == 0:
            continue
        sts = segs[i].get_spiketrains()
        for j in range(len(sts)): 
            t[j].extend(list((sts[j].magnitude / t_res).astype(int) + time_passed))
        time_passed += int((sts[0].t_stop.magnitude - sts[0].t_start.magnitude) / t_res)
    return t
    
def get_st(dsv, s_res=None, t_res=None):
    posx,posy = get_s(dsv,s_res)
    t = get_t(dsv,t_res)
    return posx, posy, t

def gen_st_array(dsv, s_res=None, t_res=None,smoothing=True):
    smoothing_scaler = 50
    
    posx, posy, t = get_st(dsv,s_res,t_res)
    s_ker = s_kernel(s_res)
    kx, ky = s_ker.shape

    A = np.zeros((max(posx)+1+kx, max(posy)+1+ky,max([v for l in t for v in l])+1))
    for i in range(len(posx)):
        for st in t[i]:
            A[posx[i]:posx[i]+kx,posy[i]:posy[i]+ky,st] += s_ker
    A /= t_res
    A_c = scipy.ndimage.convolve1d(A, t_kernel(t_res), axis=2, mode='constant', origin=-len(t_kernel(t_res))//2)
    
    if smoothing:
        sigma = 100 / s_res
        A_s = scipy.ndimage.gaussian_filter1d(A_c, sigma, axis=0, mode='constant')
        A_s = scipy.ndimage.gaussian_filter1d(A_s, sigma, axis=1, mode='constant')
        A_c += A_s * smoothing_scaler
    return A_c[kx//2:-kx//2,ky//2:-ky//2,:], A[kx//2:-kx//2,ky//2:-ky//2,:]
    
    #A = np.zeros((50,50,200))
    #A[18:22,18:22,0] = 1
    #A[18:22,18:22,50] = 1
    #A[38:42,38:42,100] = 1
    #A[38:42,38:42,199] = 1
    #print(A[40,40,100])
    #print(A.shape)
    #A_ = scipy.ndimage.convolve1d(A, t_kernel(t_res), axis=2, mode='constant', origin=-len(t_kernel(t_res))//2)
    #print(A[40,40,100])
    #print(A.shape)
    #print(A[20,20,20])
    #return A_,A

def plot_percent_of_active_pixels(A_active_sum,A,t_res,thresh=None):
    pylab.plot(np.linspace(0,len(A_active_sum)*t_res,len(A_active_sum)),A_active_sum)
    if thresh != None:
        pylab.plot([0,len(A_active_sum)*t_res],[thresh,thresh],'r')
    pylab.xlabel("Time/ms")
    pylab.ylabel("Percent of active pixels")
    pylab.show()

def percentile_thresh(A, percentile):
    A_sorted = copy.deepcopy(A)
    A_sorted.sort()
    thresh_idx = int(np.round((A.shape[-1] - 1) * percentile))
    if len(A_sorted.shape) == 1:
        return A_sorted[thresh_idx]
    elif len(A_sorted.shape) == 3:
        return A_sorted[:, :, thresh_idx]
    else:
        return None    
    
def extract_events(
    A, t_res, px_active_p=0.995, event_activity_p=0.8, min_segment_duration=100
):

    thresh = percentile_thresh(A, px_active_p)
    A_active = A.copy().transpose((2, 0, 1))
    A_active[A_active < thresh] = 0
    A_active[A_active >= thresh] = 1
    A_active = A_active.transpose((1, 2, 0))
    A_active_sum = A_active.sum(axis=(0, 1))

    thresh = percentile_thresh(A_active_sum, event_activity_p)
    A_active_zeroed = A_active_sum.copy()
    A_active_zeroed[A_active_zeroed < thresh] = 0

    segment_imgs = []
    for i in range(A.shape[2]):
        if A_active_zeroed[i] > 0:
            segment_max = 0
            segment_max_idx = 0
            segment_start = i
            while A_active_zeroed[i] != 0:
                if A_active_zeroed[i] > segment_max:
                    segment_max_idx = i
                    segment_max = A_active_zeroed[i]
                i += 1
                if i >= A.shape[2]:
                    break
            if i - segment_start > min_segment_duration // t_res:
                segment_imgs.append(A[:, :, segment_max_idx])

    if len(segment_imgs) == 0:
        return None

    event_maxima = np.array(segment_imgs)
    event_plot_params = {"N_active_pixels": A_active_sum, "N_active_pixels_thresh": thresh}
    return event_maxima.transpose((1, 2, 0)), event_plot_params    
    
def extract_events_old(A, t_res, active_px_thresh_sd=4, activity_sd_thresh=1.0,min_segment_duration=100):
    A_mean = A.mean(axis=2)
    A_std = A.std(axis=2)
    A_active = (A.transpose((2,0,1)) > active_px_thresh_sd*A_std + A_mean).transpose((1,2,0))
    A_active_sum = A_active.sum(axis=(0,1))
    #plot_percent_of_active_pixels(A_active_sum,A,t_res)
    
    thresh = A_active_sum.std() * activity_sd_thresh
    A_active_zeroed = A_active_sum
    A_active_zeroed[A_active_zeroed<thresh] = 0
    #plot_percent_of_active_pixels(A_active_zeroed,A,t_res,thresh)

    segment_imgs = []
    for i in range(A.shape[2]):
        if A_active_zeroed[i] > 0:
            segment_max = 0
            segment_max_idx = 0
            segment_start = i
            while(A_active_zeroed[i] != 0):
                if A_active_zeroed[i] > segment_max:
                    segment_max_idx = i
                    segment_max = A_active_zeroed[i]
                i += 1
                if i >= A.shape[2]:
                    break
            if i - segment_start > min_segment_duration // t_res:
                segment_imgs.append(A[:,:,segment_max_idx])
    if len(segment_imgs) == 0:
        return None
    event_maxima = np.array(segment_imgs)
    return event_maxima.transpose((1,2,0))

def log_runtime(what):
    def decorator(method):
        def wrapper(*args, **kw):
            t0 = time.time()
            result = method(*args, **kw)
            t1 = time.time()
            message = what + " took %.2f seconds" % (t1-t0)
            mozaik.getMozaikLogger().info(message)
            return result
        return wrapper
    return decorator

@log_runtime("Correlation map calculation")
def correlation_maps(A,coords):
    Av = (A.transpose((2,0,1)) - A.mean(axis=2)).transpose((1,2,0))
    Avss = (Av * Av).sum(axis=2)    
    results = []
    for i in range(len(coords)):
        x,y = coords[i]
        #print(x,y)
        result = np.matmul(Av,Av[x,y,:])/ np.sqrt(Avss[x,y] * Avss)
        result = np.nan_to_num(result)
        results.append(result)
    # bound the values to -1 to 1 in the event of precision issues
    return results

@log_runtime("Similarity map generation")
def correlation_or_map_similarity(Cmaps,coords,or_map,size=None):  
    s_map = np.zeros(Cmaps[0].shape)
    or_map_s = np.sin(or_map).flatten()
    or_map_c = np.cos(or_map).flatten()
    
    results = []
    for i in range(len(coords)):
        x,y = coords[i]
        C = Cmaps[i].flatten()
        r_s, _ = scipy.stats.pearsonr(C,or_map_s)
        r_c, _ = scipy.stats.pearsonr(C,or_map_c)
        r_s = np.nan_to_num(r_s)
        r_c = np.nan_to_num(r_c)
        s_map[x,y] = np.sqrt(r_s*r_s + r_c*r_c)
    #t2=time.time()
    #print("Entire similarity map calc time: %.2f s" % (t2-t0))
    return s_map

def resize_arr(A, new_width, new_height):
    A = np.asarray(A)
    shape = list(A.shape)
    shape[0] = new_width
    shape[1] = new_height
    ind = np.indices(shape, dtype=float)
    ind[0] *= (A.shape[0] - 1) / float(new_width - 1)
    ind[1] *= (A.shape[1] - 1) / float(new_height - 1)
    return scipy.ndimage.interpolation.map_coordinates(A, ind, order=1)

@log_runtime("Dimensionality calculation")
def dimensionality(A):
    A = A.reshape((-1,A.shape[2]))
    try:
        cov_mat = numpy.cov(A)
        e = np.linalg.eigvalsh(cov_mat)
    except:
        return -1
    return e.sum()**2 / (e*e).sum() 

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = np.real(phi) % np.pi

    return x0, y0, ap, bp, e, phi

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

@log_runtime("Local correlation eccentricity calculation")
def local_correlation_eccentricity(Cmaps,coords):
    eccentricities = np.zeros((len(coords)))
    for i in range(len(coords)):
        x,y = coords[i]
        C = Cmaps[i]

        # Crop the image to just the ellipse to make it faster!
        lw, num = scipy.ndimage.measurements.label(C>0.8)
        lw = lw==lw[x,y]
        lw = np.sqrt(scipy.ndimage.sobel(lw,axis=0)**2+scipy.ndimage.sobel(lw,axis=1)**2).astype(float) > 0
        X, Y = np.where(lw)

        try:
            coeffs = fit_ellipse(X, Y)
        except:
            return eccentricities
        if len(coeffs) == 6:
            x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
            eccentricities[i] = e

    return eccentricities

def dist_or_map(or_map):
    O = or_map.copy()
    O[O>np.pi/2] *= -1
    O[O<0] += np.pi
    return scipy.ndimage.gaussian_filter(O,2)

def rotate_or_map(or_map,angle):
    return (or_map.copy() - angle + np.pi) % np.pi 

def find_local_maxima(arr,min_dist):
    xmax0, ymax0 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=0)
    xmax1, ymax1 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=1)
    s1 = {(xmax0[i],ymax0[i],arr[xmax0[i],ymax0[i]]) for i in range(len(xmax0))}
    s2 = {(xmax1[i],ymax1[i],arr[xmax1[i],ymax1[i]]) for i in range(len(xmax1))}
    s = sorted(list(s1 & s2),key=lambda el : el[2],reverse=True)
    i = 0
    while i < len(s):
        j=i+1
        while j < len(s):
            if (s[i][0] - s[j][0])**2 + (s[i][1] - s[j][1])**2 < min_dist**2:
                s.pop(j)
            else:
                j+=1
        i+=1
    return s

@log_runtime("Kohonen map generation")
def kohonen_map(Cmaps):
    som = SOM(1,6)
    som.fit(np.array([C.flatten() for C in Cmaps]),epochs=10000,verbose=False)

    km = numpy.zeros(Cmaps[0].shape)
    for i in range(len(Cmaps)):
        km[i // km.shape[1], i % km.shape[1]] = som.winner(Cmaps[i].flatten())[1]
    return km

@log_runtime("Orientation map generation")
def gen_or_map(dsv, sheet_name, s_res):
    analysis_result = dsv.full_datastore.get_analysis_result(
        identifier="PerNeuronValue",
        value_name="LGNAfferentOrientation",
        sheet_name=sheet_name,
    )
    if len(analysis_result) == 0:
        NeuronAnnotationsToPerNeuronValues(dsv, ParameterSet({})).analyse()
    result = dsv.full_datastore.get_analysis_result(
        identifier="PerNeuronValue",
        value_name="LGNAfferentOrientation",
        sheet_name=sheet_name,
    )[0]
    st_ids = [s for s in dsv.get_segments() if len(s.spiketrains) > 0][
        0
    ].get_stored_spike_train_ids()

    orientations = result.get_value_by_id(st_ids)

    posx, posy = get_s(dsv, s_res)
    or_map_sampled = np.ones((max(posx) + 1, max(posy) + 1)) * -1

    for i in range(len(st_ids)):
        or_map_sampled[posx[i], posy[i]] = orientations[i]

    ind = scipy.ndimage.distance_transform_edt(
        or_map_sampled == -1, return_distances=False, return_indices=True
    )
    return or_map_sampled[tuple(ind)]
