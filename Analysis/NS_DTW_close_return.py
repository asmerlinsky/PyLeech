import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstUtils
from PyLeech.Utils.unitInfo import UnitInfo
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import matplotlib.pyplot as plt
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig
import os.path
import winsound
from mpl_toolkits.mplot3d import Axes3D
import PyLeech.Utils.AbfExtension as abfe
import time
from tslearn import metrics
from sklearn.manifold import TSNE
if __name__ == "__main__":
    cdd = CDU.loadDataDict()
    file_list = list(cdd)
    binning_dt = .1
    spike_kernel_sigma = .5
    rt_frac = .1
    RT_THRESHOLD = .2
    trace_list = []
    emb_list = []
    ran_files = []
    run_dict = {}

    cycle_list = []

    # file_list = file_list[-2:]
    cycle_duration = []
    i = 0
    for fn in file_list:
        # if '2019_07_22_0009' not in fn:
        #     continue
        if cdd[fn]['skipped']:
            print('{} is skipped'.format(fn))
            continue
        try:
            ns_channel = [key for key, items in cdd[fn]['channels'].items() if 'NS' == items][0]
        except Exception as e:

            print('{} raised {} \nContinuing'.format(fn, e))
            continue

        pkl_fn = abfe.getAbFileListFromBasename(fn)[0]
        try:
            arr_dict, time_vector , fs = abfe.getArraysFromAbfFiles(pkl_fn, [ns_channel])
        except Exception as e:
            print('{} raised {}\nContinuing'.format(pkl_fn, e))
            continue

        ran_files.append(os.path.splitext(os.path.basename(fn))[0])
        run_dict[os.path.splitext(os.path.basename(fn))[0]] = i

        NS_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=1 / fs)
        bl_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=45, time_range=10 * 60, dt_step=1 / fs)

        data = arr_dict[ns_channel] - np.mean(arr_dict[ns_channel])
        del arr_dict

        data[(data<-20) | (data>20)] = 0
        interval = fs* np.array(cdd[fn]['crawling_intervals'])
        # interval = np.array((0, data.shape[0]))
        
        interval = interval.astype(int)

        bl = spsig.fftconvolve(data, bl_kernel, mode='same')[interval[0]:interval[1]:int(binning_dt * fs)]
        trace = spsig.fftconvolve(data, NS_kernel, mode='same')[interval[0]:interval[1]:int(binning_dt * fs)]

        # bl = spsig.fftconvolve(data, bl_kernel, mode='same')[interval[0]:interval[1]]
        # trace = spsig.fftconvolve(data, NS_kernel, mode='same')[interval[0]:interval[1]]
        
        
        data = data[interval[0]:interval[1]:int(binning_dt * fs)]
        # bl = bl[interval[0]:interval[1]]
        trace -= bl
        
        peaks = spsig.find_peaks(trace, 2, distance=int(10/binning_dt))[0]
        start_idxs = peaks[:-1]
        end_idxs = peaks[1:]
        
        
        cycle_duration.extend(np.diff(peaks)*binning_dt)
        
        for idx in range(peaks.shape[0]-1):
            tc = trace[peaks[idx]:peaks[idx+1]]
            tc = 2*(tc - tc.min())/tc.max() - 1 
            
            cycle_list.append(trace[peaks[idx]:peaks[idx+1]])
        
                
        time_array = np.linspace(0, data.shape[0]*binning_dt, num=data.shape[0])
        
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(time_array, data)
        ax[0].plot(time_array, trace)
        ax[0].plot(time_array, bl)
        ax[1].plot(time_array, trace)
        ax[1].axhline(np.median(trace)+2)
        for pk in peaks:
            ax[1].scatter(time_array[pk], trace[pk], marker='x', c='r')
            ax[0].scatter(time_array[pk], data[pk], marker='x', c='r')
        fig.suptitle(fn)

    


    dist_matrix = metrics.cdist_dtw(cycle_list, global_constraint="sakoe_chiba", sakoe_chiba_radius=10)

    fig, ax = plt.subplots()
    mappable = ax.imshow(dist_matrix)
    fig.colorbar(mappable)
    
    tsne = TSNE(perplexity=5., early_exaggeration=100., n_iter=5000, n_iter_without_progress=1000, metric='precomputed')
    tsne_data = tsne.fit_transform(dist_matrix)
    print(tsne.kl_divergence_)
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(tsne_data[:, 0], tsne_data[:, 1])

    H = np.histogram2d(tsne_data[:, 0], tsne_data[:, 1], bins=10)

    mappable = ax[1].imshow(H[0].T, origin='lower')
    fig.colorbar(mappable)