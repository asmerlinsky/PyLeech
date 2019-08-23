import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstStorerLoader as bStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import os
import numpy as np
import scipy.signal as spsig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from itertools import combinations
import PyLeech.Utils.burstClasses as burstClasses
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    cdb = CDU.loadCrawlingDatabase()

    file_list = []
    for files in list(cdd.keys()):
        burst_obj = bStorerLoader.BurstStorerLoader(files, 'RegistrosDP_PP', mode='load')
        try:
            arr_dict, time_vector1, fs = abfe.getArraysFromAbfFiles(files, ['Vm1'])
            file_list.append(files)
        except:
            pass

    run_once = []

    # for fn1, fn2 in combinations(file_list, 2):
    for fn1 in [file_list[1]]:

        arr_dict, time_vector1, fs1 = abfe.getArraysFromAbfFiles(fn1, ['Vm1'])
        NS1 = arr_dict['Vm1']
        #
        # arr_dict, time_vector2, fs2 = abfe.getArraysFromAbfFiles(fn2, ['Vm1'])
        # NS2 = arr_dict['Vm1']

        del arr_dict


        NS1[(NS1 > -20) | (NS1 < -60)] = np.median(NS1)
        # NS2[(NS2 > -20) | (NS2 < -60)] = np.median(NS2)
        # good_neurons = [neuron for neuron, neuron_dict in info['neurons'].items() if neuron_dict['neuron_is_good']]

        crawling_intervals1 = cdd[fn1]['crawling_intervals']
        print(crawling_intervals1)
        # crawling_intervals2 = cdd[fn2]['crawling_intervals']
        # print(crawling_intervals2)

        binning_dt = 0.1

        burst_obj = bStorerLoader.BurstStorerLoader(fn1, 'RegistrosDP_PP', mode='load')

        # new_sfd = burstUtils.removeOutliers(burst_obj.spike_freq_dict, 5)
        # binned_sfd = burstUtils.digitizeSpikeFreqs(new_sfd, binning_dt, time_vector[-1], count=False)
        # cut_binned_freq_array = burstUtils.binned_spike_freq_dict_ToArray(binned_sfd, crawling_intervals, good_neurons)
        # kernel = NLD.generateGaussianKernel(sigma=2, time_range=20, dt_step=binning_dt)
        # smoothed_sfd = {}
        # for key, items in cut_binned_freq_array.items():
        #     smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])
        step1 = int(binning_dt * fs1)
        # step2 = int(binning_dt * fs2)
        idxs1 = []
        for interval in crawling_intervals1:
            idxs1.append(np.where((time_vector1 > interval[0]) & (time_vector1 < interval[1]))[0][::step1])
        idxs1 = np.hstack(idxs1)

        cut_time1 = time_vector1[idxs1]

        kernel = NLD.generateGaussianKernel(sigma=4, time_range=30, dt_step=1 / fs1)
        conv_NS1 = spsig.fftconvolve(NS1, kernel, mode='same')
        cut_NS1 = conv_NS1[idxs1]

        # idxs2 = []
        # for interval in crawling_intervals2:
        #     idxs2.append(np.where((time_vector2 > interval[0]) & (time_vector2 < interval[1]))[0][::step2])
        # idxs2 = np.hstack(idxs2)

        # cut_time2 = time_vector2[idxs2]

        # kernel = NLD.generateGaussianKernel(sigma=4, time_range=30, dt_step=1 / fs2)
        # conv_NS2 = spsig.fftconvolve(NS2, kernel, mode='same')
        # cut_NS2 = conv_NS2[idxs2]



        fig, axes = burstUtils.plotFreq(burst_obj.spike_freq_dict, color_dict=burst_obj.color_dict,
                                        template_dict=burst_obj.template_dict, scatter_plot=True,
                                        optional_trace=[time_vector1[::int(step1/4)], NS1[::int(step1/4)]],
                                        draw_list=[0],
                                        outlier_thres=5, ms=4)
        # fig.suptitle(os.path.basename(os.path.splitext(fn)[0]))
        # # reload(burstClasses)
        # segmented_data = burstClasses.SegmentandCorrelate(binned_sfd, NS, time_vector, fs, crawling_intervals,
        #                                                   intracel_peak_height=None, intracel_prominence=3, sigma=1,
        #                                                   intracel_peak_distance=10)

        embedding1 = NLD.getDerivativeEmbedding(cut_NS1, 0.1, 3)
        jumps = np.where(np.diff(idxs1) != np.diff(idxs1)[0])[0]
        if len(jumps) > 0:

            embedding1[jumps] = np.nan
            embedding1[jumps+1] = np.nan

        # NLD.plot3Dline(embedding1)
        sc = StandardScaler()
        scaled_embedding1 = sc.fit_transform(embedding1)

        embedding2 = NLD.getDerivativeEmbedding(cut_NS2, 0.1, 3)
        jumps = np.where(np.diff(idxs2) != np.diff(idxs2)[0])[0]
        if len(jumps) > 0:
            embedding2[jumps] = np.nan
            embedding2[jumps+1] = np.nan
        # NLD.plot3Dline(embedding2)
        sc = StandardScaler()
        scaled_embedding2 = sc.fit_transform(embedding2)


        # NLD.plot3Dline(scaled_embedding1)
        rt2 = NLD.getCloseReturns(scaled_embedding2)
        rt1 = NLD.getCloseReturns(scaled_embedding1)
        if fn1 not in run_once:
            fig1, ax1, fig2, ax2 = NLD.plotCloseReturns(rt1, reorder=False, masked=False, thr=.07)
            fig1.suptitle(fn1)
            fig2.suptitle(fn1)
            run_once.append(fn1)
            fig, ax = NLD.plot3Dline(scaled_embedding1)
            fig.suptitle(fn1)
        if fn2 not in run_once:
            fig1, ax1, fig2, ax2 = NLD.plotCloseReturns(rt2, reorder=False, masked=False, thr=.07)
            fig1.suptitle(fn2)
            fig2.suptitle(fn2)
            run_once.append(fn2)
            fig, ax = NLD.plot3Dline(scaled_embedding2)
            fig.suptitle(fn2)

        cross_rt = NLD.getCloseReturns(scaled_embedding1, scaled_embedding2)
        fig1, ax1, fig2, ax2 = NLD.plotCloseReturns(cross_rt, thr=.05, reorder=False, masked=False)
        fig1.suptitle(fn1 + " &\n" + fn2)
        fig2.suptitle(fn1 + "&\n" + fn2)
        fig1.tight_layout()
        fig2.tight_layout()

    fig = plt.figure()
    ax = Axes3D(fig)
    # NLD.plot3Dline(scaled_embedding1[1362:1630], [fig, ax],colorbar=False)
    NLD.plot3Dline(scaled_embedding2, [fig, ax])
    #
    fig, ax = plt.subplots()
    ax.plot(scaled_embedding1[1362-1000:1630+1000, 0])
    ax.plot(scaled_embedding2[429:1000, 0])
