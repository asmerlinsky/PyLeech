# import sys
# [sys.path.append(i) for i in ['.', '..']]

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstUtils
from PyLeech.Utils.burstStorerLoader import BurstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import matplotlib.pyplot as plt
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig

if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    file_list = list(cdd)

    for fn in file_list:
        if cdd[fn]['skipped'] or cdd[fn]["DE3"] is None or cdd[fn]["DE3"] == -1: continue

        burst_object = BurstStorerLoader(fn, 'RegistrosDP_PP', 'load')
        burst_object.isDe3 = cdd[fn]["DE3"]

        binning_dt = 0.1
        spike_kernel_sigma = 1

        spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, .1,
                                                           selected_neurons=[burst_object.isDe3],
                                                           time_length=burst_object.time[-1])

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)
        smoothed_sfd = {}
        for key, items in spike_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])

        spike_idxs = NLD.getSpikeIdxs(smoothed_sfd, cdd[fn]["crawling_intervals"])

        # fig, axes = burstUtils.plotFreq(burst_object.spike_freq_dict, color_dict=burst_object.color_dict, outlier_thres=3.5,
        #                                 template_dict=burst_object.template_dict, scatter_plot=True,
        #                                 ms=4, draw_list=[burst_object.isDe3])
        fig.suptitle(fn)
        # fig, ax = plt.subplots()
        # ax.plot(smoothed_sfd[burst_object.isDe3][1][spike_idxs], c='k')
        # ax.plot(smoothed_sfd[burst_object.isDe3][1][spike_idxs][::2], c='g')

        # ax.axvline(cdd[fn]["crawling_intervals"][0][0], c='r')
        # ax.axvline(cdd[fn]["crawling_intervals"][0][1], c='r')
        try:
            concatenated_traces = np.append(concatenated_traces, smoothed_sfd[burst_object.isDe3][1][spike_idxs])
        except NameError:
            concatenated_traces = smoothed_sfd[burst_object.isDe3][1][spike_idxs]

        # try:
        #     concatenated_embeddings = np.append(concatenated_embeddings, NLD.getDerivativeEmbedding(smoothed_sfd[burst_object.isDe3][1], dt=.1, emb_size=3), axis=0)
        # except NameError:
        #     concatenated_embeddings = NLD.getDerivativeEmbedding(smoothed_sfd[burst_object.isDe3][1], dt=.1, emb_size=3)

    fig, ax = plt.subplots()
    # ax.plot(concatenated_embeddings[:,0])
    ax.plot(concatenated_traces)

    # rt = NLD.getCloseReturns(concatenated_embeddings)

    concatenated_traces = np.array([concatenated_traces]).T
    concatenated_traces = concatenated_traces.astype(np.float32)
    concatenated_traces = concatenated_traces[::2]
    concatenated_traces[concatenated_traces<.01] = 0

    import time
    ln = concatenated_traces.shape[0]
    tm = []
    # for i in np.arange(.3, .7, 0.1):
    # i = .3
    start = time.time()
    # a = %timeit -o NLD.getCloseReturns(concatenated_traces[:int(ln * i)])
    rt = NLD.getCloseReturns(concatenated_traces)
    # rt = NLD.getCloseReturns(concatenated_traces[:int(ln * i)])
    end = time.time()
    print(end-start)
        # print(i, end-start)

