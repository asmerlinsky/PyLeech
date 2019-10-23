# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 00:02:15 2019

@author: Agustin Sanchez Merlinsky

This file segments oscilations by file using NS,
then calculates correlation between every neuron by segment and between files.
Finally clusters them by correlation


"""


import PyLeech.Utils.burstClasses as burstClasses
import PyLeech.Utils.AbfExtension as abfe
from PyLeech.Utils.burstStorerLoader import BurstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.abfUtils as abfUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload
import numpy as np
import PyLeech.Utils.filterUtils as fU
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import scipy.signal as spsig
import PyLeech.Utils.NLDUtils as NLD

segment_list = []
row_labels = []
# reload(burstClasses)
# reload(burstUtils1)

if __name__ == "__main__":
    cdd = CDU.loadDataDict()
    file_list = list(cdd.keys())
    fn = file_list[5]

    burst_object = BurstStorerLoader(fn, 'ResgistrosDP_PP', 'load')
    arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1', 'IN6'])
    # dp_trace = arr_dict['IN6']
    NS = arr_dict['Vm1']
    del arr_dict

    good_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if neuron_dict['neuron_is_good']]

    crawling_intervals = cdd[fn]['crawling_intervals']

    dt_step = 0.1
    #
    # new_sfd = burstUtils.removeOutliers(burst_object.spike_freq_dict, 5)
    # binned_sfd = burstUtils.digitizeSpikeFreqs(new_sfd, dt_step, time[-1], count=False)
    # cut_binned_freq_array = burstUtils.binned_spike_freq_dict_ToArray(binned_sfd, crawling_interval, good_neurons)
    #
    # gaussian = NLD.generateGaussianKernel(sigma=3, time_range=20, dt_step=dt_step)
    #
    # smoothed_sfd = {}
    # for key, items in cut_binned_freq_array.items():
    #     smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], gaussian, mode='same')])

    correlation_segments = burstClasses.SegmentandCorrelate(burst_object.spike_freq_dict, NS, time,
                                                            time_intervals=burst_object.crawling_segments, no_cycles=1,
                                                            intracel_peak_height=-52)


    burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]],
                          template_dict=burst_object.template_dict, outlier_thres=10, sharex=None)

    for i in np.unique(np.concatenate(correlation_segments.intervals)):
        plt.axvline(i)
    correlation_segments.concatenateRasterPlot()
    sl = []
    i = 0
    for segment in correlation_segments.resampled_segmented_neuron_frequency_list:

        for key, items in segment.items():
            if max(items) != 0.0:
                sl.append(items)
                row_labels.append('F' + str(j) + '-' + 'S' + str(i) + 'n' + str(key))
        i += 1
    segment_list.append(sl)

np_row_labels = np.array(row_labels)
new_sl1, new_sl2 = burstUtils.resampleArrayList(segment_list[0], segment_list[1])
segment_list = new_sl1 + new_sl2
corr_mat = np.corrcoef(segment_list)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_mat)
fig.colorbar(cax)
#
# ax.set_xticklabels(['']+row_labels)
# ax.set_yticklabels(['']+row_labels)
plt.gca().set_aspect('auto')
plt.show()

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score

# data, row_idx, col_idx = sg._shuffle(data, random_state=0)

i = 18
scores = []
for i in range(3, 10):
    model = SpectralCoclustering(n_clusters=i, random_state=0)
    model.fit(corr_mat)
    score = consensus_score(model.biclusters_,
                            (model.rows_[model.rows_], model.columns_[model.columns_]))
    # print(i, i*score)
    scores.append(score)
    # print("consensus score: {:.3f}".format(score))

    fit_data = corr_mat[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    # plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.matshow(fit_data)
    plt.colorbar()
    plt.gca().set_aspect('auto')
    plt.title(str(i))
plt.plot(scores)

j = 0
for i in np.unique(model.row_labels_):
    print(i, j)
    j += sum(model.row_labels_ == i)

clust_list = [3, 4, 5, 11, 13]
for cl in clust_list:
    neurons = []
    for element in np_row_labels[model.row_labels_ == cl]:
        neurons.append(element[1] + '-' + element[-2:])
    print('cluster %i' % cl, np.unique(np.array(neurons)))


##### Getting the autocorrelation function


segment_list = []
row_labels = []
reload(burstClasses)
# reload(burstUtils1)
pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)): print(j, pkl_files[j])
peak_heights = []
for j in range(6,8):
    filename = pkl_files[j]
    print(filename)

    burst_object = PyLeech.Utils.burstStorerLoader.BurstStorerLoader(filename, 'load')

    basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
    arr_dict, time, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
    NS = arr_dict['Vm1']
    del arr_dict
    #
    # correlation_segments = burstUtils.SegmentandCorrelate(burst_object.spike_freq_dict, NS, time,
    #                                                       time_intervals=burst_object.crawling_segments,
    #                                                       intracel_cutoff_freq=2,
    #                                                       no_cycles=1, intracel_peak_height=-52)
    # burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]],
    #                     template_dict=burst_object.template_dict, outlier_thres=10, sharex=None)

    binned_sfd = burstUtils.digitizeSpikeFreqs(burst_object.spike_freq_dict, 5 / fs, time[-1], counting=True)
    print('file %i' % j)
    for key, items in binned_sfd.items():
        print('neuron %i' % key)
        # correlation = spsig.correlate(items[1], items[1])
        fig, ax = plt.subplots(1,1)
        fU.plotSpectrums(items[1], sampling_rate=fs / 5)
        # ax.plot(correlation)
        # ax.set_ylim([-10,100])
        # ax.set_xlim([.49*len(correlation), .51*len(correlation)])
        fig.suptitle("file %i, neuron %i" % (j, key))

