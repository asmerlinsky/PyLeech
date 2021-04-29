# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:40:50 2018

@author: Agustin
"""

"""
Primera parte
"""
import os
import time
import PyLeech.Utils.AbfExtension as abfe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PyLeech.Utils.constants as constants
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.spsortUtils as spsortUtils
import PyLeech.Utils.SpSorter as SpSorter
import PyLeech.Utils.unitInfo as burstStorerLoader
nan = constants.nan
opp0 = constants.opp0
import PyLeech.Utils.filterUtils as filterUtils
import PyLeech.Utils.miscUtils as miscUtils

'''If loading run one of these:
sorter = SpSorter.SpSorter('RegistrosDP_PP/2019_08_30_0006.pkl')
'''

np.set_printoptions(precision=3)
plt.ion()
noise_fn = '19-08-30/2019_08_30_0000.abf'

noise_arr_dict, time_vector , fs= abfe.getArraysFromAbfFiles(noise_fn, ['IN4', 'IN5', 'IN6', 'IN7'])
fig, ax = plt.subplots()
for key, item in noise_arr_dict.items():
    spectrum = filterUtils.getPowerSpectrum(item, fs, 100000)
    print(key, np.median(spectrum[1]))
    filterUtils.plotSpectrums(item, sampling_rate=fs, nperseg=100000, ax=ax, label1=key)
ax.legend()



peak_dict = {}
every_peak = np.array([])
for key, item in noise_arr_dict.items():
    bl = np.median(filterUtils.getPowerSpectrum(item, fs, 100000)[1])
    if key == "IN4":
        peak_dict[key] = np.array(filterUtils.getSpectralPeaks(item, fs, 3000, 100000, height=bl+30, distance=10)).T
    else:
        peak_dict[key] = np.array(filterUtils.getSpectralPeaks(item, fs, 3000, 100000, height=bl+30, distance=40)).T
    every_peak = np.append(every_peak, peak_dict[key][:,0])
every_peak = np.unique(every_peak)



filenames = ['19-08-30/2019_08_30_0006.abf',
             ]

arr_dict, time_vector , fs= abfe.getArraysFromAbfFiles(filenames, ['IN4', 'IN5', 'IN6', 'IN7'])

# NS = arr_dict['Vm1']
traces = [-arr_dict['IN4'], -arr_dict['IN5'], -arr_dict['IN6'], arr_dict['IN7']]

ln = len(time_vector)

filt_traces = []
i = 0
for key, trace in arr_dict.items():
    if i == 0:
        tr_filt = filterUtils.runVariablePowerFilters(trace, peak_dict[key], fs, (.3, .4))
        # print(tr_filt)
    else:
        tr_filt = filterUtils.runVariablePowerFilters(trace, peak_dict[key], fs, (5, 10))

    tr_filt = filterUtils.runButterFilter(tr_filt, 2500, sampling_rate=fs)
    tr_filt = filterUtils.runButterFilter(tr_filt, 5, sampling_rate=fs, butt_order=4, btype='high')
    i += 1

    filt_traces.append(tr_filt)

# fig, ax = plt.subplots(4, 1, sharex=True)
fig1, ax1 = plt.subplots(4, 1, sharex=True)

for i in range(len(filt_traces)):
    # ax[i].plot(traces[i])
    # ax[i].plot(filt_traces[i])
    filterUtils.plotSpectrums(arr_dict[list(arr_dict)[i]], filt_traces[i], sampling_rate=fs, nperseg=10000, ax=ax1[i])




del arr_dict, traces
"""
i = 3
fig, ax = plt.subplots()
# ax.plot(traces[i])
ax.plot(filt_traces[i])
"""


sorter = SpSorter.SpSorter(filenames, "RegistrosDP_PP", filt_traces, time_vector, fs)
del time_vector, filt_traces

sorter.normTraces()

sorter.plotDataList()
#
# sorter.plotTraceAndStd(channel=0)
# sorter.plotTraceAndStd(channel=1)
#
#
#
sorter.smoothAndVisualizeThreshold(channel=0, vect_len=5, threshold=8, interval=[.3, .4])
sorter.smoothAndVisualizeThreshold(channel=1, vect_len=5, threshold=8, interval=[.3, .4])
sorter.smoothAndVisualizeThreshold(channel=2, vect_len=5, threshold=8, interval=[.3, .4])

sorter.smoothAndFindPeaks(vect_len=5, threshold=6, min_dist=100)

# sorter.plotDataListAndDetection()

sorter.makeEvents(99, 100)
# sorter.plotEventsMedians()

sorter.makeNoiseEvents(size=4000)

# sorter.plotEvents(plot_noise=True)
"""
Visualizing the threshold
"""



sorter.takeSubSetEvents(.3)

"""
good events in trace
"""
# sorter.plotDataListAndDetection(good_evts=True)


"""
From here I can either use event max for classification or run pca for classification
"""

sorter.getPcaBase(plot_vectors=False)

sorter.getVectorWeights(30)


sorter.GmmClusterEvents(40, use_pca=True, dim_size=25, n_init=400, max_iter=4000, save=True)

sorter.plotTemplates()
sorter.plotClusters()

## 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 37



sorter.plotTemplates([10, 12, 14, 11, 13, 24])
sorter.plotClusters([10, 12, 14, 11, 13, 24])

sorter.hideClusters(10)

sorter.getSimilarTemplates(15)[0][:10]

sorter.plotTemplates([6, 11, 12, 13, 14, 15, 16, 19, 20, 23])
sorter.plotClusters([6, 11, 12, 13, 14, 16, 15, 19, 20, 23])

sorter.hideClusters([15, 16, 19, 23])

sorter.getSimilarTemplates(17)[0][:10]

sorter.plotTemplates([3, 8, 9, 17, 20, 21, 28,33])
sorter.plotClusters([3, 8, 9, 17, 20, 21, 28, 33])


sorter.hideClusters([0, 3, 5, 6, 7, 9, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 31, 32, ])

sorter.plotTemplates()
sorter.plotClusters()

sorter.hideClusters(1)

sorter.subdivideClusters(2, 2, dim_size=20)
sorter.plotClusters([2, 44])
sorter.plotTemplates([2, 44])

sorter.plotTemplates()
sorter.plotClusters()

sorter.subdivideClusters(4, 5, dim_size=25)
sorter.plotTemplates([4, 45, 46, 47, 48])
sorter.plotClusters([4, 45, 46, 47, 48])

sorter.hideClusters([45, 46, 47, 48])


sorter.saveResults()

sorter.plotTemplates()
sorter.plotClusters()

sorter.plotTemplates([34, 35, 36, 37])
sorter.plotClusters([34, 35, 36, 37], y_lim=[-60, 60])

sorter.mergeClusters(34, [35, 36, 37])
sorter.subdivideClusters(34, 5, dim_size=25)

sorter.plotTemplates([34, 49, 50, 51, 52 ])
sorter.plotClusters([34, 49, 50, 51, 52])
sorter.hideClusters([50,52])
sorter.mergeClusters(34, [49, 51])

sorter.subdivideClusters(34, clust_no=5, dim_size=25)

sorter.plotTemplates([34, 53, 54, 55, 56])
sorter.plotClusters([34, 53, 54, 55, 56])
sorter.subdivideClusters(55, 5, 25)

sorter.plotTemplates([34, 53, 54, 55, 56, 57, 58, 59, 60])
sorter.plotClusters([34, 53, 54, 55, 56, 57, 58, 59, 60])

sorter.hideClusters([55, 58, 60])
sorter.subdivideClusters(59, 3, 25)


sorter.plotTemplates([34, 53, 54, 56, 57, 59, 60, 61, 62])
sorter.plotClusters([34, 53, 54, 56, 57, 59, 60, 61, 62])

sorter.subdivideClusters(53, 4, 25)

sorter.plotTemplates([53, 54, 56, 57, 59, 60, 61, 63, 64, 65])
sorter.plotClusters([53, 54, 56, 57, 59, 60, 61, 63, 64, 65])


sorter.hideClusters([34, 53, 57, 62, 64, 65])
sorter.subdivideClusters(56, 4, 25)

sorter.plotTemplates([54, 56, 59, 60, 61, 63, 66, 67, 68])
sorter.plotClusters([54, 56, 59, 60, 61, 63, 66, 67, 68])

sorter.hideClusters([56, 63, 66, 68])

sorter.plotTemplates()
sorter.plotClusters()

sorter.saveResults()


cl = np.unique(sorter.getSimilarTemplates(33)[0][:10])
sorter.plotTemplates(cl)
sorter.plotClusters(cl)

sorter.subdivideClusters(29, 4, 25)
sorter.plotTemplates([29, 69, 70, 71])
sorter.plotClusters([29, 69, 70, 71])

sorter.hideClusters([8, 29])

sorter.subdivideClusters(69, 3, 25)

sorter.plotTemplates([69, 72, 73])
sorter.plotClusters([69, 72, 73])

sorter.hideClusters([69])

sorter.subdivideClusters(72, 4, 25)

sorter.plotTemplates([72, 74, 75, 76])
sorter.plotClusters([72, 74, 75, 76])

sorter.hideClusters([72, 75, 76])

sorter.plotClusters()

sorter.mergeClusters(11, [12, 13])

sorter.plotClusters(11)

sorter.subdivideClusters(11, 5, 25)

sorter.plotTemplates([11, 77, 78, 79, 80])
sorter.plotClusters([11, 77, 78, 79, 80])

sorter.hideClusters([11, 78, 79, 80])

sorter.subdivideClusters(77, 4, 25)

sorter.plotClusters([81, 82, 83])
sorter.plotTemplates([81, 82, 83])

sorter.plotClusters([30, 33])
sorter.plotTemplates([30, 33])

sorter.mergeClusters(30, 33)

sorter.subdivideClusters(30, 6, 25)

sorter.plotTemplates([30, 84, 85, 86, 87, 88])
sorter.plotClusters([30, 84, 85, 86, 87, 88])

sorter.hideClusters([30, 84, 85, 86, 88])

sorter.subdivideClusters(87, 5, 25)

sorter.plotTemplates([87, 89, 90, 91, 92])
sorter.plotClusters([87, 89, 90, 91, 92])

sorter.hideClusters([87, 90])

sorter.subdivideClusters(39, 5, 25)

sorter.plotClusters([39, 93, 94, 95, 96])
sorter.plotTemplates([39, 93, 94, 95, 96])

sorter.hideClusters([39, 94])

sorter.plotClusters()

sorter.subdivideClusters(38, 2, 25)
sorter.plotClusters([38, 97])
sorter.subdivideClusters(38, 2, 25)

sorter.plotClusters([38, 97, 98])
sorter.plotTemplates([38, 97, 98])

sorter.saveResults()

sorter.renumberClusters()

sorter.hideBadEvents()
sorter.plotClusters()

sorter.subdivideClusters(22, 2, 25)
sorter.plotClusters([22, 25])

sorter.subdivideClusters(13, 2, 25)
sorter.plotClusters([13, 26])

sorter.subdivideClusters(14, 2, 25)
sorter.plotClusters([14, 27])

sorter.subdivideClusters(5, 2, 25)
sorter.plotClusters([5, 28])

sorter.subdivideClusters(2, 5, 25)

sorter.plotTemplates([2, 29, 30, 31, 32])
sorter.plotClusters([2, 29, 30, 31, 32])
sorter.hideClusters(2)

sorter.hideBadEvents()
sorter.plotClusters()

sorter.makeCenterDict(700, 800)
# sorter.plotCenterDict()


peaks = sorter.smoothAndGetPeaks(vect_len=5, threshold=20, min_dist=150)

sorter.generatePrediction(store_prediction=True, before=74, after=125, peak_idxs=peaks)

# sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
#                    pred=sorter.pred[-1][:,::4])

sorter.secondaryPeeling(vect_len=5, threshold=20, min_dist=100, store_mid_steps=False)

# sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
#                    pred=sorter.pred[-1][:,::4])

sorter.secondaryPeeling(vect_len=5, threshold=20, min_dist=100, store_mid_steps=False)

# sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
#                    pred=sorter.pred[-1][:,::4])

sorter.secondaryPeeling(vect_len=5, threshold=10, min_dist=100, store_mid_steps=False)

# sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
#                    pred=sorter.pred[-1][:,::4])

sorter.secondaryPeeling(vect_len=5, threshold=10, min_dist=100, store_mid_steps=False)

# sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
#                    pred=sorter.pred[-1][:,::4])

sorter.secondaryPeeling(vect_len=5, threshold=10, min_dist=100, store_mid_steps=False)

sorter.plotPeeling(time=sorter.time[::4], to_peel_data=sorter.peel[-2][:,::4],
                   pred=sorter.pred[-1][:,::4])


sorter.mergeRoundsResults()

sorter.getSortedTemplateDifference()[1][:10]



cl = list(sorter.final_spike_dict)
good_colors = spsortUtils.setColors(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)


cl = np.unique(sorter.getSortedTemplateDifference(list(sorter.final_spike_dict))[1][:10])
sorter.plotTemplates(cl)
sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)
cl = [13, 31]
sorter.plotTemplates(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)
sorter.mergeFinalRoundClusters([13, 31])

cl = [15, 30]
sorter.plotTemplates(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)

sorter.mergeFinalRoundClusters([15, 30])


cl = [13, 32]
sorter.plotTemplates(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)



std = sorter.getSortedTemplateDifference(list(sorter.final_spike_dict))[1][:10]
print(std)
cl = np.unique(std)
cl = [8, 25]
sorter.plotTemplates(cl)
# sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)

sorter.mergeFinalRoundClusters([8, 25])

cl = [3, 24, 12]
sorter.plotTemplates(cl)
# sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)
sorter.mergeFinalRoundClusters([3, 12])
sorter.mergeFinalRoundClusters([11, 18])

cl = [20, 22]
sorter.plotTemplates(cl)
# sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)

sorter.mergeFinalRoundClusters([20, 22])

cl = [14, 29]
sorter.plotTemplates(cl)
# sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)

sorter.mergeFinalRoundClusters([14, 29])




cl = [8, 20]
sorter.plotTemplates(cl)
# sorter.plotClusters(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)


cl = [0, 4]
sorter.plotTemplates(cl)
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
fig, ax_list = sorter.plotCompleteDetection(legend=True, step=3, clust_list=cl)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=cl)



DE3 = 0
burst_object = burstStorerLoader.UnitInfo(sorter.filename, mode='save', traces=sorter.traces, time=sorter.time, spike_dict=sorter.final_spike_dict, spike_freq_dict=spike_freqs,
                                          De3=DE3, template_dict=sorter.template_dict, color_dict=good_colors)


filename="2019_08_30_0006.pklspikes"
nerve_channels = ["DP", "PP", "AA", "MA"]
note = "\nTENGO QUE CHEQUEAR EN EL CUADERNO QUE ESTEN BIEN LOS NERVIOS\n"
burst_object = burstStorerLoader.UnitInfo(filename, mode='load')
burst_object.nerve_channels = nerve_channels
burst_object.notes = note
burst_object.saveResults()


burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, template_dict=burst_object.template_dict,
                                  scatter_plot=True, outlier_thres=3.5,  ms=2)

list(burst_object.spike_freq_dict)