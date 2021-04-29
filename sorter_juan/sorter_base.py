# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:40:50 2018

@author: Agustin
"""
import PyLeech.Utils.unitInfo
import PyLeech.Utils.burstUtils

"""
Primera parte
"""

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

'''If loading run one of these:
sorter = SpSorter.SpSorter('RegistrosDP_PP/2019_07_22_0014.pkl')
'''

np.set_printoptions(precision=3)
plt.ion()

filenames = ['19-07-22/2019_07_22_0014.abf',
             ]

arr_dict, time , fs= abfe.getArraysFromAbfFiles(filenames, ['IN5', 'IN6', 'Vm1'])

n1 = arr_dict['IN6']
n2 = arr_dict['IN5']
NS = arr_dict['Vm1']

del arr_dict

# plt.figure()
# PyLeech.filterUtils.plotSpectrums(n1, n2, sampling_rate=fs, nperseg=10000)
ln = len(time)

n1_filt = filterUtils.runButterFilter(n1, 5000, sampling_rate=fs)
n1_filt = filterUtils.runButterFilter(n1_filt, 5, sampling_rate=fs, butt_order=4, btype='high')

n2_filt = filterUtils.runButterFilter(n2, 5000, sampling_rate=fs)
n2_filt = filterUtils.runButterFilter(n2_filt, 5, sampling_rate=fs, butt_order=4, btype='high')

# PyLeech.filterUtils.plotSpectrums(n1, n1_filt, sampling_rate=fs, nperseg=10000)

sorter = SpSorter.SpSorter(filenames, "RegistrosDP_PP", [n1_filt, n2_filt], time, fs)
del time, n1, n1_filt, n2, n2_filt

sorter.normTraces()
#
# sorter.plotDataList()
#
# sorter.plotTraceAndStd(channel=0)
# sorter.plotTraceAndStd(channel=1)



# sorter.smoothAndVisualizeThreshold(channel=0, vect_len=5, threshold=5)
# sorter.smoothAndVisualizeThreshold(channel=1, vect_len=5, threshold=5)
#

sorter.smoothAndFindPeaks(vect_len=5, threshold=10, min_dist=150)

# sorter.plotDataListAndDetection()

sorter.makeEvents(99, 100)
# sorter.plotEventsMedians()

sorter.makeNoiseEvents(size=4000)

sorter.plotEvents(plot_noise=False)
"""
Visualizing the threshold
"""


sorter.takeSubSetEvents(.1)

"""
good events in trace
"""
# sorter.plotDataListAndDetection(good_evts=True)


"""
From here I can either use event max for classification or run pca for classification
"""

sorter.getPcaBase()

sorter.getVectorWeights(30)


sorter.fitTSNE(pca_dim_size=10)
sorter.viewTSNE()

sorter.GmmClusterEvents(20, use_tsne=True, n_init=400, max_iter=6000, save=True)

sorter.viewTSNE()


sorter.plotTemplates()
sorter.plotClusters()

sorter.viewTSNE(clust_list=[4])

sorter.plotClusters([4])

sorter.subdivideClusters(4, 2)
sorter.plotClusters([4, 20])

sorter.viewTSNE(clust_list=[4, 20])


sorter.subdivideClusters(7, 5)
sorter.plotClusters([4, 20], superposed=True)

sorter.hideClusters(4)



sorter.setGoodColors()
sorter.viewTSNE([7, 20, 21, 22, 23])
sorter.plotClusters([7, 20, 21, 22, 23])

sorter.subdivideClusters(14, 8, use_tsne=True)
cl = [14, 24, 25, 26, 27, 28, 29, 30]
sorter.setGoodColors()
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)
sorter.mergeClusters(14, [26, 28])

cl = [14, 24, 25, 27, 29]

sorter.viewTSNE(cl)
sorter.plotTemplates(cl)
sorter.plotClusters(cl)


sorter.plotClusters([14, 30], superposed=True)
sorter.mergeClusters(14, 30)


sorter.hideClusters(29)

cl = np.unique(sorter.getClosestsTemplates(10))
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)

cl = [2, 14, 16, 11]
sorter.plotTemplates(cl)
sorter.plotClusters(cl)

sorter.mergeClusters(2, [14, 16, 11])

cl = np.unique(sorter.getClosestsTemplates(5))
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)

cl = [0, 1, 8, 13]
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)
sorter.plotClusters(cl)

sorter.subdivideClusters(0, 4, use_tsne=True)

sorter.plotTemplates([0, 30, 31, 32])
sorter.plotClusters([0, 30, 31, 32])
sorter.mergeClusters(30, [31, 32])


cl = np.unique(sorter.getClosestsTemplates(5))
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)
sorter.plotClusters(cl)


sorter.mergeClusters(1, [13, 30])
sorter.mergeClusters(4, [6, 10, 15])

cl = np.unique(sorter.getClosestsTemplates(5))
sorter.viewTSNE(cl)
sorter.plotTemplates(cl)
sorter.plotClusters(cl)

sorter.plotClusters([5, 12], superposed=True)

sorter.plotClusteredEvents()

sorter.makeCenterDict(400, 700)
sorter.plotCenterDict()

sorter.clusteringPrediction()
sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])

sorter.before = 59
sorter.after = 60
sorter.secondaryPeeling(vect_len=5, threshold=20, min_dist=100, store_mid_steps=False)


sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])

sorter.secondaryPeeling(vect_len=5, threshold=15, min_dist=100, store_mid_steps=False)


sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])

sorter.secondaryPeeling(vect_len=5, threshold=15, min_dist=100, store_mid_steps=False)


sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])

sorter.secondaryPeeling(vect_len=5, threshold=10, min_dist=100, store_mid_steps=False)


sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])

sorter.secondaryPeeling(vect_len=5, threshold=10, min_dist=100, store_mid_steps=False)


sorter.plotPeeling(time=sorter.time[::2], to_peel_data=sorter.peel[-2][:,::2],
                   pred=sorter.pred[-1][:,::2])



sorter.mergeRoundsResults()
sorter.saveResults()


spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)
for key, items in spike_freqs.items():
    print(key, (~burstUtils.is_outlier(items[1])).sum())


selected_neurons = np.unique(sorter.getSimilarTemplates(2)[0][:5])

fig, ax_list = sorter.plotCompleteDetection(step=5,legend=True, lw=0.5, clust_list=selected_neurons)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=selected_neurons, ms=3, facecolor='k')

sorter.mergeFinalRoundClusters([2, 19])

selected_neurons = np.unique(sorter.getSimilarTemplates(9)[0][:5])

fig, ax_list = sorter.plotCompleteDetection(step=5,legend=True, lw=0.5, clust_list=selected_neurons)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=selected_neurons, ms=3, facecolor='k')

sorter.mergeFinalRoundClusters([24, 25])
spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)


selected_neurons = [1, 4, 8, 17, 20]
sorter.plotTemplates(selected_neurons)
fig, ax_list = sorter.plotCompleteDetection(step=5,legend=True, lw=0.5, clust_list=selected_neurons)
burstUtils.plotFreq(spike_freqs, template_dict=sorter.template_dict, scatter_plot=True,
                                  outlier_thres=3.5, sharex=ax_list[0], draw_list=selected_neurons, ms=3, facecolor='k')

sorter.saveResults()



good_colors = spsortUtils.setColors(list(sorter.final_spike_dict.keys()))

spike_freqs = burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)




DE3 = 2
burst_object = burstStorerLoader.UnitInfo(sorter.filename, mode='save', traces=sorter.traces, time=sorter.time, spike_dict=sorter.final_spike_dict, spike_freq_dict=spike_freqs,
                                          De3=DE3, template_dict=sorter.template_dict, color_dict=good_colors)


filename = "2019_07_22_0014.pklspikes"
nerve_channels = ["DP", "MA"]
burst_object = burstStorerLoader.UnitInfo(filename, mode='load')
burst_object.nerve_channels = nerve_channels
burst_object.notes = ""
burst_object.saveResults()



burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, template_dict=burst_object.template_dict,
                                  scatter_plot=True, outlier_thres=3.5, ms=2)

