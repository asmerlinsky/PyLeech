# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:40:50 2018

@author: Agustin
"""
import PyLeech.Utils.SpSorter
import PyLeech.Utils.burstUtils

"""
Primera parte

%load_ext autoreload
%autoreload 2
%aimport PyLeech.spsortUtils
%aimport PyLeech.filterUtils
"""

import PyLeech.Utils.AbfExtension as abfe
import numpy as np
import matplotlib.pylab as plt
import PyLeech.Utils.constants as constants
import PyLeech.Utils.filterUtils

nan = constants.nan
opp0 = constants.opp0





np.set_printoptions(precision=3)
plt.ion()


block = abfe.ExtendedAxonRawIo("RegistrosDP_PP/18n05010.abf")
ch_info = block.ch_info

[print(ch, item) for ch, item in ch_info.items()]

n1 = np.array(block.getSignal_single_segment('IN5'))
n2 = np.array(block.getSignal_single_segment('IN6'))
fs = block.get_signal_sampling_rate()
data_len = int(len(n1))
time = abfe.generateTimeVector(data_len, fs)


del block

# plt.figure()
# PyLeech.filterUtils.plotSpectrums(n1, n2, sampling_rate=fs, nperseg=10000)

n1_filt = PyLeech.filterUtils.runButterFilter(n1, 2000, sampling_rate=fs)
n2_filt = PyLeech.filterUtils.runButterFilter(n2, 2000, sampling_rate=fs)

n1_filt = PyLeech.filterUtils.runButterFilter(n1_filt, 10, sampling_rate=fs, butt_order=4, btype='high')
n2_filt = PyLeech.filterUtils.runButterFilter(n2_filt, 10, sampling_rate=fs, butt_order=4, btype='high')

#PyLeech.filterUtils.plotSpectrums(n2, n2_filt, sampling_rate=fs, nperseg=10000)

train_data = np.array(
    [n1_filt,
     n2_filt,
    ]
)
del n1, n2, n1_filt, n2_filt
train_time = time


sorter = PyLeech.Utils.SpSorter.SpSorter(train_data, None, train_time, fs)
del train_data, train_time

# sorter.plotDataList()

sorter.normTraces()

# sorter.plotTraceAndStd(channel=0)

# sorter.smoothAndVisualizeThreshold(channel=1, vect_len=7, threshold=6, interval=[0, 0.1])

sorter.smoothAndFindPeaks(vect_len=9, threshold=6, min_dist=100)

# sorter.plotDataListAndDetection()

sorter.makeEvents(149, 400)
# sorter.plotEventsMedians()

# sorter.plotEvents(evts_no=4000)

sorter.makeNoiseEvents(size=4000)

# sorter.plotEvents(plot_noise=True, evts_no=4000)

"""
Visualizing the threshold
"""

# sorter.visualizeCleanEvents(thresholds=[30])

sorter.getGoodEvents(threshold=70)


"""
good events in trace
"""
# sorter.plotDataListAndDetection(good_evts=True)


"""
From here I can either use event max for classification or run pca for classification
"""

sorter.getPcaBase(plot_vectors=False)

sorter.getVectorWeights(40)
# sorter.viewPcaClusters(go_pandas=True, clust_dim=8)

sorter.KMeansClusterEvents(20, use_pca=True, dim_size=30)

sorter.plotClusters(good_ones=True)
# sorter.meergeClusters(4, [3])

sorter.hideClusters(bad_clust=[1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 16])

# sorter.plotClusteredEvents(legend=True, clust_list=[0, 15])

sorter.generateClustersTemplate()
sorter.hideBadEvents(0.15)
sorter.generateClustersTemplate()

sorter.plotTemplates()
# sorter.viewPcaClusters(go_pandas=True, clust_dim=8,good_clusters=True)
# sorter.viewPcaClusters(ggobi=True, ggobi_clusters=True, clust_list=[])



sorter.assignChannelsToClusters()

sorter.plotClusteredEvents(skip_bad=True, legend=True, lw=0.001)

sorter.mergeClusters(6, [10])

# sorter.mergeClusters(3, [4])

sorter.makeCenterDict(149, 500)

sorter.generatePrediction(store_prediction=True)

# sorter.plotCompleteDetection(round=0, legend=True, lw=0.01)

# sorter.plotPeeling(to_peel_data=sorter.peel[0], pred=sorter.pred[0])

sorter.secondaryPeeling(vect_len=3, threshold=5, min_dist=50)
#sorter.secondaryPeeling(channels=0, vect_len=3, threshold=5, min_dist=25)

sorter.plotPeeling(to_peel_data=sorter.peel[-2], pred=sorter.pred[-1])


sorter.mergeSpikeResults()

sorter.plotCompleteDetection(round='All', legend=True, lw=0.1)

# sorter.secondaryPeeling(channels=0, vect_len=3, threshold=5, min_dist=25)



sorter.plotCompleteDetection(clust_list=[19], lw=0.1)
good_colors = PyLeech.Utils.burstUtils.setGoodColors(list(sorter.final_spike_dict.keys()))
spike_freqs = PyLeech.Utils.burstUtils.getInstFreq(sorter.time, sorter.final_spike_dict, sorter.sample_freq)

PyLeech.Utils.burstUtils.plotFreq(spike_freqs, good_colors, sorter.template_dict, thres=[1, 300], single_figure=True)

freq_bins = PyLeech.Utils.burstUtils.binSpikesFreqs(spike_freqs, sorter.time[-1], 0.1)

PyLeech.Utils.burstUtils.plotFreq(freq_bins, good_colors, sorter.template_dict, thres=[1, 100], single_figure=False)

sorter.plotCompleteDetection(round='All', legend=True, lw=0.1)



from scipy.stats.mstats import mquantiles
plt.figure()
dataQ = map(lambda x:
            mquantiles(diff, prob=np.arange(0.01,0.99,0.001)),data)
dataQsd = map(lambda diff:
              mquantiles(diff/np.std(diff), prob=np.arange(0.01,0.99,0.001)),
              data)
from scipy.stats import norm
qq = norm.ppf(np.arange(0.01,0.99,0.001))
plt.plot(np.linspace(-3,3,num=100),np.linspace(-3,3,num=100),
         color='grey')
colors = ['black', 'orange', 'blue', 'red']
for i,y in enumerate(dataQ):
    plt.plot(qq,y,color=colors[i])

for i,y in enumerate(dataQsd):
    plt.plot(qq,y,color=colors[i],linestyle="dashed")

plt.xlabel('Normal quantiles')
plt.ylabel('Empirical quantiles')