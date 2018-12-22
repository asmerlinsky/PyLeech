import PyLeech.burstUtils as BurstUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload
import PyLeech.spsortUtils as spsortUtils
"""
%load_ext autoreload
%autoreload 2

"""
pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)): print(j, pkl_files[j])
filename = pkl_files[5]
print(filename)
burst_object = BurstUtils.BurstStorerLoader(filename, 'load')

burst_object.spike_dict.keys()

# cl = [6,7,9]
# gc = spsortUtils.setGoodColors(cl)
# BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
#                                  burst_object.template_dict, gc, legend=True, interval=[0,0.5], step=2, clust_list=cl)

fig, axes = BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
                                 burst_object.template_dict, burst_object.color_dict, legend=True, step=4)

# burst_object.isDe3 = 0

BurstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, [burst_object.time[::5], NS[::5]], burst_object.template_dict,
                    outlier_thres=5, ms=2, sharex=axes[0])



reload(BurstUtils)
crawling_segmenter = BurstUtils.CrawlingSegmenter(dp_trace=burst_object.traces[0], time=burst_object.time,
                                                  spike_freq_dict=burst_object.spike_freq_dict,
                                                  de3_neuron=burst_object.isDe3, no_bursts=2, spike_max_dist=0.6)

crawling_segmenter.concatenateRasterPlot(generate_grid=True, split_raster=160, linewidths=1)
crawling_segmenter.eventplot_ax.set_facecolor('indigo')
intervals = crawling_segmenter.intervals[:,1]-crawling_segmenter.intervals[:,0]
plt.figure()
plt.hist(intervals, bins=1500, )
plt.xlim([-0.0001, 7.5])




fig, ax = plt.subplots()
i = 0
for key, items in crawling_segmenter.raster_cmap.items():
    i += 1
    ax.plot([0, 1], [i, i], color=items, label=key, linewidth=20)
fig.legend()
ax.set_facecolor('lightgray')
#################################
import glob

pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')

for filename in pkl_files:
    print(filename)
    burst_object = BurstUtils.BurstStorerLoader(filename, 'load')
    BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
                                     burst_object.template_dict, burst_object.color_dict, legend=True)
    plt.suptitle(filename)
    # ax = BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict, burst_object.template_dict, burst_object.color_dict, legend=True)

    # BurstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, burst_object.template_dict, outlier_thres=10, sharex=ax)

    # plt.figure()

    # burst_object.isDe3 = int(input("Tell me De3 channel"))

    # burst_object.saveResults()

##########################################################
t0 =
expected_loc = []

for el in expected_loc:
    plt.axvline(el, color='b')
