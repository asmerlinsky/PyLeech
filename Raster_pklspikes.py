import PyLeech.BurstUtils as BurstUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload

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

BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
                                 burst_object.template_dict, burst_object.color_dict, legend=True)

# burst_object.isDe3 = 9

BurstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, None, burst_object.template_dict,
                    outlier_thres=50, ms=2)



reload(BurstUtils)
crawling_segmenter = BurstUtils.CrawlingSegmenter(dp_trace=burst_object.traces[0], time=burst_object.time,
                                                  spike_freq_dict=burst_object.spike_freq_dict,
                                                  de3_neuron=burst_object.isDe3, no_bursts=1, spike_max_dist=0.6)

crawling_segmenter.concatenateRasterPlot(generate_grid=True, split_raster=80)

intervals = crawling_segmenter.intervals[:,1]-crawling_segmenter.intervals[:,0]
plt.figure()
plt.hist(intervals, bins=1500, )
plt.xlim([-0.0001, 7.5])



#
# plt.figure()
# i = 0
# for key, items in colors.items():
#     i += 1
#     plt.plot([0, 1], [i, i], label=key, linewidth=20)
# plt.legend()
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
