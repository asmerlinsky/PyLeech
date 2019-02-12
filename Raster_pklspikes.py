import PyLeech.Utils.abfUtils as abfUtils
import PyLeech.Utils.burstClasses as burstClasses
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.filterUtils as filterUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload
import numpy as np
import scipy.signal as spsig

"""
%load_ext autoreload
%autoreload 2

"""
pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)): print(j, pkl_files[j])
filename = pkl_files[7]
print(filename)

burst_object = burstClasses.BurstStorerLoader(filename, 'load')
basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
arr_dict, time, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
NS = arr_dict['Vm1']
del arr_dict

crawling_interval = [340, 625]

NS -= np.median(NS[np.where(time > crawling_interval[0])[0][0]:np.where(time > crawling_interval[1])[0][0]])
plt.figure()
plt.plot(time[::5], NS[::5])

NS_lp = filterUtils.runButterFilter(NS, 1, butt_order=4, sampling_rate=fs)

plt.figure()
plt.plot(time[::5], NS[::5])
plt.plot(time[::5], NS_lp[::5])

plt.figure()

idxs = spsig.find_peaks(NS_lp, 5, distance=fs * 8, prominence=1)[0]
plt.figure()
plt.plot(time, NS)
plt.plot(time[::5], NS_lp[::5])
plt.scatter(time[idxs], NS[idxs], color='r', zorder=10)

times = time[idxs][(time[idxs] > crawling_interval[0]) & (time[idxs] < crawling_interval[1])]
times_idxs = idxs[(time[idxs] > crawling_interval[0]) & (time[idxs] < crawling_interval[0])]
time_mask = np.ones(len(times_idxs), dtype=bool)
time_mask[2] = False

plt.figure()
plt.hist(np.diff(times), bins=10)

plt.figure()
plt.plot(time, NS)
plt.scatter(time[times_idxs[time_mask]], NS[times_idxs[time_mask]], color='r')
times_idxs = times_idxs[time_mask]

burst_object.spike_dict.keys()

#
# for key in burst_object.spike_freq_dict.keys():
#    spikes = burst_object.spike_freq_dict[key][0]
#    crawling_spikes = spikes[(spikes>350) & (spikes<600)]
#    diff = np.diff(crawling_spikes)
#
#    diff = diff[(diff>0.0067)]
##    diff = burst_object.spike_freq_dict[key][1]
#    plt.figure()
#    plt.hist(diff, bins=400)
#    plt.title(str(key))

# cl = [6,7,9]
# gc = spsortUtils.setGoodColors(cl)
# BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
#                                  burst_object.template_dict, gc, legend=True, interval=[0,0.5], step=2, clust_list=cl)

fig, axes = burstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
                                                     burst_object.template_dict, burst_object.color_dict, legend=True,
                                                     step=2)

# burst_object.isDe3 = 0

reload(burstClasses)

crawling_intervals = [crawling_interval, [664, 821]]
Ns_segmented = burstClasses.NSSegmenter(burst_object.spike_freq_dict, NS, time, cutoff_freq=1, peak_height=-5.5,
                                        peak_distance=10, prominence=5,
                                        time_intervals=crawling_intervals, no_cycles=1)

Ns_segmented.concatenateRasterPlot(split_raster=2, linewidths=1.5)
burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]],
                            template_dict=burst_object.template_dict, outlier_thres=3.5, ms=2, sharex=None)  # ,
# draw_list=[5, 12, 14, 21])

times = np.unique(np.concatenate(Ns_segmented.intervals))
for i in range(len(times)):
    plt.axvline(times[i])

Ns_segmented.CalculateRate(step=0.02, to_avg_no=1)

# fig, ax = plt.subplots(len(Ns_segmented.averaged_segment_list[i]), 1, sharex=True)
fig, ax = plt.subplots(len(Ns_segmented.averaged_segment_list), 1, sharex=True)
for i in range(len(Ns_segmented.averaged_segment_list)):
    for key, items in Ns_segmented.averaged_segment_list[i].items():
        # items[items > 0] = items[items>0] - np.min(items[items>0])
        # items[items > 0] /= np.max(items[items > 0])
        if key in [5, 12, 14, 21]:
            ax[i].plot(np.arange(len(items)), items, label=key, color=burst_object.color_dict[key])

            ax[i].grid()
            # ax[j].legend()

        # plt.scatter(np.arange(len(items)),items, label=key)

# df = BurstUtils.segment_listToDataFrame(Ns_segmented.time_isi_pair_segment_list)


"""
Divisor

"""

crawling_segmenter = burstClasses.CrawlingSegmenter(spike_freq_dict=burst_object.spike_freq_dict,
                                                    dp_trace=burst_object.traces[0], time=burst_object.time,
                                                    de3_neuron=burst_object.isDe3, no_cycles=2, spike_max_dist=0.6)

crawling_segmenter.concatenateRasterPlot(generate_grid=True, split_raster=160, linewidths=1)
crawling_segmenter.eventplot_ax.set_facecolor('indigo')
intervals = crawling_segmenter.intervals[:, 1] - crawling_segmenter.intervals[:, 0]
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
pkl_files = pkl_files[6:8]
for filename in pkl_files:
    print(filename)

    basename = PyLeech.Utils.abfUtils.getAbfFilenamesfrompklFilename(filename)
    arr_dict, time, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
    NS = arr_dict['Vm1']
    burst_object = burstClasses.BurstStorerLoader(filename, 'load')
    # PyLeech.burstUtils1.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
    #                                           burst_object.template_dict, burst_object.color_dict, legend=True)
    plt.suptitle(filename)
    # ax = BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict, burst_object.template_dict, burst_object.color_dict, legend=True)

    burstClasses.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]], template_dict=burst_object.template_dict, outlier_thres=10, sharex=None)
    correlation_segments = burstClasses.SegmentandCorrelate(burst_object.spike_freq_dict, NS, time,
                                                            time_intervals=burst_object.crawling_segments,
                                                            intracel_cutoff_freq=2,
                                                            no_cycles=1, intracel_peak_height=-52)
    correlation_segments.
    # plt.figure()

    # burst_object.isDe3 = int(input("Tell me De3 channel"))

    # burst_object.saveResults()

##########################################################


pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)): print(j, pkl_files[j])
filename = pkl_files[6]
print(filename)

burst_object = burstClasses.BurstStorerLoader(filename, 'load')
basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
arr_dict, time, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
NS = arr_dict['Vm1']
del arr_dict

burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]],
                            template_dict=burst_object.template_dict, outlier_thres=3.5, ms=2, sharex=None)


crawling_intervals = [[300, 700], [748, 801]]

### Generating an envelope
import scipy.signal as spsig

reload(burstClasses)
# crawling_intervals = [[340, 625], [664, 821]]
crawling_intervals = [[346, 622], [662, 82]]
correlation_segments = burstClasses.SegmentandCorrelate(burst_object.spike_freq_dict, NS, time, crawling_intervals,
                                                        no_cycles=1, intracel_peak_height=-50)

full_spike_corr = []
i=0
for key, items in correlation_segments.spike_freq_dict.items():
    print(i, key)
    full_spike_corr.append(items[1,:])
    i+=1

plt.matshow(np.corrcoef(full_spike_corr))
plt.colorbar()
plt.title('full spike signal correlation')

plt.figure()
plt.plot(full_spike_corr[0])
plt.plot(full_spike_corr[9])
plt.plot(full_spike_corr[6])

plt.figure()
plt.axvline(len(full_spike_corr[0]), color='k')
plt.plot(spsig.correlate(full_spike_corr[0], full_spike_corr[9]), label='0-9')
plt.plot(spsig.correlate(full_spike_corr[0], full_spike_corr[6]), label='0-6')
plt.legend()


neurons = {}
for key in correlation_segments.resampled_segmented_neuron_frequency_list[0].keys():
    neurons[key] = []




for element in correlation_segments.resampled_segmented_neuron_frequency_list:
    
    for key, item in element.items():
        neurons[key].append(item)

corr_sum = None
for key in [5, 10, 11, 12, 14, 18, 19, 21]:
    corr = np.corrcoef(neurons[key])
    if corr_sum is None:
        corr_sum = corr
    else:
        corr_sum += corr
    plt.matshow(corr)
    plt.colorbar()
    plt.title('neuron' + str(key))
    
plt.matshow(corr_sum)
plt.colorbar()
plt.title('added correlation')
    

concatenated_list = []
for element in correlation_segments.resampled_segmented_neuron_frequency_list:
    element_list = np.array(())
    for key, item in element.items():
            element_list = np.append(element_list, item)
    concatenated_list.append(element_list)
    
full_corr = np.corrcoef(concatenated_list)


plt.matshow(full_corr)
plt.colorbar()
plt.title('full correlation')

plt.matshow(corr_sum)
plt.colorbar()
plt.title('added correlation')



burstUtils.plotFreq(correlation_segments.spike_freq_dict, burst_object.color_dict, optional_trace=[time[::5], NS[::5]],
                            template_dict=burst_object.template_dict, outlier_thres=None, ms=2, scatter_plot=False, sharex=None)  # ,
# draw_list=[5, 12, 14, 21])

times = np.unique(np.concatenate(correlation_segments.intervals))
for i in range(len(times)):
    plt.axvline(times[i])
