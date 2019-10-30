import PyLeech.Utils.burstUtils
from PyLeech.Utils import filterUtils as filterUtils
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.burstStorerLoader as bStorerLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as spsig
from copy import deepcopy
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import warnings


def find_nearest(array, list_of_values):
    idxs = []
    for value in list_of_values:
        idxs.append(np.abs(value - array).argmin())
    return idxs


class CrawlingSegmenter():

    def __init__(self, spike_freq_dict, dp_trace=None, time=None, de3_neuron=-1, no_cycles=2, min_spike_no=10,
                 min_spike_per_sec=10, spike_max_dist=0.7):
        if dp_trace is not None:
            assert de3_neuron >= 0;
            'de3_neuron isn´t properly assigned'

        self.no_cycles = no_cycles
        self.de3_neuron = de3_neuron
        self.spike_freq_dict = spike_freq_dict
        self.intervals = None

        self.generateInterval(dp_trace, time, self.de3_neuron, min_spike_no=min_spike_no,
                              min_spike_per_sec=min_spike_per_sec, spike_max_dist=spike_max_dist, no_bursts=no_cycles)
        self.select_spikes(*list(spike_freq_dict))
        self.generateSegments()

    def generateInterval(self, dp_trace, time, de3_neuron, min_spike_no=5, min_spike_per_sec=5,
                         spike_max_dist=0.7, no_bursts=1, linewidth=2):
        burst_list = burstUtils.getBursts(np.array(self.spike_freq_dict[de3_neuron][0]), min_spike_no=min_spike_no,
                                          min_spike_per_sec=min_spike_per_sec, spike_max_dist=spike_max_dist)

        print("Check whether bursts were correctly selected")
        max_freq = self.spike_freq_dict[de3_neuron][1][~burstUtils.is_outlier(self.spike_freq_dict[de3_neuron][1])].max()
        fig, ax = burstUtils.plotFreq({'de3': np.array(self.spike_freq_dict[de3_neuron])}, outlier_thres=3.5, color='k')
        for spikes in burst_list:
            ax.plot([spikes[0], spikes[-1]], [25, 25], color='r', linewidth=linewidth)

        self.intervals = burstUtils.getInterBurstInterval(burst_list, no_bursts)

    def mergeFinalRoundClusters(self, clust_list):
        clus_obj = clust_list[0]
        clus_rm = clust_list[1:]
        for i in clus_rm:
            self.spike_freq_dict[clus_obj] = np.concatenate(
                (self.spike_freq_dict[clus_obj], self.spike_freq_dict[i]))
            del self.spike_freq_dict[i]
        self.spike_freq_dict[clus_obj] = np.sort(self.spike_freq_dict[clus_obj])

    def generateSegments(self, spike_list=None, time=None, trace=None):
        i = 0
        self.segment_list = []
        for event in self.intervals:
            event_dict = {}
            for key in self.spike_freq_dict.keys():
                if spike_list is None or key in spike_list:
                    event_dict[key] = self.spike_freq_dict[key][0][
                        (self.spike_freq_dict[key][0] > event[0]) & (self.spike_freq_dict[key][0] < event[1])]
                    event_dict[key] = self.no_cycles * (event_dict[key] - event[0]) / (event[1] - event[0])
            self.segment_list.append(event_dict)

            i += 1

    def select_spikes(self, *args):
        self.selected_spikes = []
        for arg in args:
            self.selected_spikes.append(arg)

        cmap = burstUtils.categorical_cmap(
            math.ceil(len(self.selected_spikes) / math.ceil(len(self.selected_spikes) / 10)),
            math.ceil(len(self.selected_spikes) / 10))
        self.raster_cmap = {}
        i = 0
        for sp in self.selected_spikes:
            self.raster_cmap[sp] = cmap(i)
            i += 1

    def rasterPlot(self, generate_grid=True, split_raster=1, linewidths=1):

        steps = np.arange(0, len(self.segment_list) + 1, int(len(self.segment_list) / split_raster))
        steps[-1] = len(self.segment_list)
        i = 1
        j = 0

        for i in range(len(steps) - 1):
            spike_list = []
            color_list = []
            for j in range(steps[i], steps[i + 1]):

                spike_list.append(self.segment_list[j][self.de3_neuron])
                color_list.append(self.raster_cmap[self.de3_neuron])
                for key, items in self.segment_list[j].items():
                    if key != self.de3_neuron and key in self.selected_spikes:
                        spike_list.append(items)
                        # color_list.append(burst_object.color_dict[key])
                        color_list.append(self.raster_cmap[key])

            self.fig, self.eventplot_ax = plt.subplots()
            self.eventplot_ax.eventplot(spike_list, colors=color_list, linewidths=linewidths)
            self.eventplot_ax.set_title('plot no %i' % i)
            if generate_grid:
                minor_ticks = np.arange(0, len(spike_list), 10 * len(self.selected_spikes))
                self.eventplot_ax.set_yticks(minor_ticks, minor=True)
                self.eventplot_ax.grid(which='minor', linestyle='--')

    def concatenateRasterPlot(self, generate_grid=True, split_raster=1, linewidths=1):
        if self.de3_neuron is None:
            self.de3_neuron = list(self.spike_freq_dict)[0]

        steps = np.arange(0, len(self.segment_list) + 1, int(len(self.segment_list) / split_raster))
        steps = np.append(steps, len(self.segment_list))
        i = 1
        j = 0

        spike_list = []
        color_list = []
        for i in range(len(steps) - 1):
            line_to_extend = 0
            for j in range(steps[i], steps[i + 1]):

                if i == 0:
                    spike_list.append(self.segment_list[j][self.de3_neuron])
                    color_list.append(self.raster_cmap[self.de3_neuron])
                    for key, items in self.segment_list[j].items():
                        if key != self.de3_neuron and key in self.selected_spikes:
                            spike_list.append(items)
                            # color_list.append(burst_object.color_dict[key])
                            color_list.append(self.raster_cmap[key])
                else:

                    spike_list[line_to_extend] = np.append(spike_list[line_to_extend], self.segment_list[j][
                        self.de3_neuron] + i * self.no_cycles)

                    line_to_extend += 1
                    for key, items in self.segment_list[j].items():
                        if key != self.de3_neuron and key in self.selected_spikes:
                            spike_list[line_to_extend] = np.append(spike_list[line_to_extend],
                                                                   items + i * self.no_cycles)
                            line_to_extend += 1

        self.fig, self.eventplot_ax = plt.subplots()
        self.eventplot_ax.eventplot(spike_list, colors=color_list, linewidths=linewidths)
        if generate_grid:
            minor_ticks = np.arange(-0.5, len(spike_list) - 0.5, len(self.selected_spikes))
            self.eventplot_ax.set_yticks(minor_ticks, minor=True)
            self.eventplot_ax.grid(which='minor', linestyle='--')

        vlines = np.arange(self.no_cycles, split_raster * self.no_cycles + 1, self.no_cycles)
        for ln in vlines:
            self.eventplot_ax.axvline(ln, color='k', linestyle='--', linewidth=1)


class SegmentandCorrelate(CrawlingSegmenter):
    """
    Using Inheritance only for the "concatenate raster plot method"

    This class rescales neurons times

    NS is subsampled to bin_step, but kept unrescaled to perform embeddings, embeddings could then be rescaled
    """

    def __init__(self, spike_freq_dict, intracel_signal, time_vector, intracel_fs=20000, time_intervals=None,
                 no_cycles=1, intracel_peak_height=None, intracel_peak_distance=10, intracel_prominence=1,
                 kernel_spike_sigma=1, spike_outlier_threshold=3.5, kernel_time_range=20, bin_step=0.1,
                 intracel_sigma=5, separate_by_min=False, de3_neuron=None):

        self.de3_neuron = de3_neuron
        if self.de3_neuron is None:
            warnings.warn("de3 not selected, assigning first element")
            self.de3_neuron = list(spike_freq_dict)[0]

        self.binned_and_smoothed = False
        self.spike_freq_dict = deepcopy(spike_freq_dict)
        self.separate_by_min = separate_by_min
        self.outlier_threshold = spike_outlier_threshold
        for key, item in spike_freq_dict.items():
            self.spike_freq_dict[key] = np.array(item)
        self.intracel_signal = intracel_signal
        self.time = time_vector
        if type(time_intervals[0]) is not list:
            time_intervals = [time_intervals]
        self.time_intervals = time_intervals
        self.no_cycles = no_cycles
        self.intervals = None
        self.spike_sigma = kernel_spike_sigma
        self.intra_sigma = intracel_sigma
        self.kernel_time_range = kernel_time_range
        self.bin_step = bin_step
        self.fs = intracel_fs
        self.select_spikes(*list(spike_freq_dict))

        self.generateInterval(intracel_peak_height, intracel_peak_distance, intracel_prominence,
                              no_cycles)

    def processSegments(self):
        # super().generateSegments()

        self.processNeurons()

        self.generateSegments()

        self.resampleSegments()

    def removeBadSpikes(self, outlier_threshold=5):
        self.spike_freq_dict = burstUtils.removeOutliers(self.spike_freq_dict, outlier_threshold=outlier_threshold)

    def select_spikes(self, *args):
        self.selected_spikes = []
        for arg in args:
            self.selected_spikes.append(arg)

        cmap = burstUtils.categorical_cmap(
            math.ceil(len(self.selected_spikes) / math.ceil(len(self.selected_spikes) / 10)),
            math.ceil(len(self.selected_spikes) / 10))
        self.raster_cmap = {}
        i = 0
        for sp in self.selected_spikes:
            self.raster_cmap[sp] = cmap(i)
            i += 1

    def processNeurons(self):
        self.spike_freq_dict = burstUtils.processSpikeFreqDict(self.spike_freq_dict, self.bin_step,
                                                               time_length=self.time[-1],
                                                               outlier_threshold=self.outlier_threshold)

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=self.spike_sigma, time_range=self.kernel_time_range,
                                                                 dt_step=self.bin_step)
        new_dict = {}
        for key, items in self.spike_freq_dict.items():
            new_dict[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])
        self.spike_freq_dict = new_dict
        self.binned_and_smoothed = True

    def generateInterval(self, peak_height=None, peak_distance=10, prominence=5, no_cycles=None):

        if no_cycles is None:
            no_cycles = self.no_cycles

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(self.intra_sigma, self.kernel_time_range, 1 / self.fs)
        self.filtered_intracel = spsig.fftconvolve(self.intracel_signal, kernel, mode='same')
        self.filtered_intracel = self.filtered_intracel[::int(self.bin_step * self.fs)]

        peak_distance = int(peak_distance / self.bin_step)

        intracel_extreme_idxs = []
        for interval in self.time_intervals:

            loc = np.where((self.time[::int(self.bin_step * self.fs)] > interval[0]) & (
                        self.time[::int(self.bin_step * self.fs)] < interval[1]))[0]
            loc0 = np.where(self.time[::int(self.bin_step * self.fs)] > interval[0])[0][0]

            if peak_height is None:
                if self.separate_by_min:
                    peak_height = np.mean(-self.filtered_intracel[loc])
                else:
                    peak_height = np.mean(self.filtered_intracel[loc])

                print(peak_distance, peak_height, prominence)

            if self.separate_by_min:
                intracel_interval_idxs = \
                spsig.find_peaks(-self.filtered_intracel[loc], height=peak_height, distance=peak_distance,
                                 prominence=prominence)[0] + loc0
            else:
                intracel_interval_idxs = \
                spsig.find_peaks(self.filtered_intracel[loc], height=peak_height, distance=peak_distance,
                                 prominence=prominence)[
                    0] + loc0

            intracel_extreme_idxs.append(
                intracel_interval_idxs[(self.time[::int(self.bin_step * self.fs)][intracel_interval_idxs] > interval[0])
                                       &
                                       (self.time[::int(self.bin_step * self.fs)][intracel_interval_idxs] < interval[
                                           1])])

        assert len(np.concatenate(
            intracel_extreme_idxs)) > 0, 'No peak was found for the intracelular signal and the given time interval'

        self.intervals = []

        for idx_list in intracel_extreme_idxs:
            for i in range(len(idx_list) - no_cycles):
                self.intervals.append([self.time[::int(self.bin_step * self.fs)][idx_list[i]],
                                       self.time[::int(self.bin_step * self.fs)][idx_list[i + no_cycles]]])
        self.intervals = np.array(self.intervals)

        intracel_extreme_idxs = [idx for subl in intracel_extreme_idxs for idx in subl]
        plt.figure()

        # plt.plot(self.time[::100], self.intracel_signal[::100])
        first = self.time_intervals[0][0]
        last = self.time_intervals[-1][1]
        idxs = np.where(
            (self.time[::int(self.bin_step * self.fs)] > first) & (self.time[::int(self.bin_step * self.fs)] < last))[0]
        plt.plot(self.time[::int(self.bin_step * self.fs)][idxs], self.filtered_intracel[idxs], color='k')

        plt.scatter(self.time[::int(self.bin_step * self.fs)][intracel_extreme_idxs],
                    self.filtered_intracel[intracel_extreme_idxs], color='r', zorder=50)

    def generateSegments(self):
        if self.binned_and_smoothed:
            print('Segments have already been binned and smoothed')
        else:
            print('Segments haven´t been binned and smoothed yet')

        self.segmented_neuron_frequency_list = []

        intracel_time = self.time[::int(self.bin_step * self.fs)]

        self.NS_segments = []

        for event in self.intervals:
            ns_idxs = np.where((intracel_time > event[0]) & (intracel_time < event[1]))
            self.NS_segments.append(self.filtered_intracel[ns_idxs])
            event_freq_dict = {}

            for key in self.spike_freq_dict.keys():

                idxs = np.where((self.spike_freq_dict[key][0] > event[0]) & (self.spike_freq_dict[key][0] < event[1]))[0]

                # times = self.spike_freq_dict[key][0][idxs]
                event_freqs = self.spike_freq_dict[key][1][idxs]

                # rescaled_times = self.no_cycles * (times - event[0]) / (event[1] - event[0])

                try:
                    event_freq_dict[key] = np.array(event_freqs)
                except Exception as e:
                    print(key, len(event_freqs))
                    raise e

            self.segmented_neuron_frequency_list.append(event_freq_dict)


        self.segment_list = self.segmented_neuron_frequency_list

    def resampleSegments(self):
        self.resampled_segmented_neuron_frequency_list = []
        max_len = 0
        first_key = list(self.spike_freq_dict)[0]
        for segment in self.segmented_neuron_frequency_list:
            # print(len(segment[first_key]))
            if len(segment[first_key]) > max_len:
                max_len = len(segment[first_key])

        for segment in self.segmented_neuron_frequency_list:
            new_dict = {}
            for key, item in segment.items():
                new_dict[key] = spsig.resample(item, max_len)

            self.resampled_segmented_neuron_frequency_list.append(new_dict)

    def getSegmentsEmbeddings(self, trace='NS'):
        "trace is either NS o the neuron's number"
        embedding_list = []

        if trace == 'NS':
            intracel_time = self.time[::int(self.bin_step * self.fs)]
            emb_NS = NLD.getDerivativeEmbedding(self.filtered_intracel, self.bin_step, 3)

            for event in self.intervals:
                idxs = np.where((intracel_time > event[0]) & (intracel_time < event[1]))
                embedding_list.append(emb_NS[idxs])


        else:
            intracel_time = self.spike_freq_dict[trace][0]
            emb_sp = NLD.getDerivativeEmbedding(self.spike_freq_dict[trace][1], self.bin_step, 3)

            for event in self.intervals:
                idxs = np.where((intracel_time > event[0]) & (intracel_time < event[1]))
                embedding_list.append(emb_sp[idxs])

        return embedding_list

    def plotCrawlingSegment(self):
        times = [int for intervals in self.intervals for int in intervals]
        plt.figure()
        for interval in self.time_intervals:
            idxs1 = np.where((self.time[::100] > interval[0])
                             &
                             (self.time[::100] < interval[1])
                             )
            idxs2 = np.where((self.time[::int(self.bin_step * self.fs)] > interval[0])
                             &
                             (self.time[::int(self.bin_step * self.fs)] < interval[1])
                             )
            plt.plot(self.time[::100][idxs1], self.intracel_signal[::100][idxs1], color='dodgerblue')
            plt.plot(self.time[::int(self.bin_step * self.fs)][idxs2], self.filtered_intracel[idxs2], color='k', lw=2)

        idxs = find_nearest(self.time[::int(self.bin_step * self.fs)], times)
        print(idxs)
        plt.scatter(self.time[::int(self.bin_step * self.fs)][idxs],
                    self.filtered_intracel[idxs], color='r', zorder=50)

    def concatenateRasterPlot(self, generate_grid=True, split_raster=1, linewidths=1):
        self.segment_list = []
        for segment in self.time_isi_pair_segment_list:
            new_dict = {}
            for key, items in segment.items():
                new_dict[key] = items[:, 0]
            self.segment_list.append(deepcopy(new_dict))

        self.de3_neuron = None
        # print(self.segment_list[0])
        super().concatenateRasterPlot(generate_grid, split_raster, linewidths)

        del self.segment_list


class NSSegmenter(CrawlingSegmenter):
    """
    Clase vieja, no debería usarse
    """

    def __init__(self, spike_freq_dict, intracel_signal, time_vector, cutoff_freq=1, peak_height=-5.5,
                 peak_distance=10, prominence=5, freq_threshold=150,
                 time_intervals=None, no_cycles=1, NS_is_filtered=False, segment_by_min=False, de3_neuron=None):

        self.de3_neuron = de3_neuron
        if self.de3_neuron is None:
            warnings.warn("Warning: de3 not selected")
            self.de3_neuron = list(spike_freq_dict)[0]

        self.spike_freq_dict = deepcopy(spike_freq_dict)
        for key, item in spike_freq_dict.items():
            self.spike_freq_dict[key] = np.array(item)

        self.segment_by_min = segment_by_min
        self.NS_is_filtered = NS_is_filtered
        self.intracel_signal = intracel_signal
        self.time = time_vector
        self.time_intervals = time_intervals
        self.no_cycles = no_cycles
        self.intervals = None
        self.generateInterval(cutoff_freq, peak_height, peak_distance, prominence, no_cycles)
        self.select_spikes(*list(spike_freq_dict))
        self.removeBadSpikes(freq_threshold=freq_threshold)
        self.generateSegments()

    def concatenateRasterPlot(self, generate_grid=True, split_raster=1, linewidths=1):
        self.segment_list = []
        for segment in self.time_isi_pair_segment_list:
            new_dict = {}
            for key, items in segment.items():
                new_dict[key] = items[:, 0]
            self.segment_list.append(deepcopy(new_dict))

        self.de3_neuron = None
        # print(self.segment_list[0])
        super().concatenateRasterPlot(generate_grid, split_raster, linewidths)

        del self.segment_list

    def removeBadSpikes(self, freq_threshold=150):
        for key, items in self.spike_freq_dict.items():
            self.spike_freq_dict[key] = items[:, items[1] < freq_threshold]

    def generateInterval(self, cutoff_freq, peak_height, peak_distance, prominence, no_cycles=None):
        if no_cycles is None:
            no_cycles = self.no_cycles
        sampling_rate = round(1 / (self.time[1] - self.time[0]))
        peak_distance = int(peak_distance * sampling_rate)
        if not self.NS_is_filtered:
            NS_filt = filterUtils.runButterFilter(self.intracel_signal, cuttoff_freq=cutoff_freq, butt_order=4,
                                                  sampling_rate=sampling_rate)
        else:
            NS_filt = self.intracel_signal
        intracel_max_idxs = \
            spsig.find_peaks(NS_filt, height=peak_height, distance=peak_distance, prominence=prominence)[0]

        self.intracel_idxs = []
        for interval in self.time_intervals:
            self.intracel_idxs.append(intracel_max_idxs[(self.time[intracel_max_idxs] > interval[0]) & (
                    self.time[intracel_max_idxs] < interval[1])])

        assert len(self.intracel_idxs) > 0, 'No peak was found for the intracelular signal and the given time interval'

        self.intervals = []
        for idx_list in self.intracel_idxs:
            for i in range(len(idx_list) - no_cycles):
                self.intervals.append([self.time[idx_list[i]], self.time[idx_list[i + no_cycles]]])
        self.intervals = np.array(self.intervals)

    def generateSegments(self, spike_list=None):
        i = 0
        self.time_isi_pair_segment_list = []
        for event in self.intervals:
            event_times = {}
            for key in self.spike_freq_dict.keys():
                if spike_list is None or key in spike_list:
                    times = self.spike_freq_dict[key][0][
                        (self.spike_freq_dict[key][0] > event[0]) & (self.spike_freq_dict[key][0] < event[1])]
                    ISI = np.diff(times)
                    rescaled_times = self.no_cycles * (times[1:] - event[0]) / (event[1] - event[0])
                    try:
                        event_times[key] = np.array((rescaled_times, ISI)).transpose()
                    except Exception as e:
                        print(key, len(rescaled_times), len(ISI))
                        raise e

            self.time_isi_pair_segment_list.append(event_times)

            i += 1

    def CalculateRate(self, step=0.01, spike_list=None, to_avg_no=2):
        averaged_segment_list = []
        bins = np.arange(0, self.no_cycles, step)
        i = 0
        binned_data_dict = {}
        if spike_list is None:
            spike_list = list(self.spike_freq_dict)

        for key in spike_list: binned_data_dict[key] = [[] for i in range(len(bins) - 1)]

        while i <= (len(self.time_isi_pair_segment_list) - to_avg_no):

            temp_dict = deepcopy(binned_data_dict)
            for k in range(to_avg_no):
                for key, items in self.time_isi_pair_segment_list[i + k].items():

                    for j in range(len(bins) - 1):
                        temp_dict[key][j].extend(items[:, 1][(items[:, 0] > bins[j]) & (items[:, 0] < bins[j + 1])])

            averaged_dict = {}
            for key, items in temp_dict.items():

                Averaged_ISIs = np.zeros(len(bins) - 1)
                for m in range(len(items)):
                    if len(items[m]) == 0:
                        pass
                    else:
                        Averaged_ISIs[m] = np.mean(items[m])
                averaged_dict[key] = np.reciprocal(Averaged_ISIs)
                averaged_dict[key][averaged_dict[key] == np.inf] = 0
            averaged_segment_list.append(averaged_dict)
            i += to_avg_no

        self.averaged_segment_list = averaged_segment_list


""" 
I'm only checking  SegmentandCorrelate for proper work with the NLD tests i need to run
"""
if __name__ == "__main__":
    cdd = CDU.loadDataDict()

    fn = list(cdd)[5]

    burst_obj = bStorerLoader.BurstStorerLoader(fn, 'RegistrosDP_PP', mode='load')
    arr_dict, time, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])
    NS = arr_dict['Vm1']
    del arr_dict


    good_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if neuron_dict['neuron_is_good']]

    spike_freq_dict = {key: burst_obj.spike_freq_dict[key] for key in good_neurons}
    crawling_intervals = cdd[fn]['crawling_intervals']

    segments = SegmentandCorrelate(spike_freq_dict, NS, time, intracel_fs=fs, time_intervals=crawling_intervals,
                                   separate_by_min=False)
    segments.processSegments()
    segments.generateSegments()

    #
    # segments = SegmentandCorrelate(spike_freq_dict, NS, time, intracel_fs=fs, time_intervals=crawling_intervals,
    #                                separate_by_min=True)
    # segments.processSegments()
    # segments.generateSegments()
    # segments.rasterPlot()