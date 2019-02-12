import os.path
from PyLeech import filterUtils as filterUtils
from PyLeech.Utils.burstUtils import *




class BurstStorerLoader():
    required_to_save_dict_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict']
    expected_load_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict', 'isDe3', 'crawling_segments']

    def __init__(self, filename, foldername, mode='save', traces=None, time=None, spike_dict=None, spike_freq_dict=None, De3=None,
                 template_dict=None, color_dict=None, crawling_segments=None):
        self.filename = filename
        self.traces = traces
        self.time = time
        self.spike_dict = spike_dict
        self.spike_freq_dict = spike_freq_dict
        self.isDe3 = De3
        self.template_dict = template_dict
        self.color_dict = color_dict
        self.crawling_segments = crawling_segments
        self.folder = foldername
        if str.lower(mode) == 'save':
            self.saveResults()
        elif str.lower(mode) == 'load':
            self.loadResults()
            if self.isDe3 == None:
                print('I need to set attribute \'isDe3\' before continuing')

    def saveResults(self, filename=None, folder=None):
        pkl_dict = self.generatePklDict()
        if filename is None:
            filename = self.filename
        if folder is None:
            try:
                folder = self.folder
            except:
                pass

        generatePklFilename(self.filename, folder)
        filename = os.path.splitext(filename)[0] + '.pklspikes'
        saveSpikeResults(filename, pkl_dict)

    def generatePklDict(self):
        pkl_dict = {}
        for key in BurstStorerLoader.required_to_save_dict_keys:
            assert getattr(self, key) is not None, ("I need %s" % key)
            pkl_dict.update({key: getattr(self, key)})
        if self.isDe3 is not None:
            pkl_dict.update({'isDe3': self.isDe3})
            print("Saving De3 channel")
        return pkl_dict

    def loadResults(self):
        if type(self.filename) is list:
            self.filename = generateFilenameFromList(self.filename)
        spike_dict = loadSpikeResults(self.filename)
        cp = deepcopy(BurstStorerLoader.required_to_save_dict_keys)
        for key, items in spike_dict.items():
            assert key in BurstStorerLoader.expected_load_keys, ('Unexpected key %s' % key)
            setattr(self, key, items)
        # assertMissingData(cp)


    def plotTemplates(self, signal_inversion=[1,1], clust_list=None, fig_ax=None):

        assert self.traces.shape[0] == 2, 'This method is implemented for only 2 channels'

        if fig_ax is None:
            fig, ax = plt.subplots(1,1)
            fig_ax = [fig, ax]

        if clust_list is None:
            clust_list = list(self.template_dict.keys())
        colors = self.color_dict

        ch1_max = 0
        ch2_max = 0
        for key in list(self.spike_freq_dict):
            tp_median = self.template_dict[key]['median']
            half = int(len(tp_median)/2)
            if np.abs(tp_median[:half]).max()>ch1_max:
                ch1_max = np.abs(tp_median[:half]).max()

            if np.abs(tp_median[half:]).max()>ch2_max:
                ch2_max = np.abs(tp_median[half:]).max()


        for key in list(self.spike_freq_dict):
            if key in clust_list:
                tp_median = self.template_dict[key]['median']
                tp_median[:half] = signal_inversion[0] * tp_median[:half]
                tp_median[half:] = signal_inversion[1] * tp_median[half:]

                # if len(spsig.find_peaks(tp_median, height=1, distance=30)[0])>1:

                tp_median[:half] = tp_median[:half]/ch1_max
                tp_median[half:] = tp_median[half:] / ch2_max


                fig_ax[1].plot(tp_median, color=colors[key], label=key, lw=2)

        fig_ax[0].legend()

        return fig_ax


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
                         spike_max_dist=0.7, no_bursts=1):
        burst_list = getBursts(np.array(self.spike_freq_dict[de3_neuron][0]), min_spike_no=min_spike_no,
                               min_spike_per_sec=min_spike_per_sec, spike_max_dist=spike_max_dist)
        print("Check whether bursts were correctly selected")
        plt.figure()
        plt.plot(time, dp_trace, linewidth=1, color='k')
        t0 = []

        self.intervals = getInterBurstInterval(burst_list, no_bursts)

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

        cmap = categorical_cmap(math.ceil(len(self.selected_spikes) / math.ceil(len(self.selected_spikes) / 10)),
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


class NSSegmenter(CrawlingSegmenter):

    def __init__(self, spike_freq_dict, intracel_signal, time_vector, cutoff_freq=1, peak_height=-5.5,
                 peak_distance=10, prominence=5, freq_threshold=150,
                 time_intervals=None, no_cycles=1):

        self.spike_freq_dict = deepcopy(spike_freq_dict)
        for key, item in spike_freq_dict.items():
            self.spike_freq_dict[key] = np.array(item)
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

        NS_filt = filterUtils.runButterFilter(self.intracel_signal, cuttoff_freq=cutoff_freq, butt_order=4,
                                              sampling_rate=sampling_rate)

        intracel_max_idxs = \
            spsig.find_peaks(NS_filt, height=peak_height, distance=peak_distance, prominence=prominence)[0]

        intracel_idxs = []
        for interval in self.time_intervals:
            intracel_idxs.append(intracel_max_idxs[(self.time[intracel_max_idxs] > interval[0]) & (
                    self.time[intracel_max_idxs] < interval[1])])

        assert len(intracel_idxs) > 0, 'No peak was found for the intracelular signal and the given time interval'

        self.intervals = []
        for idx_list in intracel_idxs:
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


class SegmentandCorrelate(NSSegmenter):
    ## Using Inheritance only for the "concatenate raster plot method"

    def __init__(self, spike_freq_dict, intracel_signal, time_vector, time_intervals=None, no_cycles=1,
                 intracel_cutoff_freq=1, intracel_peak_height=-50,
                 intracel_peak_distance=10, intracel_prominence=5, neuron_freq_threshold=150,
                 filt_length=7, filt_polyorder=2, bin_step=0.1
                 ):
        self.binned_and_smoothed = False
        self.spike_freq_dict = deepcopy(spike_freq_dict)
        for key, item in spike_freq_dict.items():
            self.spike_freq_dict[key] = np.array(item)
        self.intracel_signal = intracel_signal
        self.time = time_vector
        self.time_intervals = time_intervals
        self.no_cycles = no_cycles
        self.intervals = None
        self.select_spikes(*list(spike_freq_dict))
        self.removeBadSpikes(freq_threshold=neuron_freq_threshold)

        self.generateInterval(intracel_cutoff_freq, intracel_peak_height, intracel_peak_distance, intracel_prominence,
                              no_cycles)

        super().generateSegments()

        self.smoothNeurons(filt_length, filt_polyorder, bin_step)

        self.generateSegments()

        self.resampleSegments()


    def removeBadSpikes(self, freq_threshold=150):
        for key, items in self.spike_freq_dict.items():
            self.spike_freq_dict[key] = items[:, items[1] < freq_threshold]

    def select_spikes(self, *args):
        self.selected_spikes = []
        for arg in args:
            self.selected_spikes.append(arg)

        cmap = categorical_cmap(math.ceil(len(self.selected_spikes) / math.ceil(len(self.selected_spikes) / 10)),
                                math.ceil(len(self.selected_spikes) / 10))
        self.raster_cmap = {}
        i = 0
        for sp in self.selected_spikes:
            self.raster_cmap[sp] = cmap(i)
            i += 1

    def smoothNeurons(self, filt_length, filt_polyorder, bin_step=0.1):
        self.spike_freq_dict = binSpikeFromISIs(self.spike_freq_dict, self.time[-1], bin_step, full_binning=True)
        new_dict = {}
        for key, items in self.spike_freq_dict.items():


            smoothed = spsig.savgol_filter(items[1], window_length=filt_length,
                                                                    polyorder=filt_polyorder)


            new_dict[key] = np.array([items[0], smoothed])
        self.spike_freq_dict = new_dict
        self.binned_and_smoothed = True

    def generateInterval(self, cutoff_freq, peak_height, peak_distance, prominence, no_cycles=None):
        if no_cycles is None:
            no_cycles = self.no_cycles
        sampling_rate = round(1 / (self.time[1] - self.time[0]))
        peak_distance = int(peak_distance * sampling_rate)




        mean_array = np.array(())

        for interval in self.time_intervals:
            mean_array = np.concatenate((mean_array, self.intracel_signal[interval[0]:interval[1]]))

        # filtered_intracel = self.intracel_signal - np.median(mean_array)
        filtered_intracel = self.intracel_signal
        filtered_intracel = filterUtils.runButterFilter(filtered_intracel, cuttoff_freq=cutoff_freq, butt_order=4,
                                              sampling_rate=sampling_rate)



        intracel_max_idxs = \
            spsig.find_peaks(filtered_intracel, height=peak_height, distance=peak_distance, prominence=prominence)[0]


        intracel_idxs = []
        for interval in self.time_intervals:
            intracel_idxs.append(intracel_max_idxs[(self.time[intracel_max_idxs] > interval[0]) & (
                    self.time[intracel_max_idxs] < interval[1])])

        plt.figure()
        plt.plot(self.time[::5], filtered_intracel[::5], color='k')
        plt.scatter(self.time[intracel_max_idxs], filtered_intracel[intracel_max_idxs], color='r')

        assert len(np.concatenate(intracel_idxs)) > 0, 'No peak was found for the intracelular signal and the given time interval'

        self.intervals = []

        for idx_list in intracel_idxs:
            for i in range(len(idx_list) - no_cycles):
                self.intervals.append([self.time[idx_list[i]], self.time[idx_list[i + no_cycles]]])
        self.intervals = np.array(self.intervals)


    def generateSegments(self):
        if self.binned_and_smoothed:
            print('Segments have already been binned and smoothed')
        else:
            print('Segments haven´t been binned and smoothed yet')
        i = 0
        self.segmented_neuron_frequency_list = []
        for event in self.intervals:
            event_freq_dict = {}
            for key in self.spike_freq_dict.keys():

                idxs = np.where((self.spike_freq_dict[key][0] > event[0]) & (self.spike_freq_dict[key][0] < event[1]))
                # times = self.spike_freq_dict[key][0][idxs]
                event_freqs = self.spike_freq_dict[key][1][idxs]
                # rescaled_times = self.no_cycles * (times - event[0]) / (event[1] - event[0])

                try:
                    event_freq_dict[key] = np.array(event_freqs)
                except Exception as e:
                    print(key, len(event_freqs))
                    raise e

            self.segmented_neuron_frequency_list.append(event_freq_dict)

            i += 1


    def resampleSegments(self):
        self.resampled_segmented_neuron_frequency_list = []
        max_len = 0
        first_key = list(self.spike_freq_dict)[0]
        for segment in self.segmented_neuron_frequency_list:
            # print(len(segment[first_key]))
            if len(segment[first_key])>max_len:
                max_len = len(segment[first_key])

        for segment in self.segmented_neuron_frequency_list:
            new_dict = {}
            for key, item in segment.items():
                new_dict[key] = spsig.resample(item, max_len)

            self.resampled_segmented_neuron_frequency_list.append(new_dict)

