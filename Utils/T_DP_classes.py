import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as spsig
import os.path
import PyLeech.Utils.spikeUtils as sU


def normResults(par, norm_freq=15, rg=5):
    ind = np.where((par[:, 0] < norm_freq + rg / 2) & (
                par[:, 0] >= norm_freq - rg / 2))[0]
    norm_val = np.mean(par[ind, 1])
    print(norm_val)
    if np.isnan(norm_val):
        return
    par[:, 1] /= norm_val

class TIpspDpsResults():

    def __init__(self, processed_T_obj, processed_DP_obj):
        """
        Ipsps and DP DE-3 associated bursts analyzer

        TIpspDpResults(filename, T trace object, DP object, **kwargs)
        """
        self.par_area_freq = []
        self.par_area_freq_single_burst = []
        self.par_dpmatch_freq = []
        self.result_dict = {}
        self.result_dict['match_count'] = {'ipsp_y': 0, 'ipsp_n': 0, 'burst_y': 0, 'burst_n': 0}
        self.result_dict['ipsp_data'] = {'matched_size': 0, 'unmatched_size': 0, 'matched_no': 0, 'unmatched_no': 0}

        self.T = processed_T_obj
        self.DP = processed_DP_obj
        self.processData()




    def processData(self):


        self.total_ipsps_bursts = len(self.T.burst_obj_list)
        self.total_dp_bursts = len(self.DP.burst_obj_list)
        
        i = 0
        j = 0
        m = 0
        while i < self.total_dp_bursts:
            dp_matched = False
            dp_burst = self.DP.burst_obj_list[i]
            j = m
            while j < self.total_ipsps_bursts:
                ipsp_burst = self.T.burst_obj_list[j]
                if  ipsp_burst.last >= dp_burst.first and dp_burst.last >= ipsp_burst.first:
                    burst_norm_size = ipsp_burst.size / ipsp_burst.diff_from_bl
                    self.result_dict['match_count']['ipsp_y'] += 1
                    self.result_dict['ipsp_data']['matched_size'] += burst_norm_size
                    self.result_dict['ipsp_data']['matched_no'] += len(ipsp_burst.spike_list)

                    dp_matched = True
                    ipsp_burst.matched_burst = i
                    ipsp_burst.dp_freq = dp_burst.freq

                    # if burst_obj_list[j].dp_freq>30:
                    # print(key, j)
                    # if len(dp_burst)>10:

                    m = j + 1

                if ipsp_burst.first > dp_burst.last:
                    break
                j += 1

            if dp_matched:
                self.result_dict['match_count']['burst_y'] += 1
                self.par_dpmatch_freq.append((1, dp_burst.freq))
            else:
                self.result_dict['match_count']['burst_n'] += 1
                self.par_dpmatch_freq.append((0, dp_burst.freq))

            i += 1


        for i in range(self.total_dp_bursts):
            largest = 0
            for ipsp_burst in self.T.burst_obj_list:
                if hasattr(ipsp_burst, 'matched_burst') and ipsp_burst.matched_burst == i:
                    largest += ipsp_burst.size / np.abs(ipsp_burst.baseline + 60)
                    freq = ipsp_burst.dp_freq
            if largest != 0:
                self.par_area_freq_single_burst.append((freq, largest))

        for ipsp_burst in self.T.burst_obj_list:
            if hasattr(ipsp_burst, 'matched_burst'):
                self.par_area_freq.append((ipsp_burst.dp_freq, ipsp_burst.size / ipsp_burst.diff_from_bl))
            else:
                self.par_area_freq.append((0, ipsp_burst.size / ipsp_burst.diff_from_bl))
                self.par_area_freq_single_burst.append((0, ipsp_burst.size / ipsp_burst.diff_from_bl))
                self.result_dict['ipsp_data']['unmatched_size'] += ipsp_burst.size / ipsp_burst.diff_from_bl
                self.result_dict['ipsp_data']['unmatched_no'] += len(ipsp_burst.spike_list)
                self.result_dict['match_count']['ipsp_n'] += 1

        self.par_area_freq_single_burst = np.array(self.par_area_freq_single_burst)
        self.par_area_freq = np.array(self.par_area_freq)
        self.par_dpmatch_freq = np.array(self.par_dpmatch_freq)

    def getIpspPhaseShift(self):
        i = 0
        j = 0
        shift = []
        if (self.total_dp_bursts>1) and (self.total_ipsps_bursts>1):
            while i < (self.total_dp_bursts-1):
                dp_burst = self.DP.burst_obj_list[i]
                next_dp_burst = self.DP.burst_obj_list[i+1]
                while j < self.total_ipsps_bursts:
                    ipsp_burst = self.T.burst_obj_list[j]
                    ipsp_center = np.median(ipsp_burst.spike_list)
                    dp_center = np.median(dp_burst.spike_list)
                    next_dp_center = np.median(next_dp_burst.spike_list)
                    dp_tau = next_dp_center - dp_center


                    if (ipsp_center < dp_center) or (np.abs(ipsp_center-dp_center) < np.abs(ipsp_center-next_dp_center)):
                        shift.append((ipsp_center-dp_center)/ dp_tau)
                        j += 1
                    elif ipsp_center>next_dp_center:
                        break
                    else:
                        shift.append((ipsp_center - next_dp_center) / dp_tau)
                        j += 1
                        break

                i += 1

            dp_burst = self.DP.burst_obj_list[i]
            dp_center = np.median(dp_burst.spike_list)
            while j < self.total_ipsps_bursts:
                ipsp_burst = self.T.burst_obj_list[j]
                ipsp_center = np.median(ipsp_burst.spike_list)
                shift.append((ipsp_center - dp_center)/dp_tau)
                j += 1

        return np.array(shift)





class TraceWithBurstsInfo():
    """
    Parent class of DP_info and T_info to merge attributes
    and facilitate plotting
    """
    def __init__(self, fn, signal, peak_dist, fs, margin=0, source=''):
        self.filename = fn
        self.signal = signal
        self.peak_dist = peak_dist
        self.sampling_freq = fs
        self.spikes = []
        self.margin = margin
        self.source = source
        self.burst_obj_list = []
        self.burst_list = []

    def generateBurstObjList(self, spike_max_dist=0.7, min_sps=10, min_sp_no=10, height=0):
        if not self.burst_list:
            self.burst_list = sU.getBursts(self.spikes, self.sampling_freq, spike_max_dist=spike_max_dist, min_spike_per_sec=min_sps, min_spike_no=min_sp_no)
        for burst in self.burst_list:
            self.burst_obj_list.append(Burst(self.signal, burst, self.source, self.margin, height=height, fs=self.sampling_freq))






class DpInfo(TraceWithBurstsInfo):
    type = 'DP nerve'
    def __init__(self, fn, signal, thresholds, peak_dist=100, fs=5000.):

        if thresholds[0]<0:
            self.signal = -signal
            thresholds = [x * -1 for x in thresholds]
        else:
            self.signal = signal

        self.max_height = thresholds[1]
        TraceWithBurstsInfo.__init__(self, fn, signal, peak_dist, fs, source=self.type)
        self.spikes = spsig.find_peaks(self.signal, height=thresholds, distance=peak_dist)[0]

    #
    #
    #
    #
    #
    # def generateBurstObjList(self, spike_max_dist, min_sps):
    #     self.burst_list = sU.getBursts(self.spikes, self.sampling_freq, spike_max_dist=spike_max_dist, min_spike_per_sec=min_sps)
    #     self.burst_obj_list = []
    #     for burst in self.burst_list:
    #         self.burst_obj_list.append(Burst(self.signal, burst, fs=self.sampling_freq))
    #     return self.burst_obj_list


class TInfo(TraceWithBurstsInfo):
    neuron_type = 'T neuron'

    def __init__(self, fn, signal, fs=5000, step=1000, b_order=8, filt_freq=150,
                 std_thres=4, ipsp_thres=4.75, peak_dist=150, margin=400):
        """
        t_info = TInfo (filename, T trace, sampling frecuency, iteration steps,
        butterworth filter order, filter frequency, std threshold when looing for ipsps,
        ipsp amp threshold from baseline, peak min distance)

        just in case: self.spikes are ipsp peaks, self.T_spikes are actual T spikes
        """

        TraceWithBurstsInfo.__init__(self, fn, signal, peak_dist, fs, margin=margin, source=self.neuron_type)

        self.seg_len = step
        self.std_thres = std_thres
        self.ipsp_thres = ipsp_thres
        self.buttord = b_order
        self.filt_freq = filt_freq

        self.plot_results = False
        self.use_LPsignal = False
        self.getFiltSignal()

        self.T_spikes = spsig.find_peaks(self.signal, height=-5, distance=100)

    def getIpspBursts(self, spike_max_dist=0.7, min_spike_no=0, min_sps=0):
        self.burst_list = sU.getBursts(self.spikes, self.sampling_freq, spike_max_dist, min_spike_no, min_sps)
        return self.burst_list

    def getActiveSignal(self):
        if self.use_LPsignal:
            return self.LPsignal
        return self.signal

    def getFiltSignal(self):
        poli_b, poli_a = spsig.butter(self.buttord, self.filt_freq * 2 / (self.sampling_freq * np.pi))
        self.LPsignal = spsig.filtfilt(poli_b, poli_a, self.signal)
        return self.LPsignal

    def generateFigures(self, tsignal, ind, segments, full_ipsps, std):
        segments = np.array(segments)
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(self.filename)


        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        ax1.plot(tsignal, zorder=0)
        ax1.set_axisbelow(True)
        ax1.grid()
        try:
            ax1.scatter(ind[segments[:, 0]], [np.mean(tsignal)] * len(segments[:, 0]), color='k', zorder=10)
            ax1.scatter(ind[segments[:, 1]], [np.mean(tsignal)] * len(segments[:, 1]), color='k', zorder=10)
        except IndexError:
            pass
        ax1.scatter(full_ipsps, tsignal[full_ipsps], color='r', marker='*', s=50, zorder=20)

        # plt.scatter(ind[overthres], [np.mean(signal)+1]*len(ind[overthres]))

        ax2.plot(ind[:-1] + self.seg_len, std, zorder=0)
        ax2.set_axisbelow(True)
        ax2.grid()

    def getIpsps(self):

        no_spike_signal = sU.removeTSpikes(self.signal)

        if not self.use_LPsignal:
            temp_signal = self.signal
        else:
            temp_signal = self.LPsignal

        if np.nanmean(no_spike_signal) < -58 and np.nanmean(no_spike_signal) > -62:
            self.ipsp_list = []
            return self.ipsp_list

        ind = np.arange(0, len(temp_signal), self.seg_len)
        std = []
        for i in ind:
            std.append(np.std(temp_signal[i:i + self.seg_len]))

        ind = np.append(ind, len(temp_signal) - 1)
        std = np.array(std)
        thres = self.std_thres * np.mean(np.partition(std, 10)[:10])

        overthres = np.where(np.array(std) > thres)[0]
        segments = []
        full_ipsps = []
        j = 0

        if len(overthres) > 1:
            for i in range(len(overthres) - 1):

                if j == 0:
                    first = overthres[i]
                    j += 1
                    le = 1

                if (overthres[i + 1] - overthres[i]) <= 5:
                    le += overthres[i + 1] - overthres[i]
                else:
                    j = 0
                    segments.append((first, first + le))

            if j != 0:
                segments.append((first, first + le))
            elif j == 0:
                segments.append((overthres[-1], overthres[-1] + 1))

        elif len(overthres) == 1:
            segments.append((overthres[0], overthres[0] + 1))

        i = 0
        for tup in segments:
            i += 1
            try:
                ahead_mean = np.mean(temp_signal[ind[tup[1]]:ind[tup[1] + 1]])
            except:
                ahead_mean = np.nan
            try:
                back_mean = np.mean(temp_signal[ind[tup[0] - 1]:ind[tup[0]]])
            except:
                back_mean = np.nan

            baseline = np.nanmean((ahead_mean, back_mean))
            try:
                thres_std = self.ipsp_thres * std[tup[0] - 1]
            except:
                thres_std = self.ipsp_thres * std[tup[1]]

            side = getSide(baseline, temp_signal[ind[tup[0]]:ind[tup[1]]])
            height = side * baseline + thres_std

            T_indexes = spsig.find_peaks(side * temp_signal[ind[tup[0]]:ind[tup[1]]], height=height,
                                         distance=self.peak_dist, width=70)
            T_spikes = spsig.find_peaks(self.signal[ind[tup[0]]:ind[tup[1]]], height=-5, distance=100)

            bad_indexes = sU.getSpikeIndexes(T_indexes[0], T_spikes[0])  ##look for wrongly chosen peaks

            T_ipsps = np.delete(T_indexes[0], bad_indexes) + ind[tup[0]]

            full_ipsps.extend(T_ipsps)

        if self.plot_results: self.generateFigures(temp_signal, ind, segments, full_ipsps, std)
        self.spikes = full_ipsps
        return full_ipsps


class Burst():
    def __init__(self, signal, spikes, source='', margin=0, height=0, fs=5000.):
        """
            Generates burst data given signal and bursts' spikes

        burst = Burst(source, signal, spikes, margin, sample frequency)

        source should be 'T neuron' for T cell.
            Needed if you want to generate a margin on the edges
            and check the baseline
        """
        self.source = source
        self.sample_freq = fs
        self.spike_list = spikes
        self.margin = margin
        self.setEdges(len(signal))
        self.signal = signal[self.first:self.last]
        if self.source == 'T neuron':
            self.getBaseline(signal)
            self.getBurstSize()
            self.diff_from_bl = np.abs(self.baseline + 60)

        elif self.source == 'DP nerve':
            self.height = height


        if len(signal[spikes[0]:spikes[-1]])!=0:
            self.freq = len(spikes)*fs/len(signal[spikes[0]:spikes[-1]])
        else:
            self.freq = np.nan

    def setEdges(self, sl):

        if self.spike_list[0] - self.margin > 0:
            self.first = self.spike_list[0] - self.margin
        else:
            self.first = 0

        if self.spike_list[-1] + self.margin < sl:
            self.last = self.spike_list[-1] + self.margin
        else:
            self.last = sl - 1

    def getBaseline(self, signal):

        self.baseline = np.mean(self.signal[0:int(self.margin * 3 / 4)])

    #        first_seg = self.signal[0:int(margin/2)]
    #        last_seg = self.signal[-int(margin/2):]
    #        self.baseline = np.mean((first_seg, last_seg))

    def getBurstSize(self):
        dt = 1 / self.sample_freq
        base_area = len(self.signal) * dt * self.baseline
        ipsp_area = sum(self.signal) * dt
        self.size = np.abs(ipsp_area - base_area)




def getSide(bl, signal, bins=50, spdist=100):
    if bl > -60:
        return -1
    elif bl < -60:
        return 1
    else:
        no_spike_signal = sU.removeTSpikes(signal, spdist)
        nan_filtered_signal = no_spike_signal[np.logical_not(np.isnan(no_spike_signal))]
        rel_max = np.nanmax(no_spike_signal) - bl
        rel_min = bl - np.nanmin(no_spike_signal)
        if rel_max < 0:
            return -1
        elif rel_min < 0:
            return 1
        elif rel_max > rel_min:
            return 1
        else:
            return -1


def getIsi(spike_list, freq):
    for i in range(len(spike_list) - 1):
        spike_list[i] = (spike_list[i + 1] - spike_list[i]) / freq
    return spike_list[:-1]


def getSignal(block, channel):
    ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
    return ch_data[:, block.ch_info[channel]['index']]


def plotBursts(pltobj, time, twb_info:[DpInfo, TInfo], ms=4):
    """
    :type twb_info: [DP_info, T_info]
    """


    i = 0
    if twb_info.source in 'T neuron':
        iterate_height = True
        if twb_info.use_LPsignal:
            signal = twb_info.LPsignal
        else:
            signal = twb_info.signal

    else:
        signal = twb_info.signal
        iterate_height = False
        amp = twb_info.max_height*1.05

    pltobj.plot(time, signal)
    pltobj.plot(time[twb_info.spikes], signal[twb_info.spikes], '*', label='Peaks', ms=ms)

    for bl in twb_info.burst_obj_list:
        if iterate_height:
            amp = bl.baseline

        first = bl.first
        last = bl.last

        if len(bl.signal) == 1:
            pltobj.plot([time[first], time[last]], [amp, amp], color='k', linewidth=5)
        else:
            if i == 0:
                i += 1
                pltobj.plot([time[first], time[last]], [amp, amp], color='k', linewidth=5, label='bursts')
                continue

            pltobj.plot([time[first], time[last]], [amp, amp], color='k', linewidth=5)
            # pltobj.plot(time[bl], nerve_signal[bl], 'r-*' )


    pltobj.legend(loc=4, fontsize=20)
    pltobj.grid()
    # pltobj.ylim([-.1,.1])
    try:
        pltobj.tight_layout()
    except AttributeError:
        pass


class resultsPlotter():

    def __init__(self, data, bins=[], use_title=True, trace=None, neuron=None):
                
        data = np.array(data)
        self.x = np.array(data[:,0])
        self.y = np.array(data[:,1])
        self.use_title = use_title
        self.trace = trace
        self.neuron = neuron
        self.bins = bins
        self.generateTitle()

    def generateTitle(self):
        if self.trace is not None:
            self.title = 'trace ' + os.path.basename(os.path.splitext(self.trace)[0])
        elif self.neuron is not None:
            self.title = 'neuron' + self.neuron
        else:
            self.title = 'Full Results'

    def boxPlot(self):
        boxplot_list = []

        j = (self.bins[1] - self.bins[0])/2
        for i in self.bins:
            ind = np.where((self.x < i + j) & (
                        self.x >= i - j))[0]
            boxplot_list.append(self.y[ind])

        plt.boxplot(boxplot_list, positions=self.bins, widths=1.5)
        if self.use_title:
            plt.title(self.title)
        plt.xlim([self.bins[0]-j, self.bins[-1]+j])

    def meansPlot(self):
        bin_results = np.array(sU.binXYLists(self.bins, self.x, self.y, std_as_err=True))
        if self.use_title:
            plt.title(self.title)
        xerr = self.bins[1] - self.bins[0]
        plt.errorbar(bin_results[0], bin_results[1], xerr=xerr, yerr=[bin_results[2], bin_results[3]], fmt='|',
                     color='k')
        plt.xlim([-5, 40])

    def getMeansResult(self, get_median=False):
        bin_results = np.array(sU.binXYLists(self.bins, self.x, self.y, get_median=get_median, std_as_err=True))
        return (bin_results[0:2])

    def scatterPlot(self):
        plt.scatter(self.x, self.y)
        if self.use_title:
            plt.title(self.title)


