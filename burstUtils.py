import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
import PyLeech.spikeUtils as spikeUtils
import os.path
from copy import deepcopy
from PyLeech.constants import *
import _pickle as pickle
import math


def getInstFreq(time, spike_dict, fs):
    spike_freq_dict = {}
    for key, items in spike_dict.items():
        spike_freq_dict.update({key: [time[items[:-1]], np.reciprocal(np.diff(items) / fs)]})

    return spike_freq_dict


def binSpikes(spike_freq_dict, time_length, step):
    rg = np.arange(0, time_length, step)
    binned_spike_freq_dict = {}
    for key, item in spike_freq_dict.items():
        freqs = np.zeros(len(rg))
        bins = spikeUtils.binXYLists(rg, item[0], item[1], get_median=True, std_as_err=True)
        binned_spike_freq_dict.update({key: [np.array(bins[0]), np.array(bins[1])]})
    return binned_spike_freq_dict


def plotCompleteDetection(traces, time, spike_dict, template_dict, cluster_colors, legend=True, lw=1, clust_list=None,
                          hide_clust_list=None, interval=None, step=1, intracel_signal=None):
    if interval is None:
        interval = [0, len(time)]

    else:
        interval[0] = int(interval[0] * len(time))
        interval[1] = int(interval[1] * len(time))
    if intracel_signal is not None:
        hr = [2] * len(traces)
        hr.append(1)
        fig, ax_list = plt.subplots(
            nrows=(len(traces) + 1), ncols=1, sharex=True,
            gridspec_kw={'height_ratios': hr}
        )
        plot_data_list(traces[:, interval[0]:interval[1]:step], time[interval[0]:interval[1]:step],
                       linewidth=lw, fig=fig, ax_list=ax_list)
    else:
        fig, ax_list = plot_data_list(traces[:, interval[0]:interval[1]:step], time[interval[0]:interval[1]:step],
                                      linewidth=lw)

    for key, spike_ind in spike_dict.items():
        if type(hide_clust_list) is list and key in hide_clust_list:
            continue
        if (clust_list is None) or (key in clust_list):
            try:
                channels = template_dict[key]['channels']
            except KeyError:
                channels = None

            if key == nan:
                label = '?'
            else:
                label = str(key)

            plot_detection(traces,
                           time,
                           spike_ind[(spike_ind > interval[0]) & (spike_ind < interval[1])] - interval[0],
                           channels=channels,
                           peak_color=cluster_colors[key],
                           label=label,
                           ax_list=ax_list)
    hdl = []
    lbl = []
    if intracel_signal is not None:
        ax_list[2].plot(time[interval[0]:interval[1]], intracel_signal[interval[0]:interval[1]], color='k')
    for ax in ax_list:
        ax.grid(linestyle='dotted')
        removeTicksFromAxis(ax, 'y')
        removeTicksFromAxis(ax, 'x')
        handle, label = ax.get_legend_handles_labels()
        hdl.extend(handle)
        lbl.extend(label)
    showTicksFromAxis(ax_list[-1], 'x')
    if legend:
        # ax_list[0].legend(hdl, lbl, loc='upper right')
        fig.legend(hdl, lbl, loc='upper right')
        # plt.legend(loc='upper right')

    return fig, ax_list


def plot_data_list(data_list,
                   time_axes,
                   linewidth=0.2,
                   signal_color='black',
                   fig=None,
                   ax_list=None):
    """Plots data together with detected events.

    Parameters
    ----------
    data_list: a list of numpy arrays of dimension 1 that should all
               be of the same length (not checked).
    time_axes: an array with as many elements as the components of
               data_list. The time values of the abscissa.
    linewidth: the width of the lines drawing the curves.
    signal_color: the color of the curves.

    Returns
    -------
    Nothing is returned, the function is used for its side effect: a
    plot is generated.
    """
    nb_chan = len(data_list)
    if (fig is None) and (ax_list is None):
        fig, ax_list = plt.subplots(nb_chan, sharex=True)
    else:
        assert (fig is not None) or (ax_list is not None), "I need both a fig and its ax_list to work properly"

    if nb_chan == 1: ax_list = [ax_list]

    for i in range(nb_chan):
        ax_list[i].plot(time_axes, data_list[i],
                        linewidth=linewidth, color=signal_color)

    plt.xlabel("Time (s)")
    return fig, ax_list


def plot_detection(data_list,
                   time_axes,
                   evts_pos,
                   channels=None,
                   peak_color='r',
                   label=None,
                   ax_list=None):
    """Plots data together with detected events.

    Parameters
    ----------
    data_list: a list of numpy arrays of dimension 1 that should all
               be of the same length (not checked).
    time_axes: an array with as many elements as the components of
               data_list. The time values of the abscissa.
    evts_pos: a vector containing the indices of the detected
              events.
    signal_color: the color of the curves.

    Returns
    -------
    Nothing is returned, the function is used for its side effect: a
    plot is generated.
    """
    nb_chan = len(data_list)

    labeled = False
    for i in range(nb_chan):
        if (channels is None) or (i in channels) or (-1 in channels):

            if peak_color is not None:
                if not labeled:
                    ax_list[i].scatter(time_axes[evts_pos],
                                       data_list[i][evts_pos], marker='o', color=peak_color, label=label)
                    labeled = True
                else:
                    ax_list[i].scatter(time_axes[evts_pos],
                                       data_list[i][evts_pos], marker='o', color=peak_color)
            else:
                if not labeled:
                    ax_list[i].scatter(time_axes[evts_pos],
                                       data_list[i][evts_pos], marker='o', label=label)
                    labeled = True
                else:
                    ax_list[i].scatter(time_axes[evts_pos],
                                       data_list[i][evts_pos], marker='o')


def plotFreq(spike_freq_dict, color_dict, optional_trace=None, template_dict=None, scatter_plot=True,
             single_figure=False,
             skip_list=None, draw_list=None, thres=None, ms=1, outlier_thres=None, sharex=None):
    """Plots instantaneous frequency from a given dictionary of the clusters and corresponding time, frequency lists

            Parameters
            ----------
            spike_freq_dict: a dictionary of the different clusters pointing to a list of two np arrays(time, freq)
            color_dict: a dictionary of clusters containing its corresponding color sequence
            optional_trace: list of time, trace desired to add to the graph (for example [time_vector, NS neuron trace])

            Returns
            -------
            Nothing is returned, the function is used for its side effect: a
            plot is generated.
            """

    if skip_list is None:
        skip_list = []
    if draw_list is None:
        draw_list = list(spike_freq_dict.keys())

    fig = plt.figure(figsize=(12, 6))
    fig.tight_layout()
    i = 1
    if single_figure:
        j = 1
    else:
        j = len(draw_list) - len(skip_list)
    if optional_trace is not None:
        j += 1

    hdl = []
    lbl = []
    for key, items in spike_freq_dict.items():
        if key in draw_list and key not in skip_list:
            if thres is None:
                mask = [True] * len(items[1])
            else:
                mask = (items[1] > thres[0]) & (items[1] < thres[1])

            label = str(key)
            if template_dict is not None:
                if len(template_dict[key]['channels']) == 1:
                    if template_dict[key]['channels'][0] == -1:
                        label += ': all'
                    else:
                        label += ': ch ' + str(template_dict[key]['channels'][0])
                else:
                    label += ': chs:'
                    for i in template_dict[key]['channels']:
                        label += ' ' + str(i)

            data0 = items[0][mask]
            data1 = items[1][mask]
            if outlier_thres is not None:
                data0 = data0[~is_outlier(data1, outlier_thres)]
                data1 = data1[~is_outlier(data1, outlier_thres)]

            if sharex is not None:
                ax = plt.subplot(j, 1, i, sharex=sharex)
            elif i == 1:
                ax = plt.subplot(j, 1, i)
            else:
                ax = plt.subplot(j, 1, i, sharex=ax)

            if scatter_plot:
                ax.scatter(data0, data1, color=color_dict[key], label=label, s=ms)
            else:
                ax.plot(data0, data1, color=color_dict[key], label=label, ms=ms)
            # ax.legend()
            ax.grid(linestyle='dotted')
            removeTicksFromAxis(ax, 'x')
            ax.set_facecolor('lightgray')
            ax.legend()
            if not single_figure:
                i += 1
    if optional_trace is not None:
        if sharex is not None:
            ax = plt.subplot(j, 1, i, sharex=sharex)
        elif i == 1:
            ax = plt.subplot(j, 1, i)
        else:
            ax = plt.subplot(j, 1, i, sharex=ax)

        ax.plot(optional_trace[0], optional_trace[1], color='k', lw=1)
    ax.grid(linestyle='dotted')
    removeTicksFromAxis(ax, 'y')
    showTicksFromAxis(ax, 'x')

def removeTicksFromAxis(ax, axis='y'):
    axis = getattr(ax, axis + 'axis')
    for tic in axis.get_major_ticks():
        tic.label1On = tic.label2On = False
        tic.tick1On = tic.tick2On = False

def showTicksFromAxis(ax, axis='y'):
    axis = getattr(ax, axis + 'axis')
    for tic in axis.get_major_ticks():
        # tic.label1On = tic.label2On = True
        # tic.tick1On = tic.tick2On = True
        tic.label1On = True
        tic.tick1On = True


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


class BurstStorerLoader():
    required_to_save_dict_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict']
    expected_load_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict', 'isDe3']

    def __init__(self, filename, mode='save', traces=None, time=None, spike_dict=None, spike_freq_dict=None, De3=None,
                 template_dict=None, color_dict=None):
        self.filename = filename
        self.traces = traces
        self.time = time
        self.spike_dict = spike_dict
        self.spike_freq_dict = spike_freq_dict
        self.isDe3 = De3
        self.template_dict = template_dict
        self.color_dict = color_dict

        if str.lower(mode) == 'save':
            self.saveResults()
        elif str.lower(mode) == 'load':
            self.loadResults()
            if self.isDe3 == None:
                print('I need to set attribute \'isDe3\' before continuing')

    def saveResults(self, filename=None):
        pkl_dict = self.generatePklDict()
        if filename is None:
            filename = generatePklFilename(self.filename)
        else:
            filename = generatePklFilename(filename)
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


def assertMissingData(needed_keys):
    try:
        assert len(needed_keys) < 2, 'This data is missing:'
    except AssertionError as e:
        e.args += tuple([str(mk) for mk in needed_keys])
        raise


def saveSpikeResults(filename, json_dict):
    os.path.splitext(filename)[0] + '.pklspikes'
    print('Saved in %s' % filename)
    with open(filename, 'wb') as pfile:
        pickle.dump(json_dict, pfile)


def loadSpikeResults(filename):
    assert os.path.splitext(filename)[1] == '.pklspikes', 'Wrong file extension, I need a .pklspikes file'
    with open(filename, 'rb') as pfile:
        results = pickle.load(pfile)
    return results


def getBursts(times, spike_max_dist=0.7, min_spike_no=10, min_spike_per_sec=10):
    """ Returns list of bursts if there are any or empty if none

    :param times:
    :type np.ndarray
    :param spike_max_dist:
    :param min_spike_no:
    :param min_spike_per_sec:
    :return: list of lists
    """
    if len(times) == 0:
        return []
    burst_list = []
    burst_index = []
    for i in range(len(times) - 1):
        burst_index.append(times[i])
        if (times[i + 1] - times[i]) > spike_max_dist:
            if checkBurst(burst_index, min_spike_no, min_spike_per_sec):
                burst_list.append(burst_index)
            else:
                new_bursts = checkBurstWithin(burst_index, min_spike_no, min_spike_per_sec)
                burst_list.extend(new_bursts)

            burst_index = []

    ## just in case the last one is considered a "burst" and was forgotten because it wasn't part of the previous burst
    burst_index.append(times[-1])

    ##checking the last one
    if checkBurst(burst_index, min_spike_no, min_spike_per_sec):
        burst_list.append(burst_index)
    else:
        new_bursts = checkBurstWithin(burst_index, min_spike_no, min_spike_per_sec)
        burst_list.extend(new_bursts)

    return burst_list


def checkBurst(burst_index, min_spike_no, min_spike_per_sec):
    if len(burst_index) < min_spike_no:
        return False
    elif len(burst_index) == 1:
        return True
    elif len(burst_index) / (burst_index[-1] - burst_index[0]) > min_spike_per_sec:
        return True
    else:
        return False


def checkBurstWithin(times_list, min_spike_no, min_spike_per_sec):
    bursts = []
    if len(times_list) <= min_spike_no:
        return []

    i = 0
    while i < len(times_list) - min_spike_no:

        for j in range(i + min_spike_no, len(times_list)):

            if not checkBurst(times_list[i:j + 1], min_spike_no, min_spike_per_sec) and checkBurst(times_list[i:j],
                                                                                                   min_spike_no,
                                                                                                   min_spike_per_sec):

                bursts.append(times_list[i:j])
                i = j
            elif (j == len(times_list) - 1) and checkBurst(times_list[i: len(times_list) + 1], min_spike_no,
                                                           min_spike_per_sec):
                bursts.append(times_list[i:j + 1])

                i = j
            if i == j:
                break
        i += 1

    return bursts


def getInterBurstInterval(burst_lists, no_burst=2):
    burst_int = np.zeros((len(burst_lists) - no_burst, 2))

    for i in range(len(burst_lists) - no_burst):
        burst_int[i, 0] = burst_lists[i][0]
        burst_int[i, 1] = burst_lists[i + no_burst][0]
    return burst_int


class CrawlingSegmenter():

    def __init__(self, dp_trace, time, spike_freq_dict, de3_neuron, no_bursts=2, min_spike_no=10, min_spike_per_sec=10,
                 spike_max_dist=0.7):
        self.de3_neuron = de3_neuron
        self.spike_freq_dict = spike_freq_dict
        self.generateInterval(dp_trace, time, self.de3_neuron, no_bursts, min_spike_no=min_spike_no,
                              min_spike_per_sec=min_spike_per_sec, spike_max_dist=spike_max_dist)
        self.select_spikes(*list(spike_freq_dict.keys()))
        self.generateSegments()

    def generateInterval(self, dp_trace, time, de3_neuron, no_bursts=2, min_spike_no=5, min_spike_per_sec=5,
                         spike_max_dist=0.7):
        burst_list = getBursts(np.array(self.spike_freq_dict[de3_neuron][0]), min_spike_no=min_spike_no,
                               min_spike_per_sec=min_spike_per_sec, spike_max_dist=spike_max_dist)
        print("Check whether bursts were correctly selected")
        plt.figure()
        plt.plot(time, dp_trace, linewidth=1, color='k')
        self.no_bursts = no_bursts
        self.dp_intervals = []
        t0 = []
        for bl in burst_list:
            plt.plot(bl, [0] * len(bl), color='red', linewidth=5)
            self.dp_intervals.append([bl[0], bl[-1]])
        self.dp_intervals = np.array(self.dp_intervals)

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
                    event_dict[key] = self.no_bursts * (event_dict[key] - event[0]) / (event[1] - event[0])
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
                        self.de3_neuron] + i * self.no_bursts)

                    line_to_extend += 1
                    for key, items in self.segment_list[j].items():
                        if key != self.de3_neuron and key in self.selected_spikes:
                            spike_list[line_to_extend] = np.append(spike_list[line_to_extend],
                                                                   items + i * self.no_bursts)
                            line_to_extend += 1

        self.fig, self.eventplot_ax = plt.subplots()
        self.eventplot_ax.eventplot(spike_list, colors=color_list, linewidths=linewidths)
        if generate_grid:
            minor_ticks = np.arange(-0.5, len(spike_list) - 0.5, len(self.selected_spikes))
            self.eventplot_ax.set_yticks(minor_ticks, minor=True)
            self.eventplot_ax.grid(which='minor', linestyle='--')

        vlines = np.arange(self.no_bursts, split_raster * self.no_bursts + 1, self.no_bursts)
        for ln in vlines:
            self.eventplot_ax.axvline(ln, color='k', linestyle='--', linewidth=1)


def generateFilenameFromList(filename):
    new_filename = os.path.basename(filename[0]).split('_')
    new_filename = "_".join(new_filename[:-1])

    for fn in filename:
        num = os.path.splitext(fn.split("_")[-1])[0]
        new_filename += '_' + num

    return new_filename


def generatePklFilename(filename):
    if type(filename) is list:
        filename = generateFilenameFromList(filename)

    else:
        filename = os.path.basename(os.path.splitext(filename)[0])

    filename = 'RegistrosDP_PP/' + filename
    return filename


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
