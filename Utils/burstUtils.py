import math
import os.path
import pickle as pickle
import pandas as pd
import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as spsig
from PyLeech.Utils import spikeUtils as spikeUtils
from PyLeech.Utils.constants import nan
from copy import deepcopy
import warnings


def getInstFreq(time, spike_dict, fs):
    """
    Generates an spike frequency dictionary of the type {key:[spike times, spike instataneous frequency]} where `key` is
    its corrresponding neuron number
    Parameters
    ----------
    time ndarray:
        time array where the spikes were acquired from
    spike_dict dict:
        spike dictionary of the type {key: [spike times, spike indexes]
    fs float:
        data sampling frequency

    Returns
    -------
    spike_freq_dict dict
    """
    spike_freq_dict = {}
    for key, items in spike_dict.items():
        spike_freq_dict.update({key: [time[items[:-1]], np.reciprocal(np.diff(items) / fs)]})

    return spike_freq_dict


def getSpikeTimesDict(spike_freq_dict):
    """
    Generates a dictionary of spike times from an spike_freq_dict
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type [spike times, spike instataneous frequency]} where `key` is
    its corrresponding neuron number

    Returns
    -------
    SpikeTimesDict dict
    """
    return {key: items[0] for key, items in spike_freq_dict.items()}


def binSpikesFreqs(spike_freq_dict, time_length, step, full_binning=False, use_median=False):
    """
    Generates digitized spike frequency dictionary from a non digitized spike_freq_dict
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is
    its corrresponding neuron number
    time_length float:
        time_length from where the signal was acquired
    step float:
        binning step to be used for the digitalization process
    full_binning bool:
        Parameter for controlling the digitalization length, if True it will generate the binning up to 'time_length'
        even if spiking activity ended before that value
    use_median bool:
        if True, the binned value will be computed by calculating the median spike period rather than the average

    Returns
    -------
    binned_spike_freq_dict dict:
        The corresponding digitalization of the spike_freq_dict
    """

    # Generates bin edges
    rg = np.arange(0, time_length, step)
    binned_spike_freq_dict = {}
    for key, item in spike_freq_dict.items():

        # returns the bining for each unit
        bins = spikeUtils.binXYLists(rg, item[0], item[1], get_median=use_median, std_as_err=True)

        # generates the actual dictionary
        if full_binning:
            freqs = np.zeros(len(rg))
            freqs[np.in1d(rg, bins[0])] = bins[1]
            binned_spike_freq_dict.update({key: [rg, freqs]})
        else:
            binned_spike_freq_dict.update({key: [np.array(bins[0]), np.array(bins[1])]})
    return binned_spike_freq_dict


def removeOutliers(spike_freq_dict, outlier_threshold=3.5):
    """
    Removes detected outliers by comparing the instant frequency with respect to the average instant frequency.

    Parameters
    ----------
     spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is
        its corrresponding neuron number

    outlier_threshold float:
        In units of standard deviation

    Returns
    An updated spike_freq_dict without spikes where its instataneous frequency was above the selected threshold
    -------

    """
    no_outlier_sfd = {}
    for key, items in spike_freq_dict.items():
        ##This function compares each value with the threshold and returns a boolean array
        outlier_mask = is_outlier(items[1], thresh=outlier_threshold)
        no_outlier_sfd[key] = [items[0][~outlier_mask], items[1][~outlier_mask]]
    return no_outlier_sfd


def digitizeSpikeFreqs(spike_freq_dict, num, time_length=None, counting=False, freq_threshold=None):
    """
    Generates digitized spike frequency dictionary from a non digitized spike_freq_dict. It will always return a full
    binning.
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where `key` is
        its corresponding neuron number
    num int:
        Number of bins to be used
    time_length float:
        Original signal duration in seconds.
    counting bool:
         If set to True, the function will return spike count by bin. If False, it will return mean frequency.
    freq_threshold float:
        frequency threshold to be used in outlier removal before digitalization

    Returns
    -------
    binned_spike_freq_dict
    """

    if time_length is None:
        time_length = max([items[1].max() for key, items in spike_freq_dict.items()])

    rg = np.linspace(0, time_length, num, endpoint=False)

    binned_spike_freq_dict = {}
    freqs = np.zeros(len(rg))

    for key, item in spike_freq_dict.items():
        freqs[:] = 0
        times = np.array(item[0])

        freqs_arr = np.array(item[1])
        if freq_threshold is not None:
            times = times[freqs_arr < freq_threshold]
            freqs_arr = freqs_arr[freqs_arr < freq_threshold]

        digitalization = np.digitize(times, rg) - 1

        uid = np.unique(digitalization)
        if not counting:

            for i in uid:
                # freqs[i] = 1 / np.mean(1 / freqs_arr[np.where(digitalization == i)[0]])
                bin_idxs = np.where(digitalization == i)[0]
                if bin_idxs.size > 1:
                    isi = np.mean(np.diff(times[bin_idxs]))
                    freqs[i] = 1 / isi
                elif bin_idxs.size == 1:
                    freqs[i] = freqs_arr[bin_idxs[0]]
                else:
                    freqs[i] = 0

                if np.isnan(freqs[i]): print(freqs[i])



        else:
            freqs = np.bincount(digitalization, minlength=len(freqs))
            if counting == 'bool':
                freqs[freqs != 0] = 1

        if (counting == 'bool') and ((freqs == 0).sum() < 15):
            pass
        else:
            binned_spike_freq_dict[key] = np.array([rg[:], deepcopy(freqs)])

    return binned_spike_freq_dict


def digitizeSpikeTimes(spike_times, step, time_length, counting=False):
    """
    Generates digitized spike counts from an array corresponding to the spike times of a single unit. If `counting` is
    False, the function returns an binary array indicating activity/no activity in the correspoding bin.
    Parameters
    ----------
    spike_times ndarray:
        array with every spike time for the corresponding neuron.
    step int:
        binning step.
    time_length float:
        Original signal duration in seconds.
    counting bool:
        if True, the function will return bin edges and spikes counts. if False, it will return a binarized array instead
        of spike counts.
    Returns
    -------
    rg ndarray:
        bin edges generated for the digitalization.
    counts ndarray:
        array with spike counts for each bin. Returned only if `counting` is True.
    binary_count ndarray:
        binary array accounting only for spiking activity. Returned only if counting is False.
    """
    rg = np.arange(0, time_length, step)

    times = np.array(spike_times)
    digitalization = np.digitize(times, rg) - 1
    uid = np.unique(digitalization)
    if not counting:
        counts = np.zeros(len(rg))
        counts[uid] = 1
        return counts
    else:
        binary_count = np.bincount(digitalization, minlength=len(rg))
        return rg, binary_count


def getSpISIdictFromSpFreqdict(spike_freq_dict):
    """
    A resumed way to get inter spike intervals for spike from a spike_freq_dict. The method may incur in numeric errors.
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where `key` is
        its corresponding neuron number

    Returns
    -------
    sISId dict:
        A dictionary like a spike_freq_dict, but with ISIs instead of instant frequency.

    """
    sISId = {key: [items[0], 1 / items[1]] for key, items in spike_freq_dict.items()}

    return sISId


def binSpikeISIdict(spike_ISI_dict, time_length, step, full_binning=False):
    """
    Generates a binned inter spike interval dict, just like the binned_spike_freq_dict is generated
    Parameters.
    ----------
    spike_ISI_dict dict:
        A spike dict of the type {key: [spike times, inter spike interval]}.
    time_length float:
        Original signal duration in seconds.
    step float:
        binning step to be used for the digitalization process.
    full_binning bool:
        Parameter for controlling the digitalization length, if True it will generate the binning up to 'time_length'
        even if spiking activity ended before that value.

    Returns
    -------
    binned_spike_ISI_dict dict:
        the corresponding binned inter spike interval dictionary.
    """
    rg = np.arange(0, time_length, step)
    binned_spike_ISI_dict = {}
    for key, item in spike_ISI_dict.items():

        bins = spikeUtils.binXYLists(rg, item[0], item[1], get_median=False, std_as_err=True)
        if full_binning:
            ISIs = np.zeros(len(rg))
            ISIs[np.in1d(rg, bins[0])] = bins[1]
            binned_spike_ISI_dict.update({key: [rg, ISIs]})
        else:
            binned_spike_ISI_dict.update({key: [np.array(bins[0]), np.array(bins[1])]})
    return binned_spike_ISI_dict


def getNonZeroIdxs(trace, threshold):
    """
    Gets burst start and end indexes from a trace of binned spike frequencies

    Parameters
    ----------
    trace ndarray:
        numpy array with binned spike frequency
    threshold:
        spike rate threshold

    Returns
    -------
    indexes where bursts start
    """

    non_zero_idxs = np.where(trace > threshold)[0]
    jumps = np.append([0], np.where(np.diff(non_zero_idxs) > 1)[0])
    jumps = np.append(jumps, jumps + 1)
    return np.sort(non_zero_idxs[jumps[1:-1]])


def plotDataPredictionAndResult(time, data, pred):

    """
    plot trace, prediction and peeling results from the spike sorting using the arrays stored in `Spsorter.time`,
    `Spsorter.pred` and `Spsorter.peel`.
    Parameters
    ----------
    time ndarray:
        Time array to be used for the x axis, it can be usually used by passing `Spsorter.time`.
    data ndarray:
        Prediction array. Usually `Spsorter.peel[-1]` to plot the last prediction.
    pred ndarray:
        After peeling prediction. Usually `Spsorter.pred[-1]` to plot the most updated step

    Returns
    -------
    Nothing. It only generates a Figure.
    """
    res = data - pred
    for i in range(len(data)):
        fig = plt.figure(figsize=(16, 8))
        try:
            ax1 = fig.add_subplot(3, 1, 1, sharex=ax1)
        except UnboundLocalError:
            ax1 = fig.add_subplot(3, 1, 1)

        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
        fig.suptitle('channel ' + str(i))
        ax1.set_title('original data')
        ax1.plot(time, data[i,], color='black')
        ax2.set_title('prediction')
        ax2.plot(time, pred[i,], color='black')
        ax3.set_title('peeled')
        ax3.plot(time, res[i,], color='black')
        for ax in [ax1, ax2]:
            removeTicksFromAxis(ax, 'x')
            ax.grid(linestyle='dotted')
        ax3.grid(linestyle='dotted')


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
    linewidth: set the plot linewidths.
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

    if (type(ax_list) != np.ndarray) and (type(ax_list) != list): ax_list = [ax_list]

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


def plotFreq(spike_freq_dict, color_dict=None, label_dict=None, optional_trace=None, template_dict=None,
             scatter_plot=True,
             single_figure=False, skip_list=None, draw_list=None, thres=None, ms=1, outlier_thres=None, sharex=None,
             facecolor='lightgray', legend=True):
    """
    Plots instantaneous frequency from a given dictionary of the clusters and corresponding time, frequency lists

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
    if color_dict is None:
        keys = [key for key in list(spike_freq_dict) if (key in draw_list) and (key not in skip_list)]
        keys.sort()
        color_dict = setGoodColors(keys)
    elif (color_dict is 'single_color') or (color_dict is 'k'):
        color_dict = {key: 'k' for key in spike_freq_dict.keys()}
    elif color_dict is 'jet':
        if not scatter_plot:
            warnings.wanr('time coloured function only works with scatter plots\nSetting to scatter plot')
            scatter_plot = True
        # N = 21
        cmap = plt.cm.jet
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        data_len = spike_freq_dict[list(spike_freq_dict)[0]][1].shape[0]
        color_dict = {key: cmap(np.linspace(0, 1, data_len)) for key in spike_freq_dict.keys()}

    if label_dict is not None:
        color_dict = {key: color_dict[mapped_key] for key, mapped_key in label_dict.items()}
    fig = plt.figure(figsize=(12, 6))
    fig.tight_layout()
    ax_list = []
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

            if label_dict is None:
                label = key
            else:
                label = label_dict[key]
            if template_dict is not None:
                try:

                    if len(template_dict[key]['channels']) == 1:
                        if template_dict[key]['channels'][0] == -1:
                            label += ': all'
                        else:
                            label += ': ch ' + str(template_dict[key]['channels'][0])
                    else:
                        label += ': chs:'
                        for chan in template_dict[key]['channels']:
                            label += ' ' + str(chan)
                except:
                    pass

            data0 = items[0][mask]
            data1 = items[1][mask]
            if outlier_thres is not None:
                data0 = data0[~is_outlier(data1, outlier_thres)]
                data1 = data1[~is_outlier(data1, outlier_thres)]

            if sharex is not None:
                ax = fig.add_subplot(j, 1, i, sharex=sharex, label=i)
            elif i == 1:
                ax = fig.add_subplot(j, 1, i, label=i)
            else:
                ax = fig.add_subplot(j, 1, i, sharex=ax, label=i)

            if scatter_plot:
                ax.scatter(data0, data1, color=color_dict[key], label=label, s=ms)
            else:

                if (type(color_dict[key]) is str) or (type(color_dict[key]) is tuple):
                    ax.plot(data0, data1, color=color_dict[key], label=label, ms=ms)
                elif type(color_dict[key]) is list:
                    ax.plot(data0, data1, color=color_dict[key][0], label=label, ms=ms)

            # ax.legend()
            ax.grid(linestyle='dotted')
            removeTicksFromAxis(ax, 'x')
            ax.set_facecolor(facecolor)
            if legend:
                ax.legend()
            if not single_figure:
                i += 1
            ax_list.append(ax)

    if optional_trace is not None:
        if single_figure:
            i += 1
        if sharex is not None:
            ax = fig.add_subplot(j, 1, i, sharex=sharex, label=i)
        elif i == 1:
            ax = fig.add_subplot(j, 1, i, label=i)
        else:
            ax = fig.add_subplot(j, 1, i, sharex=ax, label=i)

        ax.plot(optional_trace[0], optional_trace[1], color='k', lw=1)
        ax.grid(linestyle='dotted')

        ax_list.append(ax)

    #    removeTicksFromAxis(ax, 'y')
    showTicksFromAxis(ax, 'x')
    return fig, ax_list


def plotFreqByNerve(spike_freq_dict, color_dict=None, label_dict=None, nerve_unit_dict=None,
                    scatter_plot=True, draw_list=None, thres=None, ms=1, outlier_thres=None, sharex=None, De3=None,
                    facecolor='lightgray', legend=True):
    """
    Plots instantaneous frequency from a given dictionary of the clusters and corresponding time, frequency lists

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

    if color_dict is None:
        keys = [key for key in list(spike_freq_dict) if (key in draw_list)]
        keys.sort()
        color_dict = setGoodColors(keys)
    elif (color_dict is 'single_color') or (color_dict is 'k'):
        color_dict = {key: 'k' for key in spike_freq_dict.keys()}
    elif color_dict is 'jet':
        if not scatter_plot:
            warnings.warn('time coloured function only works with scatter plots\nSetting to scatter plot')
            scatter_plot = True
        # N = 21
        cmap = plt.cm.jet
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        data_len = spike_freq_dict[list(spike_freq_dict)[0]][1].shape[0]
        color_dict = {key: cmap(np.linspace(0, 1, data_len)) for key in spike_freq_dict.keys()}

    if label_dict is not None:
        color_dict = {key: color_dict[mapped_key] for key, mapped_key in label_dict.items()}

    fig_dict = {}

    for nerve, units in nerve_unit_dict.items():

        unit_list = units[:, 0].astype(int)

        i = 1
        if draw_list is not None:
            j = min(len([elem for elem in unit_list if elem in draw_list]), len(list(spike_freq_dict)))
        else:
            j = unit_list.shape[0]

        fig = plt.figure(figsize=(12, 6))
        fig.tight_layout()
        ax_list = []

        for unit in unit_list:

            if (unit in draw_list) and (unit in list(spike_freq_dict)):

                if thres is None:
                    mask = [True] * len(spike_freq_dict[unit][1])
                else:
                    mask = (spike_freq_dict[unit][1] > thres[0]) & (spike_freq_dict[unit][1] < thres[1])

                if label_dict is None:
                    label = str(unit)
                else:
                    label = str(label_dict[unit])
                if De3 is not None and int(label) == De3:
                    label = 'De3'
                data0 = spike_freq_dict[unit][0][mask]
                data1 = spike_freq_dict[unit][1][mask]
                if outlier_thres is not None:
                    data0 = data0[~is_outlier(data1, outlier_thres)]
                    data1 = data1[~is_outlier(data1, outlier_thres)]

                if sharex is not None:
                    ax = fig.add_subplot(j, 1, i, sharex=sharex, label=i)
                elif i == 1:
                    ax = fig.add_subplot(j, 1, i, label=i)
                else:
                    ax = fig.add_subplot(j, 1, i, sharex=ax, label=i)

                if scatter_plot:
                    ax.scatter(data0, data1, color=color_dict[unit], label=label, s=ms)
                else:

                    if (type(color_dict[unit]) is str) or (type(color_dict[unit]) is tuple):
                        ax.plot(data0, data1, color=color_dict[unit], label=label, ms=ms)
                    elif type(color_dict[unit]) is list:
                        ax.plot(data0, data1, color=color_dict[unit][0], label=label, ms=ms)

                ax.grid(linestyle='dotted')
                removeTicksFromAxis(ax, 'x')
                ax.set_facecolor(facecolor)
                if legend:
                    ax.legend()

                i += 1

                ax_list.append(ax)

        fig.suptitle(nerve)

        ax.grid(linestyle='dotted')
        showTicksFromAxis(ax, 'x')
        fig_dict[nerve] = (fig, ax_list)

    return fig_dict


def plotCorrByNerve(corr_dict, color_dict=None, label_dict=None, nerve_unit_dict=None,
                    scatter_plot=True, draw_list=None, thres=None, ms=1, outlier_thres=None, sharex=None, De3=None,
                    burst_info=None, facecolor='lightgray', legend=True):
    """
    Plots instantaneous frequency from a given dictionary of the clusters and corresponding time, frequency lists

    Parameters
    ----------
    corr_dict: a dictionary of the different clusters pointing to a list of two np arrays(time, freq)
    color_dict: a dictionary of clusters containing its corresponding color sequence
    optional_trace: list of time, trace desired to add to the graph (for example [time_vector, NS neuron trace])

    Returns
    -------
    Nothing is returned, the function is used for its side effect: a
    plot is generated.
    """

    if color_dict is None:
        keys = [key for key in list(corr_dict) if (key in draw_list)]
        keys.sort()
        color_dict = setGoodColors(keys)
    elif (color_dict is 'single_color') or (color_dict is 'k'):
        color_dict = {key: 'k' for key in corr_dict.keys()}
    elif color_dict is 'jet':
        if not scatter_plot:
            warnings.warn('time coloured function only works with scatter plots\nSetting to scatter plot')
            scatter_plot = True
        # N = 21
        cmap = plt.cm.jet
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        data_len = corr_dict[list(corr_dict)[0]][1].shape[0]
        color_dict = {key: cmap(np.linspace(0, 1, data_len)) for key in corr_dict.keys()}

    if label_dict is not None:
        color_dict = {key: color_dict[mapped_key] for key, mapped_key in label_dict.items()}

    fig_dict = {}

    for nerve, units in nerve_unit_dict.items():
        unit_list = units[:, 0].astype(int)

        i = 1

        j = len([elem for elem in unit_list if elem in list(corr_dict)])

        if j > 0:
            if burst_info is not None:
                j += 2

            fig = plt.figure(figsize=(12, 6))
            fig.tight_layout()
            ax_list = []

            for unit in unit_list:
                if unit in list(corr_dict):
                    if thres is None:
                        mask = [True] * len(corr_dict[unit][1])
                    else:
                        mask = (corr_dict[unit][1] > thres[0]) & (corr_dict[unit][1] < thres[1])

                    if label_dict is None:
                        label = str(unit)
                    else:
                        label = str(label_dict[unit])
                    if De3 is not None and int(label) == De3:
                        label = 'De3'
                    data0 = corr_dict[unit][0][mask]
                    data1 = corr_dict[unit][1][mask]
                    if outlier_thres is not None:
                        data0 = data0[~is_outlier(data1, outlier_thres)]
                        data1 = data1[~is_outlier(data1, outlier_thres)]

                    if sharex is not None:
                        ax = fig.add_subplot(j, 1, i, sharex=sharex, label=i)
                    elif i == 1:
                        ax = fig.add_subplot(j, 1, i, label=i)
                    else:
                        ax = fig.add_subplot(j, 1, i, sharex=ax, label=i)

                    if scatter_plot:
                        ax.scatter(data0, data1, color=color_dict[unit], label=label, s=ms)
                    else:

                        if (type(color_dict[unit]) is str) or (type(color_dict[unit]) is tuple):
                            ax.plot(data0, data1, color=color_dict[unit], label=label, ms=ms)
                        elif type(color_dict[unit]) is list:
                            ax.plot(data0, data1, color=color_dict[unit][0], label=label, ms=ms)

                    ax.grid(linestyle='dotted')
                    removeTicksFromAxis(ax, 'x')
                    ax.set_facecolor(facecolor)
                    if legend:
                        ax.legend()

                    i += 1

                    ax_list.append(ax)
            if burst_info is not None:
                if sharex is not None:
                    ax = fig.add_subplot(j, 1, i, sharex=sharex, label=i)
                    ax1 = fig.add_subplot(j, 1, i + 1, sharex=sharex, label=i)
                else:
                    ax = fig.add_subplot(j, 1, i, sharex=ax, label=i)
                    ax1 = fig.add_subplot(j, 1, i + 1, sharex=ax, label=i)

                ax.scatter(burst_info["burst median time"][:-1], burst_info["cycle period"], label='DP cycle period',
                           s=ms)
                ax1.scatter(burst_info["burst median time"][:-1], burst_info["burst duty cycle"], label='DP duty cycle',
                            s=ms)
                ax_list.append(ax)
                ax_list.append(ax1)
                if legend:
                    ax.legend()
                    ax1.legend()

                ax.grid(linestyle='dotted')
                ax1.grid(linestyle='dotted')
                removeTicksFromAxis(ax, 'x')
                # showTicksFromAxis(ax1, 'x')
            fig.suptitle(nerve)
            showTicksFromAxis(ax_list[-1], 'x')
            fig_dict[nerve] = (fig, ax_list)

    return fig_dict


def removeTicksFromAxis(ax, axis='y'):
    """
    removes tick from passed axes and axis.
    Parameters
    ----------
    ax maptplotlib.axes.AxesSubplot:
        axes from where the ticks will be removed.
    axis str:
        axis name from where the ticks will be removed: should be either 'x' or 'y'.

    Returns
    -------
    Nothing.
    """
    axis = getattr(ax, axis + 'axis')
    for tic in axis.get_major_ticks():
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)

        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)

        # tic.label1On = tic.label2On = False
        # tic.tick1On = tic.tick2On = False


def showTicksFromAxis(ax, axis='y'):
    """
    removes tick from passed axes and axis.
    Parameters
    ----------
    ax maptplotlib.axes.AxesSubplot:
        axes to where the ticks will be added.
    axis str:
        axis name to where the ticks will be added: should be either 'x' or 'y'.

    Returns
    -------
    Nothing.
    """
    axis = getattr(ax, axis + 'axis')
    for tic in axis.get_major_ticks():
        tic.label1.set_visible(True)
        tic.tick1line.set_visible(True)


def saveSpikeResults(filename, json_dict):
    """
    Saves a json dict in pickled format
    Parameters
    ----------
    filename str:
        data filename.
    json_dict dict:
        Dictionary to be stored

    Returns
    -------

    """
    os.path.splitext(filename)[0] + '.pklspikes'
    print('Saved in %s' % filename)
    with open(filename, 'wb') as pfile:
        pickle.dump(json_dict, pfile)


def loadSpikeResults(filename):
    """
    Loads a pklspikes file saved in pickled format. It usually stores information of a PyLeech.Utils.unitInfo.UnitInfo
    instance.
    Parameters
    ----------
    filename str:
        path to file

    Returns
    -------
    dictionary of where keys are usually attributes of a UnitInfo instace.
    """
    assert os.path.splitext(filename)[1] == '.pklspikes', 'Wrong file extension, I need a .pklspikes file'

    with open(filename, 'rb') as pfile:
        results = pickle.load(pfile)

    return results


def getBursts(times, spike_max_dist=0.7, min_spike_no=10, min_spike_per_sec=10):
    """
    Returns list of bursts if there are any given the conditions passed through the parameter, or empty if it finds none.
    Parameters
    ----------
    times array:
        Spike times of a single unit.
    spike_max_dist float:
        Maximum time inverval between two spikes for them to be considered part of the same burst.
    min_spike_no int:
        Minimum number of spikes for a set of spikes to be considered a burst.
    min_spike_per_sec:
        Minimum spike rate in burst for it to be considered as such.

    Returns
    -------
    bursts_list list:
        list of lists in which each list contains the spike times of the correspoding burst.

    """
    # if there are no spikes, just returns an empty list
    if len(times) == 0:
        return []

    burst_list = []
    spike_times = []

    for i in range(len(times) - 1):

        spike_times.append(times[i])
        ## it appends spikes times to the burst until the next one appears after `spike_max_dist`
        if (times[i + 1] - times[i]) > spike_max_dist:

            # Uses checkBurst function to check whether the conditions are satisfied
            if checkBurst(spike_times, min_spike_no, min_spike_per_sec):
                burst_list.append(spike_times)
            # if not, it goes through checking a subset of bursts (Usually done in case the fire rate is below the
            # threshold and removing the last few spikes satifies every condition
            else:
                new_bursts = checkBurstWithin(spike_times, min_spike_no, min_spike_per_sec)
                burst_list.extend(new_bursts)

            spike_times = []

    ## just in case the last one is considered a "burst" and was forgotten because it wasn't part of the previous burst
    spike_times.append(times[-1])

    ##checking the last one
    if checkBurst(spike_times, min_spike_no, min_spike_per_sec):
        burst_list.append(spike_times)
    else:
        new_bursts = checkBurstWithin(spike_times, min_spike_no, min_spike_per_sec)
        burst_list.extend(new_bursts)

    return burst_list


def checkBurst(spike_times, min_spike_no, min_spike_per_sec):
    """
    check whether the list of spikes can be considered a burst given the conditions
    Parameters
    ----------
    spike_times list:
        List of burst times to be checked
    min_spike_no int:
        Minimum number of spikes for a set of spikes to be considered a burst.
    min_spike_per_sec:
        Minimum spike rate in burst for it to be considered as such.

    Returns
    -------
    Boolean value
    """
    if len(spike_times) < min_spike_no:
        return False
    elif len(spike_times) == 1:
        return True
    elif len(spike_times) / (spike_times[-1] - spike_times[0]) > min_spike_per_sec:
        return True
    else:
        return False


def checkBurstWithin(spike_times, min_spike_no, min_spike_per_sec):
    """
    check whether there is a subset of spikes that can be considered a burst within the spike list. Usually run when
    checkBurst function return False
    Parameters
    ----------
    spike_times list:
        List of burst times to be checked
    min_spike_no int:
        Minimum number of spikes for a set of spikes to be considered a burst.
    min_spike_per_sec:
        Minimum spike rate in burst for it to be considered as such.

    Returns
    -------
    bursts list:
        list of spikes times where a burst was found. If no burst is found it returns an empty list
    """
    bursts = []
    if len(spike_times) <= min_spike_no:
        return []

    i = 0
    # The subset of spikes being evaluated starts from the first spike
    while i < len(spike_times) - min_spike_no:

        for j in range(i + min_spike_no, len(spike_times)):
            #check whether the current set can be considered a burst BUT not when one more spike is added.
            if not checkBurst(spike_times[i:j + 1],
                              min_spike_no,
                              min_spike_per_sec) and checkBurst(spike_times[i:j],
                                                                min_spike_no,
                                                                min_spike_per_sec):

                bursts.append(spike_times[i:j])
                i = j
            # IF thats not the case and it reached the end, this line checks a last time
            elif (j == len(spike_times) - 1) and checkBurst(spike_times[i: len(spike_times) + 1], min_spike_no,
                                                            min_spike_per_sec):
                bursts.append(spike_times[i:j + 1])

                i = j

            # If it found a burst, it will just break this iteration and start again from the first non checked spike.
            if i == j:
                break
        i += 1

    return bursts


def getInterBurstInterval(burst_list, no_burst=2):
    """
    Gets each burst time interval as the times where the current and following burst starts
    Parameters
    ----------
    burst_list list of lists:
        A list where each element is a bursts containing a list with its corresponding spikes
    no_burst int:
        Number of bursts to be used for interval calculation. If 1, the result is the InterburstInterval. If larger
        than 1, it simply returns the time interval between the burst and `no_burst` bursts ahead

    Returns
    -------
    burst_int ndarray:
        array of shape (number of bursts-`no_bursts`, 2) where the first column has the burst start time and the second
        one the burst end time (next burst start time)

    """
    burst_int = np.zeros((len(burst_list) - no_burst, 2))

    for i in range(len(burst_list) - no_burst):
        burst_int[i, 0] = burst_list[i][0]
        burst_int[i, 1] = burst_list[i + no_burst][0]
    return burst_int


def generateFilenameFromList(filename):
    """
    Generates a new filename from a list of filenames, where each filename is of the type'YYYY_MM_DD_NNNN' and every
    filename was generated on the same date
    Parameters
    ----------
    filename list:
        list of filenames

    Returns
    -------
    new_filename str:
        string in which each NNNN of each filename was append to it.
    """
    new_filename = os.path.basename(filename[0]).split('_')
    new_filename = "_".join(new_filename[:-1])

    for fn in filename:
        num = os.path.splitext(fn.split("_")[-1])[0]
        new_filename += '_' + num

    return new_filename


def generatePklFilename(filename, folder):
    """
    Generates a filename for a .pkl file given a path and a filename.
    Parameters
    ----------
    filename list,str:
        Filename from where data was extracted. Usually a '.abf' file. It can be a list if the data was stored in several
        files.
    folder
        Path to where the file will be stored.
    Returns
    -------
    filename str:
        Return filename with its relative path.
    """
    if folder is None:
        folder = 'RegistrosDP_PP/'
    else:
        folder = folder + '/'
    if type(filename) is list:
        filename = generateFilenameFromList(filename)

    else:
        filename = os.path.basename(os.path.splitext(filename)[0])

    filename = folder + filename
    return filename


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Taken from here https://stackoverflow.com/a/47232942
    Generates a categorical colormap to get the most different set of possible colors. It tries to generate as many
    distiguinshible colors as it can.

    Parameters
    ----------
    nc int:
        Number of color categories.
    nsc int:
        Number of color subcategories.
    cmap str:
        Original cmap from where the new cmap will be generated.
    continuous bool:
        Whether to generate the categories in a continuous fashion or a discrete one.

    Returns
    -------
    cmap func:
        cmap with number of qualitative different required colors.

    """
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


def setGoodColors(good_clusters, num_col=10):
    """
    Generates a set of different colors for the `good_clusters` list
    Parameters
    ----------
    good_clusters list:
        List of different clusters that will be drawn in future figures.
    num_col int:
        Number of colors categories to be used.
    Returns
    -------
    cluster_color dict:
        Dictionary of the type {key: RGBA tuple} where key is the cluster associated with that corresponding color.
    """
    cmap = categorical_cmap(num_col, math.ceil(len(good_clusters) / num_col))
    j = 0
    cluster_color = {}
    for i in good_clusters:
        cluster_color.update({i: cmap(j)})
        j += 1
    return cluster_color


def spike_freq_dictToDataFrame(spike_freq_dict):
    """
    Adapts the spike_freq_dict into a DataFrame with three columns: neuron number, spike time, instant firing rate
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is
        its corrresponding neuron number.

    Returns
    -------
    df pandas.DataFrame:
        Dataframe with the same information as the `spike_freq_dict` but condensed into three columns.
    """
    for key in spike_freq_dict.keys():

        Tsfd = np.array(spike_freq_dict[key]).transpose()

        keyarray = np.array([key] * Tsfd.shape[0])

        temp_array = np.column_stack((keyarray, Tsfd))
        try:
            spike_freq_array = np.vstack((spike_freq_array, temp_array))
        except:
            spike_freq_array = temp_array

    df = pd.DataFrame(spike_freq_array, columns=['neuron', 'time', 'IFR'])

    dtype = {'neuron': np.int32, 'time': np.float64, 'IFR': np.float64}
    for k, v in dtype.items():
        df[k] = df[k].astype(v)

    return df


def binned_sfd_to_dict_array(binned_spike_freq_dict, time_interval=None, good_neurons=None):
    """
    Converts the items in a `binned_spike_freq_dict` wich are two list within a list into a numpy array of
    shape (2, num_bins)

    Parameters
    ----------
    binned_spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is
        its corrresponding neuron number.
    time_interval float, list:
        If it's a list, it will return the dictionary of arrays cut into those intervals. If its a number it will range
        from 0 up to this time value at that time.
    good_neurons list, None:
        If None, it will return the dictionary with every key. If not, it will return a dictionary containing only the
        passed units.

    Returns
    -------
    new_sfd dict:
        New spike frequency dict pointing to ndarrays insted of lists.
    """
    new_sfd = {}
    first_key = list(binned_spike_freq_dict)[0]
    if time_interval is not None:
        if type(time_interval[0]) is list:

            mask = np.zeros(len(binned_spike_freq_dict[first_key][0]), dtype=bool)

            for interval in time_interval:
                idxs = np.where((binned_spike_freq_dict[first_key][0] > interval[0]) & (binned_spike_freq_dict[first_key][0] < interval[1]))[0]
                mask[idxs] = 1

        else:

            mask = (binned_spike_freq_dict[first_key][0] > time_interval[0]) & (binned_spike_freq_dict[first_key][0] < time_interval[1])
    else:

        mask = np.ones(binned_spike_freq_dict[first_key][0].shape[0], dtype=bool)

    for key, items in binned_spike_freq_dict.items():
        if good_neurons is None or key in good_neurons:
            new_sfd[key] = np.array([items[0][mask], items[1][mask]])

    return new_sfd


def processSpikeFreqDict(spike_freq_dict, step, num=None, outlier_threshold=3.5, selected_neurons=None,
                         time_length=None,
                         time_interval=None, counting=False, freq_threshold=None):
    """
    Returns a full processing of a raw `spike_freq_dict`, integrating the necessary functions to to return a processed
    spike frequenncy array of the type {key: ndarray(binned times, spike firing rate)}

    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is its corrresponding
        neuron number.
    step float:
        time step to be used for the binning interval
    num Nonetype, int:
        Number of bins to be used for binning. Alternative to step. Use only when a specific number of bins are needed.
        It is best to avoid using it.
    outlier_threshold float:
        Outlier threshold in standard deviation units to be used in outlier removal before digitalization.
    selected_neurons Nonetype, list:
        If its None, every unit will be processed. If it's a list, then only data for those units will be returned
    time_length Nonetype, float:
        Original signal duration in seconds. If None, the function will use largest time found in the spike_freq_dict.
    time_interval float, list:
        If it's a list, it will return the dictionary of arrays cut into those intervals. If its a number it will range
        from 0 up to this time value at that time.
    counting bool:
         If set to True, the function will return spike count by bin. If False, it will return mean frequency.
    freq_threshold float:
        frequency threshold to be used in outlier removal before digitalization. It usually won't be needed as the
        the outlier removal will be done by using the outlier threshold.

    Returns
    -------
    new_sfd dict:
        A processed spike frequency dict pointing to ndarrays insted of lists.
    """

    if num is None:
        num = int(time_length / step)
        time_length -= time_length % step

    new_sfd = removeOutliers(spike_freq_dict, outlier_threshold=outlier_threshold)

    new_sfd = digitizeSpikeFreqs(new_sfd, num=num, time_length=time_length, counting=counting,
                                 freq_threshold=freq_threshold)

    return binned_sfd_to_dict_array(new_sfd, time_interval=time_interval, good_neurons=selected_neurons)


def cutSpikeFreqDict(spike_freq_dict, time_interval, outlier_threshold=3.5):
    """
    Cuts a spike_freq_dict into an specified time inverval
    Parameters
    ----------
    spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is its corrresponding
        neuron number.
    time_interval list:
        Interval into which the `spike_freq_dict` will be cut
    outlier_threshold float:
        Outlier threshold in standard deviation units to be used in outlier removal before digitalization.

    Returns
    -------
    new_sfd dict:
        A new spike_freq_dict containing spikes only in the selected interval.
    """
    new_sfd = {}
    for key, items in spike_freq_dict.items():
        mask = (items[0] > time_interval[0]) & (items[0] < time_interval[1])
        outlier_mask = ~is_outlier(items[1], thresh=outlier_threshold)
        new_sfd[key] = [items[0][mask & outlier_mask], items[1][mask & outlier_mask]]
    return new_sfd


def saveSpikeFreqDictToBinnedMat(spike_freq_dict, step, filename, num=None, outlier_threshold=3.5,
                                 selected_neurons=None, time_length=None,
                                 time_interval=None, counting=False, freq_threshold=None):

    """
    Processes and stores a spike_freq_dict into a condensed matrix of shape (time steps, number of units)
    Parameters
    ----------
     spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is its corrresponding
        neuron number.
    step float:
        Time step to be used for the binning interval
    filename str:
        Path into which the .csv will be saved.
    num Nonetype, int:
        Number of bins to be used for binning. Alternative to step. Use only when a specific number of bins are needed.
        It is best to avoid using it.
    outlier_threshold float:
        Outlier threshold in standard deviation units to be used in outlier removal before digitalization.
    selected_neurons Nonetype, list:
        If its None, every unit will be processed. If it's a list, then only data for those units will be returned.
    time_length Nonetype, float:
        Original signal duration in seconds. If None, the function will use largest time found in the spike_freq_dict.
    time_interval float, list:
        If it's a list, it will return the dictionary of arrays cut into those intervals. If its a number it will range
        from 0 up to this time value at that time.
    counting bool:
         If set to True, the function will return spike count by bin. If False, it will return mean frequency.
    freq_threshold float:
        frequency threshold to be used in outlier removal before digitalization. It usually won't be needed as the
        the outlier removal will be done by using the outlier threshold.

    Returns
    -------
    Nothing. Only stores data into a .csv
    """
    binned_sfd = processSpikeFreqDict(spike_freq_dict, step=step, num=num, outlier_threshold=outlier_threshold,
                                      selected_neurons=selected_neurons, time_length=time_length,
                                      time_interval=time_interval, counting=counting, freq_threshold=freq_threshold)
    burst_array = []
    for key, items in binned_sfd.items():
        burst_array.append(items[1])
    burst_array = np.array(burst_array).T
    filename = os.path.splitext(filename)[0]
    pd.DataFrame(burst_array, columns=list(binned_sfd)).to_csv(filename + '.csv', index=False)


def saveBinnedSfdToBinnedMat(binned_sfd, filename):
    """
    Stores a processed binned spike frequency dictionary into a csv table
    Parameters
    ----------
    binned_spike_freq_dict dict:
        Dictionary of the type {key: [spike times, spike instantaneous frequency]} where 'key' is
        its corrresponding neuron number.
    filename str:
        Path into which the .csv will be saved.

    Returns
    -------
    Nothing. Only stores data into a .csv
    """
    burst_array = []
    for key, items in binned_sfd.items():
        burst_array.append(items[1])
    burst_array = np.array(burst_array).T
    filename = os.path.splitext(filename)[0] + '.csv'
    print('Saving in %s' % filename)
    pd.DataFrame(burst_array, columns=list(binned_sfd)).to_csv(filename, index=False)


def generateBurstSegmentsFromManyNeurons(smoothed_sfd, intervals_dict):
    burst_list = []
    target_neuron = []
    max_len = 0
    for key, items in smoothed_sfd.items():
        for i, j in intervals_dict[key]:
            segment = items[1][i:j]
            burst_list.append(segment)
            if len(segment) > max_len:
                max_len = len(segment)
            target_neuron.append(key)

    return np.array([spsig.resample(burst * len(burst) / max_len, max_len) for burst in burst_list]), np.array(
        target_neuron)


def resampleSegmentList(segment_list):
    max_len = 0
    for segment in segment_list:
        if len(segment) > max_len:
            max_len = len(segment)

    return np.array([spsig.resample(segment * segment.shape[0] / max_len, max_len) for segment in segment_list])


def segment_listToDataFrame(segment_list):
    for i in range(len(segment_list)):
        for key, items in segment_list[i].items():

            keyarray = np.array([key] * items.shape[0])

            temp_array = np.column_stack((keyarray, items))
            try:
                spike_freq_array = np.vstack((spike_freq_array, temp_array))
            except:
                spike_freq_array = temp_array

        segment_array = np.array([i] * spike_freq_array.shape[0])

        full_segment_array = np.column_stack((segment_array, spike_freq_array))

        try:
            final_array = np.vstack((final_array, full_segment_array))

        except:
            final_array = full_segment_array

    df = pd.DataFrame(final_array, columns=['cycle_no', 'neuron', 'time', 'IFR'])

    dtype = {'cycle_no': np.int32, 'neuron': np.int32, 'time': np.float64, 'IFR': np.float64}
    for k, v in dtype.items():
        df[k] = df[k].astype(v)

    return df


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
                           # spike_ind[(spike_ind > interval[0]) & (spike_ind < interval[1])]- interval[0],
                           spike_ind[(spike_ind > interval[0]) & (spike_ind < interval[1])],
                           channels=channels,
                           peak_color=cluster_colors[key],
                           label=label,
                           ax_list=ax_list)
    hdl = []
    lbl = []
    if intracel_signal is not None:
        ax_list[-1].plot(time[interval[0]:interval[1]], intracel_signal[interval[0]:interval[1]], color='k')
    for ax in ax_list:
        ax.grid(linestyle='dotted')
        removeTicksFromAxis(ax, 'y')
        removeTicksFromAxis(ax, 'x')
        handle, label = ax.get_legend_handles_labels()
        hdl.extend(handle)
        lbl.extend(label)

    showTicksFromAxis(ax_list[-1], 'x')

    if intracel_signal is not None:
        showTicksFromAxis(ax_list[-1], 'y')
    if legend:
        # ax_list[0].legend(hdl, lbl, loc='upper right')
        fig.legend(hdl, lbl, loc='upper right')
        # plt.legend(loc='upper right')

    return fig, ax_list


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


def assertMissingData(needed_keys):
    try:
        assert len(needed_keys) < 2, 'This data is missing:'
    except AssertionError as e:
        e.args += tuple([str(mk) for mk in needed_keys])
        raise


def resampleArrayList(arr_list1, arr_list2):
    l1 = len(arr_list1[0])
    l2 = len(arr_list2[0])

    if l1 > l2:
        resampled_arr_list2 = []
        resampled_arr_list1 = arr_list1
        for array in arr_list2:
            resampled_arr_list2.append(spsig.resample(array, l1))
    elif l2 > l1:
        resampled_arr_list2 = arr_list2
        resampled_arr_list1 = []
        for array in arr_list1:
            resampled_arr_list1.append(spsig.resample(array, l2))
    else:
        resampled_arr_list1 = arr_list1
        resampled_arr_list2 = arr_list2

    return resampled_arr_list1, resampled_arr_list2


def generateGaussianKernel(sigma, time_range, dt_step, half_gaussian=False):
    sigma = sigma
    time_range = time_range
    x_range = np.arange(-time_range, time_range + dt_step, dt_step)
    if half_gaussian:
        x_range = x_range[int(np.floor(x_range.shape[0]/2)):]

    gaussian = np.exp(-(x_range / sigma) ** 2)
    gaussian /= gaussian.sum()
    return gaussian


def smoothBinnedSpikeFreqDict(binned_sfd, sigma, time_range, dt_step, half_gaussian=False):
    smoothed_sfd = {}
    kernel = generateGaussianKernel(sigma=sigma, time_range=time_range, dt_step=dt_step, half_gaussian=half_gaussian)

    for key, items in binned_sfd.items():
        smoothed_sfd[key] = np.array([items[0], spsig.convolve(items[1], kernel, mode='same', method='direct')])

    return smoothed_sfd

def sfdToArray(spike_freq_dict):
    burst_array = []
    for key, items in spike_freq_dict.items():
        burst_array.append(spike_freq_dict[key][1])

    return np.array(burst_array).T


def processed_sfd_to_array(processed_sfd):
    burst_array = []
    for key, items in processed_sfd.items():
        burst_array.append(processed_sfd[key][1])
    return np.array(burst_array).T
