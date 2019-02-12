import math
import os.path
import pickle as pickle

import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as spsig
from PyLeech.Utils import spikeUtils as spikeUtils
from PyLeech.Utils.constants import nan
from copy import deepcopy

def getInstFreq(time, spike_dict, fs):
    spike_freq_dict = {}
    for key, items in spike_dict.items():
        spike_freq_dict.update({key: [time[items[:-1]], np.reciprocal(np.diff(items) / fs)]})

    return spike_freq_dict

def getSpikeTimesDict(spike_freq_dict):
    return {key: items[0] for key, items in spike_freq_dict.items()}

def binSpikesFreqs(spike_freq_dict, time_length, step, full_binning=False):
    rg = np.arange(0, time_length, step)
    binned_spike_freq_dict = {}
    for key, item in spike_freq_dict.items():

        bins = spikeUtils.binXYLists(rg, item[0], item[1], get_median=True, std_as_err=True)
        if full_binning:
            freqs = np.zeros(len(rg))
            freqs[np.in1d(rg, bins[0])] = bins[1]
            binned_spike_freq_dict.update({key: [rg, freqs]})
        else:
            binned_spike_freq_dict.update({key: [np.array(bins[0]), np.array(bins[1])]})
    return binned_spike_freq_dict

def digitizeSpikeFreqs(spike_freq_dict, time_length, step, count=False, freq_threshold=150):
    """ Always returns full binning
    If count is set to True, it will return spike count by bin.
    If false, it will return mean freq"""
    rg = np.arange(0, time_length, step)
    binned_spike_freq_dict = {}
    freqs = np.zeros(len(rg) - 1)
    for key, item in spike_freq_dict.items():
        freqs[:] = 0
        times = np.array(item[0])
        freqs_arr = np.array(item[1])

        times = times[freqs_arr<freq_threshold]
        freqs_arr = freqs_arr[freqs_arr<freq_threshold]
        digitalization = np.digitize(times, rg)

        uid = np.unique(digitalization)

        for i in uid:
            if not count:
                freqs[i] = np.mean(freqs_arr[np.where(digitalization==i)[0]])
            else:
                freqs[i] = np.where(digitalization == i)[0].size
        binned_spike_freq_dict[key] = [rg[:-1], deepcopy(freqs)]

    return binned_spike_freq_dict

def binSpikeFromISIs(spike_freq_dict, time_length, step, full_binning=False):
    rg = np.arange(0, time_length, step)
    binned_spike_freq_dict = {}
    for key, item in spike_freq_dict.items():

        bins = spikeUtils.binXYLists(rg, item[0][:-1], np.diff(item[0]), get_median=True, std_as_err=True)
        if full_binning:
            freqs = np.zeros(len(rg))
            freqs[np.in1d(rg, bins[0])] = np.reciprocal(bins[1])
            binned_spike_freq_dict.update({key: [rg, freqs]})
        else:
            binned_spike_freq_dict.update({key: [np.array(bins[0]), np.reciprocal(bins[1])]})
    return binned_spike_freq_dict


def plotDataPredictionAndResult(time, data, pred):
    res = data - pred
    for i in range(len(data)):
        fig = plt.figure(figsize=(16, 8))
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


def plotFreq(spike_freq_dict, color_dict=None, optional_trace=None, template_dict=None, scatter_plot=True,
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

    if color_dict is None:
        keys = [key for key in list(spike_freq_dict) if (key in draw_list) and (key not in skip_list)]
        keys.sort()
        color_dict = setGoodColors(keys)

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
    #    removeTicksFromAxis(ax, 'y')
    showTicksFromAxis(ax, 'x')
    return fig


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


def generateFilenameFromList(filename):
    new_filename = os.path.basename(filename[0]).split('_')
    new_filename = "_".join(new_filename[:-1])

    for fn in filename:
        num = os.path.splitext(fn.split("_")[-1])[0]
        new_filename += '_' + num

    return new_filename


def generatePklFilename(filename, folder):
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
    cmap = categorical_cmap(num_col, math.ceil(len(good_clusters) / num_col))
    j = 0
    cluster_color = {}
    for i in good_clusters:
        cluster_color.update({i: cmap(j)})
        j += 1
    return cluster_color


def spike_freq_dictToArray(sfd):
    for key in sfd.keys():
        if type(sfd[key]) is list:
            Tsfd = np.array(sfd[key]).transpose()
        else:
            Tsfd = np.array(sfd[key]).transpose()

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
                           spike_ind[(spike_ind > interval[0]) & (spike_ind < interval[1])] - interval[0],
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
    resampled_arr_list1 = []
    resampled_arr_list2 = []

    if l1>l2:
        resampled_arr_list2 = []
        resampled_arr_list1 = arr_list1
        for array in arr_list2:
            resampled_arr_list2.append(spsig.resample(array, l1))
    elif l2>l1:
        resampled_arr_list2 = arr_list2
        resampled_arr_list1 = []
        for array in arr_list1:
            resampled_arr_list1.append(spsig.resample(array, l2))
    else:
        resampled_arr_list1 = arr_list1
        resampled_arr_list2 = arr_list2

    return  resampled_arr_list1, resampled_arr_list2