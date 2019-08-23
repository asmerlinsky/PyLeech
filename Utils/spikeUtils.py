# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:57:59 2018

@author: Agustin Sanchez Merlinsky
"""
import numpy as np
import scipy.signal as spsig
import matplotlib as mpl


mpl.rcParams['agg.path.chunksize'] = 10000


##Conseguir la duracion, frecuencia de burst
def getBursts(indexes, fs=5000., spike_max_dist=0.7, min_spike_no=10, min_spike_per_sec=10):
    if len(indexes) == 0:
        return []
    burst_list = []
    burst_index = []
    for i in range(len(indexes) - 1):
        burst_index.append(indexes[i])
        if (indexes[i + 1] - indexes[i]) / fs > spike_max_dist:
            if check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):
                burst_list.append(burst_index)
            burst_index = []
            continue

    ## just in case the last one is considered a "burst" and was forgotten because it wasn't part of the previous burst
    if not burst_index:
        burst_index.append(indexes[-1])
        ##checking the last one
    if check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):
        burst_index.append(indexes[i + 1])
        burst_list.append(burst_index)

    return burst_list


def check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):
    if len(burst_index) < min_spike_no:
        return False
    elif len(burst_index) == 1:
        return True
    elif fs * len(burst_index) / (burst_index[-1] - burst_index[0]) > min_spike_per_sec:
        return True
    else:
        return False


from copy import deepcopy


def removeTSpikes(sig, dist=100, amp=-5, ):
    first = 200
    last = 150
    signal = deepcopy(sig)
    spikes = list(spsig.find_peaks(signal, height=amp, distance=dist)[0])
    spikes.sort(reverse=True)
    for spike in spikes:

        if spike < first:
            signal[0:spike + 50] = [signal[spike + 50]] * len(signal[0:spike + 50])
        elif spike > len(signal) - last:
            signal[spike - first:] = [signal[spike - first]] * len(signal[spike - first:])
        else:
            # del signal[spike-first:spike+last]
            signal[spike - first:spike + last] = np.zeros(first + last) * np.nan
            # np.delete(signal, [x for x in range(spike-first,spike+last)])
    # no_spike_mean = np.nanmean(signal)
    # signal = deepcopy(sig)

    # for spike in spikes:
    #    if (first < spike) and (spike < len(signal)-last):
    #        signal[spike-first:spike+last] = [no_spike_mean]*(first + last)
    return signal


def getSpikeIndexes(indexes, spike_indexes):
    spikes = []
    for i in range(len(indexes)):
        for j in range(len(spike_indexes)):
            if -200 <= (indexes[i] - spike_indexes[j]) and (indexes[i] - spike_indexes[j]) < 800:
                spikes.append(i)
                break
    return spikes


def getBaselineSideStd(signal, bins=50, spdist=100, side=0):
    """
    Gets baseline value looking on one side of the most frequent valuefrom the last pct of the segment
    First will remove spikes which may significative alter the baseline
    Also looks whether peak is positive or negative
    """
    '''converts spikes into nans'''
    no_spike_signal = removeTSpikes(signal, spdist)
    nan_filtered_signal = no_spike_signal[np.logical_not(np.isnan(no_spike_signal))]
    hist = np.histogram(nan_filtered_signal, bins=bins)
    argmax = np.argmax(hist[0])
    most_frequent = hist[1][argmax]
    if side == 0:
        rel_max = np.nanmax(no_spike_signal) - most_frequent
        rel_min = most_frequent - np.nanmin(no_spike_signal)
        if rel_max > rel_min:
            indexes = np.where(no_spike_signal < most_frequent)
            side = 1
        else:
            indexes = np.where(no_spike_signal > most_frequent)
            side = -1

    elif side == 1:
        indexes = np.where(no_spike_signal < most_frequent)
    elif side == -1:
        indexes = np.where(no_spike_signal > most_frequent)

    return np.mean(no_spike_signal[indexes]), side, np.std(no_spike_signal[indexes])


def binXYLists(bins, xvaluelist, yvaluelist, get_median=False, std_as_err=False, full_binning=False):
    np_xval = np.asarray(xvaluelist)
    np_yval = np.asarray(yvaluelist)
    bin_interval = np.abs(bins[1] - bins[0])
    filled_bins = []
    bin_mean = []
    rel_bin_max = []
    rel_bin_min = []
    for center in bins:
        upper = center + bin_interval / 2
        lower = center - bin_interval / 2

        inbin = np.where((np_xval <= upper) & (np_xval > lower))[0]
        if len(inbin) > 0:
            filled_bins.append(center)

            if not get_median:
                bmean = np.mean(np_yval[inbin])
            else:
                bmean = np.median(np_yval[inbin])

            bmax = np.max(np_yval[inbin])
            bmin = np.min(np_yval[inbin])

            bin_mean.append(bmean)
            if not std_as_err:
                rel_bin_max.append(bmax - bmean)
                rel_bin_min.append(bmean - bmin)
            else:

                std_as_err = np.std(np_yval[inbin])
                rel_bin_max.append(std_as_err)
                rel_bin_min.append(std_as_err)
        elif full_binning:
            filled_bins.append(center)
            bin_mean.append(0)
            rel_bin_max.append(0)
            rel_bin_min.append(0)
    return filled_bins, bin_mean, rel_bin_min, rel_bin_max
