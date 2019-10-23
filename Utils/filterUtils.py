# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:49:37 2018

@author: Agustin Sanchez Merlinsky
"""

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt
import warnings

def getFreqSpectrum(data, sampling_rate, nperseg=1000):
    return spsig.welch(data, fs=sampling_rate, nperseg=nperseg)

def getPowerSpectrum(data, sampling_rate, nperseg=1000):
    spectrum = list(getFreqSpectrum(data, sampling_rate, nperseg=nperseg))
    spectrum[1] = 20 * (np.log10(spectrum[1]) - np.log10(spectrum[1][0]))
    return spectrum

def getbutterPolinomials(cuttoff_freq, butt_order=8, sampling_rate=5000, btype='low'):
    assert  cuttoff_freq<(sampling_rate/2), "cutoff frequency must be < Nyquist frequency"
    return spsig.butter(butt_order, cuttoff_freq*2/sampling_rate, btype=btype)

def runButterFilter(data, cuttoff_freq, butt_order=8, sampling_rate=5000, btype='low'):
    poli_b, poli_a = getbutterPolinomials(cuttoff_freq, butt_order, sampling_rate, btype=btype)
    return spsig.filtfilt(poli_b, poli_a, data)

def plotSpectrums(data1, data2=None, sampling_rate=5000, nperseg=1000, ax=None, label1=None, label2=None):

    filt_spec1 = list(getFreqSpectrum(data1, sampling_rate, nperseg=nperseg))
    filt_spec1[1] = 20 * (np.log10(filt_spec1[1]) - np.log10(filt_spec1[1][0]))
    if ax is None:
        fig, ax = plt.subplots()
    if label1 is None:
        label1 = 'data1'

    ax.plot(filt_spec1[0], filt_spec1[1], label=label1)

    if data2 is None:
        return
    if label2 is None:
        label2 = 'data2'

    filt_spec2 = list(getFreqSpectrum(data2, sampling_rate, nperseg=nperseg))
    filt_spec2[1] = 20 * (np.log10(filt_spec2[1]) - np.log10(filt_spec2[1][0]))


    ax.plot(filt_spec2[0], filt_spec2[1], label=label2)
    ax.legend()

def getSpectralPeaks(trace, fs, lim_freq, nperseg=1000, height=5, distance=40):
    frec_step = 2 * fs/ nperseg
    distance /= frec_step
    spectrum = getPowerSpectrum(trace, fs, nperseg)
    idxs = np.where(spectrum[0]<lim_freq)
    peaks = spsig.find_peaks(spectrum[1][idxs], height=height, distance=distance)[0]
    return spectrum[0][idxs][peaks], spectrum[1][idxs][peaks]


def getFilterFromPowerSpectrum(freq, intensity):
    """ 
    Gets a few main noise frecuencies in the [35, 300] Hz range
    pass the power spectrum of some baseline data
    """
    start_index = np.where(freq >= 35)[0][0]
    filter_freqs = [freq[start_index]]
    j = 0
    for i in range(start_index, np.where(freq <= 260)[0][-1]):
        if intensity[i] > intensity[np.where(freq == filter_freqs[j])[0][0]]:
            filter_freqs[j] = freq[i]
        if freq[i] > (j + 1) * 100:
            filter_freqs.append(freq[i])
            j += 1
    return filter_freqs


def getFreqInd(freqs_array, freq):
    for i in range(len(freqs_array)):
        if freqs_array[i] >= freq:
            return i


def getNoiseFreqsFromEveryChannel(channels, sample_freq, peak_dist, max_freq=340):
    """ 
    Gets noise frecuencies for every channel in a segment.
    """
    freqs = []
    for i in range(len(channels[0, :])):
        freq_spec = getFreqSpectrum(channels[:, i], sample_freq)
        indini = getFreqInd(freq_spec[0], 35)
        indend = getFreqInd(freq_spec[0], max_freq)
        delta_f = freq_spec[0][1] - freq_spec[0][0]
        dist = int(peak_dist / delta_f)
        indexes = spsig.argrelmax(freq_spec[1][indini:indend], order=dist)[0] + indini
        freqs.append(freq_spec[0][indexes])
        # freqs.append(getFilterFromPowerSpectrum(freq_spec[0], freq_spec[1]))

    freqs = [item for sublist in freqs for item in sublist]
    freqs = list(set(freqs))
    freqs.sort()
    return freqs


def getNotchFilterPolinomials(freqs, sample_freq, Q_factors):
    polinomials = []
    try:
        for filt_freq, Q_f in zip(freqs, Q_factors):
            w0 = 2 * filt_freq / sample_freq
            polinomials.append(spsig.iirnotch(w0, Q_f * filt_freq))  ##this would be 1hz
    except TypeError:
        warnings.warn("Using same Q_factor for every frequency")
        for filt_freq in freqs:
            w0 = 2 * filt_freq / sample_freq
            polinomials.append(spsig.iirnotch(w0, Q_factors * filt_freq))  ##this would be 1hz
    return polinomials


def runFilter(data, filt_freqs, sample_freq, atenuation):
    """

    :param data: trace to filter
    :type data: np.array
    :param filt_freqs: list of frecuencies to filter
    :type filt_freqs: list
    :param sample_freq: sampling frequency
    :type sample_freq: float
    :param atenuation: Q_factor for atenuation, can be single value or list
    :type atenuation: int or list
    :return: filtered trace
    :rtype: np.array
    """
    polinomlist = getNotchFilterPolinomials(filt_freqs, sample_freq, atenuation)
    for poli in polinomlist:
        data = spsig.filtfilt(poli[0], poli[1], data)
    return data

def runVariablePowerFilters(data, freq_pow_arr, sample_freq, atenuation_rg):
    min, max = freq_pow_arr[:, 1].min(), freq_pow_arr[:, 1].max()
    rescaled_atennuation = (freq_pow_arr[:, 1].copy() - min)/ (max-min)

    rescaled_atennuation *= (atenuation_rg[0] - atenuation_rg[1])
    rescaled_atennuation -= atenuation_rg[0]
    return runFilter(data, freq_pow_arr[:,0], sample_freq, np.abs(rescaled_atennuation))


def testIirNotch(polinom_list, fs):
    i = 0  # Frequency response
    for poli in polinom_list:
        if i == 0:
            w, h = spsig.freqz(poli[0], poli[1])
            i += 1
        else:
            w, h0 = spsig.freqz(poli[0], poli[1])
            h = h * h0
            # Generate frequency axis

        # w = np.linspace(0,1, len(h))
        freq = w * fs / (2 * np.pi)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        # ax[0].set_xlim([0, 100])
        # ax[0].set_ylim([-25, 10])
        ax[0].grid()
        ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        # ax[1].set_xlim([0, 350])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid()
        plt.show()
