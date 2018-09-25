# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 14:49:37 2018

@author: Agustin Sanchez Merlinsky
"""

import scipy.signal as spsig
import numpy as np
import matplotlib.pyplot as plt


def getFreqSpectrum(data, samplefreq, nperseg=1000):
    return spsig.welch(data, fs=samplefreq, nperseg=nperseg)

def getFilterFromPowerSpectrum(freq, intensity):
    """ 
    Gets a few main noise frecuencies in the [35, 300] Hz range
    pass the power spectrum of some baseline data
    """
    start_index =np.where(freq >=35)[0][0]
    filter_freqs = [freq[start_index]]
    j = 0
    for i in range(start_index, np.where(freq <=260)[0][-1]):
        if intensity[i]> intensity[np.where(freq==filter_freqs[j])[0][0]]:  
            filter_freqs[j] = freq[i]
        if freq[i] > (j+1)*100:
            filter_freqs.append(freq[i])
            j += 1
    return filter_freqs

def getFreqInd(freqs_array, freq):
    for i in range(len(freqs_array)):
        if freqs_array[i] >= freq:
            return i


def getNoiseFreqsFromEveryChannel(channels, sample_freq, peak_dist):
    """ 
    Gets noise frecuencies for every channel in a segment.
    """
    freqs = []
    for i in range(len(channels[0,:])):
        freq_spec = getFreqSpectrum(channels[:,i], sample_freq)
        indini = getFreqInd(freq_spec[0], 35)
        indend = getFreqInd(freq_spec[0], 340)
        delta_f = freq_spec[0][1]-freq_spec[0][0]
        dist = int(peak_dist/delta_f)
        indexes = spsig.argrelmax(freq_spec[1][indini:indend],order=dist)[0]+indini
        freqs.append(freq_spec[0][indexes])
        #freqs.append(getFilterFromPowerSpectrum(freq_spec[0], freq_spec[1]))
    
    freqs = [item for sublist in freqs for item in sublist] 
    freqs = list(set(freqs))
    freqs.sort()
    return freqs
        
def getFilterPolinomials(freqs, sample_freq, Q_factor):
    polinomials = []
    for filt_freq in freqs:
        w0 = 2*filt_freq/sample_freq
        polinomials.append(spsig.iirnotch(w0, Q_factor*filt_freq)) ##this would be 1hz
    return polinomials
    
    
def runFilter(data, filt_freqs, sample_freq, atenuation):
    polinomlist = getFilterPolinomials(filt_freqs, sample_freq, atenuation)
    for poli in polinomlist:
        data = spsig.filtfilt(poli[0], poli[1], data)
    return data
    
def testIirNotch(polinom_list, fs):
    i = 0# Frequency response
    for poli in polinom_list:
        if i == 0:
            w, h = spsig.freqz(poli[0], poli[1])
            i += 1
        else:
            w, h0 = spsig.freqz(poli[0], poli[1])
            h = h*h0        
    # Generate frequency axis
        
        #w = np.linspace(0,1, len(h))    
        freq = w*fs/(2*np.pi)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20*np.log10(abs(h)) , color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        #ax[0].set_xlim([0, 100])
        #ax[0].set_ylim([-25, 10])
        ax[0].grid()
        ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        #ax[1].set_xlim([0, 350])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid()
        plt.show()