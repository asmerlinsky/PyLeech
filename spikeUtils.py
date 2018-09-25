# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:57:59 2018

@author: Agustin Sanchez Merlinsky
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
    

##Conseguir la duracion, frecuencia de burst
def getBursts(indexes, fs, spike_max_dist=0.7, min_spike_no = 10, min_spike_per_sec = 10):
    burst_list = []
    burst_index = []
    for i in range(len(indexes)-1):
        burst_index.append(indexes[i])
        if (indexes[i+1] - indexes[i])/fs > spike_max_dist:    
            if check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):
                burst_list.append(burst_index)
            burst_index = []
            continue
               
    ##checking the last one
    if check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):
        burst_index.append(indexes[i+1])
        burst_list.append(burst_index)
    
    return burst_list
            
            
            
def check_burst(burst_index, min_spike_no, min_spike_per_sec, fs):        
        if len(burst_index) < min_spike_no: 
            return False
        elif fs*len(burst_index)/(burst_index[-1]-burst_index[0]) > min_spike_per_sec:
            return True
        else:
            return False
        
from copy import deepcopy        
def removeTSpikes(sig, dist = 100, amp = -5, ):
    first = 200
    last = 150
    signal = deepcopy(sig)
    spikes = list(spsig.find_peaks(signal, height=amp, distance=dist)[0])
    spikes.sort(reverse=True)
    for spike in spikes:
        
        if spike < first: 
            signal[0:spike+50] = [signal[spike+50]]*len(signal[0:spike+50])
        elif spike > len(signal) - last:
            signal[spike - first:] = [signal[spike - first]]*len(signal[spike - first:])
        else:
            #del signal[spike-first:spike+last]   
            signal[spike-first:spike+last] = np.zeros(first+last)*np.nan
            #np.delete(signal, [x for x in range(spike-first,spike+last)])
    #no_spike_mean = np.nanmean(signal)
    #signal = deepcopy(sig)
    
    #for spike in spikes:
    #    if (first < spike) and (spike < len(signal)-last):
    #        signal[spike-first:spike+last] = [no_spike_mean]*(first + last)
    return signal
        
def getSpikeIndexes(indexes, spike_indexes):
    spikes = []
    for i in range(len(indexes)):
        for j in range(len(spike_indexes)):
            if -200<=(indexes[i]-spike_indexes[j]) and (indexes[i]-spike_indexes[j])<800:
                spikes.append(i)
                break
    return spikes
    
        
        
def getBaselineSideStd(signal, bins=50, spdist=100):
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
    rel_max = np.nanmax(no_spike_signal) - most_frequent
    rel_min = most_frequent - np.nanmin(no_spike_signal)
    if rel_max > rel_min:
        indexes = np.where(no_spike_signal < most_frequent)
        side = 1
    else:
        indexes = np.where(no_spike_signal > most_frequent)
        side = -1 
    return np.mean(no_spike_signal[indexes]), side, np.std(no_spike_signal[indexes])
    
    
                   
def plotBursts(pltobj, time, nerve_signal, indexes, burst_list, amp, ms=4):
    pltobj.plot(time, nerve_signal)
    pltobj.plot(time[indexes], nerve_signal[indexes], '*', label='Peaks', ms=ms)

    i = 0
    for bl in burst_list:
        if i == 0:
            i += 1
            pltobj.plot([time[bl[0]], time[bl[-1]]], [amp, amp], color='k', linewidth=5, label='bursts')
            #pltobj.plot(time[bl], nerve_signal[bl], 'r-*' )
            continue
            
        
        pltobj.plot([time[bl[0]], time[bl[-1]]], [amp, amp], color='k', linewidth=5)
        #pltobj.plot(time[bl], nerve_signal[bl], 'r-*' )
        
    pltobj.legend(loc=4, fontsize=20)
    pltobj.grid()
    #pltobj.ylim([-.1,.1])
    try:
        pltobj.tight_layout()
    except AttributeError:
        pass
    
             