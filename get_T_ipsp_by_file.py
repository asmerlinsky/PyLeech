# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:19:41 2018

@author: Agustin Sanchez Merlinsky
"""

import inspect
import os
import sys
import json

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
sys.path.append(wdir)
sys.path.append(file_dir)

import AbfExtension as AbfE
import scipy.signal as spsig
import spikeUtils as sU
import numpy as np
from shutil import copyfile
import pandas as pd
import glob
import json_numpy
'''
pass filename : spike_amp (as [])
 '''
file_list = {
        'T1':[0.16, 0.21],
        'T2':[0.16, 0.21],
        'T3':[0.1, 0.15],
        'T4':[0.04, 0.07],
        'T5':[0.13, 0.25],
        'T6':[0.18, 0.26],
        'T7':[0.03, 0.05],
        }



def getIpspResults(filename, spike_amp = [0.3, 0.5], min_spike_per_sec=10, channels=['Vm1', 'IN5'], spike_max_dist=0.7):
    '''
    This function reads a file with an intra and an extracellular recording. 
    -Analizes data to retrieve bursts.
    -Removes spikes from intraceullar recording
    -Identifies and calculates ipsp for each burst
    -Saves results in a file
    '''
    
    copyfile(filename, 'temp.abf')
    block = AbfE.ExtendedAxonRawIo('temp.abf')
    
    ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
    nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
    intra_signal_1 = ch_data[:, block.ch_info['Vm1'] ['index']]
    time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())
    
    '''
    retrieving burst indexes from extracellular electrode
    '''
    indexes = spsig.find_peaks(nerve_signal, height=spike_amp, distance=40)[0]
    burst_list = sU.getBursts(indexes, block.get_signal_sampling_rate(), spike_max_dist, min_spike_no=20, min_spike_per_sec=min_spike_per_sec)
    result_dict = {}
    i=1
    for burst_indexes in burst_list:
        burst_dict = {}
        T_dict = {}
        
        if burst_indexes[0]>=1250 and burst_indexes[-1]<(len(intra_signal_1)-1250):
            T_when_burst = intra_signal_1[burst_indexes[0]-1250:burst_indexes[-1]+1250]
        elif burst_indexes[0]>=1250:
            T_when_burst = intra_signal_1[burst_indexes[0]-1250:burst_indexes[-1]]
        elif burst_indexes[-1]<(len(intra_signal_1)-1250):
            T_when_burst = intra_signal_1[burst_indexes[0]:burst_indexes[-1]+1250]
        
        burst_dict['t_ini'] = time[burst_indexes[0]]
        burst_dict['t_end'] = time[burst_indexes[-1]]
        burst_dict['duration'] = time[burst_indexes[-1]]-time[burst_indexes[0]]
        burst_dict['freq'] = len(burst_indexes)/burst_dict['duration']
        
        baseline, side, std= sU.getBaselineSideStd(T_when_burst, spdist=100)
        T_dict['baseline'] = baseline
        
        height = side*baseline + 6 * std
        
        T_indexes = spsig.find_peaks(side*T_when_burst, height=height, distance=400, width=70)
        T_spikes = spsig.find_peaks(T_when_burst, height=-5, distance=100)
        bad_indexes = sU.getSpikeIndexes(T_indexes[0], T_spikes[0]) ##look for wrongly chosen peaks
        T_no_spike_indexes = np.delete(T_indexes[0], bad_indexes)
        
        T_dict['mean Ipsp'] = getMeanIpspAmplitude(T_when_burst, T_no_spike_indexes, baseline)
        T_dict['Ipsp count'] = len(T_no_spike_indexes)
        
               
        result_dict[i] = {'burst data' : burst_dict, 'T data' : T_dict}
        i += 1
    del(block)    
    return result_dict
        
        


def getMeanIpspAmplitude(T_data, Ipsp_indexes, baseline):
    Ipsp_amplitude = T_data[Ipsp_indexes]
    Ipsp_amplitude -= baseline
    
    if np.isnan(np.mean(Ipsp_amplitude)):
        return 0
    
    return np.mean(Ipsp_amplitude)



if __name__ == "__main__":
      

    full_file_results = {}            
    for neuron, spike_amp in file_list.items():
        #file_results = getIpspResults(filename, spike_amp)
        #full_file_results[filename] = file_results
        trace_results = {}
        neuron_traces  =glob.glob('*/' + neuron + '*.abf')
        for trace in neuron_traces:
        

            file_results = getIpspResults(trace, spike_amp)
            trace_results[os.path.splitext(os.path.basename(trace))[0]] = file_results

        full_file_results[neuron] = trace_results
    
    dumped = json.dumps(full_file_results, cls=json_numpy.NumpyAwareJSONEncoder)
    f=open("results.json","w")
    f.write(dumped)
    f.close()






