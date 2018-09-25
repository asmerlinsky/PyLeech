# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:39:15 2018

@author: Agustin
"""

import inspect
import os
import sys

if True: #run when starting ipython instance
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd()+'\\PyLeech')
        
    
file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
import AbfExtension as AbfE
import matplotlib.pyplot as plt
import matplotlib as mpl
from filterUtils import *
#mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import spikeUtils as sU
from shutil import copyfile
import importlib
import glob



dirname = 'T_ipsp'

file_list = glob.glob(dirname + '/*4.1.abf')


for abf_file in file_list:
    
        
        
        
    try:
        del(block)
    except:
        pass    
    copyfile(abf_file, "temp.abf")
    
    block = AbfE.ExtendedAxonRawIo("temp.abf")
    
    ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
    time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info[list(block.ch_info)[0]]['index']]), block.get_signal_sampling_rate())
    



    
    for key, ch_params in block.ch_info.items():
        if 'Vm1' in key:
            signal = ch_data[:, block.ch_info[key]['index']]
            
            time = time[0:len(signal)]
            spec = list(spsig.welch(signal, fs=int(ch_params['sampling_rate']), nperseg=10000))
            spec[1] = 20*(np.log10(spec[1])-np.log10(spec[1][0]))
            
            
            plt.figure()
            plt.title(abf_file.split('\\')[1].split('.')[0] + ', ' + key)
            plt.plot(spec[0], spec[1], label='unfiltered')
            plt.grid()
            
            
            poli_b, poli_a = spsig.butter(8, 100*2/(ch_params['sampling_rate']*np.pi))
            lpfiltered = spsig.filtfilt(poli_b, poli_a, signal)
            #lpfiltered = runFilter(lpfiltered, [50], int(ch_params['sampling_rate']), 0.01)
            lpfilt_spec = list(spsig.welch(lpfiltered, fs=int(ch_params['sampling_rate']), nperseg=10000))
            lpfilt_spec[1] = 20*(np.log10(lpfilt_spec[1])-np.log10(lpfilt_spec[1][0]))
            plt.plot(lpfilt_spec[0], lpfilt_spec[1], label='low pass filtered')
            
            
            poli_b, poli_a = spsig.butter(2, 1/(ch_params['sampling_rate']*np.pi))
            filtered = spsig.filtfilt(poli_b, poli_a, signal)
            filt_spec = list(spsig.welch(filtered, fs=int(ch_params['sampling_rate']), nperseg=10000))
            filt_spec[1] = 20*(np.log10(filt_spec[1])-np.log10(filt_spec[1][0]))
            
            plt.plot(filt_spec[0], filt_spec[1], label='baseline only')
            plt.legend()
            #plt.close()
            
            
            
            fig = plt.figure(figsize=(16,10))
            fig.suptitle(abf_file.split('\\')[1].split('.')[0] + ', ' + key)
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)
            
            ax1.plot(signal)
            ax1.grid()
            
            ax2.plot(lpfiltered)
            ax2.grid()
            
            ax3.plot(filtered)
            ax3.grid()
            
            
            fig = plt.figure(figsize=(16,10))     
            fig.suptitle(abf_file.split('\\')[1].split('.')[0] + ', ' + key)
            
            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
            ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
            
            ax1.plot(lpfiltered)
            ax1.grid()
            std = []
            ind = []
            step = 2000
            num_steps = np.arange(step, len(lpfiltered), step)
            for i in num_steps:
                ind.append(i-int(step/2))
                std.append(np.std(lpfiltered[i-step:i]))
            ax2.plot(ind,std, marker='x')
            
            
            threshold = 8*np.min(np.partition(std, 10)[:10])
            ax2.plot([0, len(lpfiltered)], [threshold, threshold], color='r')
            ax2.grid()
            
            dfiltered = np.gradient(filtered)
            ax3.plot(dfiltered)
            ax3.grid()