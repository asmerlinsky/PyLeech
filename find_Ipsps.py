# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:03:38 2018

@author: Agustin Sanchez Merlinsky
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

              
            
def getSide(bl, signal, bins=50, spdist=100):
    no_spike_signal = sU.removeTSpikes(signal, spdist)
    nan_filtered_signal = no_spike_signal[np.logical_not(np.isnan(no_spike_signal))]
    rel_max = np.nanmax(no_spike_signal) - bl
    rel_min = bl - np.nanmin(no_spike_signal)
    if rel_max<0:
        return -1
    elif rel_min<0:
        return 1
    elif rel_max>rel_min:
        return 1
    else:
        return -1 
    
                
def getIpsps(filename, channel, step, buttord=8, freq=100, std_thres=5, ipsp_thres=3):            
    
    block = AbfE.ExtendedAxonRawIo(filename)
    
    ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
    time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info[list(block.ch_info)[0]]['index']]), block.get_signal_sampling_rate())
    
    for key, ch_params in block.ch_info.items():
        if channel in key:
            signal = ch_data[:, block.ch_info[key]['index']]
            poli_b, poli_a = spsig.butter(buttord, freq*2/(ch_params['sampling_rate']*np.pi))
            LPsignal = spsig.filtfilt(poli_b, poli_a, signal)
            
            
            ind = np.arange(0, len(signal), step)
            std = []
            for i in ind:
                std.append(np.std(LPsignal[i:i+step]))
                
            ind = np.append(ind, len(signal)-1)
            std = np.array(std)
            thres = std_thres*np.mean(np.partition(std, 10)[:10])
            
            overthres = np.where(np.array(std)>thres)[0]
            segments = []
            full_ipsps = []
            '''
            segments will be a list of tuples where
            tup = (first index over std, las ind over std)
            '''
            j = 0
            skip_next = False
            if len(overthres)>1:
                for i in range(len(overthres)-1):
                    
                    if j==0:
                        first = overthres[i]
                        j += 1
                        le = 1
                        
                    if (overthres[i+1]-overthres[i])<=3:
                        le += overthres[i+1]-overthres[i]
                    else:
                        j = 0
                        segments.append((first, first + le))
                segments.append((first, first + le))
            elif len(overthres) == 1:
                segments.append((overthres[0], overthres[0] + 1))
                    
            
            
            i = 0
            for tup in segments:
                i += 1
                try:
                    ahead_mean = np.mean(signal[ind[tup[1]]:ind[tup[1]+1]])
                except:
                    ahead_mean = np.nan
                try:
                    back_mean = np.mean(signal[ind[tup[0]-1]:ind[tup[0]]])
                except:
                    back_mean = np.nan
                
                
                baseline = np.nanmean((ahead_mean, back_mean))
                try:
                    bl_std = ipsp_thres*std[tup[0]-1]
                except:
                    bl_std = ipsp_thres*std[tup[1]]
                    
                side = getSide(baseline, LPsignal[ind[tup[0]]:ind[tup[1]]])
                height = side*baseline + bl_std
                #print(i, baseline, side, height)
                T_indexes = spsig.find_peaks(side*signal[ind[tup[0]]:ind[tup[1]]], height=height, distance=200, width=70)
                T_spikes = spsig.find_peaks(signal[ind[tup[0]]:ind[tup[1]]], height=-5, distance=100)
                
                bad_indexes = sU.getSpikeIndexes(T_indexes[0], T_spikes[0]) ##look for wrongly chosen peaks
                T_ipsps = np.delete(T_indexes[0], bad_indexes) + ind[tup[0]]
                
                full_ipsps.extend(T_ipsps)
                        

            segments = np.array(segments)   
            plt.figure()
            plt.title(filename)
            plt.plot(signal, zorder=0)
            
            #plt.scatter(ind[overthres], [np.mean(signal)+1]*len(ind[overthres]))
            try:
                plt.scatter(ind[segments[:,0]], [np.mean(signal)]*len(segments[:,0]), color='k', zorder=10)
                plt.scatter(ind[segments[:,1]], [np.mean(signal)]*len(segments[:,1]), color='k', zorder=10)
            except IndexError:
                pass
            plt.scatter(full_ipsps, signal[full_ipsps], color = 'r', marker='*', s=50, zorder=20)
        
    del block
    return full_ipsps

            

if __name__ == '__main__':
    '''
    Be sure to be reading the right channell (either Vm1 or Vm2)
    '''
    
    T_channel = 'Vm'      
    dirname = 'T_ipsp'
    
    file_list = glob.glob(dirname + '/*5.3*abf')
    
    for abf_file in file_list:
        ipsp_list = getIpsps(abf_file, channel='Vm1', step=1000, buttord=8, freq=100, std_thres=4, ipsp_thres=8) 
        
        