# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:06:48 2018

@author: Agustin
"""

import inspect
import os
import sys

if True:  # run when starting ipython instance
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd() + '/PyLeech')

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
# sys.path.append(wdir)
# sys.path.append(file_dir)
import AbfExtension as AbfE
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import spikeUtils as sU
from shutil import copyfile
import importlib
import glob

file_list = {
    'T1': [0.16, 0.21],
    'T2': [0.16, 0.21],
    'T3': [0.1, 0.15],
    'T4': [0.03, 0.06],
    'T5': [0.13, 0.26],
    'T6': [0.18, 0.26],
    'T7': [0.03, 0.05],
}

for neuron, amp in file_list.items():
    for trace in glob.glob('*/' + neuron + '*.abf'):

        try:
            del (block)
        except:
            pass
        copyfile(trace, "temp.abf")

        spike_amp = amp  # works for 18-06-28/2018_06_28_0002.abf
        ##copyfile("18-06-29/2018_06_29_0007.abf", "temp.abf")
        ##spike_amp = [0.03, 0.06] # works for 18-06-29/2018_06_29_0007.abf

        # copyfile("18-06-21/2002_01_01_0004.abf", "temp.abf")
        # spike_amp = [0.03, 0.06] # works for 18-06-29/2018_06_29_0007.abf

        plt_channels = ['Vm1', 'IN5']

        block = AbfE.ExtendedAxonRawIo("temp.abf")
        # print(block.ch_info)
        ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
        nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
        time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())

        # block.plotEveryChannelFromSegmentNo(plt_channels)

        intra_signal_1 = ch_data[:, block.ch_info['Vm1']['index']]
        nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
        indexes = spsig.find_peaks(nerve_signal, height=spike_amp, distance=40)[0]
        burst_list = sU.getBursts(indexes, block.get_signal_sampling_rate(), spike_max_dist=0.6, min_spike_no=20,
                                  min_spike_per_sec=10)
        full_index_list = np.empty((1, 0), dtype='int64')
        i = 0

        '''
        I have a list with a list of spikes separated in bursts
        This will iterate through the lists and get the corresponding ipsp for each burst
        '''
        ipsp_list = []
        baseline_list = []
        for burst_indexes in burst_list:
            if burst_indexes[0] >= 1250 and burst_indexes[-1] < (len(intra_signal_1) - 1250):
                T_when_burst = intra_signal_1[burst_indexes[0] - 1250:burst_indexes[-1] + 1250]
            elif burst_indexes[0] >= 1250:
                T_when_burst = intra_signal_1[burst_indexes[0] - 1250:burst_indexes[-1]]
            elif burst_indexes[-1] < (len(intra_signal_1) - 1250):
                T_when_burst = intra_signal_1[burst_indexes[0]:burst_indexes[-1] + 1250]

            ''' 
            Removing spikes before looking 
            for the baseline and the side the ipsp go
            '''
            baseline, side, std = sU.getBaselineSideStd(T_when_burst, spdist=100)
            height = side * baseline + 6 * std
            T_indexes = spsig.find_peaks(side * T_when_burst, height=height, distance=400, width=70)
            T_spikes = spsig.find_peaks(T_when_burst, height=-5, distance=100)
            bad_indexes = sU.getSpikeIndexes(T_indexes[0], T_spikes[0])
            T_no_spike_indexes = np.delete(T_indexes[0], bad_indexes)

            if burst_indexes[0] >= 1250:
                actual_indexes = T_no_spike_indexes + burst_indexes[0] - 1250
            else:
                actual_indexes = T_no_spike_indexes + burst_indexes[0]

            # actual_indexes = T_indexes[0] + burst_indexes[0]
            baseline_list.append(baseline)
            ipsp_list.append(actual_indexes)
            full_index_list = np.append(full_index_list, actual_indexes)

        mpl.rcParams.update({'font.size': 15})

        fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        fig.suptitle(trace)
        ax[0].plot(time, intra_signal_1, color='k')
        ax[0].plot(time[full_index_list], intra_signal_1[full_index_list], '*', color='darkorange', label='Peaks', ms=8)
        j = 0
        if False:
            for i in range(0, len(burst_list)):
                if j == 0:
                    ax[0].plot([time[burst_list[i][0]], time[burst_list[i][-1]]], [baseline_list[i], baseline_list[i]],
                               color='r', label='baseline')
                    j += 1
                ax[0].plot([time[burst_list[i][0]], time[burst_list[i][-1]]], [baseline_list[i], baseline_list[i]],
                           color='r')

        # ax[0].grid()
        # ax[0].legend(loc=1)
        ax[0].set_ylabel('T (mV)', rotation=0, labelpad=40)
        if False:
            sU.plotBursts(ax[1], time, nerve_signal, indexes, burst_list, 1.1 * amp[1], ms=8)
        else:
            ax[1].plot(time, nerve_signal, color='k')
            # ax[1].grid()
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel('DP (mV)', rotation=0, labelpad=40)
        # ax[1].set_xticklabels([])
        # ax[1].legend(loc=1, fontsize=18)

        # ax[1].grid()
        plt.show()
        del (block)
