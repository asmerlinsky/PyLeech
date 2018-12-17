# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:09:00 2018

@author: Agustin Sanchez Merlinsky
"""

import inspect
import os
import sys

if True:  # run when starting ipython instance
    sys.path.append(os.getcwd())
    sys.path.append(os.getcwd() + '\\PyLeech')

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
import AbfExtension as AbfE
import matplotlib.pyplot as plt
import matplotlib as mpl
from filterUtils import *
# mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import spikeUtils as sU
from shutil import copyfile
import importlib
import glob

file_list = {
    # 'T1':[0.16, 0.21],
    # 'T2':[0.16, 0.21],A
    # 'T3':[0.1, 0.15],
    'T4': [0.03, 0.06],
    # 'T5':[0.13, 0.26],
    'T6': [0.18, 0.26],
    'T7': [0.03, 0.05],
}

# copyfile(file, "temp.abf")


for neuron, amp in file_list.items():
    for trace in glob.glob('*/' + neuron + '*.abf'):

        # if not (('5.1' in trace) or ('5.1' in trace)): continue

        try:
            del (block)
        except:
            pass
        copyfile(trace, "temp.abf")

        block = AbfE.ExtendedAxonRawIo("temp.abf")

        ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
        time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())

        for key, ch_params in block.ch_info.items():
            if 'Vm' in key:
                signal = ch_data[:, block.ch_info[key]['index']]

                time = time[0:len(signal)]
                spec = list(spsig.welch(signal, fs=int(ch_params['sampling_rate']), nperseg=10000))
                spec[1] = 20 * (np.log10(spec[1]) - np.log10(spec[1][0]))
                plt.figure()
                plt.title(trace)
                plt.plot(spec[0], spec[1], label='unfiltered')

                poli_b, poli_a = spsig.butter(10, 600 * 2 / (ch_params['sampling_rate'] * np.pi))
                lpfiltered = spsig.filtfilt(poli_b, poli_a, signal)
                lpfiltered = runFilter(lpfiltered, [50], int(ch_params['sampling_rate']), 0.1)
                lpfilt_spec = list(spsig.welch(lpfiltered, fs=int(ch_params['sampling_rate']), nperseg=10000))
                lpfilt_spec[1] = 20 * (np.log10(lpfilt_spec[1]) - np.log10(lpfilt_spec[1][0]))
                plt.plot(lpfilt_spec[0], lpfilt_spec[1], label='low pass filtered')

                poli_b, poli_a = spsig.butter(2, 1 / (ch_params['sampling_rate'] * np.pi))
                filtered = spsig.filtfilt(poli_b, poli_a, signal)
                filt_spec = list(spsig.welch(filtered, fs=int(ch_params['sampling_rate']), nperseg=10000))
                filt_spec[1] = 20 * (np.log10(filt_spec[1]) - np.log10(filt_spec[1][0]))

                plt.plot(filt_spec[0], filt_spec[1], label='baseline only')
                plt.legend()
                # plt.close()
                fig = plt.figure(figsize=(16, 10))
                fig.suptitle(trace.split('\\')[1])
                ax1 = fig.add_subplot(3, 1, 1)
                ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
                ax3 = fig.add_subplot(3, 1, 3, sharex=ax1, sharey=ax1)

                ax1.plot(signal)
                ax1.grid()

                ax2.plot(lpfiltered)
                ax2.grid()

                ax3.plot(filtered)
                ax3.grid()

                fig = plt.figure(figsize=(16, 10))
                fig.suptitle(trace.split('\\')[1])

                ax1 = fig.add_subplot(3, 1, 1)
                ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
                ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

                ax1.plot(lpfiltered)
                ax1.grid()
                std = []
                ind = []
                step = 4000
                num_steps = np.arange(step, len(lpfiltered), step)
                for i in num_steps:
                    ind.append(i - int(step / 2))
                    std.append(np.std(lpfiltered[i - step:i]))
                ax2.plot(ind, std, marker='x')
                threshold = 5 * np.min(std)
                ax2.plot([0, len(lpfiltered)], [threshold, threshold], color='r')
                ax2.grid()

                dfiltered = np.gradient(filtered)
                ax3.plot(dfiltered)
                ax3.grid()
