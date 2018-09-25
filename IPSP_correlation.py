# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:49:48 2018

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

#mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import spikeUtils as sU
from shutil import copyfile
import importlib
import glob




#copyfile(file, "temp.abf")

block = AbfE.ExtendedAxonRawIo("temp.abf")

ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))

nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())
intra_1 = ch_data[:, block.ch_info['Vm1'] ['index']]

channels = 'Vm1'

ipsp_list = spsig.find_peaks(intra_1, threshold=.1, distance=600)

#plt.figure()
#plt.plot(time, intra_1)
#plt.scatter(time[ipsp_list[0]], intra_1[ipsp_list[0]],color='r')
    
plt.figure()
plt.plot(intra_1)
plt.scatter(ipsp_list[0], intra_1[ipsp_list[0]],color='r')    
    
    