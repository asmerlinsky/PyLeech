# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:03:38 2018

@author: Agustin Sanchez Merlinsky
"""

import inspect
import os
import sys

from PyLeech.Utils.T_DP_classes import TInfo, getIsi

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '\\PyLeech')

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)

import PyLeech.Utils.AbfExtension as AbfE
from PyLeech.Utils.filterUtils import *
import glob

block = None

if __name__ == '__main__':
    '''
    Be sure to be reading the right channell (either Vm1 or Vm2)
    '''

    T_channel = 'Vm'
    dirname = 'T_ipsp'

    file_list = glob.glob(dirname + '/T6*.abf')
    ISI = []
    for abf_file in file_list:
        block = AbfE.ExtendedAxonRawIo(abf_file)

        T_signal = block.getSignal_single_segment('Vm1')
        fs = block.get_signal_sampling_rate()

        time = AbfE.generateTimeVector(len(T_signal), fs)

        # getIpsps(abf_file, T_signal, time, fs, 1000, buttord=8, filt_freq=100, std_thres=6, ipsp_thres=6, peak_dist=150, plot_results=True)

        T_obj = TInfo(abf_file, T_signal, fs, 1000, b_order=8, filt_freq=150, std_thres=3.75, ipsp_thres=4.75,
                      peak_dist=150)
        T_obj.plot_results = True
        T_obj.use_LPsignal = True
        T_obj.getFiltSignal()
        ipsp_list = T_obj.getIpsps()

        ISI.extend(getIsi(ipsp_list, fs))

        del block
    plt.figure()
    ISI = [i for i in ISI if i < 2]
    plt.hist(ISI, bins=200)
