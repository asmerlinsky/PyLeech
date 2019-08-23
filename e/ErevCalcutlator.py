# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 12:31:35 2018

@author: Agustin
"""
% matplotlib
tk
import os

os.chdir(r'C:\Users\Agustin\Documents\Doctorado\NeuroData')
import LeechPy.AbfExtension as AbfE
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import LeechPy.spikeUtils as sU
from shutil import copyfile

copyfile("18-06-28/2018_06_28_0003.abf", "temp.abf")

plt_channels = ['Vm1', 'IN5']
spike_amp = [0.18, 0.26]

block = AbfE.ExtendedAxonRawIo("temp.abf")
print(block.ch_info)
ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())

block.plotEveryChannelFromSegmentNo(plt_channels)

###Single Burst
burstini = np.where(time > 100)[0][0]
burstend = np.where(time > 104)[0][0]
plt.figure()
burst = nerve_signal[burstini:burstend]
plt.plot(time[burstini:burstend], nerve_signal[burstini:burstend])
# burst_ind = spsig.find_peaks(burst, height=spike_amp, distance=40, width=[5,35])[0]
burst_ind = spsig.find_peaks(burst, height=spike_amp, distance=40)[0]
plt.figure()
plt.plot(burst)
plt.plot(burst_ind, burst[burst_ind], '*')
plt.grid()

###Every Burst

indexes = spsig.find_peaks(nerve_signal, height=spike_amp, distance=40)[0]
burst_list = sU.getBursts(indexes, block.get_signal_sampling_rate(), spike_max_dist=0.7, min_spike_per_sec=15)
plt.figure(figsize=[16, 10])
sU.plotBursts(plt, time, nerve_signal, indexes, burst_list)

fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ax[0].plot(time, ch_data[:, 0], color='k')
ax[0].grid()
sU.plotBursts(ax[1], time, nerve_signal, indexes, burst_list)
ax[1].set_xlabel("Time (s)")
ax[1].grid()
ax[1].legend(loc=1, fontsize=18)
fig.tight_layout()

###### Checking how signal noise looks
copyfile("18-06-28/2018_06_28_0000.abf", "temp1.abf")
noise_block = AbfE.ExtendedAxonRawIo("temp1.abf")
noise_data = noise_block.rescale_signal_raw_to_float(noise_block.get_analogsignal_chunk(0, 0))
noise_time = AbfE.generateTimeVector(len(noise_data[:, noise_block.ch_info['Vm1']['index']]),
                                     noise_block.get_signal_sampling_rate())
index_ini = np.where(noise_time > 52)[0][0]
index_end = np.where(noise_time > 57)[0][0]
intra_noise_signal = noise_data[index_ini:index_end, noise_block.ch_info['Vm1']['index']]
plt.figure()
plt.plot(intra_noise_signal)

plt.figure()
n, bins, patches = plt.hist(intra_noise_signal, num_bins, facecolor='blue', alpha=0.5)
plt.show()

####☺


intra_signal_1 = ch_data[:, 0]
burst_time = time[burst_list[1][0]:burst_list[1][-1]]
T_signal_in_burst = intra_signal_1[burst_list[1][0]:burst_list[1][-1]]
plt.figure()
num_bins = 50
n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)
plt.show()

plt.figure()
plt.plot(burst_time, T_signal_in_burst)
baseline, side = sU.getBaselineAndSide(T_signal_in_burst)
height = side * baseline + 2 * np.std(T_signal_in_burst)
T_indexes = spsig.find_peaks(side * T_signal_in_burst, height=height, distance=400, width=70)
##T_indexes = spsig.find_peaks(T_signal_in_burst, height=height, distance=500, prominence=[6,10])
plt.plot(burst_time[T_indexes[0]], T_signal_in_burst[T_indexes[0]], 'r*')

### looking  a different burst
burst_time = time[burst_list[0][0]:burst_list[0][-1]]
T_signal_in_burst = intra_signal_1[burst_list[0][0]:burst_list[0][-1]]
plt.figure()
num_bins = 100
n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)
plt.figure()
plt.plot(burst_time, T_signal_in_burst)
baseline, side = sU.getBaselineAndSide(T_signal_in_burst)
print(baseline, side)
height = side * baseline + 2 * np.std(T_signal_in_burst)
T_indexes = spsig.find_peaks(side * T_signal_in_burst, height=height, distance=500, width=70)
# T_indexes = spsig.find_peaks(T_signal_in_burst, height=height, distance=500, prominence=[6,10])
plt.plot(burst_time[T_indexes[0]], T_signal_in_burst[T_indexes[0]], 'r*')

##Analizing T neuron inhibition for every burst

###This won´t work. getInhibitionBaseline doesn´t exist anymore
intra_signal_1 = ch_data[:, 0]
i = 0

for burst_indexes in burst_list:
    burst_time = time[burst_indexes[0]:burst_indexes[-1]]
    T_when_burst = intra_signal_1[burst_indexes[0]:burst_indexes[-1]]
    plt.figure()
    plt.title("burst %i" % i)
    num_bins = 50
    n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)

    plt.figure()
    plt.title("burst %i" % i)
    plt.plot(burst_time, T_when_burst)
    baseline, baselinestd = sU.getInhibitionBaseline(T_when_burst, 0.25)
    print("burst %i:\t baseline=%f, std=%f" % (i, baseline, baselinestd))
    rel_max = np.max(T_when_burst) - np.mean(T_when_burst)
    rel_min = np.mean(T_when_burst) - np.min(T_when_burst)
    if rel_min > rel_max:
        T_when_burst = -T_when_burst
        baseline = -baseline
    height = -baseline + 3 * baselinestd
    T_indexes = spsig.find_peaks(T_when_burst, height=height, distance=400, width=70)
    if rel_min > rel_max:
        T_when_burst = -T_when_burst
        baseline = -baseline

    plt.plot(burst_time[T_indexes[0]], T_when_burst[T_indexes[0]], 'r*')

    i += 1

### Trting to plot everything

spike_amp = [0.18, 0.26]
intra_signal_1 = ch_data[:, block.ch_info['Vm1']['index']]
nerve_signal = ch_data[:, block.ch_info['IN5']['index']]
indexes = spsig.find_peaks(nerve_signal, height=spike_amp, distance=40)[0]
burst_list = sU.getBursts(indexes, block.get_signal_sampling_rate(), spike_max_dist=0.7, min_spike_per_sec=15)
full_index_list = np.empty((1, 0), dtype='int64')
i = 0
for burst_indexes in burst_list:
    T_when_burst = intra_signal_1[burst_indexes[0]:burst_indexes[-1]]
    baseline, side = sU.getBaselineAndSide(T_when_burst)
    height = side * baseline + 2 * np.std(T_when_burst)
    T_indexes = spsig.find_peaks(side * T_when_burst, height=height, distance=400, width=70)
    actual_indexes = T_indexes[0] + burst_indexes[0]
    full_index_list = np.append(full_index_list, actual_indexes)

fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ax[0].plot(time, intra_signal_1)
ax[0].plot(time[full_index_list], intra_signal_1[full_index_list], '*', label='peaks')
ax[0].grid()
ax[0].legend(loc=1, fontsize=18)
sU.plotBursts(ax[1], time, nerve_signal, indexes, burst_list)
ax[1].set_xlabel("Time (s)")
ax[1].grid()
ax[1].legend(loc=1, fontsize=18)
fig.tight_layout()
