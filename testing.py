# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 20:37:53 2018

@author: Agustin Sanchez Merlinsky
"""

import os
import sys
import matplotlib.pyplot as plt
os.chdir('C:/Users/Agustin/Documents/Doctorado/Neurodata/PyLeech')
sys.path.append('C:/Users/Agustin/Documents/Doctorado/Neurodata/PyLeech')
from abfUtils import * ##this would be and older version
import scipy.signal as spsig
from filterUtils import *
from AbfExtension import *

os.chdir("..")
### Basic Loading
blockwsegments = loadAbfFile('18-06-12/2002_01_01_0009.abf')
blockw4signals = loadAbfFile('18-06-15/2002_01_01_0024.abf')
block = loadAbfFile('18-06-21/2002_01_01_0004.abf')


data = concatenateSegments('Vm1', blockwsegments)
time = generateTimeVector(len(data), blockwsegments._sampling_rate)


chdict = associateChannels(blockw4signals)
channels = getDataBySegment(blockw4signals, 0)
time = generateTimeVector(len(channels[:,0]), blockw4signals._sampling_rate)


## Getting noise spectrum
for key, item in chdict.items():
    partial_data = spsig.welch(channels[-5000:, item], fs=int(blockw4signals._sampling_rate), nperseg=1000)
    plt.figure()
    plt.title('Unfiltered %s' % key)
    #plt.plot(time, channels[:, item])
    plt.plot(channels[-5000:, item])
    
    plt.figure()
    plt.title(key)
    plt.axis((0,350,0,max(partial_data[1])))
    plt.plot(partial_data[0], partial_data[1])
    
## Notch filter test
freqss = getNoiseFreqsFromEveryChannel(channels[-5000:], int(blockw4signals._sampling_rate), 50)
freqss = [50, 250]
for key, item in chdict.items():
    output = runFilter(channels[:,item], freqss, int(blockw4signals._sampling_rate), 0.25)
    plt.figure()
    plt.title('Filtered %s' % key)
    plt.plot(time, output)
    
    plt.figure()
    plt.title(key)
    plt.plot(time, output, label='filtered %s' % key)
    plt.plot(time, channels[:,item], label='unfiltered %s' % key)
    plt.legend()
    
poli_list = getFilterPolinomials(freqss, int(blockw4signals._sampling_rate), 0.25)
fs = 5000
testIirNotch(poli_list, fs)
    
burst_signal = getDataBySegment(block, 0)
burst_time = generateTimeVector(len(burst_signal[:,0]), int(block._sampling_rate))
plt.plot(burst_time, burst_signal[:,1])    

'''
         Extended class testing
'''

%matplotlib tk
from LeechPy.AbfExtension import *
import matplotlib.pyplot as plt
block = ExtendedAxonRawIo('18-06-21/2002_01_01_0004.abf')
ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
time = generateTimeVector(len(ch_data[:,block.ch_info['Vm1']['index']]), block.get_signal_sampling_rate())
plt.plot(time, ch_data[:,block.ch_info['Vm1']['index']])
plt.figure()
plt.plot(time, ch_data[:,block.ch_info['IN5']['index']])

'''
         PeakUtils
'''

signal = ch_data[:,block.ch_info['IN5']['index']]
signal_amp = np.max(signal) - np.abs(np.min(signal))
dist = 0.01*block._sampling_rate
indexes = spsig.find_peaks(signal, height=0.035, distance=dist)
plt.figure()
#pplot(time, signal, indexes)
plt.plot(time[indexes[0]], signal[indexes[0]], '.')
plt.plot(time,signal)

'''
        finding peaks in a  burst
'''
import os
os.chdir(r'C:\Users\Agustin\Documents\Doctorado\NeuroData')
import LeechPy.AbfExtension as AbfE
%matplotlib tk
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import numpy as np
import scipy.signal as spsig
import LeechPy.spikeUtils as sU
block = AbfE.ExtendedAxonRawIo('18-06-21/2002_01_01_0004.abf')
block.ch_info
ch_data = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, 0))
signal = -ch_data[:,1]
time = AbfE.generateTimeVector(len(ch_data[:, block.ch_info['IN5']['index']]), block.get_signal_sampling_rate())
plt.figure()
plt.plot(time, ch_data[:,1])

burstini = np.where(time>50)[0][0]
burstend = np.where(time>58.5)[0][0]
plt.figure()
burst = signal[burstini:burstend]
plt.plot(time[burstini:burstend], signal[burstini:burstend])
burst_ind = spsig.find_peaks(burst, height=[0.03,0.05], distance=40, width=[5,35])[0]
plt.figure()
plt.plot(burst)
plt.plot(burst_ind, burst[burst_ind], '*')

indexes = spsig.find_peaks(signal, height=[0.03,0.05], distance=40, width=[5,35])[0]

plt.figure(figsize=[16,10])
plt.plot(time, signal)
plt.plot(time[indexes], signal[indexes], '*', label='detected peaks')
burst_list = sU.getBursts(indexes, block.get_signal_sampling_rate())
i = 0
for bl in burst_list:
    if i == 0:
        i += 1
        plt.plot([time[bl[0]], time[bl[-1]]], [.075,.075], color='k', linewidth=5, label='bursts')
    
    plt.plot([time[bl[0]], time[bl[-1]]], [.075,.075], color='k', linewidth=5)
    
plt.legend(loc=4, fontsize=20)
plt.grid()
plt.ylim([-.1,.1])
plt.tight_layout()
plt.savefig('bursts.png', dpi=500)

fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
ax[0].plot(time, ch_data[:,0], color='k', )
#ax[0].set_ylim([-25, 10])
ax[0].grid()
i = 0
ax[1].plot(time, signal)
ax[1].plot(time[indexes], signal[indexes], '*', label='detected peaks')
for bl in burst_list:
    if i == 0:
        i += 1
        ax[1].plot([time[bl[0]], time[bl[-1]]], [.075,.075], color='k', linewidth=5, label='bursts')
    
    ax[1].plot([time[bl[0]], time[bl[-1]]], [.075,.075], color='k', linewidth=5)
    

ax[1].set_xlabel("Time (s)")
#ax[1].set_xlim([0, 350])
#ax[1].set_ylim([-90, 90])
ax[1].grid()
ax[1].legend(loc=1, fontsize=18)
fig.tight_layout()
fig.savefig('signal-bursts.png', dpi=500)

'''
        Analizing T neuron inhibition
'''

T_signal = ch_data[:,0]
burst_time = time[burst_list[6][0]:burst_list[6][-1]]
T_signal_in_burst = T_signal[burst_list[6][0]:burst_list[6][-1]]
plt.figure()
num_bins = 100
n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)
plt.show()

baseline, side, baselinestd = sU.getBaselineSideStd(T_signal_in_burst, 50)
plt.figure()
plt.plot(burst_time, T_signal_in_burst)
height = side*baseline + 3*baselinestd
T_indexes = spsig.find_peaks(side*T_signal_in_burst, height=height, distance=500, width=70)
##T_indexes = spsig.find_peaks(T_signal_in_burst, height=height, distance=500, prominence=[6,10])
plt.plot(burst_time[T_indexes[0]], T_signal_in_burst[T_indexes[0]], 'r*')
plt.plot([burst_time[0], burst_time[-1]],[baseline, baseline])



burst_time = time[burst_list[0][0]:burst_list[0][-1]]
T_signal_in_burst = T_signal[burst_list[0][0]:burst_list[0][-1]]
plt.figure()
num_bins = 100
n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)
plt.figure()
plt.plot(burst_time, T_signal_in_burst)
baseline, baselinestd = sU.getInhibitionBaseline(T_signal_in_burst, 0.25)
print(baseline, baselinestd)
height = -baseline + 3*baselinestd
T_indexes = spsig.find_peaks(-T_signal_in_burst, height=height, distance=500, width=70)
#T_indexes = spsig.find_peaks(T_signal_in_burst, height=height, distance=500, prominence=[6,10])
plt.plot(burst_time[T_indexes[0]], T_signal_in_burst[T_indexes[0]], 'r*')


'''
Analizing T neuron inhibition for every burst
'''
T_signal = ch_data[:,0]
i=0
for burst_indexes in burst_list:
    burst_time = time[burst_indexes[0]:burst_indexes[-1]]
    T_when_burst = T_signal[burst_indexes[0]:burst_indexes[-1]]
    plt.figure()
    plt.title("burst %i" % i)
    num_bins = 100
    n, bins, patches = plt.hist(T_signal_in_burst, num_bins, facecolor='blue', alpha=0.5)
    
    plt.figure()
    plt.title("burst %i" % i)
    plt.plot(burst_time, T_when_burst)
    baseline, baselinestd = sU.getInhibitionBaseline(T_when_burst, 0.25)
    print("burst %i:\t baseline=%f, std=%f"%(i, baseline, baselinestd))
    rel_max = np.max(T_when_burst)-np.mean(T_when_burst)
    rel_min = np.mean(T_when_burst)-np.min(T_when_burst)
    if rel_min>rel_max:
        T_when_burst = -T_when_burst
        baseline = -baseline
    height = -baseline + 3*baselinestd
    T_indexes = spsig.find_peaks(T_when_burst, height=height, distance=400, width=70)
    T_spikes = spsig.find_peaks(T_when_burst, height=0, distance=300)
    
    
    if rel_min>rel_max:
        T_when_burst = -T_when_burst
        baseline = -baseline
    
    plt.plot(burst_time[T_indexes[0]], T_when_burst[T_indexes[0]], 'r*')
    
    i+=1
    
    
    
    
    
    
    
    
    

