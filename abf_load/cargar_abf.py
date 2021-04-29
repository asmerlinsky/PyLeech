# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:40:50 2018

@author: Agustin
"""

"""
Primera parte
"""

import PyLeech.Utils.AbfExtension as abfe
import matplotlib.pyplot as plt
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np

filenames = ['19-08-30/2019_08_30_0003.abf']

arr_dict, time , fs= abfe.getArraysFromAbfFiles(filenames, ['IN4', 'IN5', 'IN6', 'IN7'])

# NS = arr_dict['Vm1']
traces = [arr_dict['IN4'], arr_dict['IN5'], arr_dict['IN6'], arr_dict['IN7']]

fig, ax = plt.subplots()
ax.plot(time, -traces[2])

from scipy.signal import find_peaks


tiempo = time[:int(time.shape[0]/2)]
traza = -traces[2][:int(time.shape[0]/2)]

fig, ax = plt.subplots()
ax.plot(tiempo, traza)

peaks = find_peaks(traza, .15)
tiempo_peaks = tiempo[peaks[0]]
isi_peaks = np.diff(tiempo_peaks)

#
diccionario_spikes = {}
diccionario_spikes[0] = [tiempo_peaks[:-1], isi_peaks]

# diccionario_spikes = {0: [tiempo_peaks[:-1], isi_peaks]}

digit_spikes = burstUtils.processSpikeFreqDict(diccionario_spikes, step=1, time_length=tiempo_peaks[-1], time_interval=(), outlier_threshold=100, counting=False)

# fig, ax = plt.subplots()
# ax.plot(digit_spikes[0][0], digit_spikes[0][1])

fig, ax = plt.subplots()
ax.scatter(digit_spikes[0][0], digit_spikes[0][1])