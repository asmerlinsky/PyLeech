
import PyLeech.Utils.AbfExtension as abfe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PyLeech.Utils.constants as constants
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.spsortUtils as spsortUtils
import PyLeech.Utils.SpSorter as SpSorter
import PyLeech.Utils.unitInfo as burstStorerLoader
nan = constants.nan
opp0 = constants.opp0
import PyLeech.Utils.filterUtils as filterUtils

np.set_printoptions(precision=3)
if __name__ == "__main__":

    plt.ion()

    filename = '19-08-30/2019_08_30_0000.abf'

    arr_dict, time , fs= abfe.getArraysFromAbfFiles(filename, ['IN4', 'IN5'])
    trace = arr_dict[list(arr_dict)[0]]
    low_noise_trace = arr_dict["IN5"]
    filterUtils.plotSpectrums(trace, low_noise_trace, sampling_rate=fs, nperseg=20000)
    spect = filterUtils.getFreqSpectrum(trace, fs, nperseg=20000)
    plt.plot(spect[0], np.log(spect[1]))

    ln = len(time)

        tr_filt = filterUtils.runButterFilter(trace, 2000, sampling_rate=fs)
        tr_filt = filterUtils.runButterFilter(tr_filt, 5, sampling_rate=fs, butt_order=4, btype='high')
        if i == 0:
            tr_filt = filterUtils.runFilter(tr_filt, np.arange(50, 3000, 100), fs, .05)
            i =+ 1
        else:
            tr_filt = filterUtils.runFilter(tr_filt, np.arange(50, 2000, 100.0625), fs, .2)
        filt_traces.append(tr_filt)

    # fig, ax = plt.subplots(4, 1, sharex=True)
    fig1, ax1 = plt.subplots(4, 1, sharex=True)

    for i in range(len(filt_traces)):
        # ax[i].plot(traces[i])
        # ax[i].plot(filt_traces[i])
        filterUtils.plotSpectrums(traces[i], filt_traces[i], sampling_rate=fs, nperseg=10000, ax=ax1[i])
