import matplotlib.pyplot as plt
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.NLDUtils as NLD
import numpy as np
import scipy.signal as spsig
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.burstUtils
from PyLeech.Utils.unitInfo import UnitInfo
import PyLeech.Utils.filterUtils as filterUtils
import PyLeech.Utils.burstUtils as burstUtils





if __name__ == "__main__":

    binning_dt = .5
    spike_kernel_sigma = 3

    cdd = CDU.loadDataDict()

    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]


    for fn, data in cdd.items():

        try:
            ns_channel = [key for key, items in data['channels'].items() if 'NS' == items][0]
            print("Running %s" % fn)
        except IndexError:
            continue

        arr_dict, time_vector1, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])

        NS_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=1 / fs)
        conv_NS = spsig.fftconvolve(arr_dict[ns_channel], NS_kernel, mode='same')[::int(binning_dt * fs)]
        time_vector1 = time_vector1[::int(binning_dt * fs)]
        del arr_dict


        burst_object = UnitInfo(fn, 'RegistrosDP_PP', 'load')

        good_neurons = [neuron for neuron, neuron_dict in data['neurons'].items() if neuron_dict['neuron_is_good']]

        spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=int(binning_dt * fs) / fs,
                                                           num=conv_NS.shape[0],
                                                           time_length=burst_object.time[-1])




        smoothed_sfd = {}
        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)
        for key, items in spike_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])
        burst_array = []

        fig, ax_list = burstUtils.plotFreq(smoothed_sfd, scatter_plot=False, color_dict='single_color',
                                           draw_list=good_neurons,
                                           optional_trace=[time_vector1, conv_NS])
        fig.suptitle(fn)

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)
        N_neurons = len(list(spike_freq_array)) + 1
        burst_array.append(conv_NS)
        fig = plt.figure()
        fig1 = plt.figure()

        j = 1
        for key, items in spike_freq_array.items():
            smoothed_spikes = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])
            # filterUtils.plotSpectrums(smoothed_spikes[1]-smoothed_spikes[1].mean(), sampling_rate=1/binning_dt, nperseg=1000000, pltobj=ax[j])
            f, pxx = filterUtils.getFreqSpectrum(smoothed_spikes[1]-smoothed_spikes[1].mean(), sampling_rate=1/binning_dt, nperseg=1000000)
            if j == 1:
                ax = fig.add_subplot(N_neurons, 1, j)
            else:
                ax = fig.add_subplot(N_neurons, 1, j, sharex=ax)
            ax1 = fig1.add_subplot(N_neurons, 1, j, sharex=ax)
            ax.semilogy(f, pxx)
            ax1.plot(f, pxx)
            j += 1

        # filterUtils.plotSpectrums(conv_NS - conv_NS.mean(), sampling_rate=1/binning_dt, nperseg=1000000, pltobj=ax[j])
        filterUtils.getFreqSpectrum(conv_NS - conv_NS.mean(), sampling_rate=1/binning_dt, nperseg=1000000)

        ax = fig.add_subplot(N_neurons, 1, j, sharex=ax)
        ax1 = fig1.add_subplot(N_neurons, 1, j, sharex=ax)

        ax.semilogy(f, pxx)
        ax1.plot(f, pxx)
        fig.suptitle(fn)
        fig1.suptitle(fn)