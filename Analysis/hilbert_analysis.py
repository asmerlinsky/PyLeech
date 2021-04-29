import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.correlationUtils as corrUtils
import scipy.signal as spsig
import PyLeech.Utils.filterUtils as filterUtils
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    time_step = .1
    kernel_sigma = 5
    NS_kernel_sigma = 5
    time_range = 50
    cdd = CDU.loadDataDict()
    in_neurons = False
    in_envelope = True
    to_run = '2019_07_22_0009'


    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]

    w_NS = [fn for fn in list(cdd) if 'NS' in cdd[fn]['channels'].values()]
    #
    # for fn in w_NS: print(fn)


    if in_neurons:
        for filename in list(cdd):
            if to_run in filename:
                # ns_channel = [key for key, items in cdd[filename]['channels'].items() if 'NS' == items][0]
                selected_neurons = [neuron for neuron, neuron_dict in cdd[filename]['neurons'].items() if neuron_dict["neuron_is_good"] ]
                filename = filename.replace('/', '/')


                # arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(filename, ['IN5'])
                # NS = arr_dict[ns_channel]
                # NS = corrUtils.processContinuousSignal(NS, dt_step=1/fs, kernel_sigma=kernel_sigma, time_range=time_range)[::int(time_step * fs)]

                burst_object = burstStorerLoader.UnitInfo(filename, mode='load')

                binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=.1,
                                                             selected_neurons=selected_neurons,
                                                             time_length=burst_object.time[-1])

                processed_spike_freq_dict = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=kernel_sigma, time_range=time_range, dt_step=.1)

                units_hilbert_dict = {}
                phase_dict = {}

                for key, items in processed_spike_freq_dict.items():

                    units_hilbert_dict[key] = spsig.hilbert(items[1])
                    phase_dict[key] = np.unwrap(np.angle(units_hilbert_dict[key]))


                for key, items in processed_spike_freq_dict.items():
                    fig, ax = plt.subplots(3, 1, sharex=True)
                    ax[0].plot(items[0], items[1])
                    ax[1].plot(items[0], np.abs(units_hilbert_dict[key]))
                    ax[2].plot(items[0],  np.mod(phase_dict[key], np.sign(phase_dict[key]) * 2*np.pi))

                    fig.suptitle(str(key))

    if in_envelope:
        for filename in w_NS:
            # if to_run in filename:
            if True:
                ch_dict = {ch: info for ch, info in cdd[filename]['channels'].items() if len(info) >= 2}
                filename = filename.replace("/", '/')
                basename = os.path.splitext(os.path.basename(filename))[0]

                arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(filename, list(ch_dict))



                envelope_hilbert_dict = {}
                envelope_phase_dict = {}

                if fs > 15000:
                    step = 2
                    fs /= 2
                else:
                    step = 1

                for key in arr_dict.keys():
                    if ch_dict[key] in ['-', '', ' ']:
                        continue

                    if ch_dict[key] == 'NS':
                        arr_dict[key] = corrUtils.processContinuousSignal(arr_dict[key][::step], dt_step=1 / fs,
                                                                          kernel_sigma=NS_kernel_sigma)
                    else:
                        arr_dict[key] = arr_dict[key][::step]
                        if key == 'IN4':
                            line_peaks = np.array([50., 150., 250., 350.2, 450.2, 550.2, 650.2, 750.2,
                                                   850.2, 950.4, 1050.4, 1150.4, 1250.4, 1350.4, 1450.6, 1550.4,
                                                   1650.6, 1750.6,
                                                   1850.6, 1950.6, 2050.8, 2150.8, 2250.8, 2350.8, 2450.8, 2551.,
                                                   2651., 2751., 2851., 2951.])
                            line_peaks = line_peaks[line_peaks < fs / 2]
                            arr_dict[key] = filterUtils.runFilter(arr_dict[key], line_peaks, fs, .1)
                            arr_dict[key] = filterUtils.runButterFilter(arr_dict[key], 1500, sampling_rate=fs)
                            arr_dict[key] = filterUtils.runButterFilter(arr_dict[key], 5, sampling_rate=fs,
                                                                        butt_order=4, btype='high')

                        arr_dict[key] = corrUtils.processNerveSignal(arr_dict[key], kernel_sigma=kernel_sigma,
                                                                     time_range=time_range, dt_step=1 / fs)

                    arr_dict[key] = arr_dict[key][int(.1 * arr_dict[key].shape[0]):int(.9 * arr_dict[key].shape[0])]
                    arr_dict[key] = arr_dict[key][::int(time_step * fs)]
                    # arr_dict[key] = arr_dict[key][int(.1 * arr_dict[key].shape[0]):int(.9 * arr_dict[key].shape[0])]



                    envelope_hilbert_dict[key] = spsig.hilbert(arr_dict[key])
                    envelope_phase_dict[key] = np.unwrap(np.angle(envelope_hilbert_dict[key]))

                time_vector = np.linspace(0, arr_dict[key].shape[0] * time_step, arr_dict[key].shape[0], endpoint=False)

                for key, items in arr_dict.items():
                    if ch_dict[key] == 'NS':
                        fig, ax = plt.subplots(3, 1, sharex=True)
                        ax[0].plot(time_vector, items)
                        ax[1].plot(time_vector, np.abs(envelope_hilbert_dict[key]))
                        ax[2].plot(time_vector, np.mod(envelope_phase_dict[key], np.sign(envelope_phase_dict[key]) * 2 * np.pi))
                        # ax[2].plot(time_vector, envelope_phase_dict[key])
                        fig.suptitle(os.path.splitext(os.path.basename(filename))[0] + '\n' + ch_dict[key])
                        [ax[i].grid(linestyle='dotted') for i in range(3)]
                        fig.subplots_adjust(wspace=0, hspace=0)

                fig, axes = plt.subplots(len(envelope_phase_dict), 1, sharex=True)

                i = 0
                for key, items in envelope_phase_dict.items():
                    axes[i].plot(np.mod(items, np.sign(items) * 2 * np.pi), label=ch_dict[key])
                    axes[i].grid(linestyle='dotted')
                    axes[i].legend()
                    i += 1
                fig.subplots_adjust(wspace=0, hspace=0)
                fig.suptitle(os.path.splitext(os.path.basename(filename))[0])