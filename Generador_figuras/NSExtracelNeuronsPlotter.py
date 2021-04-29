import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.correlationUtils as corrUtils
import scipy.signal as spsig

if __name__ == "__main__":
    time_step = .1
    kernel_sigma = 2
    time_range = 20
    cdd = CDU.loadDataDict()

    # del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    # del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    # del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    # del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]

    # to_plot_list = [fn for fn in list(cdd) if 'NS' in cdd[fn]['channels'].values()]

    for filename in list(cdd):
        # ns_channel = [key for key, items in cdd[filename]['channels'].items() if 'NS' == items][0]
        selected_neurons = [neuron for neuron, neuron_dict in cdd[filename]['neurons'].items() if neuron_dict["neuron_is_good"] ]
        filename = filename.replace('/', '/')


        # arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(filename, ['IN5'])
        # NS = arr_dict[ns_channel]
        # NS = corrUtils.processContinuousSignal(NS, dt_step=1/fs, kernel_sigma=kernel_sigma, time_range=time_range)[::int(time_step * fs)]

        burst_object = burstStorerLoader.UnitInfo(filename, mode='load')
        binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=.1, selected_neurons=selected_neurons, time_length=burst_object.time[-1])
        processed_spike_freq_dict = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=kernel_sigma, time_range=time_range, dt_step=.1)

        start, end = .2, .8
        processed_spike_freq_dict = {key: array[:, int(start * array.shape[1]):int(end * array.shape[1])] for key, array in processed_spike_freq_dict.items()}
        # NS = NS[int(start * NS.shape[0]):int(end * NS.shape[0])]
        # time_vector = time_vector[::int(time_step * fs)]
        # time_vector = time_vector[int(start * time_vector.shape[0]):int(end * time_vector.shape[0])]

        fig, ax = burstUtils.plotFreq(burst_object.spike_freq_dict, #optional_trace=[time_vector, NS],
                                      template_dict=None, outlier_thres=3.5, ms=2, color_dict='k', facecolor='white',
                                      scatter_plot=True, legend=True)
        fig.suptitle(filename)
        fig.subplots_adjust(wspace=0, hspace=0)
        print(filename)