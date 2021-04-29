import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import PyLeech.Utils.correlationUtils as corrUtils
import PyLeech.Utils.AbfExtension as abfe
import gc
import matplotlib
# font = {'size'   : 5}

# matplotlib.rc('font', **font)
# plt.ioff()
def partPearson(x, y):
    return pearsonr(x, y)[0]


if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/14217000.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]
    del cdd["RegistrosDP_PP/cont10.pklspikes"]
    del cdd["RegistrosDP_PP/2018_12_04_0003_0004_0005_0006b.pklspikes"]
    plot_list = ["2019_08_28_0005", "2019_08_30_0003", "2019_08_30_0006"]
    kernel_sigma = 2
    time_range = 20
    time_step = .5
    corr_step = int(60*5 / time_step)
    bin_step = .5
    num = 6
    cols = 3
    neuron_correlation_dict_by_time = {}
    count = 0
    full_corr = []
    for i in range(num):
        neuron_correlation_dict_by_time[i] = []
        corr_by_file = {}
    if False:
        for fn in list(cdd):
            # if not any([select in fn for select in ["2019_08_28_0005", "2019_08_30_0003", "2019_08_30_0006"] ]):
            #     continue
            ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}


            if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
                print("file %s has no De3 assigned" % fn)
                continue

            cdd_de3 = cdd[fn]['DE3']
            selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                                neuron_dict["neuron_is_good"]]
            fn = fn.replace("/", '/')






            basename = os.path.splitext(os.path.basen@me(fn))[0]

            burst_object = burstStorerLoader.BurstStorerLoader(fn, mode='load')

            if cdd_de3 != burst_object.isDe3:
                print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn, cdd_de3, burst_object.isDe3) )

            binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=.1,
                                                         selected_neurons=selected_neurons,
                                                         time_length=burst_object.time[-1])

            processed_spike_freq_dict = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=kernel_sigma,
                                                                             time_range=time_range, dt_step=.1)

            # burstUtils.plotFreq(processed_spike_freq_dict, scatter_plot=False, color_dict='k')

            data_len = processed_spike_freq_dict[selected_neurons[0]].shape[1]
            centers = np.arange(0, data_len, corr_step, dtype=int) + int(corr_step / 2)

            array_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, channels=ch_dict)
            if (time_vector[-1]>1500) or time_vector[-1]<1100:
                continue
            idxs = np.linspace(0, data_len, num=num + 1, dtype=int)


            # fig0, ax0 = plt.subplots(len(array_dict), 1, sharex=True)
            # i = 0
            # for key, item in array_dict.items():
            #     ax0[i].plot(time_vector[::10], item[::10], color='k')
            #     for id in idxs:
            #         ax0[i].axvline(id*.1, color='r')
            #     i += 1
            #
            # fig0.subplots_adjust(wspace=0, hspace=0)
            # fig0.suptitle(basename)
            #
            #
            # fig_ax_list = []
            neurons  = list(processed_spike_freq_dict)
            corr_dict = {}
            if any([select in fn for select in ["2019_08_28_0005", "2019_08_30_0003", "2019_08_30_0006"]]):
                fig2, ax2 = plt.subplots(2, 3)
                fig2.suptitle(basename)
            for i in range(num):
                neuron_k = burst_object.isDe3

                corr_mat = np.zeros((len(processed_spike_freq_dict), len(processed_spike_freq_dict)))
                corrs = []
                for j in range(len(neurons)):

                    neuron_j = neurons[j]
                    if neuron_j==neuron_k:
                        continue

                    r, p = spearmanr(processed_spike_freq_dict[neuron_k][1][idxs[i]:idxs[i + 1]],
                                     processed_spike_freq_dict[neuron_j][1][idxs[i]:idxs[i + 1]])

                    # r, p = pearsonr(processed_spike_freq_dict[neuron_k][1][idxs[i]:idxs[i + 1]],
                    #                                  processed_spike_freq_dict[neuron_j][1][idxs[i]:idxs[i + 1]])
                    corrs.append(r)
                    #
                    # if i == 0:
                    #     corr_dict[neuron_j] = np.zeros((num,2))
                    # corr_dict[neuron_j][i] = (r, p)
                neuron_correlation_dict_by_time[i].extend(corrs)
                if any([select in fn for select in plot_list]):
                    ax2.flatten()[i].hist(corrs, bins=np.arange(-1.1, 1.1, .2))
                    ax2.flatten()[i].set_title(str(i))

            part_corr = []
            for j in range(len(neurons)):

                neuron_j = neurons[j]
                if neuron_j == neuron_k:
                    continue

                r, p = spearmanr(processed_spike_freq_dict[neuron_k][1],
                                 processed_spike_freq_dict[neuron_j][1])
                full_corr.append(r)
                part_corr.append(r)
            count += 1
            corr_by_file[basename] = part_corr

            if any([select in fn for select in plot_list]):
                fig2, ax2 = plt.subplots()
                fig2.suptitle(basename)
                ax2.hist(part_corr, bins=np.arange(-1.1, 1.1, .2))




        fig2, ax2 = plt.subplots(2, 3)
        fig2.suptitle('every file')
        for i in range(num):
            print(np.mean(neuron_correlation_dict_by_time[i]),
                np.std(neuron_correlation_dict_by_time[i])
                )
            ax2.flatten()[i].hist(neuron_correlation_dict_by_time[i], bins=np.arange(-1.1, 1.1, .2))
            ax2.flatten()[i].set_title(str(i))

        fig2, ax2 = plt.subplots()
        fig2.suptitle('every file')
        ax2.hist(full_corr, bins=np.arange(-1.1, 1.1, .2))


    """
    r(t) by file
    """

    for fn in list(cdd):
        # if not any([select in fn for select in ["2018_12_13_0001", "2019_07_22_0014", "2019_08_28_0005", "2019_07_22_0009" ] ]):
        #     continue
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}
        # if not any([select in fn for select in plot_list]):
        #     continue

        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            print("file %s has no De3 assigned" % fn)
            continue

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("/", '/')






        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        if cdd_de3 != burst_object.isDe3:
            print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn, cdd_de3, burst_object.isDe3) )

        binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                                     selected_neurons=selected_neurons,
                                                     time_length=burst_object.time[-1], counting=True)

        spike_count_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                        selected_neurons=selected_neurons,
                                        time_length=burst_object.time[-1], counting=True)


        processed_spike_freq_dict = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=kernel_sigma,
                                                                         time_range=time_range, dt_step=bin_step)

        # processed_spike_freq_dict = binned_sfd

        data_len = processed_spike_freq_dict[selected_neurons[0]].shape[1]
        centers = np.arange(0, data_len, corr_step, dtype=int) + int(corr_step / 2)

        array_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, channels=ch_dict)
        if (time_vector[-1]>1500) or time_vector[-1]<1100:
            continue
        time_vector = time_vector[::int(fs * bin_step)]
        idxs = np.linspace(0, data_len, num=num + 1, dtype=int)
        #
        # fig1, axes1 = burstUtils.plotFreq(burst_object.spike_freq_dict, color_dict=burst_object.color_dict, scatter_plot=True,
        #                                   draw_list=selected_neurons, outlier_thres=2, facecolor='lightgray', legend=False)

        fig1, axes1 = burstUtils.plotFreq(spike_count_dict, color_dict=burst_object.color_dict, legend=False)

        # fig1, axes1 = burstUtils.plotFreq(processed_spike_freq_dict, color_dict=burst_object.color_dict, scatter_plot=False,
        #                                 draw_list=selected_neurons, facecolor='lightgray', legend=False)

        fig1.suptitle("%s\nDE-3 is %i" % (basename, burst_object.isDe3))
        fig1.subplots_adjust(wspace=0, hspace=0)
        for ax in axes1:
            ax.grid(None)
        fig1.savefig("correlation_figs/pearsons_correlation/" + basename + "_traces.png", transparent=True,
                     dpi=600)
        axes1[0].set_xlim([200, 600])
        fig1.savefig("correlation_figs/pearsons_correlation/" + basename + "_traces_zoom.png", transparent=True,
                     dpi=600)

        #     for id in idxs:
        #         ax.axvline(id * bin_step, color='r')

        # fig1.legend()

        neuron_k = burst_object.isDe3
        print(basename, burst_object.isDe3)
        neurons = np.sort(selected_neurons)

        # neurons = list(selected_neurons)
        # if any([select in fn for select in plot_list]):
        # fig1.savefig("correlation_figs/pearsons_correlation/" + basename+"_traces_zoom.png", transparent=True, dpi=600)

        # fig2, ax2 = plt.subplots(len(neurons) - 1, 1, sharex=True)
        # fig2, ax2 = plt.subplots()
        # fig2.suptitle("%s\nDE-3 is %i, cosine" % (basename, burst_object.isDe3))
        fig3, ax3 = plt.subplots()
        fig3.suptitle("%s\nDE-3 is %i, pearson" % (basename, burst_object.isDe3))
        # i = 0
        for j in range(len(neurons)):
            neuron_j = neurons[j]

            if neuron_j == neuron_k:
                continue
            # idxs, corr = corrUtils.slidingWindow(processed_spike_freq_dict[neuron_k][1], processed_spike_freq_dict[neuron_j][1], corrUtils.cosineSimilarity, step=100, window_size=200)
            idxs2, corr2 = corrUtils.pairwiseSlidingWindow(spike_count_dict[neuron_k][1], spike_count_dict[neuron_j][1], partPearson, step=100, window_size=200)
            # idxs, corr = corrUtils.slidingWindow(binned_sfd[neuron_k][1], binned_sfd[neuron_j][1], corrUtils.cosineSimilarity, step=40, window_size=300)
            # idxs, corr = corrUtils.pairwiseSlidingWindow(spike_count_dict[neuron_k][1], spike_count_dict[neuron_j][1], corrUtils.cosineSimilarity, step=40, window_size=300)
            # idxs2, corr2 = corrUtils.slidingWindow(binned_sfd[neuron_k][1], binned_sfd[neuron_j][1], partPearson, step=100, window_size=200)

            if type(burst_object.color_dict[neuron_j]) is list:
                c = burst_object.color_dict[neuron_j][0]
            else:
                c = burst_object.color_dict[neuron_j]
            # ax2.plot(time_vector[idxs], corr, marker='o', label=neuron_j, c=c)
            # ax2.set_ylim([-.1, 1])
            ax3.plot(time_vector[idxs2], corr2, marker='o', label=neuron_j, c=c)
            ax3.set_ylim([-1, 1])



            # i += 1

        # fig2.legend()
        fig3.legend(loc='lower right')
        # axes1.get_shared_x_axes().join(axes1, ax2)
        axes1[0].get_shared_x_axes().join(axes1[0], ax3)
        fig3.savefig("correlation_figs/pearsons_correlation/" + basename + "_correlations.png", transparent=True,
                     dpi=600)
        ax3.set_xlim([200, 600])
        # axes1[0].get_shared_x_axes().join(axes1[0], ax3)
        fig3.savefig("correlation_figs/pearsons_correlation/" + basename + "_correlations_zoom.png", transparent=True, dpi=600)

        # del burst_object, processed_spike_freq_dict, binned_sfd, array_dict
        # gc.collect()
