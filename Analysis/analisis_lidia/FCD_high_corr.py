import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import PyLeech.Utils.correlationUtils as corrUtils
import PyLeech.Utils.planUtils as pU
import PyLeech.Utils.NLDUtils as NLD
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


def partPearson(x, y):
    return pearsonr(x, y)[0]


def varCorrCoef(x, y=None):
    if y is None:
        return np.corrcoef(x, rowvar=False)

    return np.corrcoef(x, rowvar=False)


if __name__ == "__main__":

    cdd = CDU.loadDataDict()

    # kernel_sigma = 2
    # time_range = 20
    time_step = 1

    savefig = False

    bin_step = time_step
    corr_step = 1

    window_time_size = 120

    min_sample_time = 120
    diff_to_next = 15
    pairplot = False
    clust_dist = 5
    exp_var = .8

    """
    FC(t) by file, only if file has MA recording
    """

    # run_list = list(cdd)
    run_list = []
    for fn in list(cdd):
        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            pass

        elif 'MA' in cdd[fn]['channels'].values():
            run_list.append(fn)

    # run_list = [list(cdd)[0]]
    # print(run_list[0])!
    save_folder = "lidia_figs/MA_fcd/"

    #
    # for fn in [run_list[1]]:
    for fn in run_list:
        # for fn in [run_li;st[0], run_list[-1]]:

        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        if cdd_de3 != burst_object.isDe3:
            print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn,
                                                                                                               cdd_de3,
                                                                                                               burst_object.isDe3))

        spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]
        spike_freqs = burst_object.spike_freq_dict[burst_object.isDe3][1]
        spike_times = spike_times[~burstUtils.is_outlier(spike_freqs, 5)]

        burst_info_dict = pU.getBurstingInfo(spike_times, min_spike_no=15, min_spike_per_sec=10.)

        mean_period = np.round(np.mean(burst_info_dict['cycle period']))

        processed_spike_bool_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                                                    selected_neurons=selected_neurons,
                                                                    time_length=burst_object.time_length,
                                                                    counting='bool')

        processed_spike_freq_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                                                    selected_neurons=selected_neurons,
                                                                    time_length=burst_object.time_length, counting=True)

        processed_spike_bool_array = burstUtils.processed_sfd_to_array(processed_spike_bool_dict)

        processed_spike_freq_array = burstUtils.processed_sfd_to_array(processed_spike_freq_dict)

        i = 0

        ###tengo que arreglar esto, es demasiado redundante

        # window_time_size = window_periods * int(np.percentile(burst_info_dict['cycle period'], 95))

        ws = int(window_time_size / time_step)
        t_min = window_time_size / 2
        t_max = processed_spike_bool_dict[list(processed_spike_bool_dict)[0]][0][-1] + bin_step - window_time_size / 2
        times = np.arange(t_min, t_max, time_step)

        first_qt, last_qt = int(burst_info_dict['burst ini'][2] / time_step), int(
            burst_info_dict['burst ini'][-2] / time_step)

        unweighted_FCM = corrUtils.getFCM(processed_spike_freq_array, window_size=ws, corr_step=corr_step,
                                          corr_lengths=(window_time_size,), time_step=time_step, unweighted=False,
                                          p_thres=(25, 75))

        corr_idxs, sorted_idxs = corrUtils.getSortedCorrelationIdxs(target_unit_idx=burst_object.isDe3,
                                                                    unit_list=list(processed_spike_freq_dict),
                                                                    processed_spike_freq_array=processed_spike_freq_array,
                                                                    unweighted_FCM=unweighted_FCM,
                                                                    comparison_interval=(first_qt, last_qt))

        avg_const_dist = np.mean(
            np.sqrt(np.power(np.diff(unweighted_FCM[first_qt:last_qt], axis=0), 2).sum(axis=1)))
        # avg_const_dist = 1
        cluster_array = corrUtils.distanceClustering(FCM=unweighted_FCM[:, corr_idxs[sorted_idxs[:]]], diff_to_next=diff_to_next, clust_dist=clust_dist,
                                                     min_sample_time=min_sample_time, time_step=time_step,
                                                     avg_const_dist=avg_const_dist)

        no_de3 = list(processed_spike_freq_dict)
        no_de3.remove(burst_object.isDe3)
        colors = []

        fig, ax = plt.subplots()
        i = 0
        for idx in sorted_idxs[:]:
            # for idx in sorted_idxs:
            c = burst_object.color_dict[no_de3[idx]]
            if type(c) is list:
                ax.plot(times, unweighted_FCM[:, corr_idxs[idx]], c=c[0], label=no_de3[idx])
            else:
                ax.plot(times, unweighted_FCM[:, corr_idxs[idx]], c=c, label=no_de3[idx])

        fig.suptitle(basename)
        fig.legend()

        state_switch = np.where(np.diff(cluster_array) != 0)[0]
        clust_order = cluster_array[state_switch]
        try:

            clust_order = np.append(clust_order, cluster_array[state_switch[-1] + 1])
        except IndexError:
            clust_order = []
        #
        fig0, ax0 = burstUtils.plotFreq(processed_spike_freq_dict, scatter_plot=False,
                                        color_dict=burst_object.color_dict, optional_trace=[times, cluster_array],
                                        legend=True)

        fig0.suptitle(basename + '\n' + "/".join([str(i) for i in clust_order]) + '\ndist=%2.2f' % (clust_dist))
        ax0[0].get_shared_x_axes().join(ax, ax0[0])
        for a in ax0:
            for tm in times[state_switch]:
                a.axvline(tm, c='r')
        for tm in times[state_switch]:
            ax.axvline(tm, c='r')
        # plt.close(fig)

        fig0.subplots_adjust(wspace=0, hspace=0)
        fig0.canvas.draw()
        fig0.savefig("fig_estados_solo_de3/" + basename + ".png")
        fig.savefig("fig_estados_solo_de3/"+ basename + "_corrs.png")