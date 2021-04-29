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
from sklearn.mixture import BayesianGaussianMixture
import matplotlib
from sklearn.decomposition import PCA
import PyLeech.Utils.miscUtils as miscUtils
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
# matplotlib.rc('font', **font)

from sklearn.manifold import TSNE
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
    time_step = .5

    savefig = False

    bin_step = time_step
    corr_step = 1

    window_time_size = 340

    min_sample_time = 120
    diff_to_next = 15
    pairplot = False
    clust_dist = 15
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
    for fn in [run_list[1]]:
        # for fn in run_list:
        # for fn in [run_list[0], run_list[-1]]:

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

        unweighted_FCM = corrUtils.getFCM(processed_spike_freq_array, window_size=ws, corr_step=corr_step,
                                          corr_lengths=(window_time_size,), time_step=time_step, unweighted=False,
                                          p_thres=(25, 75))

        # rast_fig, rast_ax = corrUtils.rasterPlot(processed_spike_bool_dict, color_dict=burst_object.color_dict)

        df = pd.DataFrame(unweighted_FCM.T)

        pca = PCA(exp_var)
        sc = StandardScaler()

        tf_FCM = sc.fit_transform(unweighted_FCM)
        # tf_FCM = pca.fit_transform(tf_FCM)
        pca.fit(tf_FCM)
        times = np.arange(t_min, t_max, time_step)

        first_qt, last_qt = int(burst_info_dict['burst ini'][2] / time_step), int(
            burst_info_dict['burst ini'][-2] / time_step)
        # avg_cont_dist = np.mean(np.sqrt(np.power(np.diff(tf_FCM[first_qt:last_qt], axis=0), 2).sum(axis=1)))
        # tf_FCM = pca.fit_transform(unweighted_FCM)

        # min_samples = int(min_samp_time / time_step)
        #
        #
        # i = int(diff_to_next / time_step)
        #
        # FC_dist = tf_FCM[:-i] - tf_FCM[i:]
        # # FC_dist = np.diff(tf_FCM, axis=0)
        # FC_dist = np.sqrt((FC_dist ** 2).sum(axis=1))
        # # FC_dist = np.diff(FC_dist)
        # close = FC_dist < clust_dist * avg_cont_dist
        # cluster_array = -np.ones(tf_FCM.shape[0])
        # j = 0
        # for i in range(1, FC_dist.shape[0]):
        #     if not close[i-1]:
        #         j += 1
        #     cluster_array[i] = j
        #
        # clusters, counts = np.unique(cluster_array, return_counts=True)
        # bad_clusters = clusters[np.where(counts < min_samples)[0]]
        # good_clusters = clusters[np.where(counts >= min_samples)[0]]
        #
        #
        # j = 0
        # new_cluster_array = np.zeros(cluster_array.shape[0], dtype=int)
        # for cl in good_clusters:
        #     new_cluster_array[cluster_array == cl] = j
        #     j += 1
        #
        #
        # new_cluster_array[np.in1d(cluster_array, bad_clusters)] = -1
        # cluster_array = new_cluster_array




        cluster_array = corrUtils.distanceClustering(FCM=tf_FCM, diff_to_next=diff_to_next, clust_dist=clust_dist,
                                                     min_sample_time=min_sample_time, time_step=time_step,
                                                     comparison_interval=(first_qt, last_qt))

        state_switch = np.where(np.diff(cluster_array) != 0)[0]
        clust_order = cluster_array[state_switch]
        try:

            clust_order = np.append(clust_order, cluster_array[state_switch[-1] + 1])
        except IndexError:
            clust_order = []

        fig0, ax0 = burstUtils.plotFreq(processed_spike_freq_dict, scatter_plot=False,
                                        color_dict=burst_object.color_dict, optional_trace=[times, cluster_array],
                                        legend=False)

        fig0.suptitle(basename + '\n' + "/".join([str(i) for i in clust_order]) + '\ndist=%2.2f' % (clust_dist))

        for a in ax0:
            for tm in times[state_switch]:
                a.axvline(tm, c='r')

        norm = miscUtils.powNorm(pow=1., vmin=-1, vmax=1, vmid=0)
        # fig, ax = plt.subplots()
        # # fig.suptitle(basename + ' period=' + str(int(mean_period)))
        # fig.suptitle(basename + ' period=' + str(int(mean_period)))
        # mappable = ax.matshow(cosine_similarity(tf_FCM), cmap='bwr', vmin=-1, vmax=1,
        #                       extent=[t_min, t_max, t_max, t_min], aspect='auto', norm=norm)
        # fig.colorbar(mappable=mappable)
        # ax0[0].get_shared_x_axes().join(ax, ax0[0])

        prt_tup = (basename, mean_period, window_time_size, unweighted_FCM.shape[1], pca.n_components_, exp_var,
                   pca.n_components_ / unweighted_FCM.shape[1], (np.unique(cluster_array) != -1).sum(),
                   clust_dist * avg_cont_dist, pca.explained_variance_ratio_[:3].sum())
        print(
            "File: %s \n\tMean period is %1.2f, wts is %1.2f\n\tIt has %i connections \n\tWas reduced to %i components for %.1f explained var (%0.2f %%)\n\tGot %i clusters, avg_dist=%1.2f\n\t 3 comp explain %1.2f%% of the variance" % prt_tup)

        fig0.subplots_adjust(wspace=0, hspace=0)
        fig0.canvas.draw()
        fig0.savefig('figs_estados/' + basename + "_units.png", dpi=600, transparent=True)
        # fig.savefig("figs_estados/" + basename + "_mat.png", dpi=600, transparent=True)
        # fig.savefig("figs_estados/" + basename + "_mat.png", dpi=600, transparent=True)
