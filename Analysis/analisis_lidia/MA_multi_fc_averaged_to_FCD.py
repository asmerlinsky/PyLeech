import os
import PyLeech.Utils.miscUtils as miscUtils
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import PyLeech.Utils.correlationUtils as corrUtils
import PyLeech.Utils.planUtils as pU
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
# font = {'size'   : 5}
import matplotlib.animation as animation
# matplotlib.rc('font', **font)
# plt.ioff()

import warnings
warnings.filterwarnings("ignore")

# plt.rcParams['animation.ffmpeg_path'] = "C:/ffmpeg/bin/ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"

def varCorrCoef(x):
    return np.corrcoef(x, rowvar=False)

def only_spearmanR(x):
    return spearmanr(x)[0]

if __name__ == "__main__":

    cdd = CDU.loadDataDict()

    # kernel_sigma = 2
    # time_range = 20
    time_step = 1

    corr_mean = 60

    savefig = False
    corr_step = 1
    bin_step = time_step
    num = 6
    cols = 3
    neuron_correlation_dict_by_time = {}
    count = 0
    full_corr = []
    corr_thres = .3
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
    # print(run_list[0])
    save_folder = "lidia_figs/MA_fcd/"

    run_list = [run_list[2]]
    for fn in run_list:


        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        if cdd_de3 != burst_object.isDe3:
            print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn, cdd_de3, burst_object.isDe3) )
        #
        # spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]
        # spike_freqs = burst_object.spike_freq_dict[burst_object.isDe3][1]
        # spike_times = spike_times[~burstUtils.is_outlier(spike_freqs, 5)]

        # burst_info_dict = pU.getBurstingInfo(spike_times, min_spike_no=15, min_spike_per_sec=10.)


        # mean_period = np.mean(burst_info_dict['cycle period'])

        corr_times = np.linspace(-6, 6, endpoint=True, num=7) + corr_mean
        corr_times = corr_times[::-1]

        spike_count_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                        selected_neurons=selected_neurons,
                                        time_length=burst_object.time_length, counting=True)

        processed_spike_freq_dict = spike_count_dict

        fig1, ax1 = burstUtils.plotFreq(processed_spike_freq_dict, scatter_plot=True,
                                        color_dict=burst_object.color_dict)

        processed_spike_freq_array = burstUtils.processed_sfd_to_array(processed_spike_freq_dict)
        time_length = spike_count_dict[list(spike_count_dict)[0]][0][-1]
        i = 0

        for corr_time in corr_times:
            ws = int(corr_time / time_step)
            t_min = corr_time / 2
            t_max = time_length - corr_time / 2

            fc_matrixes = corrUtils.multiUnitSlidingWindow(processed_spike_freq_array, func=varCorrCoef, window_size=ws, step=corr_step)
            fc_matrixes[np.isnan(fc_matrixes)] = 0

            flat_idxs = corrUtils.getFlattenedUpperTriangIdxs(fc_matrixes.shape[1], 1)

            if i == 0:
                t_min_longest = t_min
                t_max_longest = t_max
                min_len = fc_matrixes.shape[0]



                if min_len % 2 != 0:
                    multi_corr_mat = np.zeros((min_len-1, len(corr_times), flat_idxs.shape[0]))

                else:
                    multi_corr_mat = np.zeros((min_len, len(corr_times), flat_idxs.shape[0]))


                half_min_len = math.floor(min_len/2)



            mat_center = math.floor(fc_matrixes.shape[0] / 2) # queda en el medio o
            multi_corr_mat[:, i, :] = fc_matrixes.reshape(fc_matrixes.shape[0], fc_matrixes.shape[1] ** 2)[mat_center-half_min_len:mat_center+half_min_len, flat_idxs]

            # multi_layer_mat[i] = spearman_corr[mat_center-half_min_len:mat_center+half_min_len, mat_center-half_min_len:mat_center+half_min_len] # si impar me faltaria el ultimo lugar pero yafu

            i += 1
            # fig_sin1, ax_sin1 = plt.subplots()
            # ax_sin1.matshow(np.abs(fc_matrixes[int(550/time_step)]>0.05).astype(float), vmin=-1, vmax=1, cmap='bwr')
            # fig_sin2, ax_sin2 = plt.subplots()
            # ax_sin2.matshow(fc_matrixes[int(650/time_step)], vmin=-1, vmax=1, cmap='bwr')


        if min_len % 2 == 0:
            time_steps = np.arange(t_min_longest, t_max_longest+bin_step, bin_step)
        else:
            time_steps = np.arange(t_min_longest, t_max_longest, bin_step)
        #
        avgd_corr_mat = multi_corr_mat.mean(axis=1)


        corrUtils.getPValues(processed_spike_freq_array, p_thres=(5, 95), corr_times=corr_times, time_step=time_step)


        # avgd_corr_mat[avgd_corr_mat > corr_thres] = 1.
        # avgd_corr_mat[avgd_corr_mat < -corr_thres] = -1.
        # avgd_corr_mat[(avgd_corr_mat != 1.) & (avgd_corr_mat != -1.)] = 0.


        # sc = StandardScaler()
        # scaled_mat = sc.fit_transform(avgd_corr_mat)
        # pca = PCA(n_components=.8)
        #
        # tf_mat = pca.fit_transform(avgd_corr_mat)
        # print(basename, pca.n_components_)
        # print(pca.explained_variance_ratio_.sum())
        #
        # print(pca.explained_variance_ratio_[:10])
        #
        # dbscan = DBSCAN(eps=1, min_samples=10)
        # pred = dbscan.fit_predict(tf_mat)
        #
        #
        # df = pd.DataFrame(tf_mat, columns=['PC' + str(i) for i in range(tf_mat.shape[1])])
        # df['clusters'] = pred
        # # sns.pairplot(df, hue='clusters', vars=df.columns[:5])
        # sns.pairplot(df, hue='clusters', vars=df.columns[:3], diag_kind='hist')


        method = 'pearson'
        # df = pd.DataFrame(avgd_corr_mat.T)
        # spearman_fcd_mat = df.corr(method=method).values
        spearman_fcd_mat = cosine_similarity(avgd_corr_mat)

        # mean_f = plt.figure(figsize=(8, 8), constrained_layout=True)
        # gs = mean_f.add_gridspec(4, 3)
        #
        # mat_ax = mean_f.add_subplot(gs[:3, :])
        # cluster_ax = mean_f.add_subplot(gs[-1, :])
        # mappable = mat_ax.matshow(spearman_fcd_mat, vmin=-1, vmax=1,
        #                           extent=[t_min_longest, t_max_longest + bin_step, t_max_longest + bin_step,
        #                                   t_min_longest],
        #                           cmap='bwr', aspect='auto')
        # cluster_ax.plot(time_steps, pred)
        # cluster_ax.get_shared_x_axes().join(cluster_ax, mat_ax)
        # #
        # # tsne = TSNE()
        # # fitted_mat = tsne.fit_transform(avgd_corr_mat)
        #
        #
        # # # dbscan = DBSCAN(eps=5, min_samples=2)
        # # # pred = dbscan.fit_predict(tf_mat)
        # # # print(np.unique(pred).shape[0])
        # # # fig, ax = plt.subplots()
        # # # scatter = ax.scatter(fitted_mat[:, 0], fitted_mat[:, 1], c=pred)
        # #
        # # legend1 = ax.legend(*scatter.legend_elements(),
        # #                     loc="lower left", title="Classes")
        # # fig.add_artist(legend1)
        #
        # # fcd_mat = np.corrcoef(multi_corr_mat.mean(axis=1))
        # # df = pd.DataFrame(multi_corr_mat.mean(axis=1).T)
        # df = pd.DataFrame(avgd_corr_mat.T)
        #
        #
        # method = 'pearson'
        # spearman_fcd_mat = df.corr(method=method).values
        # # df = pd.DataFrame(tf_mat.T)
        # # spearman_fcd_mat = df.corr(method='spearman').values
        fig, ax = plt.subplots()
        ax.matshow(spearman_fcd_mat, vmin=-1, vmax=1, extent=[t_min_longest, t_max_longest+bin_step, t_max_longest+bin_step, t_min_longest],
                                   cmap='bwr', aspect='auto')


        #
        # mean_f = plt.figure(figsize=(8, 8), constrained_layout=True)
        # gs = mean_f.add_gridspec(4, 3)
        #
        # mat_ax = mean_f.add_subplot(gs[:3, :])
        # cluster_ax = mean_f.add_subplot(gs[-1, :])
        #
        # mappable = mat_ax.matshow(spearman_fcd_mat, vmin=-1, vmax=1, extent=[t_min_longest, t_max_longest+bin_step, t_max_longest+bin_step, t_min_longest],
        #                            cmap='bwr', aspect='auto')
        #
        #
        # # mappable = mat_ax.matshow((spearman_fcd_mat>.25).astype(int), vmin=-1, vmax=1, extent=[t_min_longest, t_max_longest, t_max_longest, t_min_longest],
        # #                            cmap='bwr', aspect='auto')
        #
        # cluster_ax.plot(time_steps, pred)
        # cluster_ax.get_shared_x_axes().join(cluster_ax, mat_ax)
        # mean_f.suptitle(basename + method)
        #
        # #
        # #
        # # pearson_fcd_mat = df.corr(method='pearson').values
        # #
        # # mean_f = plt.figure(figsize=(8, 8), constrained_layout=True)
        # # gs = mean_f.add_gridspec(4, 3)
        # #
        # # mat_ax = mean_f.add_subplot(gs[:3, :])
        # # cluster_ax = mean_f.add_subplot(gs[-1, :])
        # # #
        # # # mappable = mat_ax.matshow(pearson_fcd_mat, vmin=-1, vmax=1,
        # # #                           extent=[t_min_longest, t_max_longest + bin_step, t_max_longest + bin_step,
        # # #                                   t_min_longest],
        # # #                           cmap='bwr', aspect='auto')
        # # mappable = mat_ax.matshow((pearson_fcd_mat > .25).astype(int), vmin=-1, vmax=1,
        # #                           extent=[t_min_longest, t_max_longest, t_max_longest, t_min_longest],
        # #                           cmap='bwr', aspect='auto')
        # # time_steps = np.arange(t_min_longest, t_max_longest + bin_step, bin_step)
        # # cluster_ax.plot(time_steps, pred)
        # # cluster_ax.get_shared_x_axes().join(cluster_ax, mat_ax)
        # # mean_f.suptitle(basename + 'pearson')
