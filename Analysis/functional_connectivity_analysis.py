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
import PyLeech.Utils.AbfExtension as abfe
import gc
import matplotlib
# font = {'size'   : 5}
import matplotlib.animation as animation
# matplotlib.rc('font', **font)
# plt.ioff()

# plt.rcParams['animation.ffmpeg_path'] = "C:/ffmpeg/bin/ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"

def partPearson(x, y):
    return pearsonr(x, y)[0]


if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    # del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    # del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    # del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    # del cdd["RegistrosDP_PP/14217000.pklspikes"]
    # del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]
    # del cdd["RegistrosDP_PP/cont10.pklspikes"]
    # del cdd["RegistrosDP_PP/2018_12_04_0003_0004_0005_0006b.pklspikes"]


    kernel_sigma = 2
    time_range = 20
    time_step = 1.
    corr_times = np.array([
        40,
        60,
        80,
        100,
        120,
        140,
    ])

    corr_step = 1
    bin_step = time_step
    num = 6
    cols = 3
    neuron_correlation_dict_by_time = {}
    count = 0
    full_corr = []
    corr_thres = .7
    """
    FC(t) by file
    """

    # run_list = []
    # # sel = [
    #     "2018_12_04_0003_0004_0005_0006b",
    #     "2019_07_22_0009",
    #     "2019_08_26_0002",
    #     "2019_08_30_0006",
    #     "2019_08_30_0003",
    #     "2019_07_23_0008",
    #        ]
    # for fn in list(cdd):
    #     if any([s in fn for s in sel]):
    #         run_list.append(fn)
    run_list = list(cdd)


    # run_list = [list(cdd)[0]]
    # print(run_list[0])
    save_folder = "fc_figs/"
    for fn in run_list:

        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        if cdd[fn]['skipped']:# or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
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


        # processed_spike_freq_dict = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=kernel_sigma,
        #                                                                  time_range=time_range, dt_step=bin_step)

        processed_spike_freq_dict = spike_count_dict
        processed_spike_freq_array = burstUtils.processed_sfd_to_array(processed_spike_freq_dict)

        fig1, ax1 = burstUtils.plotFreq(processed_spike_freq_dict, scatter_plot=False, color_dict=burst_object.color_dict)
        if burst_object.isDe3 is not None:
            fig1.suptitle(basename + "\n De3 is %i" % burst_object.isDe3)
        else:
            fig1.suptitle(basename + "\nDe3 is None")
        # fig1.savefig(save_folder + basename +"_units.png", dpi=600, transparent=True)

        i = 0
        fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(2,3, sharex=True, sharey=True)
        for corr_time in corr_times:
            ws = int(corr_time / time_step)
            t_min = corr_time / 2
            t_max = burst_object.time[-1] - corr_time / 2

            fc_matrixes = corrUtils.multiUnitSlidingWxindow(processed_spike_freq_array, window_size=ws, step=corr_step)
            fc_matrixes[np.isnan(fc_matrixes)] = 0

            flat_idxs = corrUtils.getFlattenedUpperTriangIdxs(fc_matrixes.shape[1], 1)
            df = pd.DataFrame(fc_matrixes.reshape(fc_matrixes.shape[0], fc_matrixes.shape[1]**2)[:, flat_idxs].T)

            spearman_corr = [df.corr().values]

            mappable = ax2.flatten()[i].matshow(spearman_corr[0], vmin=-1, vmax=1, extent=[t_min, t_max, t_max, t_min], cmap='bwr')

            # fc_matrixes = (np.abs(fc_matrixes) > 0.5).astype(float)
            # df = pd.DataFrame(fc_matrixes.reshape(fc_matrixes.shape[0], fc_matrixes.shape[1]**2)[:, flat_idxs].T)
            #
            # spearman_corr = [df.corr().values]
            # print(spearman_corr[0].shape)
            mappable = ax.flatten()[i].matshow((spearman_corr[0] > corr_thres).astype(float), vmin=-1, vmax=1, extent=[t_min, t_max, t_max, t_min], cmap='bwr')

            ax.flatten()[i].set_xticklabels([''])
            ax2.flatten()[i].set_xticklabels([''])
            ax.flatten()[i].set_title("window_size: " + str(corr_times[i]))
            ax.flatten()[i].set_aspect('equal', adjustable='box')
            ax.flatten()[i].get_shared_x_axes().join(ax.flatten()[i], ax1[0])
            ax2.flatten()[i].set_title("window_size: " + str(corr_times[i]))
            ax2.flatten()[i].set_aspect('equal', adjustable='box')
            ax2.flatten()[i].get_shared_x_axes().join(ax2.flatten()[i], ax1[0])
            fig.suptitle(basename)
            fig2.suptitle(basename)

            i += 1
            # fig_sin1, ax_sin1 = plt.subplots()
            # ax_sin1.matshow(np.abs(fc_matrixes[int(550/time_step)]>0.05).astype(float), vmin=-1, vmax=1, cmap='bwr')
            # fig_sin2, ax_sin2 = plt.subplots()
            # ax_sin2.matshow(fc_matrixes[int(650/time_step)], vmin=-1, vmax=1, cmap='bwr')
        f, a = plt.subplots()
        a.matshow(fc_matrixes[int(fc_matrixes.shape[0]/2)], vmin=-1, vmax=1, cmap='bwr')
        f.suptitle(basename)
        reordered_sfd, label_dict = corrUtils.reorderSpikeFreqDict(fc_matrixes[500], processed_spike_freq_dict, int(fc_matrixes.shape[1]/4), plot_reordered_mat=True)
        fig1, ax1 = burstUtils.plotFreq(reordered_sfd, scatter_plot=False, color_dict=burst_object.color_dict, label_dict=label_dict)
        fig1.suptitle(basename)
        # fig.colorbar(mappable=mappable)
        # fig.savefig(save_folder + basename + "_w_corr_matrices.png", dpi=600, transparent=True)
        # fig2.savefig(save_folder + basename + "_uw_corr_matrices.png", dpi=600, transparent=True)



    # miscUtils.sleepPc(600)

        # fig, ax = plt.subplots()
        # mappable = ax.matshow(spearman_corr[0], vmin=-1, vmax=1, extent=[t_min, t_max, t_min, t_max])
        # mappable = ax.matshow((np.abs(spearman_corr[0])>corr_thres).astype(int), extent=[t_min, t_max, t_max, t_min], cmap='bwr')
        # fig.colorbar(mappable=mappable)
        # fig.suptitle(basename)


        #
        # fig, ax = plt.subplots()
        # subm = fc_matrixes
        # def animate(i):
        #
        #
        #     ax.matshow(subm[i], vmin=-1, vmax=1)
        #
        #     return fig,
        # # anim = animation.FuncAnimation(fig, animate, frames=(fc_matrixes, np.arange(0,fc_matrixes.shape[0],1)),
        # #                                interval=100, blit=True)
        # anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, subm.shape[0], 1),
        #                                interval=100, blit=False)
        # mywriter = animation.FFMpegWriter(fps=100, extra_args=['-vcodec', 'libx264'])
        # anim.save("vid_prueba.mp4",  writer=mywriter)
        #



    #
    # if False:
    #
    #     if False:
    #
    #
    #
    #
    #
    #
    #         data_len = processed_spike_freq_dict[selected_neurons[0]].shape[1]
    #         centers = np.arange(0, data_len, corr_step, dtype=int) + int(corr_step / 2)
    #
    #         array_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, channels=ch_dict)
    #
    #         if (time_vector[-1]>1500) or time_vector[-1]<1100:
    #             continue
    #         time_vector = time_vector[::int(fs * bin_step)]
    #         idxs = np.linspace(0, data_len, num=num + 1, dtype=int)
    #
    #         fig1, axes1 = corrUtils.rasterPlot(spike_count_dict, color_dict=burst_object.color_dict, linewidths=.5)
    #
    #         # fig1, axes1 = burstUtils.plotFreq(processed_spike_freq_dict, color_dict=burst_object.color_dict, scatter_plot=False,
    #         #                                 draw_list=selected_neurons, facecolor='lightgray', legend=False)
    #
    #         fig1.suptitle("%s\nDE-3 is %i" % (basename, burst_object.isDe3))
    #         # axes[0].set_xlim([200, 600])
    #         # for ax in axes1:
    #         #     ax.grid(None)
    #         #     for id in idxs:
    #         #         ax.axvline(id * bin_step, color='r')
    #         # fig1.subplots_adjust(wspace=0, hspace=0)
    #         # fig1.legend()
    #
    #
    #         neuron_k = burst_object.isDe3
    #         print(basename, burst_object.isDe3)
    #         neurons = np.sort(selected_neurons)
    #
    #         # neurons = list(selected_neurons)
    #         # if any([select in fn for select in plot_list]):
    #         # fig.savefig("correlation_figs/pearsons_correlation/" + basename + "_segments_scatter.png", transparent=True, dpi=600)
    #
    #         # fig2, ax2 = plt.subplots(len(neurons) - 1, 1, sharex=True)
    #         # fig2, ax2 = plt.subplots()
    #         # fig2.suptitle("%s\nDE-3 is %i, cosine" % (basename, burst_object.isDe3))
    #         # fig3, ax3 = plt.subplots()
    #         # fig3.suptitle("%s\nDE-3 is %i, pearson" % (basename, burst_object.isDe3))
    #         # i = 0
    #
    #         # corrUtils.multiUnitSlidingWindow()
    #         #
    #         # if type(burst_object.color_dict[neuron_j]) is list:
    #         #     c = burst_object.color_dict[neuron_j][0]
    #         # else:
    #         #     c = burst_object.color_dict[neuron_j]
    #         # ax2.plot(time_vector[idxs], corr, marker='o', label=neuron_j, c=c)
    #         # ax2.set_ylim([-.1, 1])
    #         # ax3.plot(time_vector[idxs2], corr2, marker='o', label=neuron_j, c=c)
    #         # ax3.set_ylim([-1, 1])
    #         #
    #
    #
    #             # i += 1
    #
    #         # fig2.legend()
    #         # fig3.legend()
    #         # axes1.get_shared_x_axes().join(axes1, ax2)
    #         # axes1.get_shared_x_axes().join(axes1, ax3)
    #         # axes1[0].get_shared_x_axes().join(axes1[0], ax3)
    #         # fig.savefig("correlation_figs/pearsons_correlation/" + basename + "_correlations.png", transparent=True, dpi=600)
    #
    #         # del burst_object, processed_spike_freq_dict, binned_sfd, array_dict
    #         # gc.collect()
