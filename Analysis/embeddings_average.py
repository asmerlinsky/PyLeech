# import sys
# [sys.path.append(i) for i in ['.', '..']]

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstClasses as burstClasses
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyLeech.Utils.AbfExtension as abfe
from PyLeech.Utils.burstStorerLoader import BurstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig
import matplotlib as mpl
import os
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"

import seaborn as sns
sns.jointplot()

if __name__ == "__main__":

    cdd = CDU.loadDataDict("RegistrosDP_PP/embeddings_CDD.json")
    # cdd = CDU.loadDataDict("RegistrosDP_PP/CrawlingDataDict.json")
    file_list = list(cdd)
    k = 1
    draw_everything = False
    draw_segments_by_file = True
    mean_tray_dict = {}
    for fn in [file_list[2]]:
        # if cdd[fn]["skipped"]:
        #     continue
        # print(fn)
        try:
            burst_object = BurstStorerLoader(fn, 'RegistrosDP_PP', 'load')
            try:
                arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, ["IN5"])
            # except IndexError:
            #     arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(os.path.splitext(fn)[0] + ".abf", ["IN5"])
            except:
                pass
        except:
            print("final exception")
            print(Exception)
            continue

        #     dp_trace = arr_dict['IN6']
        #     NS = arr_dict['Vm1']
        #     del arr_dict
        #     NS[(NS > -20) | (NS < -70)] = np.median(NS)
        #     print("running %s" % fn)
        #
        # except IndexError:
        #     print("file %s failed" % fn)
        #     NS = None
        #     pass

        good_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if neuron_dict['neuron_is_good']]

        crawling_intervals = cdd[fn]['crawling_intervals']
        print(crawling_intervals)
        binning_dt = 0.1
        spike_kernel_sigma = 1
        if fn == file_list[2]: ## esto se quedo viejo
            crawling_intervals[0][1] = 960

        NS_kernel_sigma = 5

        spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, .1, selected_neurons=good_neurons,
                                                           time_length=burst_object.time[-1])

        kernel = NLD.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)
        smoothed_sfd = {}
        for key, items in spike_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])

        step = int(binning_dt * fs)
        neuron_idxs = []
        spike_idxs = []
        analized_neuron = list(smoothed_sfd)[0]
        for interval in crawling_intervals:
            # for interval in [[450, 550], [662, 725]]:
            neuron_idxs.append(np.where((burst_object.time > interval[0]) & (burst_object.time < interval[1]))[0][::step])
            spike_idxs.append(np.where(
                (smoothed_sfd[analized_neuron][0] > interval[0]) & (smoothed_sfd[analized_neuron][0] < interval[1]))[0])
        neuron_idxs = np.hstack(neuron_idxs)
        spike_idxs = np.hstack(spike_idxs)
        neuron_cut_time = burst_object.time[neuron_idxs]

        fig, axes = burstUtils.plotFreq(burst_object.spike_freq_dict, color_dict=burst_object.color_dict,
                                        template_dict=burst_object.template_dict, scatter_plot=True,
                                        outlier_thres=3, ms=4, draw_list=good_neurons)

        fig.suptitle(fn)
        sp_embedding = NLD.getDerivativeEmbedding(smoothed_sfd[analized_neuron][1][spike_idxs], .1, 3)
        freq_threshold = 1
        if fn == file_list[2]:
            num = 50
            freq_threshold = 1
        elif fn == file_list[0]:
            num = 250
            freq_threshold = 1
        elif fn == "RegistrosDP_PP\\2018_11_06_0004.pklspikes":
            analized_neuron = 22
        else:
            num = 250

        sp_embedding[(sp_embedding[:, 2] > num) | (sp_embedding[:, 2] < -num)] = np.nan

        # NLD.plot3Dline(sp_embedding)

        idxs = burstUtils.getNonZeroIdxs(sp_embedding[:, 0], freq_threshold)
        intervals = idxs.reshape(int(idxs.shape[0] / 2), 2)

        fig, ax = plt.subplots()
        ax.plot(smoothed_sfd[analized_neuron][0][spike_idxs], sp_embedding[:, 0])
        j = 0
        for i in intervals.flatten():
            ax.axvline(smoothed_sfd[analized_neuron][0][spike_idxs][i], c='r')
            if j % 2 == 0:
                ax.text(smoothed_sfd[analized_neuron][0][spike_idxs][i], 0, str(int(j / 2)))
            j += 1
        fig.suptitle(fn)
        step = 5

        if draw_everything:
            j = 0
            for i in np.arange(0, intervals.shape[0], step):

                fig = plt.figure()
                ax = Axes3D(fig)

                cmap = plt.get_cmap('jet')
                cNorm = mpl.colors.Normalize(vmin=0, vmax=step - 1)
                scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

                data_mean = NLD.getSegmentsMean(sp_embedding, intervals[i:i + step])
                iter = 0
                max_int = np.max(np.diff(intervals[i:i + step]))

                for start, end in intervals[i:i + step]:
                    diff = end - start
                    NLD.plot3Dline(sp_embedding[start:end] * diff / max_int, fig_ax_pair=[fig, ax], colorbar=False,
                                   color=scalarMap.to_rgba(iter), label=str(j))
                    iter += 1
                    j += 1

                NLD.plot3Dline(data_mean, fig_ax_pair=[fig, ax], colorbar=False, color=None)

                fig.suptitle(os.path.basename(fn) + '\n' + str(i) + ':' + str(i + step))
                fig.legend()
            plt.show()
            plt.ginput(n=0, timeout=0, mouse_pop=2, mouse_stop=3)
            plt.close('all')

        if fn == file_list[0]:
            segments = [[1, 4], [2, 7, 9], [15, 18, 19, 20, 22]
                        ]
        elif fn == file_list[1]:
            segments = [[0, 4, 6, 8, 18, 25], [5, 7, 11, 13, 16], [20, 21, 26, 28, 29]
                        ]
        elif fn == file_list[2]:
            segments = [[1, 4, 5, 6, 7, 10], [9, 12],  # [14, 17],
                        [11, 16, 19, 21, 22, 27, 30, 33, 35, 36, 38, 42], [26, 28, 44], [31, 37, 39, 41]
                        ]
        elif fn == file_list[3]:
            segments = [[0, 1, 4, 5], [2, 3, 5, 6, 7, 8, 10, 11], [19, 25], [17, 20, 21], [26, 27, 28]

                        ]
        elif fn == file_list[4]:
            segments = [[0, 1, 2, 4, 5, 6, 7, 8], [3, 9], [17, 20, 21], [14, 19, 24, 25], [16, 18, 22, 26, 27, 28],

                        ]
        elif fn == file_list[5]:
            segments = [[1, 4], [8, 10], [5, 7, 9], [0, 2, 3],

                        ]
        else:
            segments = []

        # NLD.plot3Dline(sp_embedding[intervals[20,0]:intervals[27,1]])

        # if draw_everything:
        figuras = []
        if draw_segments_by_file:
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            j = 0
            means_list = []
            for interval_list in segments:
                data_mean = NLD.getSegmentsMean(sp_embedding, intervals[interval_list])

                means_list.append(data_mean)

                cmap = plt.get_cmap('jet')
                cNorm = mpl.colors.Normalize(vmin=0, vmax=len(interval_list) - 1)
                scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

                fig = plt.figure(figsize=(15,15))
                ax = Axes3D(fig)
                iter = 0
                max_int = np.max(np.diff(intervals[interval_list]))
                for elem in interval_list:
                    diff = np.diff(intervals[elem])
                    NLD.plot3Dline(sp_embedding[intervals[elem, 0]:intervals[elem, 1]] * diff / max_int,
                                   fig_ax_pair=[fig, ax],
                                   color=scalarMap.to_rgba(iter),
                                   colorbar=False, label=str(elem))
                    iter += 1
                fig.suptitle(str(interval_list))
                NLD.plot3Dline(data_mean, fig_ax_pair=[fig, ax], colorbar=False, color='k')


                NLD.plot3Dline(data_mean, fig_ax_pair=[fig1, ax1], colorbar=False, color='k')
                figuras.append([str(interval_list), fig])
                # fig.savefig("figuras_charla/" + str(interval_list) + ".png", dpi=600, transparent=True)

                # def animate(i):
                #     ax.view_init(azim=i)
                #     return fig,
                # anim = animation.FuncAnimation(fig, animate, frames=np.arange(0,362,1),
                #                                interval=100, blit=True)
                # mywriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
                # anim.save("figuras_charla/" + str(interval_list) + ".mp4",  writer=mywriter)


                j += 1

                # fig.legend()

            plt.show()
            # plt.ginput(n=0, timeout=0, mouse_pop=2, mouse_stop=3)
            # plt.close('all')
            mean_tray_dict[fn] = means_list
            fig1.savefig("figuras_charla/" + str(interval_list) + ".png", dpi=600, transparent=True)
        else:
            means_list = []
            for interval_list in segments:
                data_mean = NLD.getSegmentsMean(sp_embedding, intervals[interval_list])

                means_list.append(data_mean)
            mean_tray_dict[fn] = means_list

    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.get_cmap('jet')
    cNorm = mpl.colors.Normalize(vmin=0, vmax=len(list(mean_tray_dict)) - 1)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    for key, items in mean_tray_dict.items():
        first = True
        j = 0
        cNorm = mpl.colors.Normalize(vmin=0, vmax=len(items) - 1)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        for orbit in items:
            if first:
                NLD.plot3Dline(orbit, fig_ax_pair=[fig, ax], colorbar=False, color=scalarMap.to_rgba(j),
                               label=os.path.basename(os.path.splitext(key)[0]))
                first = False
            else:
                NLD.plot3Dline(orbit, fig_ax_pair=[fig, ax], colorbar=False, color=scalarMap.to_rgba(j))
            j += 1
    # fig.legend()
    # fig.savefig("figuras_charla/every_embedding.png", dpi=600, transparent=True)
    plt.show()

    ##### Averaging NS embeddings

    # if False:
    #     kernel = NLD.generateGaussianKernel(sigma=NS_kernel_sigma, time_range=30, dt_step=1 / fs)
    #     conv_NS = spsig.fftconvolve(NS, kernel, mode='same')
    #
    #     ns_segmented = burstClasses.NSSegmenter(burst_object.spike_freq_dict, conv_NS, time_vector, NS_is_filtered=True,
    #                                             time_intervals=crawling_intervals, peak_height=np.median(NS[neuron_idxs]),
    #                                             prominence=1)
    #     ns_segmented.intervals
    #
    #     NS_interval_idxs = []
    #     for interval in ns_segmented.intervals:
    #         NS_interval_idxs.append(np.argmin(np.abs(time_vector[neuron_idxs, np.newaxis] - interval), axis=0))
    #
    #     NS_interval_idxs = np.array(NS_interval_idxs)[:-2]
    #     fig, ax = plt.subplots()
    #     ax.plot(time_vector[::50], conv_NS[::50])
    #     for ln in time_vector[neuron_idxs][NS_interval_idxs].flatten():
    #         ax.axvline(ln, c='r')
    #
    #     ml = []
    #     NS_embedding = NLD.getDerivativeEmbedding(conv_NS[neuron_idxs], .1, 3)
    #     NS_embedding[NS_embedding[:, 1] < -2] = np.nan
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     for start, end in NS_interval_idxs:
    #         NLD.plot3Dline(NS_embedding[start:end], fig_ax_pair=[fig, ax], colorbar=False)
    #
    #     for i in np.arange(0, NS_interval_idxs.shape[0], 3):
    #         fig = plt.figure()
    #         ax = Axes3D(fig)
    #         data_mean = NLD.getSegmentsMean(NS_embedding, NS_interval_idxs[i:i + 3])
    #         ml.append(data_mean)
    #         for start, end in NS_interval_idxs[i:i + 3]:
    #             NLD.plot3Dline(NS_embedding[start:end], fig_ax_pair=[fig, ax], colorbar=False)
    #         NLD.plot3Dline(data_mean, fig_ax_pair=[fig, ax], colorbar=False, color=None)
    #         fig.suptitle(str(i) + ':' + str(i + 3))
    #
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     j = 0
    #     for arr in ml:
    #         NLD.plot3Dline(arr, fig_ax_pair=[fig, ax], colorbar=False, color=scalarMap.to_rgba(j), label=str(j))
    #         j += 1
    #     fig.legend()
