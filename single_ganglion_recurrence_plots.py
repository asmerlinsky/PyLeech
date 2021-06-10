import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as bStorerLoader
import PyLeech.Utils.burstUtils
import PyLeech.Utils.burstUtils as burstUtils

import numpy as np
import scipy.signal as spsig

import matplotlib.pyplot as plt

from itertools import combinations


if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    cdb = CDU.loadCrawlingDatabase()

    file_list = []
    for files in list(cdd.keys()):
        burst_obj = bStorerLoader.UnitInfo(files, 'RegistrosDP_PP', mode='load')
        try:
            arr_dict, time_vector1, fs = abfe.getArraysFromAbfFiles(files, ['Vm1'])
            file_list.append(files)
        except:
            pass



    # for fn1, fn2 in combinations(file_list, 2):
    for fn in [file_list[2]]:

        arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])
        NS = arr_dict['Vm1']

        del arr_dict


        NS[(NS > -20) | (NS < -60)] = np.median(NS)

        good_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if neuron_dict['neuron_is_good']]

        crawling_intervals = cdd[fn]['crawling_intervals']
        print(crawling_intervals)

        binning_dt = 0.1

        burst_obj = bStorerLoader.UnitInfo(fn, 'RegistrosDP_PP', mode='load')

        kernel_sigma = 1

        new_sfd = burstUtils.removeOutliers(burst_obj.spike_freq_dict, 5)
        binned_sfd = burstUtils.digitizeSpikeFreqs(new_sfd, time_vector[-1], counting=False)
        cut_binned_freq_array = burstUtils.binned_sfd_to_dict_array(binned_sfd, crawling_intervals, good_neurons)
        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=20, dt_step=binning_dt)
        smoothed_sfd = {}
        for key, items in binned_sfd.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])

        step = int(binning_dt * fs)

        neuron_idxs = []
        for interval in crawling_intervals:
            neuron_idxs.append(np.where((time_vector > interval[0]) & (time_vector < interval[1]))[0][::step])
        neuron_idxs = np.hstack(neuron_idxs)

        cut_time1 = time_vector[neuron_idxs]

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=30, dt_step=1 / fs)
        conv_NS = spsig.fftconvolve(NS, kernel, mode='same')
        cut_NS = conv_NS[neuron_idxs]
        #
        # fig, axes = burstUtils.plotFreq(smoothed_sfd, color_dict=burst_obj.color_dict,
        #                                 template_dict=burst_obj.template_dict, scatter_plot=False,
        #                                 optional_trace=[time_vector[::int(step / 4)], conv_NS[::int(step / 4)]],
        #                                 # draw_list=[list(smoothed_sfd)[0]],
        #                                 outlier_thres=None, ms=4)
        fig, axes = burstUtils.plotFreq(burst_obj.spike_freq_dict, color_dict=burst_obj.color_dict,
                                        template_dict=burst_obj.template_dict, scatter_plot=True,
                                        optional_trace=[time_vector[::int(step / 4)], conv_NS[::int(step / 4)]],
                                        # draw_list=[list(smoothed_sfd)[0]],
                                        outlier_thres=10, ms=4)
        # fig.suptitle(os.path.basename(os.path.splitext(fn)[0]))
        # # reload(burstClasses)
        # segmented_data = burstClasses.SegmentandCorrelate(binned_sfd, NS, time_vector, fs, crawling_intervals,
        #                                                   intracel_peak_height=None, intracel_prominence=3, sigma=1,
        #                                                   intracel_peak_distance=10)

        # NS_embedding = NLD.getDerivativeEmbedding(cut_NS, 0.1, 3)
        # jumps = np.where(np.diff(neuron_idxs) != np.diff(neuron_idxs)[0])[0]
        # if len(jumps) > 0:
        #     NS_embedding[jumps] = np.nan
        #     NS_embedding[jumps + 1] = np.nan
        #
        # NLD.plot3Dscatter(NS_embedding)
        #
        #
        #
        # # NLD.plot3Dline(scaled_embedding1)
        # segment_counts = StandardScaler()
        # scaled_embedding = segment_counts.fit_transform(NS_embedding)
        # for i in [1, 1.2, 1.5, 1.8, 2, 2.5, 3, 3.5, 4, 5]:
        # # for i in [.8, 1, 1.2]:
        #     NS_embedding = NLD.getTraceEmbedding(cut_NS[:], int(i/.1), 3)
        #     fig, ax = NLD.plot3Dline(NS_embedding, colorbar=False)
        #     fig.suptitle(str(i))
        #
        # NS_embedding = NLD.getTraceEmbedding(cut_NS[:], int(3 / .1), 3)
        # rt1 = NLD.getCloseReturns(NS_embedding)
        #
        # reord_rt, fig1, ax1, fig2, ax2 = NLD.plotCloseReturns(rt1, reorder=True, masked=True, thr=.05, return_reordered=True)
        # segment_dict, segment_counts = NLD.getCloseReturnsSegments(reord_rt, 25)
        # close_return_idxs = np.where(rt1< .01 * np.max(rt1))
        # fig1.suptitle(fn)
        # fig2.suptitle(fn)
        #
        # fig, ax = NLD.plot3Dline(NS_embedding[2750:3400], colorbar=False)

        spikes_returns_dict = {}

        neuron_idxs = []
        for interval in crawling_intervals:
            neuron_idxs.append(np.where((smoothed_sfd[list(smoothed_sfd)[0]][0] > interval[0]) & (smoothed_sfd[list(smoothed_sfd)[0]][0] < interval[1]))[0])
        neuron_idxs = np.hstack(neuron_idxs)
        jumps = np.where(np.diff(neuron_idxs) != np.diff(neuron_idxs)[0])[0]

        for key, items in smoothed_sfd.items():
                if key in [5, 10, 11, 12, 14, 21]:
                    # items[1][items[1]<10**(-10)] = 0.
                    spike_embedding = NLD.getTraceEmbedding(items[1][neuron_idxs], int(3 / .1), 3)

                    if len(jumps) > 0:
                        spike_embedding[jumps] = np.nan
                        spike_embedding[jumps + 1] = np.nan
                    rt1 = NLD.getCloseReturns(spike_embedding)
                    reord_rt, fig1, ax1, fig2, ax2 = NLD.plotCloseReturns(rt1, reorder=False, masked=True, thr=.1,
                                                                          return_reordered=True)

                    # fig1.suptitle(key)
                    plt.close(fig1)
                    fig2.suptitle(key)
                    spikes_returns_dict[key] = reord_rt

            # spike_embedding = NLD.getTraceEmbedding(smoothed_sfd[21][1][neuron_idxs], int(3 / .1), 3)
            # fig, ax = NLD.plot3Dline(spike_embedding[650:1300], colorbar=False)
            # spike_embedding = NLD.getTraceEmbedding(smoothed_sfd[28][1][neuron_idxs], int(3 / .1), 3)
            # fig, ax = NLD.plot3Dline(spike_embedding[1500:2250], colorbar=False)
            #
            # spike_embedding = NLD.getTraceEmbedding(smoothed_sfd[20][1][neuron_idxs], int(3 / .1), 3)
            # fig, ax = NLD.plot3Dline(spike_embedding[1080:1400], colorbar=False)
            # spike_embedding = NLD.getTraceEmbedding(smoothed_sfd[19][1][neuron_idxs], int(3 / .1), 3)
            # fig, ax = NLD.plot3Dline(spike_embedding[1080:1400], colorbar=False)



        mat = np.ones(spikes_returns_dict[list(spikes_returns_dict)[0]].shape)
        for (n1, item1), (n2, item2) in combinations(spikes_returns_dict.items(), 2):

            fig, ax = plt.subplots()
            ax.matshow(item1 * item2)
            fig.gca().set_aspect('auto')
            fig.suptitle('neuron pair: ' + str(n1) + '-' + str(n2))