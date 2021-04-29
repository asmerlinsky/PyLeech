import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as unitInfo
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.planUtils as planUtils
import PyLeech.Utils.correlationUtils as corrUtils
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import math

def partspearman(x, y):
    return spearmanr(x, y)[0]

if __name__ == '__main__':
    cdd = CDU.loadDataDict()


    folder = 'lidia_figs/freq_ordenada_por_nervio/'
    plot_freq = True
    plot_corr = True
    plot_completo = True
    plot_crawling = True

    min_spike_per_sec = 10
    min_spike_no = 15
    period_range = (10., 30.)

    bin_step = .1
    sigma = 1
    run_list = []

    for fn in list(cdd):
        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            pass
        else:
            run_list.append(fn)


    # run_list = [run_list[7]]

    for fn in run_list:
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        crawling_segment = cdd[fn]['crawling_intervals']
        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = unitInfo.UnitInfo(fn, mode='load')
        binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                        time_length=burst_object.time_length, counting=True, selected_neurons=selected_neurons)

        smoothed_sfd = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=bin_step, time_range=bin_step, dt_step=bin_step)

        ############
        spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]

        burst_start = planUtils.getBurstsStart(spike_times, isi_threshold=1 / min_spike_per_sec, above_threshold_tol=.1)

        cs, dc_sub50 = planUtils.getCrawlingSegments(spike_times[spike_times > burst_start],
                                                                   period_range=period_range,
                                                                   min_spike_per_sec=min_spike_per_sec,
                                                                   min_spike_no=min_spike_no)

        burst_info = planUtils.getBurstingInfo(spike_times[spike_times > burst_start], min_spike_no=min_spike_no,
                                               min_spike_per_sec=min_spike_per_sec)


        ############
        corr_dict = {}
        for key, items in smoothed_sfd.items():
            if key != burst_object.isDe3:
                time, counts = items
                idxs, corr = corrUtils.pairwiseSlidingWindow(binned_sfd[burst_object.isDe3][1], counts, func=partspearman, step=1, window_size=60)
                corr_dict[key] = np.array((time[idxs], corr))

        ############
        if plot_freq:
            fig_dict = burstUtils.plotFreqByNerve(smoothed_sfd, color_dict='k', draw_list=selected_neurons, scatter_plot=False,
                                                  nerve_unit_dict=burst_object.nerve_unit_dict, outlier_thres=None,
                                                  facecolor='white', De3=burst_object.isDe3)

            for nerve, figs in fig_dict.items():
                figs[0].suptitle(basename + '\n' + nerve + ' frequencies')
                if plot_completo:
                    figs[0].savefig(folder + 'completos/' + nerve + '_' + basename + '.png', dpi=600)#, transparent=True)
                if plot_crawling:
                    if crawling_segment[0] != -1:
                        figs[1][0].set_xlim(crawling_segment)
                        figs[0].savefig(folder + 'solo_crawling/' + nerve + '_' + basename + '.png', dpi=600)#, transparent=True)

        ############
        if plot_corr:
            fig_dict = burstUtils.plotCorrByNerve(corr_dict, color_dict='k', draw_list=selected_neurons, scatter_plot=False,
                                                  nerve_unit_dict=burst_object.nerve_unit_dict, outlier_thres=None, facecolor='white',
                                                  De3=burst_object.isDe3, burst_info=burst_info, ms=10)

            for nerve, figs in fig_dict.items():
                figs[0].suptitle(basename + '\n' + nerve + ' correlations')
                if plot_completo:
                    figs[0].savefig(folder + 'completos/' + nerve + '_corr_' + basename + '.png', dpi=600)#, transparent=True)
                if plot_crawling:
                    if crawling_segment[0] != -1:
                        figs[1][0].set_xlim(crawling_segment)
                        figs[0].savefig(folder + 'solo_crawling/' + nerve + '_corr_' + basename + '.png', dpi=600)#, transparent=True)




    # if plot_crawling:
    #     for fn in run_list:
    #         ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}
    #
    #         selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
    #                             neuron_dict["neuron_is_good"]]
    #
    #         fn = fn.replace("/", '/')
    #
    #         basename = os.path.splitext(os.path.basename(fn))[0]
    #
    #         burst_object = unitInfo.UnitInfo(fn, mode='load')
    #
    #         fig_dict = burstUtils.plotFreqByNerve(burst_object.spike_freq_dict, color_dict=burst_object.color_dict,
    #                                               nerve_unit_dict=burst_object.nerve_unit_dict, outlier_thres=3.5)
    #
    #         for nerve, figs in fig_dict.items():
    #             figs[1].set_xlim(burst_object)
    #             figs[0].savefig(folder + 'solo_crawling/' + nerve + '_' + basename + '.png', dpi=600, transparent=True)