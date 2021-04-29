import os
import numpy as np
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as unitInfo
import PyLeech.Utils.burstUtils as burstUtils
from scipy.signal import correlate
import matplotlib.pyplot as plt

import time
import PyLeech.Utils.correlationUtils as corrUtils

if __name__ == '__main__':
    start = time.time()
    cdd = CDU.loadDataDict()


    folder = 'lidia_figs/freq_ordenada_por_nervio/'
    plot_freq = True
    plot_corr = True
    plot_completo = True
    plot_crawling = True
    save_fig = False
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




    for fn in run_list:
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        crawling_segment = cdd[fn]['crawling_intervals']
        basename = os.path.splitext(os.path.basename(fn))[0]
        if crawling_segment[0] != -1:

            extended_segement = (crawling_segment[0] - 30, crawling_segment[1] + 30)
            burst_object = unitInfo.UnitInfo(fn, mode='load')

            binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                            time_length=burst_object.time_length, counting=True, selected_neurons=selected_neurons,
                                                         time_interval=extended_segement)

            crawling_idxs = (binned_sfd[burst_object.isDe3][0]>crawling_segment[0]) & (binned_sfd[burst_object.isDe3][0]<crawling_segment[1])
            De3_counts = binned_sfd[burst_object.isDe3][1][crawling_idxs]
            # smoothed_sfd = burstUtils.smoothBinnedSpikeFreqDict(binned_sfd, sigma=bin_step, time_range=bin_step, dt_step=bin_step)

            ############
            # spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]
            #
            # burst_start = planUtils.getBurstsStart(spike_times, isi_threshold=1/min_spike_per_sec, above_threshold_tol=.1)



            # burstUtils.plotFreq(binned_sfd, color_dict='k', scatter_plot=False, outlier_thres=None, facecolor='white')
            cc_dict, fig, ax = corrUtils.plotCrossCorrelation(binned_sfd, bin_step, burst_object.isDe3, De3_counts=De3_counts)#, shift_range=None)

            fig.suptitle(basename + "\nDe3 cross correlation")

            # if save_fig:
            #     fig.set_size_inches(18.5, 10.5)
            #     fig.savefig(folder + 'solo_crawling/' + 'crosscorr_' + basename + '.png', dpi=800)#, transparent=True)

            fig_dict = burstUtils.plotCorrByNerve(cc_dict, color_dict='k', draw_list=selected_neurons,
                                                  scatter_plot=False,
                                                  nerve_unit_dict=burst_object.nerve_unit_dict, outlier_thres=None,
                                                  facecolor='white',
                                                  De3=burst_object.isDe3, ms=10)

            for nerve, figs in fig_dict.items():
                figs[0].suptitle(basename + '\n' + nerve + ' cross correlations')
                figs[0].set_size_inches(18.5, 10.5)
                figs[1][-1].set_xlabel("Shift (s)")
                # figs[0].savefig(folder + 'solo_crawling/' + nerve + "_" + basename + '_crosscorr.png', dpi=600)#, transparent=True)

    print(time.time()-start)