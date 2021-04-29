import os

import PyLeech.Utils.planUtils as pU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.CrawlingDatabaseUtils as CDU






if __name__ == "__main__":

    bin_step = .2

    cdd = CDU.loadDataDict()
    #
    # sel = "2019_08_26_0002"
    # run_list = []
    # for fn in list(cdd):
    #     if sel in fn:
    #         run_list.append(fn)

    run_list = list(cdd)
    tot = 0
    no_dc_sub50 = 0
    no_start = 0
    period_range = (5, 25)
    for fn in run_list:
        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            print("file %s has no De3 assigned" % fn)
            continue

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("\\", '/')

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]
        spike_freqs = burst_object.spike_freq_dict[burst_object.isDe3][1]
        spike_times = spike_times[~burstUtils.is_outlier(spike_freqs, 5)]

        start_time = pU.getBurstsStart(spike_times, isi_threshold=.1, window_size=200, step=50)
        if start_time is None:
            print("didn't` find proper start time")
        spike_times = spike_times[spike_times>start_time]

        median_time, duration, cycle_period, duty_cycle, burst_ini, burst_end = pU.getBurstingInfo(spike_times,
                                                                                                   min_spike_no=15,
                                                                                                   min_spike_per_sec=10.)

        crawling_segment = pU.getCrawlingSegments(spike_times, period_range, min_spike_per_sec=10, min_spike_no=15,)
        print(crawling_segment)
        #
        # f, a = burstUtils.plotFreq(burst_object.spike_freq_dict, draw_list=[burst_object.isDe3], scatter_plot=True,
        #                            outlier_thres=3.5)
        # f.suptitle(basename)
        # a[0].axvline(burst_ini[crawling_segment[0]], c='r')
        # a[0].axvline(burst_ini[crawling_segment[1] + 1], c='r')

        crawling_interval = (burst_ini[crawling_segment[0]], burst_ini[crawling_segment[1] + 1])



        cut_spike_freq_dict = burstUtils.cutSpikeFreqDict(burst_object.spike_freq_dict, crawling_interval, outlier_threshold=10)

        binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                                     selected_neurons=selected_neurons,
                                                     time_length=burst_object.time[-1], counting=False, time_interval=crawling_interval)




        f, a = burstUtils.plotFreq(cut_spike_freq_dict, scatter_plot=True)
        f.suptitle(basename)

    # print(no_dc_sub50, no_start, tot)