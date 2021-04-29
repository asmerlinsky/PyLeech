import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as unitInfo
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.planUtils as planUtils
import glob
import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == "__main__":

    cdd = CDU.loadDataDict()

    min_spike_per_sec = 10
    min_spike_no = 15
    period_range = (10., 30.)
    bin_step = 2
    plot_list = ["burst duration", "cycle period", "burst duty cycle"]
    folder_name = "lidia_figs/todos_los_registros/per_10_30/"
    correr_todo = False
    correr_crawling = True
    save_figs = False
    run_list = []

    # sel = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob("registros_abf_compartidos/*.abf")]
    #
    for fn in list(cdd):
        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            pass
        else:
            run_list.append(fn)


        # if any([s in fn for s in sel]):
        #     run_list.append(fn)

    # for fn in sel:
    #     if not any([fn in s for s in run_list]):
    #         print("%s is not in run list" % fn)


    promedios = {key: [] for key in plot_list}

    """
    ## este los corre enteros
    """
    if correr_todo:
        no_dc_sub50 = 0
        for fn in run_list:

            ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

            cdd_de3 = cdd[fn]['DE3']
            selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                                neuron_dict["neuron_is_good"]]

            fn = fn.replace("/", '/')

            basename = os.path.splitext(os.path.basename(fn))[0]

            burst_object = unitInfo.UnitInfo(fn, mode='load')

            if cdd_de3 != burst_object.isDe3:
                print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn,
                                                                                                                   cdd_de3,
                                                                                                                   burst_object.isDe3))

            spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]

            burst_info = planUtils.getBurstingInfo(spike_times, min_spike_no=min_spike_no,
                                                   min_spike_per_sec=min_spike_per_sec)

            median_time = burst_info["burst median time"]
            duration = burst_info["burst duration"]
            cycle_period = burst_info["cycle period"]
            duty_cycle = burst_info["burst duty cycle"]
            burst_ini = burst_info["burst ini"]
            burst_end = burst_info["burst end"]
            fig_title = basename
            fig_title += "\n%i ciclos" % len(cycle_period)
            fig, ax = plt.subplots(1, len(plot_list), figsize=(20, 10))
            i = 0
            for key in plot_list:
                ax[i].boxplot(burst_info[key])
                ax[i].set_title(key)
                ax[i].set_xlabel("mediana = %1.1f" % np.median(burst_info[key]))
                promedios[key].append(np.mean(burst_info[key]))
                i += 1
            fig.suptitle(fig_title)

            fig.savefig(folder_name + "completos/%s_boxplots.png" % basename, dpi=600, transparent=True)

            fig, ax = plt.subplots(3, 1, figsize=(20, 10))
            fig.suptitle(fig_title)
            ax[0].set_title("cycle period")
            ax[0].hist(cycle_period, bins=np.arange(0, math.ceil(max(burst_info["cycle period"]))+bin_step, bin_step)
                    )

            ax[1].hist(duty_cycle, bins=np.arange(0, 1, .025))
            ax[1].set_title("duty cycle")


            instant_freq =  1/np.diff(spike_times)
            instant_freq = instant_freq[~np.isnan(instant_freq)]
            instant_freq = instant_freq[instant_freq<500]
            ax[2].hist(instant_freq, bins=np.arange(0, 200, 5))
            ax[2].set_title("instant frequency")
            fig.savefig(folder_name + "completos/%s_histograms.png" % basename, dpi=600,
                        transparent=True)

            burst_start = planUtils.getBurstsStart(spike_times, isi_threshold=1 / min_spike_per_sec,
                                                   above_threshold_tol=.1)
            crawling_segment, dc_sub50 = planUtils.getCrawlingSegments(spike_times, period_range=period_range, min_spike_per_sec=min_spike_per_sec,
                                                             min_spike_no=min_spike_no)

            fig1, ax1 = burstUtils.plotFreq(burst_object.spike_freq_dict, draw_list=[burst_object.isDe3],
                                            color_dict='k', outlier_thres=5)

            ax1[0].axvline(burst_start, c='blue')
            textstr = "min spike/s = %i, min_spike_no=%i,\nperiod time range = (%.1f, %.1f)" % (min_spike_per_sec, min_spike_no, period_range[0], period_range[1])

            ax1[0].set_xlabel(textstr)
            if crawling_segment is not None:

                if not dc_sub50:
                    no_dc_sub50 += 1
                    c = 'r'
                else:
                    c = 'g'

                ax1[0].axvline(burst_ini[crawling_segment[0]], c=c)
                ax1[0].axvline(burst_ini[crawling_segment[1] + 1], c=c)


            for ini,end, dc in zip(burst_ini[:-1], burst_end[:-1], duty_cycle):
                if dc < .5:
                    c = 'k'
                else:
                    c = 'r'

                ax1[0].axvline(ini, c=c, ymin=.7)
                ax1[0].axvline(end, c=c, ymin=.85)


            ax1[0].axvline(burst_ini[-1], c=c, ymin=.7)
            ax1[0].axvline(burst_end[-1], c=c, ymin=.85)
            fig1.suptitle(fig_title)
            fig1.savefig(folder_name + "completos/%s_time_serie.png" % basename, dpi=600, transparent=True)

        plt.close('all')
        fig, ax = plt.subplots(1, len(plot_list), figsize=(20, 10))
        i = 0
        for key in plot_list:
            ax[i].boxplot(promedios[key])
            ax[i].set_title(key)


            i += 1
        fig.suptitle('Todos los registros, completos')
        fig.savefig(folder_name + "completos/boxplot_de_los_promedios.png", dpi=600, transparent=True)

        fig, ax = plt.subplots()
        ax.hist(promedios["cycle period"])
        fig.suptitle("histograma cycle period promedio")
        fig.savefig(folder_name + "completos/histograma_cycle_period.png", dpi=600, transparent=True)
    ##############################################################################################
    """
    Este los recorta
    """
    if correr_crawling:
        promedios = {key: [] for key in plot_list}
        crawl_dur = []
        no_dc_sub50 = 0
        no_crawling = 0

        for fn in run_list:

            ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}


            cdd_de3 = cdd[fn]['DE3']
            selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                                neuron_dict["neuron_is_good"]]

            fn = fn.replace("\\", '/')

            basename = os.path.splitext(os.path.basename(fn))[0]
            fig_title = basename
            burst_object = unitInfo.UnitInfo(fn, mode='load')

            if cdd_de3 != burst_object.isDe3:
                print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn,
                                                                                                                   cdd_de3,
                                                                                                                   burst_object.isDe3))

            spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]


            burst_start = planUtils.getBurstsStart(spike_times, isi_threshold=1/min_spike_per_sec, above_threshold_tol=.1)




            crawling_segment, dc_sub50 = planUtils.getCrawlingSegments(spike_times[spike_times>burst_start], period_range=period_range,
                                                                       min_spike_per_sec=min_spike_per_sec,
                                                                       min_spike_no=min_spike_no)



            burst_info = planUtils.getBurstingInfo(spike_times[spike_times>burst_start], min_spike_no=min_spike_no,
                                                   min_spike_per_sec=min_spike_per_sec)
            if crawling_segment is not None:
                t_start = burst_info["burst ini"][crawling_segment[0]]
                t_end = burst_info["burst ini"][crawling_segment[1] + 1]
                spike_times = spike_times[(spike_times>t_start) & (spike_times < t_end)]

                burst_info = planUtils.getBurstingInfo(spike_times, min_spike_no=min_spike_no,
                                                       min_spike_per_sec=min_spike_per_sec)
                crawl_dur.append(t_end-t_start)
            else:
                fig_title += "\nsin segmento de crawling"
                t_start = 100
                t_end = burst_object.time_length-100


            median_time = burst_info["burst median time"]
            duration = burst_info["burst duration"]
            cycle_period = burst_info["cycle period"]
            duty_cycle = burst_info["burst duty cycle"]
            burst_ini = burst_info["burst ini"]
            burst_end = burst_info["burst end"]

            fig_title += "\n%i ciclos" % len(cycle_period)

            fig, ax = plt.subplots(1, len(plot_list), figsize=(20, 10))
            i = 0
            for key in plot_list:
                ax[i].boxplot(burst_info[key])
                ax[i].set_title(key)
                ax[i].set_xlabel("mediana = %1.1f" % np.median(burst_info[key]))
                promedios[key].append(np.mean(burst_info[key]))
                i += 1
            fig.suptitle(fig_title)
            if save_figs:
                fig.savefig(folder_name + "solo_crawling/%s_boxplots.png" % basename, dpi=600, transparent=True)

            fig, ax = plt.subplots(3, 1, figsize=(20, 10))
            fig.suptitle(fig_title)
            ax[0].set_title("cycle period")
            ax[0].hist(cycle_period, bins=np.arange(0, math.ceil(max(burst_info["cycle period"])) + bin_step, bin_step)
                       )

            ax[1].hist(duty_cycle, bins=np.arange(0, 1, .025))
            ax[1].set_title("duty cycle")
            instant_freq = 1 / np.diff(spike_times)
            instant_freq = instant_freq[~np.isnan(instant_freq)]
            instant_freq = instant_freq[instant_freq < 500]
            ax[2].hist(instant_freq, bins=np.arange(0, 200, 5))
            ax[2].set_title("instant frequency")
            if save_figs:
                fig.savefig(folder_name + "%s_histograms.png" % basename, dpi=600,
                            transparent=True)


            planUtils.getBurstingInfo(spike_times)
            fig1, ax1 = burstUtils.plotFreq(burst_object.spike_freq_dict, draw_list=[burst_object.isDe3],
                                            color_dict='k', outlier_thres=5)
            ax1[0].set_xlim((t_start-100, t_end+100))

            textstr = "min spike/s = %i, min_spike_no=%i,\nperiod time range = (%.1f, %.1f)" % (min_spike_per_sec, min_spike_no, period_range[0], period_range[1])

            ax1[0].set_xlabel(textstr)

            if crawling_segment is not None:
                cdd[fn]["crawling_intervals"] = (t_start, t_end)
                burst_object.crawling_segments = (t_start, t_end)

                if not dc_sub50:
                    no_dc_sub50 += 1
                    c = 'r'
                else:
                    c = 'g'

                ax1[0].axvline(t_start, c=c)
                ax1[0].axvline(t_end, c=c)
            else:
                cdd[fn]["crawling_intervals"] = (-1, -1)
                burst_object.crawling_segments = (-1, -1)

                no_dc_sub50 += 1
                no_crawling += 1
            print(burst_object.filename, burst_object.crawling_segments)
            burst_object.saveResults()
            for ini,end, dc in zip(burst_ini[:-1], burst_end[:-1], duty_cycle):
                if dc < .5:
                    c = 'k'
                else:
                    c = 'r'

                ax1[0].axvline(ini, c=c, ymin=.7)
                ax1[0].axvline(end, c=c, ymin=.85)

            ax1[0].axvline(burst_ini[-1], c=c, ymin=.7)
            ax1[0].axvline(burst_end[-1], c=c, ymin=.85)
            fig1.suptitle(fig_title)
            if save_figs:
                fig1.savefig(folder_name + "solo_crawling/%s_time_serie.png" % basename, dpi=600, transparent=True)



        plt.close('all')
        fig, ax = plt.subplots(1, len(plot_list), figsize=(20, 10))
        i = 0
        for key in plot_list:
            ax[i].boxplot(promedios[key])
            ax[i].set_xlabel("mediana = %1.1f" % np.median(promedios[key]))
            ax[i].set_title(key)

            i += 1
        fig.suptitle("Todos los registros, solo 'crawling'")
        if save_figs:
            fig.savefig(folder_name + "solo_crawling/boxplot_de_los_promedios.png", dpi=600, transparent=True)

        fig, ax = plt.subplots(1, 2)
        ax[0].hist(promedios["cycle period"])
        ax[0].set_title("avg cycle period")
        ax[0].set_xlabel('tiempo (s)')
        crawl_dur = np.array(crawl_dur)
        ax[1].hist(crawl_dur/60)
        ax[1].set_title("crawling duration")
        ax[1].set_xlabel('tiempo (min)')
        if save_figs:
            fig.savefig(folder_name + "solo_crawling/hist_cycle_period.png", dpi=600, transparent=True)



        dc_sub50 = len(run_list) - no_dc_sub50
        crawling = len(run_list) - no_crawling

        fig, ax = plt.subplots(1, 2)
        ax[0].bar(("no crawling", "crawling"), (no_crawling, crawling), width=1, edgecolor="black", align="center")
        ax[1].bar(("dc sub", "no dc sub"), (dc_sub50, no_dc_sub50), width=1, edgecolor="black", align="center")
        if save_figs:
            fig.savefig(folder_name + "solo_crawling/barplot.png", dpi=600, transparent=True)
        plt.close('all')
        CDU.appendToDataDict(cdd, filename="RegistrosDP_PP/CrawlingDataDict_updated_crawling_segments.json")

