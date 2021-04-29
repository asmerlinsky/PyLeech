import os
import warnings
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import more_itertools as mit
import glob

def getBurstingInfo(spike_times, spike_max_dist=.7, min_spike_no=10, min_spike_per_sec=10):

    burst_list = burstUtils.getBursts(spike_times, spike_max_dist=spike_max_dist, min_spike_no=min_spike_no,
                                      min_spike_per_sec=min_spike_per_sec)

    burst_median_time = np.zeros(len(burst_list))
    burst_duration = np.zeros(len(burst_list))
    burst_ini = np.zeros(len(burst_list))
    burst_end = np.zeros(len(burst_list))
    for i in range(len(burst_list)):
        burst_ini[i] = burst_list[i][0]
        burst_end[i] = burst_list[i][-1]
        burst_median_time[i] = np.median(burst_list[i])
        burst_duration[i] = burst_end[i] - burst_ini[i]

    cycle_period = np.zeros(len(burst_list) - 1)
    burst_duty_cycle = np.zeros(len(burst_list) - 1)

    for i in range(burst_median_time.shape[0] - 1):
        cycle_period[i] = burst_median_time[i + 1] - burst_median_time[i]

        burst_duty_cycle[i] = burst_duration[i] / cycle_period[i]

    burst_info_dict = {
        "burst median time": burst_median_time,
        "burst duration": burst_duration,
        "cycle period": cycle_period,
        "burst duty cycle": burst_duty_cycle,
        "burst ini": burst_ini,
        "burst end": burst_end
    }

    return burst_info_dict

def get_consecutive_cycles(good_cycles, min_num=3):
    good_cycles_idxs = np.where(good_cycles)[0]
    consecutive_idxs = []
    for elem in mit.consecutive_groups(good_cycles_idxs):
        idx_list = list(elem)
        if len(idx_list) >= min_num:
            consecutive_idxs.append((idx_list[0], idx_list[-1]))
    return tuple(consecutive_idxs)

def getCrawlingStart(cycle_period, period_range=(10, 30), min_num=3):
    mask = (cycle_period>period_range[0]) & (cycle_period < period_range[1])
    lower_than_max_idxs = np.where(mask)[0]
    consecutive_lower_than_max_idxs = mit.consecutive_groups(lower_than_max_idxs)
    for elem in consecutive_lower_than_max_idxs:
        lista = list(elem)
        if len(lista) >= min_num:
            return lista[2]

def getCrawlingEnd(cycle_period, start_idx, period_range=(10,30), min_num=3):
    lower_than_min_idxs = np.where(cycle_period < period_range[0])[0]
    lower_than_min_idxs = lower_than_min_idxs[lower_than_min_idxs>start_idx]
    consecutive_lower_than_min_idxs = mit.consecutive_groups(lower_than_min_idxs)

    for elem in consecutive_lower_than_min_idxs:
        lista = list(elem)
        if len(lista) >= min_num:
            return lista[2]


def getCrawlingSegments(spike_times, period_range=(10., 30.), duty_cycle_max=.5, spike_max_dist=.7, min_spike_no=10, min_spike_per_sec=10., min_num=3):

    burst_info_dict = getBurstingInfo(spike_times, spike_max_dist, min_spike_no, min_spike_per_sec)


    start = getCrawlingStart(burst_info_dict["cycle period"], period_range=period_range, min_num=min_num)
    if (start is not None):
        end = getCrawlingEnd(burst_info_dict["cycle period"], start_idx=start, period_range=period_range, min_num=min_num)

        if end is None:
            end = len(burst_info_dict["cycle period"])-1

        return (start, end), all(burst_info_dict["burst duty cycle"][start:end+1]<duty_cycle_max)

    return None, None

def getBurstsStart(spike_times, isi_threshold=.1, window_size=200, step=5, above_threshold_tol=.1):
    times = np.arange(0, spike_times[-1]-window_size, step=step)
    for tm in times:
        mask = (spike_times > tm) & (spike_times < tm+window_size)

        total = spike_times[mask].shape[0]-1
        over_thres = (np.diff(spike_times[mask]) > isi_threshold).sum()


        if over_thres/total < above_threshold_tol:
            return tm
    warnings.warn("Didn't find threshold for this file")
    return 0

# Estoy agarrando el ultimo ciclo o el último burst y me falta el ciclo?
if __name__ == "__main__":

    bin_step = .2

    cdd = CDU.loadDataDict()


    run_list = []

    sel = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob("registros_abf_compartidos/*.abf")]

    for fn in list(cdd):

        if any([s in fn for s in sel]):
            run_list.append(fn)

    for fn in sel:
        if not any([fn in s for s in run_list]):
            print("%s is not in run list" % fn)

    tot = 0
    no_dc_sub50 = 0
    no_start = 0
    period_range = 5, 25

    for fn in run_list:
        # if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
        #     print("file %s has no De3 assigned" % fn)
        #     continue

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("\\", '/')

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        spike_times = burst_object.spike_freq_dict[burst_object.isDe3][0]
        spike_freqs = burst_object.spike_freq_dict[burst_object.isDe3][1]
        spike_times = spike_times[~burstUtils.is_outlier(spike_freqs, 5)]

        start_time = getBurstsStart(spike_times, isi_threshold=.1, window_size=200, step=50)
        if start_time is None:
            print("didn't`")

        spike_times = spike_times[spike_times>start_time]
        burst_info = getBurstingInfo(spike_times, min_spike_no=15, min_spike_per_sec=10.)

        crawling_segment, dc_sub50 = getCrawlingSegments(spike_times, period_range, min_spike_per_sec=10, min_spike_no=15,)
        print(crawling_segment)
        binned_sfd = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                                     selected_neurons=selected_neurons,
                                                     time_length=burst_object.time[-1], counting=False)

        # f, a = burstUtils.plotFreq(binned_sfd, draw_list=[burst_object.isDe3], scatter_plot=False)
        f, a = burstUtils.plotFreq(burst_object.spike_freq_dict, draw_list=[burst_object.isDe3], scatter_plot=True, outlier_thres=3.5)
        f.suptitle(basename)

        a[0].axvline(start_time, c='magenta')

        if crawling_segment is not None:

            if not dc_sub50:
                no_dc_sub50 += 1
                c = 'r'
            else:
                c = 'g'

            a[0].axvline(burst_ini[crawling_segment[0]], c=c)
            a[0].axvline(burst_ini[crawling_segment[1] + 1], c=c)
        else:
            no_start += 1

        for ini,end, dc in zip(burst_ini[:-1], burst_end[:-1], duty_cycle):
            if dc < .5:
                c = 'k'
            else:
                c = 'r'

            a[0].axvline(ini, c=c, ymin=.7)
            a[0].axvline(end, c=c, ymin=.85)

        a[0].axvline(burst_ini[-1], c=c, ymin=.7)
        a[0].axvline(burst_end[-1], c=c, ymin=.85)

        tot += 1
    print(no_dc_sub50, no_start, tot)

