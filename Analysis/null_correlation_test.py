
import os


import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import PyLeech.Utils.correlationUtils as corrUtils

import math
from scipy.stats.stats import pearsonr, spearmanr


# font = {'size'   : 5}
import matplotlib.animation as animation
# matplotlib.rc('font', **font)
# plt.ioff()

import warnings
warnings.filterwarnings("ignore")

# plt.rcParams['animation.ffmpeg_path'] = "C:/ffmpeg/bin/ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"

def varcorrcoef(x):
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
        df = []
        for i in range(processed_spike_freq_array.shape[0]):
        # for i in range(2):
            for j in range(i, processed_spike_freq_array.shape[0]):
            # for j in range(i, 2):
                null_correlations = corrUtils.getRandomCorrelationStats(processed_spike_freq_array[:, i], processed_spike_freq_array[:, j], time_step=time_step, corr_lengths=corr_times, num_samples=5000)
                pct = np.percentile(null_correlations[~np.isnan(null_correlations)], [1, 5, 95, 99])
                df.append([i, j, pct[0], pct[1], pct[2], pct[3]])
                fig, ax = plt.subplots()
                ax.hist(null_correlations, bins = 50)

                fig.suptitle(str(i) + '    ' + str(j) + '\n 5pct = %1.2f, 95pct = %1.2f'%(pct[0], pct[1]) )

        df = pd.DataFrame(df, columns=['i', 'j', '1pct', '5pct', '95pct', '99pct'])
        print(df)
        print(df.describe())