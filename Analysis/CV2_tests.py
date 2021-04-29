import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
from itertools import combinations

import matplotlib
font = {'size'   : 5}

matplotlib.rc('font', **font)
# plt.ioff()
if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/14217000.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]
    del cdd["RegistrosDP_PP/cont10.pklspikes"]

    kernel_sigma = 2
    time_range = 20
    time_step = .1
    corr_step = int(60*5 / time_step)


    for fn in list(cdd):
        # if "2019_08_28_0005" not in fn:
        #     continue
        # ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("/", '/')
        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        i = 0
        num = len(selected_neurons)
        cols = 1
        fig, ax = plt.subplots(int(math.ceil(num / cols)), cols, sharex=True, sharey=True)
        fig.suptitle(basename)

        for neuron, spikes in burst_object.spike_freq_dict.items():


            if neuron in selected_neurons:
                # print(neuron)
                good_times = spikes[0][~burstUtils.is_outlier(spikes[1])]
                isi = np.diff(good_times)
                isi_mask = isi<1
                # fig, ax = plt.subplots()
                cv2 = np.abs(2 * np.diff(isi) / (isi[1:] + isi[:-1]))
                cv2 = cv2[isi_mask[:-1]]
                ax.flatten()[i].scatter(good_times[:-2][isi_mask[:-1]], cv2, color='k', s=1, label=str(neuron))
                ax.flatten()[i].legend()
                i += 1
