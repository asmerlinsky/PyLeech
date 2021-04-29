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

    num = 6
    cols = 3

    for fn in list(cdd):
        if "2019_08_28_0005" not in fn:
            continue
        # ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("/", '/')
        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')
        times = np.linspace(0, burst_object.time[-1], num=num + 1, dtype=int)

        for neuron, spikes in burst_object.spike_freq_dict.items():
            if neuron in selected_neurons:


                fig, ax = plt.subplots(int(math.ceil(num / cols)), cols, sharex=True, sharey=True)
                i = 0
                spikes_array = np.array(spikes)
                spikes_array = spikes_array[:, ~burstUtils.is_outlier(spikes_array[1,:])]
                for i in range(num):
                    iff = spikes_array[1][(spikes_array[0]>times[i]) & (spikes_array[0]<times[i+1])]
                    iff = iff[iff>1]
                    ax.flatten()[i].hist(iff, bins=25)
                fig.suptitle(str(neuron))
