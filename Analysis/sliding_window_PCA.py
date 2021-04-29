import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyLeech.Utils.correlationUtils as corrUtils
import PyLeech.Utils.AbfExtension as abfe

import gc
from sklearn.decomposition import PCA
# font = {'size'   : 5}

# matplotlib.rc('font', **font)
# plt.ioff()

if __name__ == "__main__":

    cdd = CDU.loadDataDict()
    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/14217000.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]
    del cdd["RegistrosDP_PP/cont10.pklspikes"]
    del cdd["RegistrosDP_PP/2018_12_04_0003_0004_0005_0006b.pklspikes"]
    plot_list = ["2019_08_28_0005", "2019_08_30_0003", "2019_08_30_0006"]
    kernel_sigma = 2
    time_range = 20
    time_step = .1
    window_size = 300
    bin_step = .5
    n_neighbors = 3


    for fn in list(cdd):
        # if not any([select in fn for select in ["2018_12_13_0001", "2019_07_22_0014", "2019_08_28_0005", "2019_07_22_0009" ] ]):
        #     continue
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}
        # if not any([select in fn for select in plot_list]):
        #     continue

        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            print("file %s has no De3 assigned" % fn)
            continue

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]
        fn = fn.replace("/", '/')

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = burstStorerLoader.UnitInfo(fn, mode='load')

        if cdd_de3 != burst_object.isDe3:
            print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn, cdd_de3, burst_object.isDe3) )

        spike_count_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                        selected_neurons=selected_neurons,
                                        time_length=burst_object.time[-1], counting=True)


        temp_dict = {}
        for key, items in spike_count_dict.items():
            temp_dict[key] = np.array([items[0], np.sqrt(items[1])])

        smoothed_scd = burstUtils.smoothBinnedSpikeFreqDict(spike_count_dict, 3*bin_step, time_range=20, dt_step=bin_step)


        data_array = burstUtils.sfdToArray(smoothed_scd)
        fig, ax = burstUtils.plotFreq(smoothed_scd, color_dict=burst_object.color_dict)

        fig.suptitle("%s\nDE-3 is %i" % (basename, burst_object.isDe3))
        array_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, channels=ch_dict)
        if (time_vector[-1]>1500) or time_vector[-1]<1100:
            continue

        pca_emb = PCA(n_components=3)



        data_array = data_array[int(200/bin_step):int(1000/bin_step)]
        rg = np.arange(data_array.shape[0]-window_size, step=5)
        explained_var_ratio = []
        for i in rg:
            pca_emb.fit(data_array[i:i+window_size])
            explained_var_ratio.append(pca_emb.explained_variance_ratio_)
        explained_var_ratio = np.cumsum(np.array(explained_var_ratio), axis=1).T

        fig1, ax1 = plt.subplots()
        i = 1
        for row in explained_var_ratio:
            ax1.plot((rg + int(window_size/2))*bin_step, row, label="%i comp"%i)
            i += 1

        ax1.get_shared_x_axes().join(ax1, ax[0])
        fig1.suptitle(basename)


