import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as burstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyLeech.Utils.NLDUtils as NLDUtils
import PyLeech.Utils.correlationUtils as corrUtils
import PyLeech.Utils.AbfExtension as abfe


from sklearn.manifold import Isomap
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
    n_neighbors  = 7
    bin_step = .5



    for fn in list(cdd):
        # if not any([select in fn for select in ["2018_12_13_0001", "2019_07_22_0014", "2019_08_28_0005", "2019_07_22_0009" ] ]):
        #     continue
        #
        # if "2018_12_13_0001" not in fn:
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

        array_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, channels=ch_dict)
        if (time_vector[-1] > 1500) or time_vector[-1] < 1100:
            continue

        if cdd_de3 != burst_object.isDe3:
            print("file {} has different de3 assignment between datadict and pklspikes file:\n\t{}\t{}".format(fn, cdd_de3, burst_object.isDe3) )

        spike_count_dict = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=bin_step,
                                        selected_neurons=selected_neurons,
                                        time_length=burst_object.time[-1], counting=True)

        
        temp_dict = {}
        for key, items in spike_count_dict.items():
            temp_dict[key] = np.array([items[0], np.sqrt(items[1])])
        spike_count_dict = temp_dict
        smoothed_scd = burstUtils.smoothBinnedSpikeFreqDict(spike_count_dict, 3*bin_step, time_range=20, dt_step=bin_step)


        data_array = burstUtils.sfdToArray(smoothed_scd)
        fig0, ax0 = burstUtils.plotFreq(smoothed_scd, color_dict='jet', facecolor='k')

        fig0.suptitle("%s\nDE-3 is %i" % (basename, burst_object.isDe3))



        #
        #
        # fig1, axes1 = corrUtils.rasterPlot(spike_count_dict, color_dict=burst_object.color_dict, linewidths=.5)
        #
        # fig1.suptitle("%s\nDE-3 is %i" % (basename, burst_object.isDe3))

        embedding = Isomap(n_neighbors=n_neighbors, n_components=3, n_jobs=mp.cpu_count()-2, )
        pca_emb = PCA(n_components=3)

        pcad_count = pca_emb.fit_transform(data_array)
        transformed_count = embedding.fit_transform(data_array)


        fig, ax = NLDUtils.plot3Dscatter(transformed_count)
        ax.set_facecolor('k')
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # # n_splits = 8
        # ax.scatter(transformed_count[:, 0], transformed_count[:, 1], transformed_count[:, 2],
        #            c=plt.cm.jet(np.linspace(0, 1, transformed_count.shape[0])), s=10)
        # ax.scatter(transformed_count[:, 0], transformed_count[:, 1], transformed_count[:, 2],
        #            c=plt.cm.jet(np.repeat(np.linspace(0, 1, n_splits), np.ceil(transformed_count.shape[0] / n_splits))[
        #     :transformed_count.shape[0]]), s=10)
        # ax.set_title(os.path.basename(os.path.splitext(key)[0]))
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        fig.suptitle(basename + "\n%i neigbours Isomap" % n_neighbors, c='white')
        #
        # fig1 = plt.figure()
        # ax1 = Axes3D(fig1)
        # n_splits = 8
        # ax1.scatter(pcad_count[:, 0], pcad_count[:, 1], pcad_count[:, 2], c=plt.cm.jet(
        #     np.repeat(np.linspace(.1, .9, n_splits), np.ceil(pcad_count.shape[0] / n_splits))[
        #     :pcad_count.shape[0]]), s=10)
        # # ax.set_title(os.path.basename(os.path.splitext(key)[0]))
        # # ax.xaxis.set_visible(False)
        # # ax.yaxis.set_visible(False)
        # fig1.suptitle(basename + " Pca")
        #


        # break
