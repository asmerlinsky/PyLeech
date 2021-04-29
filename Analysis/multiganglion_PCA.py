import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstUtils
from PyLeech.Utils.unitInfo import UnitInfo
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import scipy.signal as spsig
import PyLeech.Utils.NLDUtils as NLD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, FactorAnalysis, PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import os
import pandas as pd


if __name__ == '__main__':

    n_splits = 6
    n_colors = np.linspace(0, 1, n_splits)
    N = 21
    cmap = plt.cm.jet


    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    dim_2 = True
    dim_3 = True
    binning_dt = .5
    spike_kernel_sigma = 3
    n_components = 5

    cdd = CDU.loadDataDict()
    sc = StandardScaler()
    ica = FastICA(n_components=n_components, max_iter=15000)
    fa = FactorAnalysis(n_components=n_components)
    pca = PCA(n_components=n_components)
    model_dict = {
        "ICA" : {},
        "PCA" : {},
        "FA" : {}

        }

    del cdd["RegistrosDP_PP/NS_DP_PP_0.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_0_cut.pklspikes"]
    del cdd["RegistrosDP_PP/NS_T_DP_PP_1.pklspikes"]
    del cdd["RegistrosDP_PP/2019_01_28_0001.pklspikes"]
    del cdd['RegistrosDP_PP/18n05010.pklspikes']
    del cdd['RegistrosDP_PP/14217000.pklspikes']
    del cdd['RegistrosDP_PP/cont10.pklspikes']

    for fn, data in cdd.items():
        if not any([select in fn for select in ["2019_07_22_0009", "2019_07_23_0008"] ]):
            continue
        try:
            ns_channel = [key for key, items in data['channels'].items() if 'NS' == items][0]
            # print("Running %s" % fn)
        except IndexError:
            ns_channel = None


        good_neurons = [neuron for neuron, neuron_dict in data['neurons'].items() if neuron_dict['neuron_is_good']]
        if len(good_neurons)<7: continue
        fn = fn.replace("/", '/')

        if ns_channel is not None:
            arr_dict, time_vector1, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])
            NS_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=1 / fs)
            conv_NS = spsig.fftconvolve(arr_dict[ns_channel], NS_kernel, mode='same')[::int(binning_dt * fs)]
        else:
            arr_dict, time_vector1, fs = abfe.getArraysFromAbfFiles(fn, ['IN5'])
        time_vector1 = time_vector1[::int(binning_dt * fs)]

        del arr_dict

        burst_object = UnitInfo(fn, 'RegistrosDP_PP', 'load')



        spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, step=binning_dt,
                                                     selected_neurons=good_neurons,
                                                     time_length=burst_object.time[-1])

        smoothed_sfd = burstUtils.smoothBinnedSpikeFreqDict(spike_freq_array, sigma=spike_kernel_sigma,
                                                                         time_range=20, dt_step=binning_dt)


        burst_array = []
        for key, items in smoothed_sfd.items():

            burst_array.append(smoothed_sfd[key][1])
        burst_array = np.array(burst_array).T

        fig, ax_list = burstUtils.plotFreq(burst_object.spike_freq_dict, color_dict=burst_object.color_dict,
                                        scatter_plot=True, draw_list=good_neurons, outlier_thres=2, facecolor=None,
                                        legend=False)
        # fig, ax_list = burstUtils.plotFreq(smoothed_sfd, scatter_plot=False, color_dict='single_color',
        #                                    draw_list=good_neurons, legend=False, outlier_thres=None)

        # split_times = np.linspace(cdd[fn.replace("/", "/")]["crawling_intervals"][0][0], cdd[fn.replace("/", "/")]["crawling_intervals"][-1][-1], num=n_splits+1)
        # boxes = [matplotlib.patches.Rectangle((split_times[i],-1000), split_times[i+1]-split_times[i], 2000) for i in range(n_splits)]

        #
        for ax in ax_list:
            ax.grid(None)
        #     ax.set_facecolor('white')
        #     pc = matplotlib.collections.PatchCollection(boxes, facecolor=cmap(n_colors), alpha=.5)
        #     ax.add_collection(pc)
        #
        #     ax.legend().set_visible(False)
        fig.suptitle(fn)
        # ax.set_xlim([cdd[fn.replace("/", "/")]["crawling_intervals"][0][0], cdd[fn.replace("/", "/")]["crawling_intervals"][-1][-1]])
        fig.subplots_adjust(wspace=0, hspace=0)
        spike_idxs = NLD.getSpikeIdxs(smoothed_sfd, cdd[fn.replace("/", "/")]["crawling_intervals"])
        #
        # if ns_channel is not None:
        #     ax.set_ylim((conv_NS[spike_idxs].min(), conv_NS[spike_idxs].max()))
        basename = os.path.splitext(os.path.basename(fn))[0]


        save_name = "PCA_figs/" + basename + ".png"
        fig.savefig(save_name, dpi=600, transparent=True)
        # cols = ['NS'] + [str(i) for i in list(smoothed_sfd)]
        # df = pd.DataFrame(burst_array, columns=cols)
        # df.to_csv(save_name, index=False)




        scaled_data = sc.fit_transform(burst_array)
        # scaled_data = sc.fit_transform(burst_array[spike_idxs])
        model_dict["ICA"][fn] = ica.fit_transform(scaled_data)
        model_dict["FA"][fn] = fa.fit_transform(scaled_data)
        model_dict["PCA"][fn] = pca.fit_transform(scaled_data)
        val = pca.explained_variance_ratio_[:3].sum()
        print(basename, val)
        n_plots = 3
        fig, ax = plt.subplots(n_plots,1, sharex=True)
        for i in range(n_plots):
            # ax[i].plot(time_vector1[spike_idxs], model_dict["FA"][fn][spike_idxs,i])
            try:
                ax[i].plot(time_vector1[:-1], model_dict["PCA"][fn][:,i], color='k')
            except:
                ax[i].plot(time_vector1, model_dict["PCA"][fn][:, i], color='k')
            ax[i].set_xticks([])
            ax[i].set_yticks([])


        ax[0].set_xlim([100, 900])
        fig.suptitle(fn + '\n explica el %1.2f de la varianza' % val)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig("PCA_figs/" + basename + "componentes.png", transparent=True, dpi=600)
        # break

    plot_model = "PCA"
    if dim_2:
        for model, current_dict in model_dict.items():
            if model == plot_model:
                #         fig, ax = plt.subplots(3, 4)
                j = 0
                i = 0
                for key, items in current_dict.items():
                    fig, ax = plt.subplots()
                    #         ax[i, j].scatter(items[:, 0], items[:, 1], c=plt.cm.jet(np.linspace(0, 1, items.shape[0])), s=15)
                    #             ax[i, j].scatter(items[:, 0], items[:, 1], c=cmap(np.repeat(np.linspace(0, 1, num_colors), np.ceil(items.shape[0]/num_colors))[:items.shape[0]]), s=8)
                    #             ax[i, j].set_title(os.path.basename(os.path.splitext(key)[0]))
                    ax.scatter(items[:, 0], items[:, 1], c=cmap(
                        np.repeat(np.linspace(0, 1, n_splits), np.ceil(items.shape[0] / n_splits))[
                        :items.shape[0]]), s=8)
                    ax.set_title(os.path.basename(os.path.splitext(key)[0]))
                    ax.set_facecolor('gray')
                    fig.suptitle(model)

                    save_name = "PCA_figs/" + os.path.splitext(os.path.basename(key))[0] + "_"+ plot_model + ".png"
                    # fig.savefig(save_name, dpi=600)


                # fig.subplots_adjust(right=0.8)
                # cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
                # fig.colorbar(sm, ticks=np.linspace(0, 1, N), boundaries=np.arange(-0.05, 1.1, .05), cax=cbar_ax)


    if dim_3:
        for model, current_dict in model_dict.items():
            if model == plot_model:

                i = 1
                for key, items in current_dict.items():
                    fig = plt.figure()
                    ax = Axes3D(fig)

                    ax.scatter(items[:, 0], items[:, 1], items[:, 2], c=plt.cm.jet(
                        np.repeat(np.linspace(.1, .9, n_splits), np.ceil(items.shape[0] / n_splits))[
                        :items.shape[0]]), s=10)
                    ax.set_title(os.path.basename(os.path.splitext(key)[0]))
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    fig.suptitle(model)
                    save_name = "PCA_figs/" + os.path.splitext(os.path.basename(key))[0] + "_" + plot_model + ".png"
                    # fig.savefig(save_name, dpi=600, transparent=True)

                    i += 1
