import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe
from functools import partial
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as bStorerLoader
import PyLeech.Utils.burstUtils
import PyLeech.Utils.burstUtils as burstUtils
import os
import math
import numpy as np
import scipy.signal as spsig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import PyLeech.Utils.burstClasses as burstClasses

def on_move(ax_list, event):
    for ax in ax_list:
        if event.inaxes == ax:
            for ax2 in ax_list:
                if ax2 != ax:
                    ax2.view_init(elev=ax.elev, azim=ax.azim)
    else:
        return
    fig.canvas.draw_idle()



if __name__ == "__main__":
    # fn = "RegistrosDP_PP/2018_12_03_0005.pklspikes"

    cdb = CDU.loadCrawlingDatabase()
    # print(cdb.index.levels[0])
    for fn in cdb.index.levels[0][:]:
    # for fn in cdb.index.levels[0][3:7]:
        if '2018_12_03_0005' in fn:
            continue
        burst_obj = bStorerLoader.UnitInfo(fn, 'RegistrosDP_PP', mode='load')
        try:
            arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])

        except:
            print("%s has no NS recording" % fn)
            continue

        NS = arr_dict['Vm1']
        del arr_dict
        NS[(NS>-20) | (NS<-60)] = np.nan
        cdb = CDU.loadCrawlingDatabase()

        good_neurons = cdb.loc[fn].index[cdb.loc[fn, 'neuron_is_good'].values.astype(bool)].values
        crawling_interval = [cdb.loc[fn].start_time.iloc[0], cdb.loc[fn].end_time.iloc[0]]
        print(crawling_interval)


        dt_step = 0.1


        new_sfd = burstUtils.removeOutliers(burst_obj.spike_freq_dict, 5)
        binned_sfd = burstUtils.digitizeSpikeFreqs(new_sfd, time_vector[-1], counting=False)
        idxs = np.where((time_vector>crawling_interval[0]) & (time_vector<crawling_interval[1]))[0][::1000]
        cut_time = time_vector[idxs]
        cut_NS = NS[idxs]
        cut_binned_freq_array = burstUtils.binned_sfd_to_dict_array(binned_sfd, crawling_interval, good_neurons)

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=2, time_range=20, dt_step=.1)
        smoothed_sfd = {}
        for key, items in cut_binned_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])


        fig, axes = burstUtils.plotFreq(burst_obj.spike_freq_dict, color_dict=burst_obj.color_dict,
                            template_dict=burst_obj.template_dict, scatter_plot=True,
                            optional_trace=[time_vector[::500], NS[::500]], #draw_list=[list(burst_obj.spike_freq_dict)[0]],
                            outlier_thres=5, ms=4)
        fig.suptitle(os.path.basename(os.path.splitext(fn)[0]))

        segmented_data = burstClasses.SegmentandCorrelate(binned_sfd, NS, time_vector, fs, crawling_interval,
                                                          intracel_peak_height=-43, intracel_peak_distance=10,
                                                          intracel_prominence=1, kernel_spike_sigma=1)
        segmented_data.processSegments()

        # embedding_list = segmented_data.getSegmentsEmbeddings('NS')
        # NS_plot = sc.fit_transform(NS_embedding)


        # cmap = burstUtils.categorical_cmap(len(embedding_list), 1)


        # NS_plot = NS_embedding
        for item in ['NS', list(burst_obj.spike_freq_dict)[0]]:
            embedding_list = segmented_data.getSegmentsEmbeddings(item)
            fig = plt.figure()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
            ax_list = []
            ceil = math.ceil(len(embedding_list)/3)
            for i in range(len(embedding_list)):
                NS_plot = embedding_list[i]
                ax_list.append(fig.add_subplot(ceil,3,i+1, projection='3d'))
                ax_list[i].scatter(NS_plot[:, 0], NS_plot[:, 1], NS_plot[:, 2], c=plt.cm.jet(np.linspace(0, 2, NS_plot.shape[0])), label=str(i), lw=2, s=2)
                ax_list[i].legend()
                # ax_list[i].set_title(os.path.splitext(os.path.basename(fn))[0])
            fig.suptitle(os.path.splitext(os.path.basename(fn))[0])
            part_on_move = partial(on_move, ax_list)
            c1 = fig.canvas.mpl_connect('motion_notify_event', part_on_move)


        N = 21
        cmap = plt.cm.jet
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for key in list(burst_obj.spike_freq_dict):
            if key == 'NS':
                cut_smoothed_NS = segmented_data.filtered_intracel[(segmented_data.time[::int(.1*fs)]>crawling_interval[0]) & (segmented_data.time[::int(.1*fs)] < crawling_interval[1])]
                embedding = NLD.getDerivativeEmbedding(cut_smoothed_NS[::int(.1*fs)], .1, 3)
            else:
                embedding = NLD.getDerivativeEmbedding(smoothed_sfd[key][1], 0.1, 3)
            fig = plt.figure()
            fig.suptitle(key)
            ax = Axes3D(fig)
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                       c=plt.cm.jet(np.linspace(0, 2, embedding.shape[0])), label=str(key), s=5)
            plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                     boundaries=np.arange(-0.05, 2.1, .1))
            fig.legend()

