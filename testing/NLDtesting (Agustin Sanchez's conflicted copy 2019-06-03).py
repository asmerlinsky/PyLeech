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
from sklearn.preprocessing import StandardScaler




if __name__ == "__main__":
    # fn = "RegistrosDP_PP/2018_12_03_0005.pklspikes"

    cdd = CDU.loadDataDict()

    for fn, info in cdd.items():

        # for fn in cdb.index.levels[0][3:7]:

        burst_obj = bStorerLoader.UnitInfo(fn, 'RegistrosDP_PP', mode='load')
        try:
            arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])

        except:
            print("%s has no NS recording" % fn)
            continue

        NS = arr_dict['Vm1']
        del arr_dict
        NS[(NS>-20) | (NS<-60)] = np.median(NS)

        good_neurons = [neuron for neuron, neuron_dict in info['neurons'].items() if neuron_dict['neuron_is_good']]



        crawling_interval = [cdb.loc[fn].start_time.iloc[0], cdb.loc[fn].end_time.iloc[1]]

        crawling_interval = [crawling_interval, [650, 817]]
        print(crawling_interval)
        dt_step = 0.1


        new_sfd = burstUtils.removeOutliers(burst_obj.spike_freq_dict, 5)
        binned_sfd = burstUtils.digitizeSpikeFreqs(new_sfd, dt_step, time_vector[-1], count=False)
        idxs= []
        for interval in crawling_interval:
            idxs.append(np.where((time_vector>interval[0]) & (time_vector<interval[1]))[0][::1000])
        idxs = np.hstack(idxs)


        cut_time = time_vector[idxs]

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=4, time_range=30, dt_step=1 / fs)
        conv_NS = spsig.fftconvolve(NS, kernel, mode='same')
        cut_NS = conv_NS[idxs]
        cut_binned_freq_array = burstUtils.binned_sfd_to_dict_array(binned_sfd, crawling_interval, good_neurons)

        kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=2, time_range=20, dt_step=.1)
        smoothed_sfd = {}
        for key, items in cut_binned_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])


        fig, axes = burstUtils.plotFreq(burst_obj.spike_freq_dict, color_dict=burst_obj.color_dict,
                                        template_dict=burst_obj.template_dict, scatter_plot=True,
                                        optional_trace=[time_vector[::2000], NS[::2000]], #draw_list=[list(burst_obj.spike_freq_dict)[0]],
                                        outlier_thres=5, ms=4)
        fig.suptitle(os.path.basename(os.path.splitext(fn)[0]))
        # reload(burstClasses)
        segmented_data = burstClasses.SegmentandCorrelate(binned_sfd, NS, time_vector, fs, crawling_interval,
                                                          intracel_peak_height=None, intracel_prominence=3, sigma=1,
                                                          intracel_peak_distance=10)

        segmented_data.processSegments()


        embedding = NLD.getDerivativeEmbedding(cut_NS, 0.1, 3)
        embedding[embedding[:, 1] > 1] = np.nan
        sc = StandardScaler()
        scaled_embedding = sc.fit_transform(embedding)
        NLD.plot3Dline(scaled_embedding)
        rt = NLD.getCloseReturns(scaled_embedding)
        NLD.plotCloseReturns(rt, .1, masked=False, reorder=True, get_counts=False)
        NLD.plot3Dline(embedding[2655:4500])
        fig = plt.figure()
        ax = Axes3D(fig)
        NLD.plot3Dline(embedding[440:715],[fig, ax], colorbar=False)
        NLD.plot3Dline(embedding[3300:3600], [fig, ax])

        NLD.plot3Dline(embedding[556:3658])
        NLD.plot3Dline(embedding[170:1500], [fig, ax], color_idx=0)
        NLD.plot3Dline(embedding[160:980])

        NLD.plot3Dline(embedding[800:1400+600], color_idx=0)
        NLD.plot3Dline(embedding[1800:2000+1800])
        NLD.plot3Dline(embedding[2200:2700+900], color_idx=0)

        for item in ['NS', list(burst_obj.spike_freq_dict)[0]]:
            embedding_list = segmented_data.getSegmentsEmbeddings(item)
            fig = plt.figure()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
            ax_list = []
            ceil = math.ceil(len(embedding_list)/3)
            for i in range(len(embedding_list)):
                NS_plot = embedding_list[i]
                ax_list.append(fig.add_subplot(ceil, 3, i + 1, projection='3d'))
                NLD.scatter3D(NS_plot, [fig, ax_list[i]], label=i, colorbar=False)
                ax_list[i].legend()
                # ax_list[i].set_title(os.path.splitext(os.path.basename(fn))[0])
            NLD.generateColorbar(fig)
            fig.suptitle(os.path.splitext(os.path.basename(fn))[0])
            part_on_move = partial(on_move, ax_list)
            c1 = fig.canvas.mpl_connect('motion_notify_event', NLD.part_on_move)


        N = 21
        cmap = plt.cm.jet
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        for key in['NS']:
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

