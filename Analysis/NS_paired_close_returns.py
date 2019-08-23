import PyLeech.Utils.CrawlingDatabaseUtils as CDU
from PyLeech.Utils.burstStorerLoader import BurstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import matplotlib.pyplot as plt
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig
import os.path
import winsound
from mpl_toolkits.mplot3d import Axes3D
import PyLeech.Utils.AbfExtension as abfe

if __name__ == "__main__":
    cdd = CDU.loadDataDict()
    file_list = list(cdd)
    trace_list = []
    emb_list = []
    ran_files = []
    run_dict = {}
    i = 0
    for fn in file_list:
        if cdd[fn]['skipped'] or cdd[fn]["DE3"] is None or cdd[fn]["DE3"] == -1: continue
        ran_files.append(os.path.splitext(os.path.basename(fn))[0])
        run_dict[os.path.splitext(os.path.basename(fn))[0]] = i
        burst_object = BurstStorerLoader(fn, 'RegistrosDP_PP', 'load')
        burst_object.isDe3 = cdd[fn]["DE3"]

        binning_dt = 0.1
        spike_kernel_sigma = 1

        spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, .1,
                                                           selected_neurons=[burst_object.isDe3],
                                                           time_length=burst_object.time[-1])

        kernel = NLD.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)
        smoothed_sfd = {}
        for key, items in spike_freq_array.items():
            smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])

        spike_idxs = NLD.getSpikeIdxs(smoothed_sfd, cdd[fn]["crawling_intervals"])

        trace = smoothed_sfd[burst_object.isDe3][1][spike_idxs]
        trace = np.array([trace]).T.astype(np.float32)
        # trace[trace<freq_threshold] = np.nan
        trace_list.append(trace)
        emb_list.append(NLD.getDerivativeEmbedding(trace, dt=.1, emb_size=3))
        i += 1

    del burst_object
    import time

    freq_threshold = 2
    rt_threshold = .03
    num_files = int(len(run_dict))
    start = time.time()
    segments = {}

    i = 3
    j = 3
    for i in range(num_files):
        for j in range(i, num_files):

            # rt = NLD.getCloseReturns(trace_list[i], trace_list[j], threshold=rt_threshold, get_mask=True)[:, :, 0]

            pos_zeros_i = np.zeros((trace_list[i].shape[0], trace_list[j].shape[0]), dtype=bool)
            pos_zeros_i[np.where(trace_list[i] < freq_threshold)[0], :] = True

            pos_zeros_j = np.zeros((len(trace_list[i]), len(trace_list[j])), dtype=bool)
            pos_zeros_j[:, np.where(trace_list[j] < freq_threshold)[0]] = True

            zeros = pos_zeros_i & pos_zeros_j

            rt = NLD.getCloseReturns(trace_list[i], trace_list[j], threshold=rt_threshold, get_mask=True)[:, :, 0]
            plots = NLD.plotCloseReturns(rt & ~zeros, masked=False, reorder=False, get_counts=False, thr=.03)

            # fig, ax = plt.subplots()
            # ax.matshow(zeros)
            # fig.suptitle(ran_files[i] + "\n" + ran_files[j])
            # fig, ax = plt.subplots()
            # ax.matshow(rt & ~zeros)
            # fig.suptitle(ran_files[i] + "\n" + ran_files[j])
            if i == j:
                sg_dict = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt , zeros, rt_len=25,
                                                                         single_file=True, min_dist=10)
            else:
                sg_dict = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt , zeros, rt_len=25)
            segments[(list(run_dict)[i], list(run_dict)[j])] = sg_dict

    print(time.time() - start)
    winsound.Beep(2000, 100)
    winsound.Beep(2000, 100)
    winsound.Beep(2000, 100)
    # plt.close('all')
    h = 0
    k = 0

    h = 0
    count = 0
    file_tup = ('2018_12_13_0015', '2018_12_13_0015')
    items1 = segments[file_tup]
    # fig1, ax1 = plt.subplots()

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    for file_tup, items1 in segments.items():
        # count = 0
        # if file_tup[0] == '2018_12_04_0003_0004_0005_0006b' or file_tup[1] == '2018_12_04_0003_0004_0005_0006b':

            # print(file_tup)

        for diagonal in list(items1):
            for tup0, tup1 in items1[diagonal]:


                h += 1
                # print(file_tup, dist, tup0, tup1)
                argmax = np.argmax(trace_list[run_dict[file_tup[0]]][tup0:tup1])
                if argmax == 0: continue
                if trace_list[run_dict[file_tup[0]]][tup0:tup0+argmax].min() < 8 and \
                        trace_list[run_dict[file_tup[0]]][tup0 + argmax:tup1].min() < 8 and trace_list[run_dict[file_tup[0]]][
                                                                                 tup0:tup1].max() > 10:



                    fig1, ax1 = plt.subplots()
                    t0 = trace_list[run_dict[file_tup[0]]][tup0:tup1]
                    t1 = trace_list[run_dict[file_tup[1]]][tup0 + diagonal:tup1 + diagonal]
                    r0 = np.arange(t0.shape[0]) - np.argmax(t0)
                    r1 = np.arange(t1.shape[0]) - np.argmax(t1)
                    ax1.plot(r0, t0, linewidth=1, color='k')
                    ax1.plot(r1, t1, linewidth=1, color='k')

                    NLD.plot3Dline(emb_list[run_dict[file_tup[0]]][0][tup0:tup1], fig_ax_pair=[fig2, ax2], color='k', linewidth=.3)
                    NLD.plot3Dline(emb_list[run_dict[file_tup[1]]][0][tup0+diagonal:tup1+diagonal], fig_ax_pair=[fig2, ax2], color='k', linewidth=.3)

                    # ax.plot(t0, color='k')
                    # ax.plot(t1, color='r')
                    fig1.suptitle(file_tup[0] + " shift: " + str(diagonal) + " " + str(tup0) + "-" + str(tup1) +
                                 "\n" + file_tup[1] + " " + str(tup0 + diagonal) + "-" + str(tup1 + diagonal))


                    if h % 30 == 0:
                        plt.ginput(n=0, timeout=0, mouse_pop=2, mouse_stop=3)
                        plt.close("all")
                        # for k in range(30):
                        #     plt.close(k)

    # trace_list[run_dict["2018_12_04_0003_0004_0005_0006b"]][1656:1711]
    # trace_list[run_dict["2018_12_13_0015"]][1656 + 3376:1711 + 3376]
    # trace_list[run_dict["2018_12_04_0003_0004_0005_0006b"]][780:823]
    # trace_list[run_dict["2018_12_13_0005"]][780 + 3878:823 + 3878]
