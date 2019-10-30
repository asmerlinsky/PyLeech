import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstUtils
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
import time

if __name__ == "__main__":
    cdd = CDU.loadDataDict()
    file_list = list(cdd)
    binning_dt = .1
    spike_kernel_sigma = 3

    trace_list = []
    emb_list = []
    ran_files = []
    run_dict = {}



    # file_list = file_list[-2:]

    i = 0
    for fn in file_list:
        if '2019_07_22_0009' not in fn:
            continue
        # if not cdd[fn]['skipped']:
        try:
            ns_channel = [key for key, items in cdd[fn]['channels'].items() if 'NS' == items][0]
        except:
            continue
        # else:
        #     continue

        fn = abfe.getAbFileListFromBasename(fn)[0]
        try:
            arr_dict, time_vector , fs = abfe.getArraysFromAbfFiles(fn, [ns_channel])
        except:
            continue

        ran_files.append(os.path.splitext(os.path.basename(fn))[0])
        run_dict[os.path.splitext(os.path.basename(fn))[0]] = i

        NS_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=1 / fs)
        bl_kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=45, time_range=10 * 60, dt_step=1 / fs)
        data = arr_dict[ns_channel] - np.mean(arr_dict[ns_channel])
        data[(data<-20) | (data>20)] = 0

        bl = spsig.fftconvolve(data, bl_kernel, mode='same')[::int(binning_dt * fs)]

        trace = spsig.fftconvolve(data, NS_kernel, mode='same')[::int(binning_dt * fs)]
        del arr_dict
        # trace_list.append((trace-bl)[1400:9500]) ### solo porque estoy correndo el 2019 08 30 0003
        # bl = np.zeros(bl.shape)
        fig, ax = plt.subplots()
        # trace_list.append((trace-bl)[1600:9200]) ### solo porque estoy correndo el 2019 07 22 0009
        trace_list.append((trace - bl))
        ax.plot(trace - bl)
        fig.suptitle(fn)
        # emb_list.append(NLD.getDerivativeEmbedding(trace, dt=binning_dt, emb_size=3))
        i += 1

    # plt.plot(trace)
    # plt.plot(bl)
    # plt.plot(trace - bl)
    # import time
    peak_height = 2.4
    # trace_list[0] = -trace_list[0]
    # resampled_trace = NLD.resampleByCycles(trace_list[0], fs=binning_dt, peak_height=peak_height)
    # fig1, ax1 = plt.subplots()
    # ax1.plot(resampled_trace)
    # NLD.plotCycles(trace_list[0], fs=binning_dt, peak_height=peak_height)
    # cycles = NLD.getCycles(trace_list[0], fs=binning_dt, peak_height=peak_height)
    # Ncycles = NLD.getNCycles(trace_list[0], fs=binning_dt, peak_height=peak_height, N=2)
    # NLD.plotCycles(trace_list[0], fs=binning_dt, peak_height=peak_height, N=2)
    # for x, y in NLD.getSimilarCycles(Ncycles, 10):
    #     fig, ax = plt.subplots()
    #     ax.plot(Ncycles[x])
    #     ax.plot(Ncycles[y])
    #     fig.suptitle('double')
    #     ax.set_title(str(x) + ' ' + str(y))
    # for x, y in NLD.getSimilarCycles(cycles, 15):
    #     fig, ax = plt.subplots()
    #     ax.plot(cycles[x])
    #     ax.plot(cycles[y])
    #     fig.suptitle('single')
    #     ax.set_title(str(x) + ' ' + str(y))



    fig1, ax1 = plt.subplots()
    rt_orig_threshold = .05
    num_files = int(len(run_dict))
    start = time.time()
    segments = {}
    # resampled_trace = NLD.resampleByCycles(trace_list[0], fs=binning_dt, peak_height=.7)
    # cycles = NLD.getCycles(trace_list[0], fs=binning_dt, peak_height=.7)
    # ax1.plot(resampled_trace)
    # kernel = NLD.generateGaussianKernel(sigma=1.5, time_range=2, dt_step=binning_dt)
    # trace_list[0] = spsig.fftconvolve(resampled_trace, kernel, mode='same')

    i = 3
    j = 3
    for i in range(num_files):
        for j in range(i, num_files):
            if i != j:
                continue
            rt_threshold = rt_orig_threshold
            rt = NLD.getCloseReturns(trace_list[i][:, np.newaxis], trace_list[j][:, np.newaxis], threshold=rt_threshold, get_mask=True)#[:, :, 0]

            while rt.sum()/rt.size > .5:
                rt_threshold /= 2
                rt = NLD.getCloseReturns(trace_list[i][:, np.newaxis], trace_list[j][:, np.newaxis],
                                         threshold=rt_threshold, get_mask=True)  # [:, :, 0]
            plots = NLD.plotCloseReturns(rt, masked=False, reorder=False, get_counts=False, thr=rt_threshold)
            plots[0].suptitle(ran_files[i])
            if i == j:
                sg_dict = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt, rt_len=25,
                                                                         single_file=True, min_dist=10)
            else:
                sg_dict = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt, rt_len=25)
            segments[(list(run_dict)[i], list(run_dict)[j])] = sg_dict

    print(time.time() - start)
    winsound.Beep(2000, 100)
    winsound.Beep(2000, 100)
    winsound.Beep(2000, 100)
    # plt.close('all')
    h = 0
    k = 0

    h = 0

    # file_tup = ('2018_12_13_0015', '2018_12_13_0015')
    # diag_tups = [(t1, t1) for t1, t2 in list(segments) if t1==t2]
    # items1 = segments[file_tup]
    # fig1, ax1 = plt.subplots()
    time_delta = 200
    length = 250
    count = 0
    count = 0
    for file_tup in list(segments)[-2:]:
        items1 = segments[file_tup]
        for diagonal in list(items1):
            for tup0, tup1 in items1[diagonal]:
                if tup1 - tup0  > length and np.abs(diagonal) > time_delta:
                    count += 1
    print(count)
    # fig2 = plt.figure()
    # ax2 = Axes3D(fig2)
    for file_tup in list(segments)[-2:]:
        items1 = segments[file_tup]
        # count = 0
        # if file_tup[0] == '2018_12_04_0003_0004_0005_0006b' or file_tup[1] == '2018_12_04_0003_0004_0005_0006b':

            # print(file_tup)

        for diagonal in list(items1):
            for tup0, tup1 in items1[diagonal]:


                h += 1
                # print(file_tup, dist, tup0, tup1)

                if tup1-tup0 > length and np.abs(diagonal)>time_delta:

                    fig1, ax1 = plt.subplots()
                    t0 = trace_list[run_dict[file_tup[0]]][tup0:tup1]
                    t1 = trace_list[run_dict[file_tup[1]]][tup0 + diagonal:tup1 + diagonal]
                    r0 = np.arange(t0.shape[0])
                    r1 = np.arange(t1.shape[0])
                    ax1.plot(r0, t0, linewidth=1, color='k')
                    ax1.plot(r1, t1, linewidth=1, color='r')
                    #
                    # NLD.plot3Dline(emb_list[run_dict[file_tup[0]]][0][tup0:tup1], fig_ax_pair=[fig2, ax2], color='k', linewidth=.3)
                    # NLD.plot3Dline(emb_list[run_dict[file_tup[1]]][0][tup0+diagonal:tup1+diagonal], fig_ax_pair=[fig2, ax2], color='k', linewidth=.3)

                    # ax.plot(t0, color='k')
                    # ax.plot(t1, color='r')
                    fig1.suptitle(file_tup[0] + " shift: " + str(diagonal) + " " + str(tup0) + "-" + str(tup1) +
                                 "\n" + file_tup[1] + " " + str(tup0 + diagonal) + "-" + str(tup1 + diagonal))

                    #
                    # if h % 30 == 0:
                    #     plt.ginput(n=0, timeout=0, mouse_pop=2, mouse_stop=3)
                    #     plt.close("all")
                        # for k in range(30):
                        #     plt.close(k)

    # trace_list[run_dict["2018_12_04_0003_0004_0005_0006b"]][1656:1711]
    # trace_list[run_dict["2018_12_13_0015"]][1656 + 3376:1711 + 3376]
    # trace_list[run_dict["2018_12_04_0003_0004_0005_0006b"]][780:823]
    # trace_list[run_dict["2018_12_13_0005"]][780 + 3878:823 + 3878]
