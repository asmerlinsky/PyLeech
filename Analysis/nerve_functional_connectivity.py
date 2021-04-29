import os

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import warnings
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import PyLeech.Utils.AbfExtension as abfe
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
import PyLeech.Utils.correlationUtils as corrUtils
import scipy.signal as spsig
import PyLeech.Utils.SpSorter as SpSorter

# plt.rcParams['animation.ffmpeg_path'] = "C:/ffmpeg/bin/ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"


def partPearson(x, y):
    return pearsonr(x, y)[0]


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


    long_kernel_sigma = 30
    kernel_sigma = 2
    time_range = 20
    time_step = .5
    corr_times = np.array([
        # 30,
        # 40,
        # 60,
        100,
    ])
    window_sizes = (corr_times / time_step).astype(int)
    corr_step = 1
    bin_step = time_step
    num = 6
    cols = 3
    neuron_correlation_dict_by_time = {}
    count = 0
    full_corr = []
    corr_thres = .85
    """
    FC(t) by file
    """
    #
    # run_list = []
    # sel = [
    #     "2018_12_13_0015",
    #     "2019_07_22_0009",
    #     "2019_08_26_0002",
    #     "2019_08_30_0006",
    #     "2019_08_30_0003",
    # ]
    #
    # for fn in list(cdd):
    #     if any([s in fn for s in sel]):
    #         run_list.append(fn)

    run_list = list(cdd)

    for fn in run_list:

        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if info.upper() in ["NS", "MA", "DP", "PP", "AA"]}
        info_dict = {v: k for k, v in ch_dict.items()}
        if cdd[fn]['skipped'] or len(info_dict)<=2:
            print("file %s was skipped" % fn)
            continue


        basename = os.path.splitext(os.path.basename(fn))[0]

        fn = fn.replace("\\\\", "/")

        abfe.getAbFileListFromBasename(fn)


        arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, list(ch_dict))
        time_vector -= time_vector[0]
        digitized_times = {}

        has_NS = True
        try:
            NS = arr_dict[info_dict["NS"]]
            NS[(NS > -20) | (NS < -60)] = np.median(NS)
            long_kernel = burstUtils.generateGaussianKernel(sigma=long_kernel_sigma, time_range=120, dt_step=1 / fs)
            kernel = burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=30, dt_step=1 / fs)
            NS -= np.median(NS)
            NS -= spsig.convolve(NS, long_kernel, mode='same')
            conv_NS = spsig.convolve(NS, kernel, 'same')
            digitized_times['NS'] = conv_NS[::int(time_step*fs)]
        except KeyError as e:
            warnings.warn("this file has no NS")
            has_NS = False


        for key, channel in info_dict.items():
            if key != "NS":

                sorter = SpSorter.SpSorter(fn, None, [arr_dict[channel]], time_vector, fs)

                sorter.normTraces()
                sorter.smoothAndFindPeaks(5, 11, 25)
                # sorter.plotDataListAndDetection()
                resampled_time_vector, digit_tm = burstUtils.digitizeSpikeTimes(sorter.time[sorter.peaks_idxs], time_step, sorter.time[-1], counting=True)
                digit_tm = spsig.convolve(digit_tm, np.array([1,1,1,1,1])/5, mode='same')
                if sorter.peaks_idxs.shape[0]>1000:
                    digitized_times[key] = digit_tm

        if len(digitized_times) <= 2:
            print("file %s was skipped" % fn)
            continue

        fig0, ax0 = plt.subplots(len(digitized_times), 1, sharex=True)
        fig0.suptitle(basename)
        i = 0
        nerve_array = []
        for key, items in digitized_times.items():
            nerve_array.append(items)
            ax0[i].plot(resampled_time_vector, items, label=key)
            ax0[i].legend()
            i += 1

        nerve_array = np.array(nerve_array).T


        i = 0
        if len(corr_times) == 4:
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots()
            ax = np.array([ax])
        for corr_time in corr_times:
            ws = int(corr_time / time_step)
            fc_matrixes = corrUtils.multiUnitSlidingWindow(nerve_array, window_size=ws, step=corr_step)

            flattened_upper_triang_idxs = corrUtils.getFlattenedUpperTriangIdxs(nerve_array.shape[1], 1)
            flattened_matrixes = fc_matrixes.reshape(fc_matrixes.shape[0], fc_matrixes.shape[1] ** 2)

            spearman_corr = np.corrcoef(flattened_matrixes[:, flattened_upper_triang_idxs])
            t_min = corr_time/2
            t_max = (resampled_time_vector[-1]+time_step) - corr_time/2

            # mappable = ax.matshow(spearman_corr[0], vmin=-1, vmax=1, extent=[t_min, t_max, t_min, t_max])
            mappable = ax.flatten()[i].matshow(spearman_corr, vmin=-1, vmax=1, extent=[t_min, t_max, t_max, t_min],
                                               cmap='bwr')

            ax.flatten()[i].set_title("window_size: " + str(corr_times[i]))
            ax.flatten()[i].set_aspect('equal', adjustable='box')
            ax.flatten()[i].get_shared_x_axes().join(ax.flatten()[i], ax0[0])
            fig.suptitle(basename)
            # fig.colorbar(mappable)
            i += 1


            fig1, ax1 = plt.subplots(flattened_upper_triang_idxs.shape[0], 1, sharex=True)
            fig1.suptitle(basename)
            j = 0
            for idx in flattened_upper_triang_idxs:
                ax1[j].plot(resampled_time_vector[int(ws/2): -int(ws/2)], flattened_matrixes[:, idx])
                ax1[j].get_shared_x_axes().join(ax1[j], ax0[0])
                fig1.suptitle(basename + ", corr length: " + str(corr_time))
                j += 1