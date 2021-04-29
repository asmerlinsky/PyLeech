import os

import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe

import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as bStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
import PyLeech.Utils.filterUtils as filterUtils
import PyLeech.Utils.correlationUtils as corrUtils
import gc
from itertools import combinations
import matplotlib.pyplot as plt
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


    for fn in list(cdd):
        if '2018_12_04_0003_0004_0005_0006b' in fn:
            continue
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}
        fn = fn.replace("/", '/')
        basename = os.path.splitext(os.path.basename(fn))[0]

        arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles(fn, list(ch_dict))


        kernel_sigma = 2
        time_range = 20
        time_step = .1
        corr_step = int(120/ time_step)

        for key in arr_dict.keys():
            if fs>15000:
                step = 2
                fs /= 2
            if ch_dict[key] == 'NS':
                arr_dict[key] = corrUtils.processContinuousSignal(arr_dict[key][::step], dt_step=1 / fs, kernel_sigma=kernel_sigma)
            else:
                arr_dict[key] = arr_dict[key][::step]
                if key == 'IN4':

                    line_peaks = np.array([  50. ,  150. ,  250. ,  350.2,  450.2,  550.2,  650.2,  750.2,
                            850.2,  950.4, 1050.4, 1150.4, 1250.4, 1350.4, 1450.6, 1550.4,
                            1650.6, 1750.6,
                            1850.6, 1950.6, 2050.8, 2150.8, 2250.8, 2350.8, 2450.8, 2551. ,
                            2651. , 2751. , 2851. , 2951. ])
                    line_peaks = line_peaks[line_peaks<fs/2]
                    arr_dict[key] = filterUtils.runFilter(arr_dict[key], line_peaks, fs, .1)
                    arr_dict[key] = filterUtils.runButterFilter(arr_dict[key], 1500, sampling_rate=fs)
                    arr_dict[key] = filterUtils.runButterFilter(arr_dict[key], 5, sampling_rate=fs, butt_order=4, btype='high')

                arr_dict[key] = corrUtils.processNerveSignal(arr_dict[key], kernel_sigma=kernel_sigma, time_range=time_range, dt_step=1 / fs)

            arr_dict[key] = arr_dict[key][int(.1 * arr_dict[key].shape[0]):int(.9 * arr_dict[key].shape[0])]
            arr_dict[key] = arr_dict[key][::int(time_step * fs)]
            arr_dict[key] = arr_dict[key][int(.1 * arr_dict[key].shape[0]):int(.9 * arr_dict[key].shape[0])]

        time_vector = np.linspace(0, arr_dict[key].shape[0] * time_step, arr_dict[key].shape[0], endpoint=False)

        centers = np.arange(0, arr_dict[key].shape[0], corr_step, dtype=int) + int(corr_step / 2)
        corr_dict = {}

        for k, j in combinations(arr_dict.keys(), 2):
            dcorr_list = []

            for i in np.arange(0, arr_dict[k].shape[0], corr_step, dtype=int):
                dcorr_list.append(corrUtils.getDistanceCorrelation(arr_dict[k][i:i + corr_step], arr_dict[j][i:i + corr_step]))

            corr_dict[(k, j)] = dcorr_list
            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(time_vector, arr_dict[k])
            # ax[0].plot(np.linspace(0, time_vector[-1], arr_dict[k][::100].shape[0]), arr_dict[k][::100])
            ax[1].plot(time_vector, arr_dict[j])

            ax[2].scatter(time_vector[centers[:-1]], dcorr_list[:-1])
            for i in range(3):
                ax[i].grid(linestyle='dotted')
            ax[2].set_ylim([0, 1])
            fig.suptitle(basename + '\n' + ch_dict[k] + '-' + ch_dict[j] + ' correlation')

            png_filename = 'correlation_figs/' + 'corr_' + basename + '_' + ch_dict[k] + '-' + ch_dict[j] + '.png'
            fig.savefig(png_filename, dpi=600)


            try:
                for a in ax:
                    a.cla()
            except:
                ax.cla()
            fig.clf()
            plt.close(fig)
            del fig, ax
            gc.collect()

        fig, ax = plt.subplots(len(corr_dict), 1, sharex=True, sharey=True)
        try:
            i = 0
            for tup in corr_dict.keys():
                ax[i].scatter(time_vector[centers[:-1]], corr_dict[tup][:-1])
                ax[i].set_title(ch_dict[tup[0]] + '-' + ch_dict[tup[1]])
                # ax[i].set_ylim([0, 1])
                ax[i].grid(linestyle='dotted')
                i += 1
                fig.suptitle(basename)
                # plt.tight_layout()
                png_filename = 'correlation_figs/' + 'corronly_' + basename + '.png'
                fig.savefig(png_filename, dpi=600)
        except TypeError:
            pass




        try:
            for a in ax:
                a.cla()
        except:
            ax.cla()
        fig.clf()
        plt.close(fig)
        del fig, ax
        gc.collect()