import PyLeech.Utils.burstStorerLoader
import PyLeech.Utils.srmFitUtils as FitUtils
import PyLeech.Utils.abfUtils as abfUtils
import PyLeech.Utils.burstClasses as burstClasses
import PyLeech.Utils.AbfExtension as abfe
import multiprocessing
import glob
import numpy as np
import itertools
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
from functools import partial
import pickle
#with open('fitter_list.pickle', 'wb') as handle:
#    pickle.dump(fitters, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
with open('fitter_list.pickle', 'rb') as handle:
   fit_list = pickle.load(handle)

#
# pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
# for j in range(len(pkl_files)): print(j, pkl_files[j])
# filenames = pkl_files[6:8]
# folder_name = 'RegistrosDP_PP'
# for filename in filenames:
#     print(filename)
#
#     burst_object = burstClasses.BurstStorerLoader(filename, folder_name, mode='load')
#     basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
#     arr_dict, time_array, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
#     NS = arr_dict['Vm1']
#     del arr_dict
# #    print(cdb.loc[filename].start_time.iloc[0], cdb.loc[filename].end_time.iloc[0])
# #    burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time_array[::5], NS[::5]],
# #                                template_dict=burst_object.template_dict, outlier_thres=3.5, ms=2, sharex=None,
# #                                draw_list=cdb.loc[filename].index[cdb.loc[filename, 'neuron_is_good'] == True].values)


if __name__ == "__main__":

    pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
    folder_name = 'RegistrosDP_PP'
    cdb = CDU.loadCrawlingDatabase()


    dt_list = [0.1, 0.2, 0.5, 1, 2]
    penalty_list = [0, 0.1 , 0.2, 0.5, 1, 2]
    lp_threshold_list = [2, 5]
    # lp_threshold_list = [5]
    timing_list = []
    it_no_list = []
    fitters = []
    processes = []
    for filename in pkl_files[6:8]:
        burst_object = PyLeech.Utils.burstStorerLoader.BurstStorerLoader(filename, folder_name, mode='load')
        basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
        arr_dict, time_array, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
        NS = arr_dict['Vm1']
        del arr_dict
        first_idx = np.where(time_array > cdb.loc[filename].start_time.iloc[0])[0][0]
        last_idx = np.where(time_array > cdb.loc[filename].end_time.iloc[0])[0][0]
        time_delta = time_array[last_idx] - time_array[first_idx]

        good_neuron_list = cdb.loc[filename].index[cdb.loc[filename, 'neuron_is_good'] == True].values

        print(good_neuron_list)

        spike_times_dict = FitUtils.getSpikeTimesDict(burst_object.spike_freq_dict, good_neuron_list,
                                                      outlier_threshold=3.5)
        spike_times_dict = FitUtils.substractTimes(spike_times_dict, time_array[first_idx])

        part_args = (time_delta, NS, spike_times_dict, 30, 30, 1, filename, first_idx, last_idx,
                     [FitUtils.penalizedNegLogLikelihood, FitUtils.gradPenalizedNegLogLikelihood], fs)

        func = partial(FitUtils.runCrawlingFitter, *part_args)
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as p:
            results = p.starmap(func,
                            itertools.product(dt_list, penalty_list, lp_threshold_list)
                            )

        fitters.extend([res[0] for res in results])
        timing_list.extend([res[1] for res in results])
        it_no_list.extend([res[2] for res in results])
            # for dt, penalty, lp_thres in itertools.product(dt_list, penalty_list, lp_threshold_list):
            #     NS = filterUtils.runButterFilter(NS, lp_thres, butt_order=4, sampling_rate=fs)
            #     step = int(fs * dt)
            #     print("\n\ndt: %f, penalty: %f" % (dt, penalty))
            #
            #     t_start = time.time()
            #
            #     re_time = np.arange(0, round(time_delta, 4), dt)
            #     re_NS = NS[first_idx:last_idx - 1:step]
            #
            #     real_fit = FitUtils.SRMFitter(re_time, re_NS, spikes_times_dict, k_duration=30, refnconn_duration=30,
            #                                   dim_red=1,
            #                                   penalty=penalty, filename=filename)
            #     t_start = time.time()
            #     real_fit.minimizeNegLogLikelihood(10 ** 5,
            #                                       [FitUtils.penalizedNegLogLikelihood,
            #                                        FitUtils.gradPenalizedNegLogLikelihood],
            #                                       verbose=False, talk=False)
            #
            #     fitters.append(real_fit)
            #     timing_list.append((time.time() - t_start) / 60)
            #     it_no_list.append(real_fit.optimization.nit)
            #     print(timing_list[-1])

            # real_fit.plotFitArray(separate_plots=True)

            # FitUtils.beep()
            # for fitter in fitters:
            #     fitter.plotFitArray(separate_plots=True)
