import PyLeech.Utils.srmFitUtils as FitUtils
import PyLeech.Utils.abfUtils as abfUtils
import PyLeech.Utils.burstClasses as burstClasses
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.AbfExtension as abfe
import matplotlib.pyplot as plt
import glob
from importlib import reload
import numpy as np
import os
import time

import PyLeech.Utils.CrawlingDatabaseUtils as CDU


pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)): print(j, pkl_files[j])
filenames = pkl_files[6:8]
folder_name = 'RegistrosDP_PP'
for filename in filenames:

    print(filename)

    burst_object = burstClasses.BurstStorerLoader(filename, folder_name, mode='load')
    basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
    arr_dict, time_array, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
    NS = arr_dict['Vm1']
    del arr_dict

    burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=[time_array[::5], NS[::5]],
                                template_dict=burst_object.template_dict, outlier_thres=3.5, ms=2, sharex=None)

cdb = CDU.loadCrawlingDataBase()

dt = 2

timing_list = []
it_no_list = []
fitters = []
for filename in pkl_files[6:8]:

    burst_object = burstClasses.BurstStorerLoader(filename, folder_name, mode='load')
    basename = abfUtils.getAbfFilenamesfrompklFilename(filename)
    arr_dict, time_array, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
    NS = arr_dict['Vm1']
    del arr_dict

    dt = 1
    print("dt:", dt)
    step = int(fs * dt)

    first_idx = np.where(time_array>cdb.loc[filename].start_time.iloc[0])[0][0]
    last_idx =  np.where(time_array>cdb.loc[filename].end_time.iloc[0])[0][0]
    time_delta = time_array[last_idx] - time_array[first_idx]
    # last_time = 400 - time_array[first_idx]
    t_start = time.time()
    re_time = np.arange(0, round(time_delta, 4), dt)
    re_NS = NS[first_idx:last_idx-1:step]
    good_neuron_list = cdb.loc[filename].index[cdb.loc[filename, 'neuron_is_good'] == True].values
    spikes_times_dict = FitUtils.getSpikeTimesDict(burst_object.spike_freq_dict, good_neuron_list, outlier_threshold=3.5)
    spikes_times_dict = FitUtils.substractTimes(spikes_times_dict, time_array[first_idx])

    real_fit = FitUtils.SRMFitter(re_time, re_NS, spikes_times_dict, k_duration=30, refnconn_duration=5, dim_red=1)
    t_start = time.time()
    real_fit.minimizeNegLogLikelihood(10**5, [FitUtils.penalizedNegLogLikelihood, FitUtils.gradPenalizedNegLogLikelihood],
                                      verbose=False, penalty_param=0.5, talk=False)

    fitters.append(real_fit)
    print((time.time()-t_start)/60)
    timing_list.append((time.time()-t_start)/60)
    it_no_list.append(real_fit.optimization.nit)
    print(timing_list)
    real_fit.plotFitArray(separate_plots=True)

    #
    #
    # plt.plot(time_steps-time_array[first_idx], np.array(times)/60)

