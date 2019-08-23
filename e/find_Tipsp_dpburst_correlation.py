
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:40:23 2018

@author: Agustin Sanchez Merlinsky
"""
import os
import inspect
import sys
import PyLeech.Utils.AbfExtension as AbfE
import PyLeech.Utils.T_DP_classes as T_DP_classes
import numpy as np
import glob
import matplotlib.pyplot as plt
import PyLeech.Utils.spikeUtils as spikeUtils

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
sys.path.append(wdir)
sys.path.append(file_dir)

'''
params:
   ipsp detection:
       
   std_steps = 1000 
   butterworth order = 8
   filt freq = 150
   std thres 3.75
   ipsp_thres = 4.75
   ipsp_max_dist = 0.7
   
   dp_bursts:
       spike_max_dist = 0.7
'''

file_list = {
     'T1':[0.16, 0.22],
     'T2': [0.16, 0.22],
     'T3': [0.11, 0.16],
     'T4':[0.04, 0.06],
     'T5':[0.15, 0.25],
     'T6':[0.18, 0.26],
     'T7':[0.18, 0.28],
     'T8':[0.2, 0.25],
     'T9':[0.06, 0.095],
     'T10':[0.035, 0.06],
     'T11':[0.15, 0.26],
     'T12':[0.08, 0.13],
     'T13':[0.17, 0.22],
     'T14':[0.16, 0.28],
     'T15':[0.28, 0.4],
     'T16':[0.16, 0.3],
     'T17':[0.18, 0.28],
     'T18':[0.18, 0.28],
     'T19':[0.06, 0.09],
}

if __name__ == '__main__':
    '''
    Be sure to be reading the right channels (either Vm1 or Vm2 and IN5, IN6)
    '''



    T_channel = 'Vm1'
    dirname = 'T_ipsp'
    plot_traces_and_segments = False
    generate_T_obj_plots = False
    plot_results_by_trace = False
    plot_results_by_neuron = False
    plot_global_results = True

    ipsp_max_dist = 0.8

    global_dict = {}
    global_dict['match_count'] = {'ipsp_y': 0, 'ipsp_n': 0, 'burst_y': 0, 'burst_n': 0}
    global_dict['ipsp_data'] = {'matched_size': 0, 'unmatched_size': 0, 'matched_no': 0, 'unmatched_no': 0}

    total_ipsps_bursts = 0
    total_dp_bursts = 0
    global_par_area_freq = []
    global_par_area_freq_single_burst = []
    global_par_dpmatch_freq = []
    global_means = []
    bin_range = 5
    bins = np.arange(0, 35, bin_range)
    neuron_medians = []
    shifts = np.array(())
    avgd_by_neuron = []
    std_by_neuron = []
    for key, nerve_thres in file_list.items():

        neuron_dict = {}
        neuron_dict['match_count'] = {'ipsp_y': 0, 'ipsp_n': 0, 'burst_y': 0, 'burst_n': 0}
        neuron_dict['ipsp_data'] = {'matched_size': 0, 'unmatched_size': 0, 'matched_no': 0, 'unmatched_no': 0}
        neuron_par_area_freq_single_burst = []
        neuron_par_area_freq = []
        neuron_par_dpmatch_freq = []

        files = glob.glob(dirname + '\\' + key + '.*.abf')

        for file in files:

            block = AbfE.ExtendedAxonRawIo(file)

            extra_cel_channels = [i for i in block.ch_info.keys() if 'IN' in i]
            if len(extra_cel_channels) == 1:
                nerve_channel = extra_cel_channels[0]
            else:
                nerve_channel = 'IN6'

            T_signal = block.getSignal_single_segment(T_channel)
            nerve_signal = block.getSignal_single_segment(nerve_channel)
            fs = block.get_signal_sampling_rate()
            time = AbfE.generateTimeVector(len(T_signal), fs)

            margin = int(ipsp_max_dist * fs / 2)

            T_obj = T_DP_classes.TInfo(file, T_signal, fs, 1000, b_order=8, filt_freq=125, std_thres=4, ipsp_thres=4.75,
                                       peak_dist=150, margin=margin)
            T_obj.getFiltSignal()
            T_obj.use_LPsignal = True
            T_obj.plot_results = generate_T_obj_plots
            ipsp_list = T_obj.getIpsps()

            T_obj.getIpspBursts(spike_max_dist=ipsp_max_dist, min_spike_no=0, min_sps=0)
            T_obj.generateBurstObjList()

            DP_obj = T_DP_classes.DpInfo(file, nerve_signal, nerve_thres, peak_dist=100, fs=fs)
            DP_obj.generateBurstObjList(spike_max_dist=0.6, min_sps=5, height=DP_obj.max_height)


            results = T_DP_classes.TIpspDpsResults(T_obj, DP_obj)
            shifts = np.append(shifts, results.getIpspPhaseShift())
            for key1 in neuron_dict.keys():
                for key2 in neuron_dict[key1].keys():
                    neuron_dict[key1][key2] += results.result_dict[key1][key2]




            # plt.figure(1)
            if plot_traces_and_segments:
                fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
                fig.suptitle(file.split('\\')[1])
                T_DP_classes.plotBursts(ax[0], time, T_obj)

                T_DP_classes.plotBursts(ax[1], time, DP_obj)

            del block
            
            if plot_results_by_trace:
                plot = T_DP_classes.resultsPlotter(results.par_area_freq_single_burst, bins, trace=file)
                plot.boxPlot()
                # plot.meansPlot()
                plot.scatterPlot()

            neuron_par_area_freq_single_burst.extend(results.par_area_freq_single_burst)
            neuron_par_area_freq.extend(results.par_area_freq)
            neuron_par_dpmatch_freq.extend(results.par_dpmatch_freq)

        """
        Here starts neuron level processing
        """

        neuron_par_area_freq_single_burst = np.array(neuron_par_area_freq_single_burst)
        # print(key)
        T_DP_classes.normResults(neuron_par_area_freq_single_burst, norm_freq=20, rg=bin_range)

        plot = T_DP_classes.resultsPlotter(neuron_par_area_freq_single_burst, bins, neuron=key)
        neuron_medians.append(plot.getMeansResult(get_median=False))
        npafsgNP = np.array(neuron_par_area_freq_single_burst).T
        bins1, bin_mean, std1, std2 = spikeUtils.binXYLists(bins, npafsgNP[0], npafsgNP[1], std_as_err=True, full_binning=True)
        avgd_by_neuron.append(bin_mean)
        std_by_neuron.append(std1)
        plt.figure()
        plt.errorbar(bins1, bin_mean, 2 * np.array(std1))


        if plot_results_by_neuron:
            # plt.figure()
            # plot.boxPlot()
            plt.figure(1)
            plot.use_title=False
            plot.meansPlot()
            plt.figure()
            plot.scatterPlot()

        for key1 in global_dict.keys():
            for key2 in global_dict[key1].keys():
                global_dict[key1][key2] += neuron_dict[key1][key2]


        global_par_area_freq_single_burst.extend(neuron_par_area_freq_single_burst)
        global_par_area_freq.extend(neuron_par_area_freq)
        global_par_dpmatch_freq.extend(neuron_par_dpmatch_freq)

    if plot_global_results:
        plot = T_DP_classes.resultsPlotter(global_par_area_freq_single_burst, bins)
        # plot.boxPlot()
        # plot.meansPlot()
        # plot.scatterPlot()

        medians_list = []
        for i in bins:
            pml = []
            for nm in neuron_medians:
                ind = np.where(nm[0] == i)[0]
                if len(ind)==1:
                    pml.append(nm[1][ind])
            medians_list.append(pml)
        plt.figure()
        plt.boxplot(medians_list, positions=bins, widths=1.5)
        plt.xlim([bins[0]-5, bins[-1]+5])

        plt.figure()
        plt.hist(shifts, bins=50)
i

avgd_by_neuron = np.array(avgd_by_neuron).T
mean = []
std = []
for i in range(avgd_by_neuron.shape[0]):
    mean.append(np.mean(avgd_by_neuron[i][avgd_by_neuron[i] != 0.]))
    std.append(np.std(avgd_by_neuron[i][avgd_by_neuron[i] != 0.]))
# bins1, bin_mean, std1, std2 = spikeUtils.binXYLists(bins, npglobal[0], npglobal[1], std_as_err=True)
plt.figure()
plt.errorbar(bins1, mean, np.array(std), marker='o')
