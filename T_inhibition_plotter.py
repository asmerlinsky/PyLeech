# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:16:07 2018

@author: Agustin Sanchez Merlinsky
"""

import inspect
import os
import sys
import json

file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
sys.path.append(wdir)
sys.path.append(file_dir)

import numpy as np
import json_numpy
import matplotlib.pyplot as plt



Run = 1
full_file_results = json.load(open('results.json'))

######################################
if Run == 1:
    for neuron, files_dict in full_file_results.items():
        for files, burst_dict in files_dict.items():
            type_separated = {}
            type_separated['baseline'] = []
            type_separated['freq'] = []
            type_separated['T data'] = []
            
            for segments, results in burst_dict.items():
                type_separated['baseline'].append(results['T data']['baseline'])
                type_separated['freq'].append(results['burst data']['freq'])
                type_separated['T data'].append(results['T data']['mean Ipsp'])
            rounded_freq = [round(x) for x in type_separated['freq']]
            plt.figure()
            plt.title(files)
            plt.scatter(type_separated['baseline'], type_separated['T data'], marker='x', s=150, c=rounded_freq, cmap='copper')        
            plt.colorbar()
    
    #####################################
    type_separated = {}
    type_separated['baseline'] = []
    type_separated['freq'] = []
    type_separated['T data'] = []
    
    for neurons, file_dict in full_file_results.items():
        for files, burst_dict in file_dict.items():    
            for segments, results in burst_dict.items():
                print(results)
                type_separated['baseline'].append(results['T data']['baseline'])
                type_separated['freq'].append(results['burst data']['freq'])
                type_separated['T data'].append(results['T data']['mean Ipsp'])
    
    
    
    plt.figure()
    plt.scatter(type_separated['baseline'], type_separated['T data'], marker='+', s=150, linewidths=4, c=type_separated['freq'], cmap='jet')
    ####################3    
    baseline_ipsp_freq = []   
    for files, data in full_file_results.items():
        
        for segments, results in data.items():
            baseline_ipsp_freq.append([files, results['T data']['baseline'], results['T data']['mean Ipsp'], results['burst data']['freq']])
    


 #########
if Run == 2:  
    single_plot = False
    if single_plot:    
        freqs = []
        ipsp_mean= []
        ipsp_count = []
        baseline = []
    for neuron, neuron_results in full_file_results.items():
        if not single_plot:    
            freqs = []
            ipsp_mean= []
            ipsp_count = []
            baseline = []
        
        for file, file_results in neuron_results.items():
            for burst, burst_info in file_results.items():
                freqs.append(burst_info['burst data']['freq'])
                baseline.append(burst_info['T data']['baseline'])
                ipsp_mean.append(burst_info['T data']['mean Ipsp'])
                ipsp_count.append(burst_info['T data']['Ipsp count'])
        
        if not single_plot:
            
            plt.figure()
            plt.title(neuron)
            plt.hist(ipsp_mean)
            plt.xlabel('Ipsp (mv)')
            
            plt.figure()
            plt.title(neuron)
            plt.hist2d(baseline, ipsp_mean, bins = 50)
            plt.xlabel('Baseline (mV)')
            plt.ylabel('Ipsp (mV)')
            plt.colorbar()
            
            plt.figure()
            plt.title(neuron)
            plt.plot(freqs, ipsp_count, 'x')
            plt.xlabel('Frecuency (Hz)')
            plt.ylabel('Ipsp peak count')
            
            plt.figure()
            plt.title(neuron)
            plt.scatter(baseline, ipsp_mean, marker='x', s=150, c=freqs, cmap='copper')
            plt.xlabel('Baseline (mV)')
            plt.ylabel('Ipsp (mV)')
            plt.colorbar()
    
    if single_plot:                
        plt.figure()
        plt.plot(freqs, ipsp_mean, 'x')
        plt.xlabel('Frecuency (Hz)')
        plt.ylabel('Ipsp (mV)')
        
        plt.figure()
        plt.hist(ipsp_mean, bins=30)
        plt.xlabel('Ipsp (mv)')
        
        plt.figure()
        plt.hist2d(freqs, ipsp_mean, bins = 50)
        plt.xlabel('Frecuency (Hz)')
        plt.ylabel('Ipsp (mV)')
        plt.colorbar()
        
        plt.figure()
        plt.plot(freqs, ipsp_count, 'x')
        plt.xlabel('Frecuency (Hz)')
        plt.ylabel('Ipsp peaks')
        
        plt.figure()
        plt.scatter(baseline, freqs, marker='x', s=150, c=ipsp_mean, cmap='copper')
        plt.xlabel('Baseline (mV)')
        plt.ylabel('Spike frequency (Hz)')
        plt.colorbar()