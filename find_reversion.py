# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:46:50 2018

@author: Agustin Sanchez Merlinsky
"""

import inspect
import os
import sys
import json
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
wdir = os.path.dirname(file_dir)
os.chdir(wdir)
sys.path.append(wdir)
sys.path.append(file_dir)

import numpy as np
import matplotlib.pyplot as plt


def binXYLists(bins, xvaluelist, yvaluelist):
    np_xval = np.asarray(xvaluelist)
    np_yval = np.asarray(yvaluelist)
    bin_interval = np.abs(bins[1]-bins[0])
    filled_bins = []
    bin_mean = []
    rel_bin_max = []
    rel_bin_min = []
    for center in bins:
        upper = center + bin_interval/2
        lower = center - bin_interval/2
        
        inbin = np.where((np_xval<= upper) & (np_xval > lower))[0]
        if len(inbin)>0:
            filled_bins.append(center)
            bmean = np.mean(np_yval[inbin])
            bmax = np.max(np_yval[inbin])
            bmin = np.min(np_yval[inbin])
            
            bin_mean.append(bmean)
            rel_bin_max.append(bmax-bmean)
            rel_bin_min.append(bmean-bmin)
    return filled_bins, bin_mean, rel_bin_min, rel_bin_max





def getReversion(json_file, plot_results=True, show_data_points=False, baseline_range=[-100, -40], invert_axis=False):
    full_file_results = json.load(open(json_file))
    baseline_bins = np.arange(baseline_range[0], baseline_range[1], 5)
    binned_results = {}
    ipsp_rev = []
    regression_error = []
    for neuron, burst_dict in full_file_results.items():
        mean_ipsp = []
        baseline = []
        freqs = []
        for burst, burst_info in burst_dict.items():
            mean_ipsp.append(burst_info['T data']['mean Ipsp'])
            baseline.append(burst_info['T data']['baseline'])
            freqs.append(burst_info['burst data']['freq'])
    
        bin_results = binXYLists(baseline_bins, baseline, mean_ipsp)
        xerror = np.abs(baseline_bins[1]-baseline_bins[0])/2
        
        bl_scaler = StandardScaler()
        ipsp_scaler = StandardScaler()
        
        scaled_bl = bl_scaler.fit_transform(np.asarray(baseline).reshape(-1,1))
        scaled_ipsp = ipsp_scaler.fit_transform(np.asarray(mean_ipsp).reshape(-1,1))
        
        regr = linear_model.LinearRegression()
        

        if invert_axis:
            ipsp_extremes = [[np.min(mean_ipsp)-.1*np.abs(np.min(mean_ipsp))],[np.max(mean_ipsp)+.1*np.abs(np.max(mean_ipsp))]]
            regr.fit(scaled_ipsp, scaled_bl)
            full_predict = bl_scaler.inverse_transform(regr.predict(scaled_ipsp))
            scaled_ipsp_extremes = ipsp_scaler.transform(ipsp_extremes)
            extremes_predict = bl_scaler.inverse_transform(regr.predict(scaled_ipsp_extremes))
            ipsp_scaled_orig = ipsp_scaler.transform([[0]])
            predicted_E_rev = bl_scaler.inverse_transform(regr.predict(ipsp_scaled_orig))[0][0]
        else:
            bl_extremes = [[bin_results[0][0]-10], [bin_results[0][-1]+10]]
            regr.fit(scaled_bl, scaled_ipsp)
            full_predict = ipsp_scaler.inverse_transform(regr.predict(scaled_bl))
            scaled_bl_extremes = bl_scaler.transform(bl_extremes)
            ipsp_scaled_orig = ipsp_scaler.transform([[0]])
            extremes_predict = ipsp_scaler.inverse_transform(regr.predict(scaled_bl_extremes))
            predicted_E_rev = bl_scaler.inverse_transform((ipsp_scaled_orig-regr.intercept_)/regr.coef_)[0][0]
             
            
        
        if invert_axis:
            err = np.sqrt(mean_squared_error(np.asarray(baseline).reshape(-1,1), full_predict))
            
            print('neuron %s : Erev = %.1f pm %.1f' % (neuron, predicted_E_rev, err))
        else:
            err = np.sqrt(mean_squared_error(np.asarray(mean_ipsp).reshape(-1,1), full_predict))
            print('neuron %s : Erev = %.1f, reg error = %.1f' % (neuron, predicted_E_rev, err))
            
        regression_error.append(err)
        ipsp_rev.append(predicted_E_rev) 
        
        
        
        
        plt.figure(1)
        
        plt.figure()
        plt.errorbar(bin_results[0], bin_results[1], xerr=xerror, yerr=[bin_results[2], bin_results[3]], fmt='|', color='k')
        
        
        plt.title(neuron)
        
        if invert_axis:
            plt.plot(extremes_predict, ipsp_extremes)
        else:
            plt.plot(bl_extremes, extremes_predict)
        plt.grid()
        plt.xlabel('Baseline (mV)')
        plt.ylabel(r'T Ipsp ($\Delta$ mV)', rotation=0, labelpad=50)
        plt.tight_layout()
        
        if show_data_points:
            plt.scatter(baseline, mean_ipsp, marker='x', s=150, color='b')
        
        
        
        plt.figure(1)
        if invert_axis:
            plt.plot(extremes_predict, ipsp_extremes, label=neuron)
        else:
            plt.plot([[bin_results[0][0]-10], [bin_results[0][-1]+10]], extremes_predict, label=neuron)
            
        
    #plt.legend()
    plt.grid()
    plt.xlim([-90, -30])
    plt.ylim([-7, 7])
    plt.xlabel('Baseline (mV)')
    plt.ylabel('T Ipsp ($\Delta$ mV)', rotation=0, labelpad=50)
    plt.tight_layout()
    return ipsp_rev, regression_error                                                                  
    

####################3    
if __name__ == '__main__':
    E_rev, error = getReversion('results_by_neuron.json', show_data_points=True, baseline_range=[-115, -40], invert_axis=False)
    print('mean E_rev = (%.1f +- %.1f) mV ' % (np.mean(E_rev), np.std(E_rev)))
    #plt.scatter(np.mean(E_rev), 0, marker='X', label='E_rev', s=100, color='k', zorder=100)
    #plt.errorbar(np.mean(E_rev), 0, xerr=(np.std(E_rev)/2), fmt='|', color='k')
    #plt.legend()
    