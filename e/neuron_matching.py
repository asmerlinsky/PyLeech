# -*- coding: utf-8 -*-

import sys
import os

import PyLeech.Utils.unitInfo

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import PyLeech.Utils.burstClasses as burstUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload
import numpy as np
import pandas as pd
"""
%load_ext autoreload
%autoreload 2

"""

plt.ion()
min_isi = 0.005
max_isi = 2
dt = min_isi/2
bins = np.arange(min_isi, max_isi, dt)
df_list =[]

pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')
for j in range(len(pkl_files)):
    print(j, pkl_files[j])
    filename = pkl_files[j]
    burst_object = PyLeech.Utils.unitInfo.UnitInfo(filename, 'load')
    # burst_object.plotTemplates()
    # burst_object.isDe3 = 0
    #
    fig = burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=None,
                        template_dict=None, outlier_thres=3.5, ms=2)
    # fig.canvas.draw_idle()
    # plt.pause(10)
    crawling_interval = [float(x) for x in input('crawling interval as min-max:\n').split('-')]

    # plt.close(fig)
    for key in burst_object.spike_freq_dict.keys():


        spikes = burst_object.spike_freq_dict[key][0]
        crawling_spikes = spikes[(spikes > crawling_interval[0]) & (spikes < crawling_interval[1])]
        diff = np.diff(crawling_spikes)
        diff = diff[(diff > min_isi) & (diff < max_isi)]
        diff /= dt
        diff = diff.round().astype('int')

        diff_hist = np.histogram(diff, bins=np.arange(30) - 0.5, density=False)[0]
        spike_count = len(spikes)
        data = np.random.poisson(np.mean(diff), 10000)
        data_hist = np.histogram(data, bins=np.arange(30) - 0.5, density=True)[0]

        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        #
        # plt.figure()
        # plt.title('mean = % d, std = %d' % (diff_mean, diff_std))
        # plt.scatter(np.arange(29), diff_hist)
        # plt.scatter(np.arange(29), data_hist)
        df_list.append([filename, crawling_interval[0], crawling_interval[1], str(key), spike_count, diff_mean, diff_std])

df = pd.DataFrame(df_list, columns=['filename', 'start_time', 'end_time', 'neuron', 'crawling_spike_count', 'ISI_mean', 'ISI_std'])
df.to_csv('test.csv')
cmap = plt.get_cmap('tab20')
for index, row in df.iterrows():
    if row.loc['file']==5 or row.loc['file'] == 6:
        plt.scatter(row.loc['ISI_mean'], row.loc['ISI_std'], color=cmap(row.loc['file']))


if False:
    import seaborn as sns
    sns.set()
    sns.scatterplot(data=df, x='ISI_mean', y='ISI_std', hue='file',style='neuron')


    filename = pkl_files[7]
    print(filename)

    # %matplotlib qt5
    sns.set()
    fig, axes = BurstUtils.plotCompleteDetection(burst_object.traces, burst_object.time, burst_object.spike_dict,
                                     burst_object.template_dict, burst_object.color_dict, legend=True, step=10)

    # burst_object.isDe3 = 0

    BurstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=None, #[burst_object.time[::5], NS[::5]]
                        template_dict=burst_object.template_dict, outlier_thres=None, ms=2, sharex=axes[0])






    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import factorial


    def poisson(k, alfa, beta):
        """poisson pdf, parameter lamb is the fit parameter"""
        return ((alfa)**(k)/factorial(k)) * np.exp(-lamb)

        return(np.exp(alfa + beta*k))


    def negLogLikelihood(params, data):
        """ the negative log-Likelohood-Function"""
        lnl = - np.sum(np.log(poisson(data, params[0])))
        return lnl


    for key in burst_object.spike_freq_dict.keys():
        if key==3:
            spikes = burst_object.spike_freq_dict[key][0]
            crawling_spikes = spikes[(spikes>350) & (spikes<600)]
            diff = np.diff(crawling_spikes)
            diff = diff[(diff>min_isi) & (diff<max_isi)]
            diff /= dt
            diff = diff.round().astype('int')
            poisson_lambda = np.mean(diff-np.min(diff))
            print(poisson_lambda)
    #
    ##        diff_hist = np.histogram(diff, bins=bins)[0]
    #        result = minimize(negLogLikelihood,  # function to minimize
    #                      x0=np.ones(1),     # start value
    #                      args=(diff,),      # additional arguments for function
    #                      method='Powell',   # minimization method, see docs
    #                      )
    #
    #        print(result)
            diff_hist = np.histogram(diff, bins=np.arange(30)-0.5, density=False)[0]

            data = np.random.poisson(np.mean(diff), 10000)
            data_hist = np.histogram(data, bins=np.arange(30)-0.5, density=True)[0]
            plt.figure()
            plt.scatter(np.arange(29), -np.log(diff_hist))
            plt.scatter(np.arange(29), -np.log(data_hist))
    #        plt.hist(diff, bins=np.arange(30)-0.5, density=True)




    #        x_plot = np.linspace(0, 100, 1000)
    #        plt.plot(x_plot, poisson(x_plot, result.x), 'r-', lw=2)

        #    plt.hist(data, bins=bins, density=True)
    #        plt.scatter(np.arange(len(bins)-1),diff_hist/sum(diff_hist))
    #
    #        plt.plot(np.arange(len(bins)), poisson(np.arange(len(bins)), result.x), 'r-', lw=2)


    """
    Fitting with sklearn logistic regression
    """


    """
    # get poisson deviated random numbers
    data = np.random.poisson(2, 1000)
    
    # minimize the negative log-Likelihood
    
    result = minimize(negLogLikelihood,  # function to minimize
                      x0=np.ones(1),     # start value
                      args=(data,),      # additional arguments for function
                      method='Powell',   # minimization method, see docs
                      )
    # result is a scipy optimize result object, the fit parameters 
    # are stored in result.x
    print(result)
    
    # plot poisson-deviation with fitted parameter
    x_plot = np.linspace(0, 20, 1000)
    
    plt.hist(data, bins=np.arange(15) - 0.5, density=True)
    plt.plot(x_plot, poisson(x_plot, result.x), 'r-', lw=2)
    """


#### Desde aca pruebo comparando templados
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.abfUtils as abfUtils
pkl_files = glob.glob('RegistrosDP_PP/*.pklspikes')

reload(burstUtils)
reload(abfe)
filename = pkl_files[7]
burst_object = PyLeech.Utils.unitInfo.UnitInfo(filename, 'load')
fig_ax = burst_object.plotTemplates(signal_inversion=[1, -1])

filename = pkl_files[6]
burst_object = PyLeech.Utils.unitInfo.UnitInfo(filename, 'load')
burst_object.plotTemplates(fig_ax=fig_ax)


for j in range(len(pkl_files)):
    # print(j, pkl_files[j])
    filename = pkl_files[j]
    burst_object = PyLeech.Utils.unitInfo.UnitInfo(filename, 'load')

    basename = abfUtils.getAbfFilenamesfrompklFilename(burst_object.filename)
    if basename:
        arr_dict, time, fs = abfe.getArraysFromAbfFiles(basename, ['Vm1'])
        NS = arr_dict['Vm1']
        del arr_dict
        try:
            opt_trace = [NS[::5], time[::5]]
        except:
            print('Exception for burst_object %s' % filename)


        burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=opt_trace,
                            template_dict=None, outlier_thres=3.5, ms=2)
        plt.suptitle(filename)
