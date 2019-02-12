import pandas as pd

import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import PyLeech.Utils.burstClasses as burstUtils
import matplotlib.pyplot as plt
import glob
from importlib import reload
import numpy as np

def loadCrawlingDataBase(filename=None):
    if filename is None:
        filename = 'RegistrosDP_PP/crawling_database.csv'

    crawling_database = pd.read_csv(filename)
    try:
        crawling_database.drop('Unnamed: 0', axis=1, inplace=True)
    except KeyError:
        pass
    if crawling_database.index.names != ['filename', 'neuron']:
        crawling_database.set_index(['filename', 'neuron'], inplace=True)

    return crawling_database

def appendToDatabase(filename_list, database, min_isi=0.005, max_isi=2):
    try:
        crawling_database = pd.read_csv(database)
    except FileNotFoundError:
        print("File not Found, I'll generate a new one in %s" % database)
        crawling_database = pd.DataFrame([])

    columns = ['filename', 'start_time', 'end_time', 'neuron', 'spike_count', 'ISI_mean', 'ISI_std']
    df_list = []

    for j in range(filename_list):
        if filename_list[j] not in crawling_database.index.unique(0):
            pass

        elif not all([c in crawling_database.columns for c in columns]):
            crawling_database.drop(index=filename_list[j], inplace=True)
        else:
            continue

        print(filename_list[j])

        burst_object = burstUtils.BurstStorerLoader(filename_list[j], 'load')

        fig = burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=None,
                            template_dict=None, outlier_thres=3.5, ms=2)

        fig.canvas.draw_idle()
        plt.pause(10)
        crawling_interval = [float(x) for x in input('crawling interval as min-max:\n').split('-')]


        for key in burst_object.spike_freq_dict.keys():
            spikes = burst_object.spike_freq_dict[key][0]
            crawling_spikes = spikes[(spikes > crawling_interval[0]) & (spikes < crawling_interval[1])]
            diff = np.diff(crawling_spikes)
            diff = diff[(diff > min_isi) & (diff < max_isi)]

            spike_count = spikes.shape[0]


            diff_mean = np.mean(diff)
            diff_std = np.std(diff)
            good_unit = str(input('Is unit %i good\n(True/False)?' % key))
            if good_unit in ['true', 'True']:
                good_unit = True
            else:
                good_unit = False
            df_list.append([filename_list[j], crawling_interval[0], crawling_interval[1], str(key), spike_count, diff_mean, diff_std, good_unit])

        plt.close(fig)

    df = pd.append(df_list, columns=['filename', 'start_time', 'end_time', 'neuron', spike_count, 'ISI_mean', 'ISI_std', 'neuron_is_good'])
    df.to_csv('test.csv')


if __name__ == '__main__':
    crawling_database = pd.read_csv('RegistrosDP_PP/crawling_database.csv')

    crawling_database.drop('Unnamed: 0', axis=1, inplace=True)
    crawling_database.set_index(['filename', 'neuron'], inplace=True)
    crawling_database.head()
