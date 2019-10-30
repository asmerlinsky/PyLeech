import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstUtils
from PyLeech.Utils.burstStorerLoader import BurstStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import matplotlib.pyplot as plt
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    cdd = CDU.loadDataDict()
    file_list = list(cdd)
    fn = file_list[5]
    print(fn)

    burst_object = BurstStorerLoader(fn, 'RegistrosDP_PP', 'load')

    binning_dt = 0.1
    spike_kernel_sigma = .7
    kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=spike_kernel_sigma, time_range=20, dt_step=binning_dt)

    good_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if neuron_dict['neuron_is_good']]
    good_neurons.remove(9)
    good_neurons.remove(20)

    fig1, ax1 = burstUtils.plotFreq(burst_object.spike_freq_dict, template_dict=burst_object.template_dict,
                                    scatter_plot=True, outlier_thres=3.5, draw_list=good_neurons, ms=2)

    spike_freq_array = burstUtils.processSpikeFreqDict(burst_object.spike_freq_dict, .1,
                                                       selected_neurons=good_neurons,
                                                       time_length=burst_object.time[-1], time_interval=[[cdd[fn]["crawling_intervals"][0][0], 632]])

    smoothed_sfd = {}
    for key, items in spike_freq_array.items():
        smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], kernel, mode='same')])



    freq_threshold = .1
    spike_interval_dict = {}
    for key, items in smoothed_sfd.items():
        idxs = burstUtils.getNonZeroIdxs(items[1], freq_threshold)
        spike_interval_dict[key] = idxs.reshape(int(idxs.shape[0] / 2), 2)


    fig, ax = burstUtils.plotFreq(smoothed_sfd, draw_list=good_neurons, scatter_plot=False, )
    j = 0
    for key, item in spike_interval_dict.items():
        for i in item.flatten():
            ax[j].axvline(smoothed_sfd[key][0][i], c='r')

        j += 1



    interval_list, target_neuron = burstUtils.generateBurstSegmentsFromManyNeurons(smoothed_sfd, spike_interval_dict)
    fig, ax = plt.subplots()
    for burst, neuron in zip(interval_list, target_neuron) :
        ax.plot(burst, color=burst_object.color_dict[neuron])


    sc = StandardScaler()
    tf = sc.fit_transform(interval_list)
    n_components = 10
    pca = PCA(n_components=n_components)
    pca_tf_bursts = pca.fit_transform(tf)

    df = pd.DataFrame(np.hstack((pca_tf_bursts, target_neuron.reshape(-1, 1))), columns=list(range(n_components)) + ['neuron'])
    print(pca.explained_variance_ratio_)
    plot_kws = {"s": 50}
    sns.pairplot(df, hue='neuron', vars=df.columns[:3], plot_kws=plot_kws)