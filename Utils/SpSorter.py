import csv
import os
import pickle as pickle
import time
from copy import deepcopy
from subprocess import Popen

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import svd
from pandas.plotting import scatter_matrix
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import PyLeech.Utils.burstUtils
from PyLeech.Utils import templateDictUtils as tdictUtils, sorting_with_python as swp
from PyLeech.Utils.spsortUtils import good_evts_fct, proc_num, generatePklFilename, beep, nan, opp0, getGoodClusters, \
    generateClustersTemplate, getTemplateDictSubset
import seaborn as sns
import warnings

# noinspection PyPep8,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit
class SpSorter:
    """Spike sorter class

    Parameters
    ----------

    data: list
        list of traces to analyze

    time_vect: ndarray
        time vector returned by AbfE.generateTimeVector

    fs: float
        sampling frequency

    ------

    Notes
    ------

    """

    attrs_list = ['filename', 'folder', 'time', 'traces', 'sample_freq', 'state', 'peaks_idxs', 'evts', 'evt_interval',
                  'ch_no', 'evt_length', 'evts_median', 'evts_mad', 'evts_max', 'original_good_evts',
                  'good_evts', 'good_evts_idxs', 'unitary', 'original_clusters', 'model', 'train_clusters',
                  'cluster_color', 'before', 'after'
                  'good_colors', 'template_dict', 'rounds', 'final_spike_dict']

    def __init__(self, filename, folder=None, data=None, time_vect=None, fs=None, verbose=False):

        if data is None or time_vect is None or fs is None:
            print('Loading data from pickle')
            self.loadResults(filename=filename)
            self.filename = filename
            try:
                self.assignChannelsToClusters()
            except:
                pass

        else:
            self.filename = filename
            self.traces = data
            self.ch_no = len(data)
            self.sample_freq = fs
            self.time = time_vect
            self.discretization = [np.min(np.diff(np.sort(np.unique(x)))) for x in self.traces]
            self.folder = folder
            self.normed = False

            self.verbose = verbose
            self.pred = []

            self.rounds = []
            self.secondary_spikes = []
            self.state = 'Unprocessed'

    def normTraces(self):
        data_mad = list(map(swp.mad, self.traces))
        self.traces = np.array(list(map(lambda x: (x - np.median(x)) / swp.mad(x), self.traces)))
        self.peel = [self.traces]
        self.normed = True

    def smoothAndFindPeaks(self, vect_len=5, threshold=5, min_dist=50):
        """ Used for smoothing and peak finding

        :param vect_len: lenght of smoothing vector
        :type vect_len: int
        :param threshold: threshold for peak detection
        :type threshold: float
        :param min_dist: minimum dist between peaks
        :type min_dist: int
        :return: peaks indexes
        :rtype: list
        """
        peak_detection_data = np.apply_along_axis(lambda x:
                                                  fftconvolve(x, np.array([1] * vect_len) / float(vect_len), 'same'),
                                                  1, np.array(self.traces))
        peak_detection_data = (peak_detection_data.transpose() /
                               np.apply_along_axis(swp.mad, 1, peak_detection_data)).transpose()
        peak_detection_data[peak_detection_data < threshold] = 0

        self.peaks_idxs = swp.peak(peak_detection_data.sum(0), minimal_dist=min_dist)
        print("found %i spikes" % self.peaks_idxs.shape[0])
        try:
            del self.random_idxs
        except AttributeError:
            pass
        return self.peaks_idxs

    def smoothAndGetPeaks(self, vect_len=5, threshold=5, min_dist=50):
        """ Used for smoothing and peak finding

        :param vect_len: lenght of smoothing vector
        :type vect_len: int
        :param threshold: threshold for peak detection
        :type threshold: float
        :param min_dist: minimum dist between peaks
        :type min_dist: int
        :return: peaks indexes
        :rtype: list
        """
        peak_detection_data = np.apply_along_axis(lambda x:
                                                  fftconvolve(x, np.array([1] * vect_len) / float(vect_len), 'same'),
                                                  1, np.array(self.traces))
        peak_detection_data = (
                    peak_detection_data.transpose() / np.apply_along_axis(swp.mad, 1, peak_detection_data)).transpose()
        peak_detection_data[peak_detection_data < threshold] = 0

        return swp.peak(peak_detection_data.sum(0), minimal_dist=min_dist)

    def smoothAndVisualizeThreshold(self, channel=0, vect_len=5, threshold=5, interval=None):

        if interval is None:
            interval = [0, len(self.time)]
        else:
            interval = [int(interval[0] * len(self.time)), int(interval[1] * len(self.time))]

        data_filtered = np.apply_along_axis(lambda x:
                                            fftconvolve(x, np.array([1] * vect_len) / float(vect_len), 'same'),
                                            1, self.traces)
        data_filtered = (data_filtered.transpose() / np.apply_along_axis(swp.mad, 1, data_filtered)).transpose()

        data_filtered[data_filtered < threshold] = 0

        plt.figure()
        plt.plot(self.time[interval[0]:interval[1]], self.traces[channel, interval[0]:interval[1]], color='black')
        plt.axhline(y=threshold, color="blue", linestyle="dashed")
        plt.plot(self.time[interval[0]:interval[1]], data_filtered[channel, interval[0]:interval[1]], color='red')
        # plt.xlim([0,0.2])
        # plt.ylim([-5,10])
        plt.xlabel('Time (s)')

    def makeEvents(self, before=49, after=50):
        self.evt_interval = [before, after]
        self.evt_length = before + after + 1
        self.evts = swp.mk_events(self.peaks_idxs, self.traces, before, after)
        self.evts_median = np.apply_along_axis(np.median, 0, self.evts)
        self.evts_mad = np.apply_along_axis(swp.mad, 0, self.evts)
        self.evts_max = self.evts[:, before + 1]

        try:
            if self.good_evts.shape[0] != self.evts.shape[0]:
                self.good_evts = np.ones(self.evts.shape[0], dtype=bool)
        except AttributeError:
            self.good_evts = np.ones(self.evts.shape[0], dtype=bool)

        return self.evts

    def makeNoiseEvents(self, safety_factor=2.5, size=2000):
        self.noise_evts = swp.mk_noise(self.peaks_idxs, self.traces, self.evt_interval[0], self.evt_interval[1],
                                       safety_factor=safety_factor, size=size)

    def getGoodEvents(self, threshold=3):
        """
        DEPRECATED
        """
        self.good_evts = np.ones(self.evts.shape[0], dtype=bool)
        return self.good_evts

    def takeSubSetEvents(self, keep_pct=0.1):
        idxs = np.where(self.good_evts)[0]

        np.random.shuffle(idxs)
        idxs = idxs[int(keep_pct * len(idxs)):]
        self.random_idxs = idxs
        self.good_evts[idxs] = False
        print("Kept %i spikes" % sum(self.good_evts))
        self.shuffle_good_idxs = np.where(self.good_evts)[0]
        self.original_good_evts = deepcopy(self.good_evts)

    def getUniqueClusters(self):
        return np.unique(self.train_clusters)

    def GmmClusterEvents(self, clust_no, use_pca=False, dim_size=0, n_init=100, max_iter=400, verbose=0,
                            n_jobs=None, save=False):
        if n_jobs is None:
            n_jobs = (proc_num - 2)

        self.model = GaussianMixture(n_components=clust_no, verbose=verbose, n_init=n_init, max_iter=max_iter)

        try:
            isfile = os.path.isfile(generatePklFilename(self.filename, self.folder))
        except:
            isfile = os.path.isfile(generatePklFilename(self.filename))

        if save and isfile:
            print('Warning, this will overwrite current pkl')
            ipt = input('enter any key if you want to exit')
            ipt = str(ipt)
            if len(ipt) > 0:
                print('exiting method')
                return

        print('Running')

        evts = self.evts

        start = time.time()
        if not use_pca:
            cluster = self.model.fit_predict(evts[self.good_evts].reshape(-1, 1))
        else:
            if dim_size == 0:
                print('Setting dimension size to 10')
                dim_size = 10
            cluster = self.model.fit_predict(np.dot(evts[self.good_evts, :], self.unitary[:, 0:dim_size]))
        self.km_dim_size = dim_size
        end = time.time()
        l = []
        for i in range(clust_no):
            if sum(cluster == i) > 0:
                l.append((i, np.apply_along_axis(np.median, 0, self.evts[self.good_evts, :][cluster == i, :])))
        cluster_median = list(l)

        cluster_size = list([np.sum(np.abs(x[1])) for x in cluster_median])
        new_order = list(reversed(np.argsort(cluster_size)))
        new_order_reverse = sorted(range(len(new_order)), key=new_order.__getitem__)
        self.train_clusters = np.array([new_order_reverse[i] for i in cluster])
        self.original_clusters = deepcopy(self.train_clusters)

        self.state = 'Postclustering'
        print(
            'Clustering took %i seconds with these settings:\ndim_size = %i,\nevt number = %i,\nn_init = %i,\nmax_iter = %i, ' % (
                int(end - start), dim_size, sum(self.good_evts), n_init, max_iter))
        beep()
        if save:
            try:
                self.saveResults()
            except Exception as e:
                print("Couldn\'t save object due to following error:")
                print(e)
                pass
        else:
            print("Not saving results")


        return self.train_clusters



    def KMeansClusterEvents(self, clust_no, use_pca=False, dim_size=0, n_init=1000, max_iter=400, verbose=0,
                            n_jobs=None, save=False):
        if n_jobs is None:
            n_jobs = (proc_num - 2)

        self.model = KMeans(n_clusters=clust_no, init='k-means++', verbose=verbose, n_init=n_init, max_iter=max_iter,
                            precompute_distances=True,
                            copy_x=True, n_jobs=n_jobs)

        try:
            isfile = os.path.isfile(generatePklFilename(self.filename, self.folder))
        except:
            isfile = os.path.isfile(generatePklFilename(self.filename))

        if save and isfile:
            print('Warning, this will overwrite current pkl')
            ipt = input('enter any key if you want to exit')
            ipt = str(ipt)
            if len(ipt) > 0:
                print('exiting method')
                return

        print('Running')

        evts = self.evts

        start = time.time()
        if not use_pca:
            cluster = self.model.fit_predict(evts[self.good_evts].reshape(-1, 1))
        else:
            if dim_size == 0:
                print('Setting dimension size to 10')
                dim_size = 10
            cluster = self.model.fit_predict(np.dot(evts[self.good_evts, :], self.unitary[:, 0:dim_size]))
        self.km_dim_size = dim_size
        end = time.time()
        l = []
        for i in range(clust_no):
            if sum(cluster == i) > 0:
                l.append((i, np.apply_along_axis(np.median, 0, self.evts[self.good_evts, :][cluster == i, :])))
        cluster_median = list(l)

        cluster_size = list([np.sum(np.abs(x[1])) for x in cluster_median])
        new_order = list(reversed(np.argsort(cluster_size)))
        new_order_reverse = sorted(range(len(new_order)), key=new_order.__getitem__)
        self.train_clusters = np.array([new_order_reverse[i] for i in cluster])
        self.original_clusters = deepcopy(self.train_clusters)

        self.state = 'Postclustering'
        print(
            'Clustering took %i seconds with these settings:\ndim_size = %i,\nevt number = %i,\nn_init = %i,\nmax_iter = %i, ' % (
                int(end - start), dim_size, sum(self.good_evts), n_init, max_iter))
        beep()
        if save:
            try:
                self.saveResults()
            except Exception as e:
                print("Couldn\'t save object due to following error:")
                print(e)
                pass
        else:
            print("Not saving results")
        return self.train_clusters

    def getKMeansFullPrediction(self):
        self.full_cluster = self.model.predict(np.dot(self.evts, self.unitary[:, 0:self.km_dim_size]))

    def subdivideClusters(self, cluster, clust_no, dim_size=0, n_init=1000, max_iter=400, kmeans=False, verbose=0):
        if kmeans:
            model = KMeans(n_clusters=clust_no, init='k-means++', n_init=n_init, max_iter=max_iter, precompute_distances=True,
                        copy_x=True, n_jobs=(proc_num - 2))
        else:
            model = GaussianMixture(n_components=clust_no, verbose=verbose, n_init=n_init, max_iter=max_iter)

        evts = self.evts[self.good_evts, :][self.train_clusters == cluster]
        if dim_size == 0:
            print('Setting dimension size to 10')
            dim_size = 10

        sub_clusters = model.fit_predict(np.dot(evts, self.unitary[:, 0:dim_size]))

        unique_clusters = np.abs(np.unique(self.train_clusters))
        max_cluster = np.max(unique_clusters[(unique_clusters != nan) & (unique_clusters != opp0)])
        clust_idx = np.where(self.train_clusters == cluster)[0]
        for i in np.unique(sub_clusters):
            if i == 0:
                max_cluster += 1
            else:
                self.train_clusters[clust_idx[sub_clusters == i]] = max_cluster
                print('added cluster  no %i' % max_cluster)
                max_cluster += 1

    def hideClusters(self, bad_clust=None, good_clust=None):

        if type(bad_clust) is int:
            if bad_clust == 0:
                self.train_clusters[self.train_clusters == bad_clust] = opp0
            else:
                self.train_clusters[self.train_clusters == bad_clust] = -bad_clust

            try:
                del self.template_dict[bad_clust]
            except (KeyError, AttributeError):
                pass

        elif type(bad_clust) is list:
            for cl in bad_clust:
                if cl == 0:
                    self.train_clusters[self.train_clusters == cl] = opp0
                else:
                    self.train_clusters[self.train_clusters == cl] = -cl
                try:
                    del self.template_dict[cl]
                except (KeyError, AttributeError):
                    pass

        elif bad_clust is None:
            pass
        else:
            assert False, "I need a cluster number or list of clusters"

        if good_clust is not None:
            for cl in np.unique(self.train_clusters):
                if cl not in good_clust:
                    if cl == 0:
                        self.train_clusters[self.train_clusters == cl] = opp0
                    else:
                        self.train_clusters[self.train_clusters == cl] = -cl
                    try:
                        del self.template_dict[cl]
                    except KeyError:
                        pass

    def renumberClusters(self):
        good_clusters = getGoodClusters(np.unique(self.train_clusters))

        good_clusters_idx = np.where(np.in1d(self.train_clusters, good_clusters))[0]

        self.train_clusters = self.train_clusters[good_clusters_idx]

        good_evts_idxs = np.where(self.good_evts)[0]

        self.good_evts = np.zeros(len(self.good_evts), dtype=bool)
        self.good_evts[good_evts_idxs[good_clusters_idx]] = True

        j = 0
        for i in np.sort(good_clusters):
            self.train_clusters[self.train_clusters == i] = j
            j += 1
        self.generateClustersTemplate()

    def mergeClusters(self, c_obj, c_merge):
        if type(c_merge) is list:
            for cl in c_merge:
                self.train_clusters[self.train_clusters == cl] = c_obj
        elif type(c_merge) is int:
            self.train_clusters[self.train_clusters == c_merge] = c_obj
        else:
            raise TypeError("clusters to merge should be an int if single or list of ints")

        self.generateClustersTemplate()

    def generateClustersTemplate(self):
        self.template_dict = generateClustersTemplate(self.train_clusters, self.evts, self.good_evts)

        for cl in np.unique(self.train_clusters):
            if cl >= 0 and (cl != nan) and (cl != opp0):
                median = np.apply_along_axis(np.median, 0, self.evts[self.good_evts, :][self.train_clusters == cl, :])
                mad = np.apply_along_axis(swp.mad, 0, self.evts[self.good_evts, :][self.train_clusters == cl, :])
                self.template_dict.update({cl: {'median': median, 'mad': mad}})
        self.assignChannelsToClusters()

    def hideBadEvents(self, max_pct_to_remove=.1, pct_from_norm=.1, clust_list=None):
        self.generateClustersTemplate()

        if clust_list is None:
            clust_list = np.unique(self.train_clusters)

        for cl in clust_list:
            if (cl >= 0) and (cl != nan) and (cl != opp0):
                clust_evts = self.evts[self.good_evts, :][self.train_clusters == cl, :]

                clust_evts -= self.template_dict[cl]['median']
                dist_to_median = np.sqrt((clust_evts * clust_evts).sum(axis=1))

                clust_norm = np.sqrt(np.dot(self.template_dict[cl]['mad'], self.template_dict[cl]['mad']))

                if (dist_to_median > pct_from_norm * clust_norm).sum() < (max_pct_to_remove * clust_evts.shape[0]):
                    self.train_clusters[self.train_clusters == cl][dist_to_median > clust_norm] = -cl
                else:
                    n_remove = int(max_pct_to_remove * clust_evts.shape[0])

                    ind1 = np.where(self.train_clusters == cl)[0]
                    inds = ind1[np.argpartition(dist_to_median, -n_remove)[-n_remove:]]
                    if cl != 0:
                        self.train_clusters[inds] = -cl
                    else:
                        self.train_clusters[inds] = opp0

    def realignMultiChannelEvents(self, clust_list, ch=0):
        for cluster in clust_list:
            center = int(self.evt_length/2 - 1)
            arg_max = np.argmax(self.template_dict[cluster]['median'][self.evt_length * ch: self.evt_length * (ch + 1)])

            shift = arg_max - center

            self.peaks_idxs[self.train_clusters == cluster] += shift
        try:
            self.makeEvents(before=self.before, after=self.after)
        except:
            self.peaks_idxs[self.train_clusters == cluster] -= shift
            raise
        self.generateClustersTemplate()




    def restoreClusters(self, clust_list=None):
        if clust_list is None:
            self.train_clusters = deepcopy(self.original_clusters)
            self.good_evts = deepcopy(self.original_good_evts)
        else:
            for i in clust_list:
                if i < 0:
                    self.train_clusters[self.train_clusters == i] = -i
                elif i == opp0:
                    self.train_clusters[self.train_clusters == i] = 0

    def restorePostClusteringState(self):

        self.train_clusters = deepcopy(self.original_clusters)
        self.good_evts = deepcopy(self.original_good_evts)

    def getPcaBase(self, vect_num=8, plot_vectors=False):

        evts_in_use = self.evts

        varcovmat = np.cov(evts_in_use[self.good_evts, :].T)
        self.unitary, self.singular, v = svd(varcovmat)
        evt_idx = range(self.evt_length * self.ch_no)

        evtsE_good_mean = np.mean(evts_in_use[self.good_evts, :], 0)

        if plot_vectors:
            j = 0
            plt.figure()
            for i in range(vect_num):
                j += 1
                if j == 5:
                    j = 1
                    plt.figure()

                plt.subplot(2, 2, j)
                plt.plot(evt_idx, evtsE_good_mean, 'black', evt_idx,
                         evtsE_good_mean + 10 * self.unitary[:, i],
                         'red', evt_idx, evtsE_good_mean - 10 * self.unitary[:, i], 'blue')
                plt.title('PC' + str(i) + ': ' + str(round(self.singular[i] / sum(self.singular) * 100)) + '%')
        self.state = 'Preclustering'

    def getVectorWeights(self, dim=10):

        noise_in_use = self.noise_evts

        noiseVar = sum(np.diag(np.cov(noise_in_use.T)))
        evtsVar = sum(self.singular)

        [print(i, sum(self.singular[:i]) + noiseVar - evtsVar) for i in range(dim)]

    def assignChannelsToClusters(self, ch_dict=None):
        """ To be applied after hiding bad clusters

        :param ch_dict: pass dict of the type {cluster number: [channels where it appears]}
        :type ch_dict:
        :return:
        :rtype:
        """
        if ch_dict is not None:
            for key, item in ch_dict.items():
                self.template_dict[key]['channels'] = item
        else:
            self.automaticAssign()

    def automaticAssign(self):
        ranges = np.arange(0, self.evt_length * self.ch_no + 1, self.evt_length)
        for cl in np.unique(self.train_clusters):
            if cl >= 0 and (cl != nan) and (cl != opp0):
                appears_in = []
                for j in range(len(ranges) - 1):
                    med_mad_difference = np.abs(self.template_dict[cl]['median'][ranges[j]:ranges[j + 1]]) - \
                                         self.template_dict[cl]['mad'][ranges[j]:ranges[j + 1]]

                    if sum(med_mad_difference > 0) > 0:
                        appears_in.append(j)

                if len(appears_in) == self.ch_no:
                    self.template_dict[cl]['channels'] = [-1]
                else:
                    self.template_dict[cl]['channels'] = appears_in

    def makeCenterDict(self, before=49, after=50):
        self.centers = {}
        dataD, dataDD = swp.mk_data_derivatives(self.traces)
        for i in np.unique(self.train_clusters):
            if (i < 0) or (i == nan) or (i == opp0):
                continue

            self.centers.update({i:
                                     swp.mk_center_dictionary(self.peaks_idxs[self.good_evts][self.train_clusters == i],
                                                              self.traces, before=before, after=after, dataD=dataD,
                                                              dataDD=dataDD)
                                 }
                                )
        beep()

    def plotCenterDict(self, clust_list=None):
        if clust_list is None:
            clust_list = list(self.centers)
        colors = PyLeech.Utils.burstUtils.setGoodColors(clust_list)

        fig, ax = plt.subplots()
        for cl in clust_list:
            ax.plot(self.centers[cl]['center'], color=colors[cl], label=str(cl))
        fig.legend()

    def generatePrediction(self, before, after, store_prediction=True, peak_idxs=None):
        self.before = before
        self.after = after
        if peak_idxs is not None:
            round0 = [swp.classify_and_align_evt(peak_idxs[i], self.traces, self.centers, self.before, self.after)
                      for i in range(len(peak_idxs))]
        else:
            round0 = [swp.classify_and_align_evt(self.peaks_idxs[i], self.traces, self.centers, self.before, self.after)
                      for i in range(len(self.peaks_idxs))]

        if store_prediction:
            self.peel = [self.traces]
            self.pred = []
            self.rounds = []
            self.secondary_spikes = []
            self.pred.append(
                swp.predict_data(round0, self.centers, nb_channels=self.ch_no, data_length=len(self.traces[0, :])))
            self.rounds.append(round0)
            self.peel.append(self.peel[-1] - self.pred[-1])
            return self.pred[-1]

        return swp.predict_data(round0, self.centers, nb_channels=self.ch_no, data_length=len(self.traces[0, :]))

    def secondaryPeeling(self, channels=None, vect_len=5, threshold=5, min_dist=50, store_mid_steps=True,
                         skip_units=None):
        centers = deepcopy(self.centers)
        if skip_units is not None:
            for unit in skip_units:
                del centers[unit]

        if channels is None:
            channels = range(self.ch_no)
        elif type(channels) is int:
            channels = [channels]
        for i in channels:
            peak_detection_data = np.apply_along_axis(lambda x:
                                                      fftconvolve(x, np.array([1] * vect_len) / float(vect_len),
                                                                  'same'),
                                                      1, np.array(self.peel[-1]))

            # peak_detection_data = (peak_detection_data.transpose() / np.apply_along_axis(swp.mad, 1,
            #                                                                              peak_detection_data)).transpose()

            peak_detection_data[peak_detection_data < threshold] = 0
            if not store_mid_steps:
                self.secondary_spikes = []
                self.pred = []
                self.peel = [self.peel[-1]]

            self.secondary_spikes.append(swp.peak(peak_detection_data[i, :], minimal_dist=min_dist))

            self.rounds.append([swp.classify_and_align_evt(self.secondary_spikes[-1][i], self.peel[-1], centers,
                                                           self.before, self.after)
                                for i in range(len(self.secondary_spikes[-1]))])
            self.pred.append(swp.predict_data(self.rounds[-1], centers, nb_channels=self.ch_no,
                                              data_length=len(self.traces[0, :])))
            self.peel.append(self.peel[-1] - self.pred[-1])

    def getSortedTemplateDifference(self, clust_list=None):

        if clust_list is None:
            clusts = np.unique(self.train_clusters)
            clust_list = clusts[(clusts >= 0) & (clusts != opp0) & (clusts != nan)]


        template_dict_subset = getTemplateDictSubset(clust_list, self.template_dict)

        self.template_dict_comparer = tdictUtils.TemplateComparer(template_dict_subset)
        dist, sim_templates = self.template_dict_comparer.getSortedTemplateDifference(clust_list=clust_list)
        return np.array(dist), np.array(sim_templates)


    def getClosestsTemplates(self, pair_no=5):

        dist, template_pairs = self.getSortedTemplateDifference(list(self.template_dict))
        return template_pairs[:pair_no]



    def getSimilarTemplates(self, cluster_no):

        try:

            return self.template_dict_comparer.getSimilarTemplates(cluster_no)
        except AttributeError:
            self.getSortedTemplateDifference()
            return self.template_dict_comparer.getSimilarTemplates(cluster_no)

    def mergeRoundsResults(self):
        round_list = [item for sublist in self.rounds for item in sublist]

        self.final_spike_dict = {n: np.unique([x[1] for x in round_list
                                               if x[0] == n]) for n in list(self.centers)}

        self.state = 'Processed'

    def mergeFinalRoundClusters(self, clust_list):
        clus_obj = clust_list[0]
        clus_rm = clust_list[1:]
        for i in clus_rm:
            try:
                self.final_spike_dict[clus_obj] = np.concatenate(
                    (self.final_spike_dict[clus_obj], self.final_spike_dict[i]))
                del self.final_spike_dict[i]
            except KeyError:
                print("%i didn't exist, it might have been merged already" % i)
        self.final_spike_dict[clus_obj] = np.sort(self.final_spike_dict[clus_obj])

    def getSpikecount(self, round_no=0):
        for key in list(self.centers.keys()):
            print(key, len([x[1] for x in self.rounds[round_no] if x[0] == key]))
        print(key, len([x[1] for x in self.rounds[round_no] if x[0] == nan]))

    def setGoodColors(self):
        good_clusters = getGoodClusters(np.unique(self.train_clusters))
        self.cluster_color = PyLeech.Utils.burstUtils.setGoodColors(good_clusters)

    def setRoundColors(self):
        round_keys = list(self.final_spike_dict.keys())
        self.cluster_color = PyLeech.Utils.burstUtils.setGoodColors(round_keys)

    def saveResults(self, filename=None, folder=None):
        if filename is None:
            filename = self.filename
        if folder is None:
            try:
                folder = self.folder
            except:
                pass
        filename = generatePklFilename(filename, folder)
        results = {}

        for attr in SpSorter.attrs_list:
            try:
                results[attr] = getattr(self, attr)
            except AttributeError:
                if attr == 'train_clusters':
                    print(" 'train_clusters' attribute wasn't there, saving 'clusters' as 'train_clusters")
                    results[attr] = self.clusters
                else:
                    print('%s wasn´t defined, it won´t be saved' % attr)
                pass
        print('Saving in %s.pkl' % filename)
        with open(filename + '.pkl', 'wb') as pfile:

            pickle.dump(results, pfile)

    def loadResults(self, filename):
        filename = os.path.splitext(filename)[0]
        with open(filename + '.pkl', 'rb') as pfile:
            results = pickle.load(pfile)
        print('Loaded:')
        for key, item in results.items():
            print(key)
            setattr(self, key, item)

    ####################################################################### Plot methods ##############################

    def viewPcaClusters(self, ggobi=False, go_pandas=False, clust_dim=8, ggobi_clusters=False, clust_list=None,
                        good_clusters=False, marker_size=20, use_hue=False):

        if use_hue:
            clusts = np.unique(self.train_clusters)
        else:
            clusts=None


        if good_clusters:
            clust_list = clusts[np.where(
                (clusts > 0) & (clusts != nan) & (
                        clusts != opp0))
            ]
        elif not clust_list:
            clust_list = clusts
        # idxs = list(range(clust_dim))
        if clust_list is not None and len(clust_list) > 0:
            ggobi_clusters = True
            mask = np.array([False] * len(self.train_clusters))
            for cl in clust_list: mask = mask | (self.train_clusters == cl)
        else:
            mask = np.array([True] * sum(self.good_evts))

        evts = self.evts

        if ggobi:

            if not ggobi_clusters:
                fn = 'csvs/evtsE.csv'
                f = open(fn, 'w+')
                w = csv.writer(f)
                w.writerows(np.dot(self.evts[self.good_evts, :], self.unitary[:, 0:clust_dim]))
            else:
                fn = 'csvs/evtsEclust.csv'
                f = open(fn, 'w')
                w = csv.writer(f)
                w.writerows(np.concatenate((np.dot(evts[self.good_evts, :][mask, :], self.unitary[:, 0:clust_dim]),
                                            np.array([self.train_clusters]).T),
                                           axis=1))

            f.close()
            proc = Popen(['ggobi.exe', fn])

        if go_pandas:
            plot_kws = {"s": marker_size}
            evts_matrix = np.dot(evts[self.good_evts, :][mask, :], self.unitary[:, :clust_dim])
            columns = list(range(clust_dim))
            if use_hue:
                try:
                    # print(evts_matrix.shape, np.array([self.train_clusters[np.in1d(self.train_clusters, clust_list)]]).T.shape)
                    evts_matrix = np.hstack((evts_matrix, np.array([
                        self.train_clusters[np.in1d(self.train_clusters, clust_list)]]).T
                                             )
                                            )
                    columns += ['cluster']
                    hue = 'cluster'
                except AttributeError:
                    warnings.warn("Warning: no cluster found for hue")
                    hue = None

            else:
                hue = None

            df = pd.DataFrame(evts_matrix, columns=columns)
            if clust_list is not None and 'cluster' in df.columns:
                df = df[df['cluster'].isin(clust_list)]
            if hue is not None:

                sns.pairplot(df, hue=hue, vars=df.columns[:-1], plot_kws=plot_kws)
            else:
                sns.pairplot(df, hue=hue, diag_kind='kde', plot_kws=plot_kws)
            # scatter_matrix(df, alpha=0.2, s=4, c='k', figsize=(6, 6),
            #                diagonal='kde', marker=".")

    def plotClusters(self, clusts=None, y_lim=None, good_ones=True, add_unclustedred_evts=False, superposed=False):

        if add_unclustedred_evts:
            self.getKMeansFullPrediction()

        if type(clusts) is int:
            plt.figure()
            plt.title('cluster %s' % str(clusts))
            if add_unclustedred_evts:
                swp.plot_events(self.evts[self.full_cluster == clusts, :], n_channels=self.ch_no)
            else:
                swp.plot_events(self.evts[self.good_evts, :][self.train_clusters == clusts, :], n_channels=self.ch_no)

            for i in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                plt.axvline(x=i, color='black', lw=1)
            return

        if superposed:
            plt.figure()
        for i in np.unique(self.train_clusters):
            if clusts is not None and i in clusts:
                if not superposed:
                    plt.figure()
                    plt.title('cluster %s' % str(i))

                if add_unclustedred_evts:
                    swp.plot_events(self.evts[self.full_cluster == i, :], n_channels=self.ch_no)
                else:
                    swp.plot_events(self.evts[self.good_evts, :][self.train_clusters == i, :], n_channels=self.ch_no)

                for j in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                    plt.axvline(x=j, color='black', lw=1)

                if y_lim is not None:
                    plt.ylim(y_lim)

            elif clusts is None:
                if ((i >= 0) and (i != nan) and (i != opp0)) or (not good_ones):
                    if not superposed:
                        plt.figure()
                        plt.title('cluster %s' % str(i))

                    if add_unclustedred_evts:
                        swp.plot_events(self.evts[self.full_cluster == i, :], n_channels=self.ch_no)
                    else:
                        swp.plot_events(self.evts[self.good_evts, :][self.train_clusters == i, :],
                                        n_channels=self.ch_no)

                    for j in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                        plt.axvline(x=j, color='black', lw=1)
                    if y_lim is not None:
                        plt.ylim(y_lim)

    def plotClusteredEvents(self, clust_list=None, legend=True, lw=1):

        if clust_list is None:
            clust_list = []
        if not clust_list:
            colors = PyLeech.Utils.burstUtils.setGoodColors(np.unique(self.train_clusters))
        else:
            colors = PyLeech.Utils.burstUtils.setGoodColors(clust_list)


        iter_clusts = np.unique(self.train_clusters)

        plt.figure()
        swp.plot_data_list(self.traces, self.time,
                           linewidth=lw)

        for i in iter_clusts:
            if (i < 0) or (i == nan) or (i == opp0):
                continue
            if (len(clust_list) == 0) or (i in clust_list):
                try:
                    channels = self.template_dict[i]['channels']
                except KeyError or AttributeError:
                    channels = None

                swp.plot_detection(self.traces, self.time,
                                   self.peaks_idxs[self.good_evts][self.train_clusters == i],
                                   channels=channels,
                                   peak_color=colors[i],
                                   label=str(i))

        if legend:
            plt.legend(loc='upper right')

    # noinspection PyUnboundLocalVariable
    def plotTemplates(self, clust_list=None, new_figure=True, skip_list=[]):
        if new_figure:
            fig = plt.figure()


        self.generateClustersTemplate()

        if clust_list is None:

            clust_list = list(self.template_dict.keys())

            self.setGoodColors()
            colors = self.cluster_color
        elif type(clust_list) is int:
            clust_list = [clust_list]
            colors = PyLeech.Utils.burstUtils.setGoodColors(clust_list)
        elif type(clust_list) in [list, np.ndarray, range]:
            colors = PyLeech.Utils.burstUtils.setGoodColors(clust_list)
        else:
            assert False, "clust_list must be either a list or an int"



        for key, item in self.template_dict.items():
            if key in clust_list and key not in skip_list:
                plt.plot(item['median'], color=colors[key], label=key, lw=2)

        for i in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
            plt.axvline(x=i, color='black', lw=1)
        plt.legend()

        return fig

    # noinspection PyTypeChecker,PyUnboundLocalVariable
    def plotCompleteDetection(self,
                              rounds='All',
                              step=1,
                              interval=None,
                              legend=False,
                              lw=1,
                              clust_list=None,
                              hide_clust_list=None,
                              intracel_signal=None):
        self.setRoundColors()

        # if interval is None:
        #     interval = [0, len(self.time)]
        #
        # else:
        #     interval[0] = int(interval[0] * len(self.time))
        #     interval[1] = int(interval[1] * len(self.time))

        if type(rounds) is str:
            spike_dict = self.final_spike_dict

        else:
            data = []
            for i in range(rounds):
                data.extend(self.rounds[i])

            spike_dict = {n: np.sort([x[1] for x in data
                                      if x[0] == n]) for n in list(self.centers)}

        if (clust_list is None) and (hide_clust_list is None):
            colors = self.cluster_color
        elif clust_list is not None:
            clust_list.sort()
            colors = PyLeech.Utils.burstUtils.setGoodColors(clust_list)
        elif hide_clust_list is not None:
            to_plot_list = [cluster for cluster in list(spike_dict) if cluster not in hide_clust_list]
            to_plot_list.sort()
            colors = PyLeech.Utils.burstUtils.setGoodColors(to_plot_list)

        fig, ax_list = PyLeech.Utils.burstUtils.plotCompleteDetection(self.traces,
                                                                      self.time,
                                                                      spike_dict=spike_dict,
                                                                      template_dict=self.template_dict,
                                                                      cluster_colors=colors,
                                                                      legend=legend,
                                                                      lw=lw,
                                                                      interval=interval,
                                                                      clust_list=clust_list,
                                                                      hide_clust_list=hide_clust_list,
                                                                      step=step,
                                                                      intracel_signal=intracel_signal)

        return fig, ax_list

    def plotTraceAndStd(self, channel=0, legend=False, interval=None):
        """ To be used after mad normalization

        :param int channel: channel to plot
        :param bool legend: optional for showing the legend
        :param [float, float] interval: interval for plotting as a fraction of the signal
        """
        assert self.normed, "data has not been normalized yet"

        if interval is None:
            first = 0
            last = len(self.time) - 1
        else:
            first = int(len(self.time) * interval[0])
            last = int((len(self.time) - 1) * interval[1])

        plt.figure()
        plt.plot(self.time, self.traces[channel], color="black")
        plt.axhline(y=1, color="red", label='mad')
        plt.axhline(y=-1, color="red")
        plt.axhline(y=np.std(self.traces[channel]), color="blue", linestyle="dashed", label='std')
        plt.axhline(y=-np.std(self.traces[channel]), color="blue", linestyle="dashed")
        plt.xlabel('Time (s)')
        plt.xlim([self.time[first], self.time[last]])
        if legend:
            plt.legend()

    def plotDataList(self, linewidth=0.5, color='black'):
        """Plots data when individual recording channels make up elements
        of a list.

        Parameters
        ----------
        linewidth: the width of the lines drawing the curves.
        color: the color of the curves.

        Returns
        -------
        Nothing is returned, the function is used for its side effect: a
        plot is generated.
        """
        nb_chan = len(self.traces)
        data_min = [np.min(x) for x in self.traces]
        data_max = [np.max(x) for x in self.traces]
        display_offset = list(np.cumsum(np.array([0] +
                                                 [data_max[i] -
                                                  data_min[i - 1]
                                                  for i in
                                                  range(1, nb_chan)])))
        plt.figure()
        for i in range(nb_chan):
            plt.plot(self.time, self.traces[i] - display_offset[i],
                     linewidth=linewidth, color=color)
        plt.yticks([])
        plt.xlabel("Time (s)")

    def plotDataListAndDetection(self, good_evts=False, linewidth=0.2):

        if good_evts:
            peaks = self.peaks_idxs[self.good_evts]
        else:
            peaks = self.peaks_idxs

        plt.figure()
        swp.plot_data_list_and_detection(self.traces, self.time, peaks, linewidth=linewidth)

    def plotEventsMedians(self):
        plt.figure()
        full_len = self.evt_length * self.ch_no
        plt.plot(self.evts_median, color='red', lw=2)
        plt.axhline(y=0, color='black')
        plt.axhline(y=1, color='black')
        for i in np.arange(self.evt_interval[0], full_len, self.evt_length):
            plt.axvline(x=i, color='black', lw=2)

        for i in np.arange(0, full_len, self.evt_length):
            plt.axvline(x=i, color='grey')

        plt.plot(self.evts_median, color='red', lw=2)
        plt.plot(self.evts_mad, color='blue', lw=2)
        plt.xlim([0, full_len])

    def plotEvents(self, plot_noise=False, evts_no=2000):
        plt.figure()
        if plot_noise:
            swp.plot_events(self.noise_evts, evts_no, n_channels=self.ch_no)
        else:
            swp.plot_events(self.evts, evts_no, n_channels=self.ch_no)

    # noinspection PyTypeChecker,PyTypeChecker
    def visualizeCleanEvents(self, thresholds=None):
        if thresholds is None:
            thresholds = [1, 10, 20]

        full_len = self.evt_length * self.ch_no
        for trs in thresholds:
            good_evts = good_evts_fct(self.evts, trs)
            print('%f:\t%i good out of %i' % (trs, len(self.evts[good_evts, :]), len(self.evts)))

            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
            fig.suptitle('Threshold = %f' % trs)

            swp.plot_events(self.evts[good_evts, :], n_channels=self.ch_no, pltobj=ax[0])
            ax[0].set_title('good events')
            for i in np.arange(self.evt_interval[0], full_len, self.evt_length):
                ax[0].axvline(x=i, color='black', lw=1)

            for i in np.arange(0, full_len, self.evt_length):
                ax[0].axvline(x=i, color='grey')

            swp.plot_events(self.evts[~good_evts, :], n_channels=self.ch_no,
                            show_median=False,
                            show_mad=False, pltobj=ax[1])

            for i in np.arange(self.evt_interval[0], full_len, self.evt_length):
                ax[1].axvline(x=i, color='black', lw=1)

            for i in np.arange(0, full_len, self.evt_length):
                ax[1].axvline(x=i, color='grey')

            ax[1].set_title('bad events')

    # noinspection PyShadowingNames
    def plotPeeling(self, peel_step=0, time=None, to_peel_data=None, pred=None):
        if time is None:
            time = self.time
        if to_peel_data is None:
            to_peel_data = self.peel[peel_step]
        if pred is None:
            pred = self.pred[peel_step]

        PyLeech.Utils.burstUtils.plotDataPredictionAndResult(time, to_peel_data, pred)