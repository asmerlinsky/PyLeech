import os

import PyLeech.Utils.SpSorter
import PyLeech.Utils.spsortUtils as spsortUtils
from copy import deepcopy
from PyLeech.Utils.constants import *
import _pickle as pickle
import numpy as np
from scipy.signal import fftconvolve
import PyLeech.Utils.sorting_with_python as swp
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans


class KMeansSorter(PyLeech.Utils.SpSorter.SpSorter):
    """KMeans sorter class

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

    attrs_list = deepcopy(PyLeech.Utils.SpSorter.SpSorter.attrs_list)
    attrs_list.append('prediction')
    attrs_list.append('train_evts')

    # def __init__(self, filename, data=None, time_vect=None, fs=None, verbose=False):
    #
    #     if data is None or time_vect is None or fs is None:
    #         print('Loading data from pickle')
    #         self.loadResults(filename=filename)
    #         self.filename = filename
    #         try:
    #             self.assignChannelsToClusters()
    #         except:
    #             pass
    #
    #     else:
    #         assert data is not None, "No trace data was given"
    #         assert time_vect is not None, "No time vector was given"
    #         assert fs is not None, "No sample frequency was given"
    #
    #         self.filename = filename
    #         self.traces = data
    #         self.ch_no = len(data)
    #         self.sample_freq = fs
    #         self.time = time_vect
    #         self.discretization = [np.min(np.diff(np.sort(np.unique(x)))) for x in self.traces]
    #
    #         self.normed = False
    #
    #         self.verbose = verbose
    #         self.pred = []
    #
    #         self.rounds = []
    #         self.secondary_spikes = []
    #         self.state = 'Unprocessed'
    #
    def takeSubSetEvents(self, pct=0.1):
        idxs = np.where(self.good_evts)[0]
        np.random.shuffle(idxs)
        idxs = idxs[int(pct * len(idxs)):]
        self.good_evts_subset = deepcopy(self.good_evts)
        self.good_evts_subset[idxs] = False
        print('Kept %i spikes' % sum(self.good_evts_subset))

    def generateTrainData(self):

        good_clusters = spsortUtils.getGoodClusters(np.unique(self.train_clusters))
        try:
            good_evts_subset_idxs = np.where(self.good_evts_subset)[0]
        except:
            print("No subset of events was found, using all of them")
            self.takeSubSetEvents(pct=1)
            good_evts_subset_idxs = np.where(self.good_evts_subset)[0]

        train_sub_idxs = np.where(np.in1d(self.train_clusters, good_clusters))[0]

        train_idxs = good_evts_subset_idxs[train_sub_idxs]
        self.train_mask = np.zeros(len(self.evts), dtype=bool)
        self.train_mask[train_idxs] = 1

        self.prediction_clusters = self.train_clusters[train_sub_idxs]
        self.cluster_map = {}
        j = 0
        for i in np.sort(np.unique(self.prediction_clusters)):
            self.prediction_clusters[self.prediction_clusters == i] = j
            self.cluster_map[i] = j
            j += 1



    def InitializeKMeansPredictor(self, n_init=1000, max_iter=400, verbose=0,
                                    n_jobs=None, save=False, dim_size=None):
        if dim_size == 0:
            print('Setting dimension size to 10')
            dim_size = 10
        self.km_dim_size = dim_size

        if save and os.path.isfile(spsortUtils.generatePklFilename(self.filename, None)):
            print('Warning, this will overwrite current pkl')
            ipt = input('enter any key if you want to exit')
            ipt = str(ipt)
            if len(ipt) > 0:
                print('exiting method')
                return


        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
        start = time.time()
        self.km = KMeans(n_clusters=len(np.unique(self.prediction_clusters)), init='k-means++', verbose=verbose,
                         n_init=n_init, max_iter=max_iter,
                         precompute_distances=True,
                         copy_x=True, n_jobs=n_jobs)

        print('Running')
        self.km.fit(np.dot(self.evts[self.train_mask, :], self.unitary[:, 0:self.km_dim_size]), self.prediction_clusters)
        fitted = time.time()

        self.state = 'Prediction by KMeans'
        print(
            "Fitting took %i seconds with these settings:"
            "\ndim_size = %i,\nevt number = %i,\nn_init = %i,\nmax_iter = %i, " % (
                int(fitted - start), self.km_dim_size, sum(self.good_evts), n_init, max_iter))

        spsortUtils.beep()

        if save:
            self.saveResults()

    def generateKMeansFirstPrediction(self):
        self.prediction = self.km.predict(np.dot(self.evts[self.good_evts, :], self.unitary[:, 0:self.km_dim_size]))

    def makeCenterDict(self, before=49, after=50):
        self.centers = {}
        dataD, dataDD = swp.mk_data_derivatives(self.traces)

        for i in np.unique(self.prediction_clusters):
            if (i < 0) or (i == nan) or (i == opp0):
                continue

            self.centers.update({i:
                                     swp.mk_center_dictionary(self.peaks_idxs[self.train_mask][self.prediction_clusters == i],
                                                              self.traces, before=before, after=after, dataD=dataD, dataDD=dataDD)
                                 }
                                )

        spsortUtils.beep()



    def alignAndPredictTrace(self, before, after, store_prediction=True):

        #
        # round0 = [swp.align_evt(self.peaks_idxs[self.good_evts][i], self.prediction[i], self.traces, self.centers,
        #             before, after)
        #           for i in range(len(self.peaks_idxs[self.good_evts]))]
        #
        before = self.before
        after = self.after
        round0 = [swp.align_evt(self.peaks_idxs[i], self.prediction[i], self.traces, self.centers,
                    before, after)
                  for i in range(len(self.peaks_idxs[self.good_evts]))]

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

    def secondaryKMeansPeeling(self, channels=None, vect_len=5, threshold=5, min_dist=50, store_mid_steps=True,
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
                                        fftconvolve(x, np.array([1] * vect_len) / float(vect_len), 'same'),
                                        1, np.array(self.peel[-1]))

            peak_detection_data = (peak_detection_data.transpose() / \
                                   np.apply_along_axis(swp.mad, 1, peak_detection_data)).transpose()

            peak_detection_data[peak_detection_data < threshold] = 0
            if not store_mid_steps:
                self.secondary_spikes = []
                self.pred = []
                self.peel = [self.peel[-1]]

            self.secondary_spikes.append(swp.peak(peak_detection_data[i, :], minimal_dist=min_dist))

            evts = swp.mk_events(self.secondary_spikes[-1], self.traces, self.before, self.after)
            prediction = self.km.predict(np.dot(evts, self.unitary[:, 0:self.km_dim_size]))

            self.rounds.append([swp.align_evt(self.secondary_spikes[-1][i], prediction[i],self.peel[-1], centers,
                                                           self.before, self.after)
                                for i in range(len(self.secondary_spikes[-1]))])
            self.pred.append(swp.predict_data(self.rounds[-1], centers, nb_channels=self.ch_no,
                                              data_length=len(self.traces[0, :])))
            self.peel.append(self.peel[-1] - self.pred[-1])

    def saveResults(self, filename=None):

        if filename is None:
            filename = spsortUtils.generatePklFilename(self.filename, None)
        else:
            filename = spsortUtils.generatePklFilename(filename, None)

        results = {}
        print(KMeansSorter.attrs_list)
        unsaved = []
        for attr in KMeansSorter.attrs_list:
            try:
                results[attr] = getattr(self, attr)
            except AttributeError:
                if attr == 'train_clusters':
                    print("'train_clusters' attribute wasn't there, saving 'clusters' as 'train_clusters")
                    results[attr] = self.clusters
                else:
                    unsaved.append(attr)
                pass
        print("these attributes weren´t there and won´t be saved: " + ", ".join(unsaved))
        print('Saving in %s.pkl' % filename)
        with open(filename + '.pkl', 'wb') as pfile:

            pickle.dump(results, pfile)

    ####################################################################### Plot methods ##############################
    def plotFirstDetectionEvents(self, clust_list=None, legend=True, lw=1):

        if clust_list is None:
            clust_list = []

        colors = spsortUtils.setGoodColors(np.unique(self.prediction))
        iter_clusts = np.unique(self.prediction)

        plt.figure()
        swp.plot_data_list(self.traces, self.time,
                           linewidth=lw)

        for i in iter_clusts:
            if (((i < 0) or (i == nan) or (i == opp0))):
                continue
            if (len(clust_list) == 0) or (i in clust_list):
                try:
                    channels = self.template_dict[i]['channels']
                except KeyError or AttributeError:
                    channels = None

                swp.plot_detection(self.traces, self.time,
                                   self.peaks_idxs[self.good_evts][self.prediction == i],
                                   channels=channels,
                                   peak_color=colors[i],
                                   label=str(i))

        if legend:
            plt.legend(loc='upper right')

    def plotFirstPredictionClusters(self, clusts=None, y_lim=None, good_ones=True):



        if type(clusts) is int:
            plt.figure()
            plt.title('cluster %s' % str(clusts))
            swp.plot_events(self.evts[self.good_evts, :][self.prediction == clusts, :], n_channels=self.ch_no)

            for i in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                plt.axvline(x=i, color='black', lw=1)
            return

        for i in np.unique(self.prediction):
            if clusts is not None and i in clusts:
                plt.figure()
                plt.title('cluster %s' % str(i))

                swp.plot_events(self.evts[self.good_evts, :][self.prediction == i, :], n_channels=self.ch_no)

                for i in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                    plt.axvline(x=i, color='black', lw=1)

                if y_lim is not None:
                    plt.ylim(y_lim)

            elif clusts is None:
                if ((i >= 0) and (i != nan) and (i != opp0)) or (not good_ones):
                    plt.figure()
                    plt.title('cluster %s' % str(i))
                    swp.plot_events(self.evts[self.good_evts, :][self.prediction == i, :],
                                    n_channels=self.ch_no)

                    for i in np.arange(self.evt_interval[0], self.evt_length * self.ch_no, self.evt_length):
                        plt.axvline(x=i, color='black', lw=1)
                    if y_lim is not None:
                        plt.ylim(y_lim)