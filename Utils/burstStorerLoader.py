import os.path
from copy import deepcopy

import numpy
from matplotlib import pyplot

from PyLeech.Utils.burstUtils import generatePklFilename, saveSpikeResults, generateFilenameFromList, loadSpikeResults


class BurstStorerLoader():
    required_to_save_dict_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict']
    expected_load_keys = ['traces', 'time', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict', 'isDe3', 'crawling_segments']

    def __init__(self, filename, foldername="RegistrosDP_PP", mode='load', traces=None, time=None, spike_dict=None, spike_freq_dict=None, De3=None,
                 template_dict=None, color_dict=None, crawling_segments=None):
        self.filename = filename
        self.traces = traces
        self.time = time
        self.spike_dict = spike_dict
        self.spike_freq_dict = spike_freq_dict
        self.isDe3 = De3
        self.template_dict = template_dict
        self.color_dict = color_dict
        self.crawling_segments = crawling_segments
        self.folder = foldername
        if str.lower(mode) == 'save':
            self.saveResults()
        elif str.lower(mode) == 'load':
            self.loadResults()
            if self.isDe3 == None:
                print('I need to set attribute \'isDe3\' before continuing')

    def saveResults(self, filename=None, folder=None):
        pkl_dict = self.generatePklDict()
        if filename is None:
            filename = self.filename
        if folder is None:
            folder = self.folder


        if type(filename) is list:
            filename = generatePklFilename(filename, folder)
        else:
            filename = folder + "/" + os.path.splitext(os.path.basename(filename))[0]
        filename += '.pklspikes'

        saveSpikeResults(filename, pkl_dict)

    def generatePklDict(self):
        pkl_dict = {}
        for key in BurstStorerLoader.required_to_save_dict_keys:
            assert getattr(self, key) is not None, ("I need %s" % key)
            pkl_dict.update({key: getattr(self, key)})
        if self.isDe3 is not None:
            pkl_dict.update({'isDe3': self.isDe3})
            print("Saving De3 channel")
        return pkl_dict

    def loadResults(self):
        if type(self.filename) is list:
            self.filename = generateFilenameFromList(self.filename)

        if not os.path.dirname(self.filename):
            spike_dict = loadSpikeResults(self.folder + "/" + self.filename)
        else:
            spike_dict = loadSpikeResults(self.filename)

        for key, items in spike_dict.items():
            assert key in BurstStorerLoader.expected_load_keys, ('Unexpected key %s' % key)
            setattr(self, key, items)



    def plotTemplates(self, signal_inversion=[1,1], clust_list=None, fig_ax=None):

        assert self.traces.shape[0] == 2, 'This method is implemented for only 2 channels'

        if fig_ax is None:
            fig, ax = plt.subplots(1,1)
            fig_ax = [fig, ax]

        if clust_list is None:
            clust_list = list(self.template_dict.keys())
        colors = self.color_dict

        ch1_max = 0
        ch2_max = 0
        for key in list(self.spike_freq_dict):
            tp_median = self.template_dict[key]['median']
            half = int(len(tp_median)/2)
            if np.abs(tp_median[:half]).max()>ch1_max:
                ch1_max = np.abs(tp_median[:half]).max()

            if np.abs(tp_median[half:]).max()>ch2_max:
                ch2_max = np.abs(tp_median[half:]).max()


        for key in list(self.spike_freq_dict):
            if key in clust_list:
                tp_median = self.template_dict[key]['median']
                tp_median[:half] = signal_inversion[0] * tp_median[:half]
                tp_median[half:] = signal_inversion[1] * tp_median[half:]

                # if len(spsig.find_peaks(tp_median, height=1, distance=30)[0])>1:

                tp_median[:half] = tp_median[:half]/ch1_max
                tp_median[half:] = tp_median[half:] / ch2_max


                fig_ax[1].plot(tp_median, color=colors[key], label=key, lw=2)

        fig_ax[0].legend()

        return fig_ax