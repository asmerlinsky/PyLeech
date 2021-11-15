import os.path
from copy import deepcopy

import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from PyLeech.Utils.burstUtils import generatePklFilename, saveSpikeResults, generateFilenameFromList, loadSpikeResults
import json


def jsonKeys2int(x):
    if isinstance(x, dict):
        try:
            return {int(k): v for k, v in x.items()}
        except ValueError:
            {k: v for k, v in x.items()}
    return x

def loadDataDict(filename='RegistrosDP_PP/CrawlingDataDict.json'):
    f = open(filename, "r")
    DataDict = json.load(f, object_hook=jsonKeys2int)
    f.close()
    return DataDict


class UnitInfo():
    required_to_save_dict_keys = ['time_length', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict',
                                  'nerve_channels', 'notes']
    expected_load_keys = ['time_length', 'spike_dict', 'spike_freq_dict', 'template_dict', 'color_dict', 'isDe3',
                          'nerve_channels', 'notes']
    expected_channels = ["DP", "PP", "MA", "AA"]

    def __init__(self, filename, foldername="RegistrosDP_PP", mode='load', time_length=None, spike_dict=None,
                 spike_freq_dict=None, De3=None,
                 template_dict=None, color_dict=None, crawling_segments=None, nerve_channels=None, notes=""):
        self.filename = filename
        assert not isinstance(time_length, Iterable);
        "time length is a sequence, this was changed to the length of the file"

        self.time_length = time_length
        self.spike_dict = spike_dict
        self.spike_freq_dict = spike_freq_dict
        self.isDe3 = De3
        self.template_dict = template_dict
        self.color_dict = color_dict
        self.crawling_segments = crawling_segments
        self.folder = foldername
        self.notes = notes
        self.nerve_channels = nerve_channels


        if str.lower(mode) == 'save':

            try:
                self.sortUnitsByNerveAndSize()
            except Exception as e:
                warnings.warn("Unable to sort units, raised error \n%s" % e)

            for ch in nerve_channels:
                assert ch in UnitInfo.expected_channels;
                "Wrong channel name in %s" % ch
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
        for key in UnitInfo.required_to_save_dict_keys:
            assert getattr(self, key) is not None, ("I need %s" % key)
            pkl_dict.update({key: getattr(self, key)})
        if self.isDe3 is not None:
            pkl_dict.update({'isDe3': self.isDe3})
            print("Saving De3 channel")
        return pkl_dict

    def loadResults(self):
        warnings.simplefilter('always', UserWarning)
        if type(self.filename) is list:
            self.filename = generateFilenameFromList(self.filename)

        if not os.path.dirname(self.filename):
            spike_dict = loadSpikeResults(self.folder + "/" + self.filename)
        else:
            spike_dict = loadSpikeResults(self.filename)

        if ('time' in spike_dict) and isinstance(spike_dict['time'], Iterable):
            spike_dict['time_length'] = spike_dict['time'][-1]
            del spike_dict['time']
        else:
            assert 'time_length' in spike_dict;
            "When loading, I couldn't find 'time' array attribute or 'time_length' attribute"

        for key, items in spike_dict.items():
            if key not in UnitInfo.expected_load_keys:
                warnings.warn("\nUnexpected key: '%s'\nWill be loaded anyways" % key)
            setattr(self, key, items)

        for attr in UnitInfo.expected_load_keys:
            if attr not in list(spike_dict):
                warnings.warn("\ndidn't load '%s'" % attr)

        try:
            if len(self.notes)>0:
                print(self.notes)
        except AttributeError:
            pass

        warnings.simplefilter('default', UserWarning)


    def sortUnitsByNerveAndSize(self,data_dict_name=None, folder='RegistrosDP_PP'):
        if data_dict_name is not None:
            db = loadDataDict(data_dict_name)
        else:
            db = loadDataDict()


        print(self.nerve_channels)
        print(len(self.nerve_channels))
        template_length = int(self.template_dict[list(self.template_dict)[0]]['median'].shape[0] / len(self.nerve_channels))

        nerve_order = ['DP', 'PP', 'MA', 'AA']
        self.nerve_unit_dict = {}
        for nerve in nerve_order:
            try:
                idx = self.nerve_channels.index(nerve)
                unsorted_units = []
                for unit in list(self.spike_freq_dict):
                    data = self.template_dict[unit]
                    if (idx in data['channels']) or -1 in data['channels']:
                        max_in_channel = data['median'][idx*template_length:(idx+1)*template_length].max()
                        unsorted_units.append((unit, max_in_channel))

                unsorted_units = np.array(unsorted_units)
                sorted_units = unsorted_units[unsorted_units[:,1].argsort()[::-1]]
                self.nerve_unit_dict[nerve] = sorted_units
            except ValueError:
                pass
        return self.nerve_unit_dict

    def plotTemplates(self, signal_inversion=[1, 1], clust_list=None, fig_ax=None):

        # assert self.traces.shape[0] == 2, 'This method is implemented for only 2 channels'

        if fig_ax is None:
            fig, ax = plt.subplots(1, 1)
            fig_ax = [fig, ax]

        if clust_list is None:
            clust_list = list(self.template_dict.keys())
        colors = self.color_dict

        ch1_max = 0
        ch2_max = 0
        for key in list(self.spike_freq_dict):
            tp_median = self.template_dict[key]['median']
            half = int(len(tp_median) / 2)
            if np.abs(tp_median[:half]).max() > ch1_max:
                ch1_max = np.abs(tp_median[:half]).max()

            if np.abs(tp_median[half:]).max() > ch2_max:
                ch2_max = np.abs(tp_median[half:]).max()

        for key in list(self.spike_freq_dict):
            if key in clust_list:
                tp_median = self.template_dict[key]['median']
                tp_median[:half] = signal_inversion[0] * tp_median[:half]
                tp_median[half:] = signal_inversion[1] * tp_median[half:]

                # if len(spsig.find_peaks(tp_median, height=1, distance=30)[0])>1:

                tp_median[:half] = tp_median[:half] / ch1_max
                tp_median[half:] = tp_median[half:] / ch2_max

                fig_ax[1].plot(tp_median, color=colors[key], label=key, lw=2)

        fig_ax[0].legend()

        return fig_ax



if __name__ == '__main__':
    cdd = loadDataDict()

    run_list = []

    correr_todo = False
    correr_crawling = True

    for fn in list(cdd):
        if cdd[fn]['skipped'] or (cdd[fn]['DE3'] == -1) or (cdd[fn]["DE3"] is None):
            pass
        else:
            run_list.append(fn)

    for fn in run_list:
        ch_dict = {ch: info for ch, info in cdd[fn]['channels'].items() if len(info) >= 2}

        cdd_de3 = cdd[fn]['DE3']
        selected_neurons = [neuron for neuron, neuron_dict in cdd[fn]['neurons'].items() if
                            neuron_dict["neuron_is_good"]]

        fn = fn.replace("/", '/')

        basename = os.path.splitext(os.path.basename(fn))[0]

        burst_object = UnitInfo(fn, mode='load')
        burst_object.sortUnitsByNerveAndSize()
        burst_object.saveResults()