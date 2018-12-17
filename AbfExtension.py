# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:45:12 2018

@author: Agustin Sanchez Merlinsky
"""

import neo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000


def generateTimeVector(datalen, sampling_rate):
    return np.linspace(2e-4, datalen / int(sampling_rate), datalen)


class ExtendedAxonRawIo(neo.rawio.AxonRawIO):

    def __init__(self, filename=''):
        super(ExtendedAxonRawIo, self).__init__(filename)
        super(ExtendedAxonRawIo, self).parse_header()
        self.associateChannels()

    def associateChannels(self):
        self.ch_info = {}
        index_dict = self.getChannelIndexes()

        for x in self.header['signal_channels']:
            self.ch_info[x['name']] = dict(zip(self.header['signal_channels'].dtype.names, x))
            del self.ch_info[x['name']]['name']
            self.ch_info[x['name']]['index'] = index_dict[x['name']]

    def getDataBySegment(self, segNo=0):
        return self.rescale_signal_raw_to_float(self.get_analogsignal_chunck(0, segNo))

    def concatenateSegments(self, ch_name):
        data = np.empty(0)
        ch_no = self.ch_dict[ch_name]

    def getChannelIndexes(self):
        chindex = {}
        indexes = self.channel_name_to_index(self.header['signal_channels']['name'])

        for i in range(len(self.header['signal_channels']['name'])):
            chindex[self.header['signal_channels']['name'][i]] = indexes[i]

        return chindex

    def plotEveryChannelFromSegmentNo(self, channels, segmentNo=0):
        ch_data = self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, segmentNo))
        fig, ax = plt.subplots(len(channels), 1, figsize=(16, 4 * len(channels)), sharex=True)
        time = generateTimeVector(len(ch_data[:, 0]), self.get_signal_sampling_rate())
        i = 0
        for key in channels:
            ax[i].plot(time, ch_data[:, self.ch_info[key]['index']], color='k', label=key)
            ax[i].grid()
            ax[i].legend(fontsize=15)
            i += 1

    def getSignal_single_segment(self, channel):
        ch_data = self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, 0))
        return ch_data[:, self.ch_info[channel]['index']]
