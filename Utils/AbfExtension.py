# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:45:12 2018

@author: Agustin Sanchez Merlinsky
"""

import neo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

mpl.rcParams['agg.path.chunksize'] = 10000


def generateTimeVector(datalen, sampling_rate):
    return np.linspace(2e-4, datalen / int(sampling_rate), datalen)

def getAbFileListFromBasename(pkl_filename, fn_folder=None):
    base_pkl_filename = os.path.basename(pkl_filename)
    base_pkl_filename = os.path.splitext(base_pkl_filename)[0]

    if "_" not in base_pkl_filename:
        assert "18n05" in base_pkl_filename, "This only works for files in folder 18-11-05"
        fn_folder = "18-11-05"

        return [fn_folder + "/" + base_pkl_filename + ".abf"]

    fn_parts = base_pkl_filename.split('_')
    fn_date = "_".join(fn_parts[:3])


    fn_folder = "-".join(fn_parts[:3])

    fn_folder = fn_folder[2:]

    fn_numbers = fn_parts[3:]

    file_list = []
    for abf_no in fn_numbers:
        file_list.append(fn_folder + "/" + fn_date + "_" + abf_no + ".abf")
    return file_list


def getArraysFromAbfFiles(file_list, channels=['IN5', 'IN6', 'Vm1', 'Vm2'], print_ch_info=False):
    array_dict = {}
    if type(file_list) is str and os.path.splitext(file_list)[1] != ".abf":
        file_list = getAbFileListFromBasename(file_list)
    elif type(file_list) is str and os.path.splitext(file_list)[1] == ".abf":
        file_list = [file_list]

    for channel in channels:
        array_dict[channel] = np.array([])
    assert len(file_list)>0, 'filename list is empty'
    for filename in file_list:


        try:
            block = ExtendedAxonRawIo(filename)
        except FileNotFoundError:
            fn = os.path.splitext(os.path.basename(filename))[0]
            # print(fn)
            # print(glob.glob("crawling/*/*"))
            fn = [s for s in glob.glob("crawling/*/*") if fn in s]
            block = ExtendedAxonRawIo(fn[0])


        ch_info = block.ch_info

        if print_ch_info:
            [print(ch, item) for ch, item in ch_info.items()]
        try:
            for channel in channels:
                array_dict[channel] = np.append(array_dict[channel], np.array(block.getSignal(channel)))
        except KeyError:
            print("Deleting block")
            del block
            raise
        fs = block.get_signal_sampling_rate()

        del block

    time = generateTimeVector(len(array_dict[list(array_dict)[0]]), fs)
    return array_dict, time, fs


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
        return self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, segNo))

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

            if len(channels) == 1:

                ax.plot(time, ch_data[:, self.ch_info[key]['index']], color='k', label=key)
                ax.grid()
                ax.legend(fontsize=15)

            else:
                ax[i].plot(time, ch_data[:, self.ch_info[key]['index']], color='k', label=key)
                ax[i].grid()
                ax[i].legend(fontsize=15)
            i += 1

    def getSignal_single_segment(self, channel):
        ch_data = self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, 0))
        try:
            return ch_data[:, self.ch_info[channel]['index']]
        except KeyError:
            ch_info_list = [str(self.ch_info[key]) for key in list(self.ch_info)]
            ch_info_string = "\n".join(ch_info_list)
            print("channel %s doesn't exists, check ch_info: \n%s" % (channel, ch_info_string))
            raise
    def getSignal(self, channel):
        sg_count = self.segment_count(0)
        for i in range(sg_count):

            try:
                ch_data = np.vstack((ch_data, self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, i))))
            except UnboundLocalError:
                ch_data = self.rescale_signal_raw_to_float(self.get_analogsignal_chunk(0, i))



        try:
            return ch_data[:, self.ch_info[channel]['index']]
        except KeyError:
            ch_info_list = [str(self.ch_info[key]) for key in list(self.ch_info)]
            ch_info_string = "\n".join(ch_info_list)
            print("channel %s doesn't exists, check ch_info: \n%s" % (channel, ch_info_string))
            raise
