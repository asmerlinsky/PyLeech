# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:57:03 2018

@author: Agustin Sanchez Merlinsky
"""
import neo
import numpy as np


def loadAbfFile(fn):
    """ Loads and parse Abf file into raw format at once
    """
    block = neo.rawio.AxonRawIO(filename=fn)
    block.parse_header()
    return block


def concatenateSegments(ch_name, block):
    """
    appends every segment data into a single array
    """

    data = np.empty(0)
    ch_no = getChannelNoByName(ch_name, block)

    if isinstance(block, neo.rawio.axonrawio.AxonRawIO):
        for i in range(block.segment_count(0)):
            # iterating over ranges
            tdata = block.rescale_signal_raw_to_float(block.get_analogsignal_chunk(0, i))[:, ch_no]
            data = np.append(data, tdata)
    return data


def getDataBySegment(obj, segNo=0):
    return obj.rescale_signal_raw_to_float(obj.get_analogsignal_chunk(0, segNo))


def associateChannels(obj):
    """ Generates a dict {channel name: channel no}
    """
    ch_dict = {}
    for chdata in obj.header['signal_channels']:
        ch_dict[chdata[0]] = chdata[1]
    return ch_dict


def getChannelNoByName(ch_name, obj):
    for chdata in obj.header['signal_channels']:
        if chdata[0] == ch_name:
            return chdata[1]


def generateTimeVector(datalen, sampling_rate):
    return np.linspace(2e-4, datalen / int(sampling_rate), datalen)
