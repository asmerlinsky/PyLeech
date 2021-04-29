# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 19:57:03 2018

@author: Agustin Sanchez Merlinsky
"""
import os

import neo
import numpy as np


def loadAbfFile(fn):
    """
    Loads and parse Abf file into raw format at once

    Parameters
    ----------
    fn str:
        Path to file

    Returns
    -------
    Parsed abf block

    """

    block = neo.rawio.AxonRawIO(filename=fn)
    block.parse_header()
    return block


def concatenateSegments(ch_name, block):
    """
    Appends every segment data into a single array

    Parameters
    ----------
    ch_name str:
        Channel name, usually 'Vm1', 'Vm2', 'IN4', etc...
    block AxonRawIO:
        Block containing the data to be concatened

    Returns
    -------
    data ndarray:
        concatenated signal

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
    """
    Gets trace for the requested segment
    Parameters
    ----------
    obj AxonRawIO:
        Block from where the trace is extracted
    segNo int, optional:
        Required segment

    Returns
    -------
    memmap where data is stored

    """
    return obj.rescale_signal_raw_to_float(obj.get_analogsignal_chunk(0, segNo))


def associateChannels(obj):
    """
    Reads a block and generates a dict pointing channel names to channel numbers as {channel name: channel no}
    Parameters
    ----------
    obj AxonRawIO:
        Block to be read

    Returns
    -------
    ch_dict dict

    """
    ch_dict = {}
    for chdata in obj.header['signal_channels']:
        ch_dict[chdata[0]] = chdata[1]
    return ch_dict


def getChannelNoByName(ch_name, obj):
    """
    Channel number

    Parameters
    ----------
    ch_name str:
        Dhannel name
    obj AxonRawIO:
        Data block

    Returns
    -------
    Channel number

    """
    for chdata in obj.header['signal_channels']:
        if chdata[0] == ch_name:
            return chdata[1]


def getAbfFilenamesfrompklFilename(pkl_filename):
    """
       Return a filename list from a given .pkl basename. For

       Parameters
       ----------
       pkl_filename: str
           pkl filename from which raw abf filenames will be retrieved

       Returns
           file_list: list
           List of filenames that were used to generate the .pkl file
       -------

       """

    no_ext_filename = os.path.splitext(os.path.basename(pkl_filename))[0]
    end_name = no_ext_filename.split('_')[3:]
    base_name = no_ext_filename[:10]
    base_path = no_ext_filename[2:10].replace('_', '-')
    file_list = []
    for element in end_name:
        file_list.append(base_path + '/' + base_name + '_' + element + '.abf')
    return file_list