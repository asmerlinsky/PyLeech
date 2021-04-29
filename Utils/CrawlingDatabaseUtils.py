import pandas as pd

import sys
import os
import json
import PyLeech.Utils.json_numpy as json_numpy

import PyLeech.Utils.unitInfo as burstStorerLoader

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import PyLeech.Utils.burstUtils as burstUtils
import matplotlib.pyplot as plt

import numpy as np


def jsonKeys2int(x):
    if isinstance(x, dict):
        try:
            return {int(k): v for k, v in x.items()}
        except ValueError:
            {k: v for k, v in x.items()}
    return x


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():
        # print(key, type(key), isinstance(key, str))
        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            # print(key, type(key))
            # try:
            #     d[str(key)] = value
            # except Exception:
            #     try:
            #         d[repr(key)] = value
            #     except Exception:
            #         raise
            d[str(key)] = value
            # delete old key
            del d[key]

    return d


def loadCrawlingDatabase(filename='RegistrosDP_PP/crawling_database.csv'):
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

        burst_object = burstStorerLoader.UnitInfo(filename_list[j], 'load')

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
            df_list.append(
                [filename_list[j], crawling_interval[0], crawling_interval[1], str(key), spike_count, diff_mean,
                 diff_std, good_unit])

        plt.close(fig)

    df = pd.append(df_list, columns=['filename', 'start_time', 'end_time', 'neuron', spike_count, 'ISI_mean', 'ISI_std',
                                     'neuron_is_good'])
    df.to_csv('test.csv')


def loadDataDict(filename='RegistrosDP_PP/CrawlingDataDict.json'):
    f = open(filename, "r")
    DataDict = json.load(f, object_hook=jsonKeys2int)
    f.close()
    return DataDict


def printKeys(d):
    for key, items in d.items():
        print(key, type(key))
        if isinstance(items, dict):
            printKeys(items)


def appendToDataDict(data_dict, filename=None):
    if filename is None:
        filename = "RegistrosDP_PP/CrawlingDataDict.json"
    try:
        f = open(filename, "r")
        DataDict = json.load(f, object_hook=jsonKeys2int)
        f.close()
    except FileNotFoundError:
        print("File not Found, I'll generate a new one in %s" % filename)
        DataDict = {}

    for key, items in data_dict.items():
        DataDict[key] = items

    str_dataDict = stringify_keys(DataDict)
    # printKeys(str_dataDict)
    dumped = json.dumps(str_dataDict, cls=json_numpy.NumpyAwareJSONEncoder)
    f = open(filename, "w")
    f.write(dumped)

    f.close()

def plotSpikesFrompklspikes(filename):
    burst_object = burstStorerLoader.UnitInfo(filename, mode='load')
    plt.close()
    fig, ax = burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=None,
                                  template_dict=None, outlier_thres=3.5, ms=2)
    fig.suptitle(filename)

def generateSingleFileDataDict(filename, min_isi=0.005, max_isi=2, data_dict=None):
    if data_dict is None:
        data_dict = {}


    print("filename is %s" % filename)

    burst_object = burstStorerLoader.UnitInfo(filename, mode='load')

    plt.close()

    fig, ax = burstUtils.plotFreq(burst_object.spike_freq_dict, burst_object.color_dict, optional_trace=None,
                                  template_dict=None, outlier_thres=3.5, ms=2)
    fig.suptitle(filename)
    fig.canvas.draw_idle()
    print("Right click on figure to continue\n")
    plt.ginput(n=0, timeout=0, mouse_pop=2, mouse_stop=3)

    crawling_intervals = []

    bad = False
    if "crawling_intervals" not in list(data_dict):
        keep_going = True
        while keep_going:

                ipt = input(
                    "input crawling interval as min-max one by one\nIf finished press enter, if useless file input \"skip\"\n")
                ipt_list = ipt.split('-')
                if len(ipt_list) == 2:
                    crawling_intervals.append([float(x) for x in ipt_list])
                elif (len(ipt_list) == 1) and ipt_list[0].lower() == 'skip':
                    bad = True
                    keep_going = False
                elif len(ipt) == 0:
                    keep_going = False
                else:
                    print("Didn´t understood input")
        data_dict["crawling_intervals"] = crawling_intervals

    if bad:
        return {"skipped": True}
    else:
        data_dict["skipped"] = False

    De3 = None
    if "DE3" not in list(data_dict) or data_dict["DE3"] is None:
        if burst_object.isDe3 is not None:
            ipt = input("De3 is %i, is this right? ('y' or pass right number)" % burst_object.isDe3)
            try:
                De3 = int(ipt)
            except ValueError:
                De3 = burst_object.isDe3
            data_dict["DE3"] = De3
        else:
            ipt = input("which no. is DE3?")
            try:
                De3 = int(ipt)
            except ValueError:
                pass

    neuron_dict = {}
    if "neurons" not in list(data_dict):
        for key in burst_object.spike_freq_dict.keys():


            ci = data_dict["crawling_intervals"][0]
            spikes = burst_object.spike_freq_dict[key][0]
            crawling_spikes = spikes[(spikes > ci[0]) & (spikes < ci[1])]
            diff = np.diff(crawling_spikes)
            diff = diff[(diff > min_isi) & (diff < max_isi)]

            diff_mean = np.mean(diff)
            diff_std = np.std(diff)
            got_it = False
            while not got_it:
                good_unit = str(input('Is unit %i good\n(y/n)?\n' % key))
                if good_unit in ['true', 'True', 'y', 'Y']:
                    good_unit = True
                    got_it = True
                elif good_unit in ['false', 'False', 'n', 'N']:
                    good_unit = False
                    got_it = True
                else:
                    print("I didn't understand the answer, Please try again\n")

            neuron_dict[str(key)] = {"ISI_mean": diff_mean, "ISI_std": diff_std, "neuron_is_good": good_unit}

        data_dict["neurons"] = neuron_dict

    if 'channels' not in list(data_dict):
        ch_list = ["Vm1", "Vm2", "IN4", "IN5", "IN6", "IN7"]
        ch_dict = {}
        for channel in ch_list:
            ch_dict[channel] = str(input("what did %s record?" % channel))
        data_dict["channels"] = ch_dict


    return data_dict


def newEntryToDataDict(filename_list, database='RegistrosDP_PP/CrawlingDataDict.json', min_isi=0.005, max_isi=2):
    try:
        crawling_data_dict = loadDataDict(database)
    except FileNotFoundError:
        print("File not Found, I'll generate a new one in %s" % database)
        crawling_data_dict = {}
    key = 'filename'
    skeys = ['neurons', 'crawling_intervals', 'DE3', "channels"]
    sskeys = ['ISI_mean', 'ISI_std', 'neuron_is_good']
    new_files = []
    updated_files = []
    file_data_dict = {}
    for j in range(len(filename_list)):
        if filename_list[j] not in list(crawling_data_dict):
            new_files.append(filename_list[j])
            single_data_dict = generateSingleFileDataDict(filename_list[j], min_isi, max_isi)

        elif crawling_data_dict[filename_list[j]]["skipped"]:
            single_data_dict = None

        elif not all([c in list(crawling_data_dict[filename_list[j]]) for c in skeys]):
            updated_files.append(filename_list[j])
            print("Re-updating entry for %s due to missing the following keys:" % filename_list[j])
            missing_keys = [c for c in skeys if c not in list(crawling_data_dict[filename_list[j]])]
            print("\n".join(missing_keys))
            ipt = input("rerun database (y) or simply update missing key(n)?")

            if str(ipt).lower == "y":
                for key in missing_keys:
                    del crawling_data_dict[filename_list[j]][key]
                single_data_dict = generateSingleFileDataDict(filename_list[j], min_isi, max_isi,
                                                              crawling_data_dict[filename_list[j]])
            else:
                single_data_dict = generateSingleFileDataDict(filename_list[j], min_isi, max_isi,
                                                              crawling_data_dict[filename_list[j]])
        else:
            single_data_dict = None

        if single_data_dict is not None:
            file_data_dict[filename_list[j]] = {}
            try:
                for key, items in single_data_dict.items():
                    if key not in crawling_data_dict[filename_list[j]]:
                        file_data_dict[filename_list[j]][key] = single_data_dict[key]
                    else:
                        file_data_dict[filename_list[j]][key] = crawling_data_dict[filename_list[j]][key]
            except KeyError:

                # raise
                file_data_dict[filename_list[j]] = single_data_dict
    plt.close("all")
    appendToDataDict(file_data_dict, database)
    if new_files:
        print("Database has been updated with the following files:")
        [print(x) for x in new_files]
    if updated_files:
        print("New entries have been added for the following files:")
        [print(x) for x in updated_files]


if __name__ == '__main__':
    crawling_database = pd.read_csv('RegistrosDP_PP/crawling_database.csv')

    crawling_database.drop('Unnamed: 0', axis=1, inplace=True)
    crawling_database.set_index(['filename', 'neuron'], inplace=True)
    crawling_database.head()
