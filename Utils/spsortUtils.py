import numpy as np

import PyLeech.Utils.burstUtils
from PyLeech.Utils import sorting_with_python as swp

try:
    import winsound


    def beep():
        for i in range(3):
            winsound.Beep(2000, 100)
except ModuleNotFoundError:
    def beep():
        return

import PyLeech.Utils.constants as constants
import math
import os

nan = constants.nan
opp0 = constants.opp0
proc_num = constants.proc_num


def hideClusterFromAx(ax, line_map, cluster):
    if type(cluster) is int:
        ax.collections[line_map[cluster]].remove()
    elif type(cluster) is list:
        for cl in cluster:
            ax.collections[line_map[cl]].remove()
    else:
        assert False, "cluster must be either int or list of ints"

    ax.legend()


def good_evts_fct(samp, thr=3):
    samp_med = np.apply_along_axis(np.median, 0, samp)
    samp_mad = np.apply_along_axis(swp.mad, 0, samp)
    above = samp_med > 0
    samp_r = samp.copy()
    for i in range(samp.shape[0]): samp_r[i, above] = 0
    samp_med[above] = 0
    res = np.apply_along_axis(lambda x:
                              np.all(abs((x - samp_med) / samp_mad) < thr),
                              1, samp_r)
    return res


def setColors(clust_no, num_col=10):
    tot_cols = 2 * len(clust_no) + 1

    cmap = PyLeech.Utils.burstUtils.categorical_cmap(num_col, math.ceil(tot_cols / num_col))
    j = 0
    cluster_color = {}
    for i in clust_no:
        cluster_color.update({i: [cmap(j)]})
        j += 1
        if i == 0:
            cluster_color.update({opp0: [cmap(j)]})
        else:
            cluster_color.update({-i: [cmap(j)]})
        j += 1
    cluster_color.update({nan: [cmap(j)]})
    return cluster_color


def getGoodClusters(unique_clust_list):
    return unique_clust_list[(unique_clust_list >= 0) & (unique_clust_list != opp0) & (unique_clust_list != nan)]


def generateFilenameFromList(filename):
    new_filename = os.path.basename(filename[0]).split('_')
    new_filename = "_".join(new_filename[:-1])

    for fn in filename:
        num = os.path.splitext(fn.split("_")[-1])[0]
        new_filename += '_' + num

    return new_filename


# noinspection PyTypeChecker


def generatePklFilename(filename, folder=None):

    if type(filename) is list:
        filename = generateFilenameFromList(filename)

    else:
        filename = os.path.basename(os.path.splitext(filename)[0])
    if folder is None:
        folder = 'RegistrosDP_PP/'
    else:
        folder = folder + '/'
    filename = folder + filename
    return filename


def generateClustersTemplate(cluster_array, evts, good_evts, check_clusters=None):
    template_dict = {}
    for cl in np.unique(cluster_array):
        if (check_clusters is None) and (cl >= 0) and (cl != nan) and (cl != opp0):
            median = np.apply_along_axis(np.median, 0, evts[good_evts, :][cluster_array == cl, :])
            mad = np.apply_along_axis(swp.mad, 0, evts[good_evts, :][cluster_array == cl, :])
            template_dict.update({cl: {'median': median, 'mad': mad}})
        elif (check_clusters is not None) and cl in check_clusters:
            median = np.apply_along_axis(np.median, 0, evts[good_evts, :][cluster_array == cl, :])
            mad = np.apply_along_axis(swp.mad, 0, evts[good_evts, :][cluster_array == cl, :])
            template_dict.update({cl: {'median': median, 'mad': mad}})

    return template_dict


def getTemplateDictSubset(clust_list, template_dict):
    temp_template_dict = {}

    for key in clust_list:
        temp_template_dict[key] = template_dict[key]
    return temp_template_dict
