import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.unitInfo as bStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import PyLeech.Utils.CrawlingDatabaseUtils as CDU

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as spsig
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
import findiff
import more_itertools as mit
from numba import njit

def generateOutputSpace(binned_spike_freq_dict, fs, intracel_signal=None):
    dt = binned_spike_freq_dict[list(binned_spike_freq_dict)[0]][0, 1] - \
         binned_spike_freq_dict[list(binned_spike_freq_dict)[0]][0, 0]
    step = int(dt * fs)

    spike_space = np.array([items[1] for key, items in binned_spike_freq_dict.items()])

    if intracel_signal is not None:
        binned_intracel = intracel_signal[::step][:-1]
        print(spike_space.shape, binned_intracel.shape)
        return np.vstack((spike_space, binned_intracel)).T
    else:
        print(spike_space.shape)
        return spike_space.T


def getSpikeIdxs(binned_sfd, crawling_intervals):
    spike_idxs = []
    key0 = list(binned_sfd)[0]
    for interval in crawling_intervals:
        spike_idxs.append(np.where(
            (binned_sfd[key0][0] > interval[0]) & (binned_sfd[key0][0] < interval[1]))[0])

    return np.hstack(spike_idxs)


def getSegmentsMean(embedding_trace, interval_list):
    max_len = np.max(np.diff(interval_list, axis=1))
    resampled_data = []
    for start, end in interval_list:
        diff = end - start
        resampled_data.append(spsig.resample(embedding_trace[start:end] * diff / max_len, max_len))
    return np.array(resampled_data).mean(axis=0)


def getSegmentsMedian(embedding_trace, interval_list):
    max_len = np.max(np.diff(interval_list, axis=1))
    resampled_data = []
    for start, end in interval_list:
        resampled_data.append(spsig.resample(embedding_trace[start:end], max_len))
    return np.median(np.array(resampled_data), axis=0)


def getCloseReturns(embedding_space1, embedding_space2=None, get_mask=False, threshold=.1, low_memory=False):
    if embedding_space1.dtype == np.float64:
        embedding_space1 = embedding_space1.astype(np.float32, copy=False)
    if embedding_space2 is not None and embedding_space2.dtype == np.float64:
        embedding_space2 = embedding_space2.astype(np.float32, copy=False)

    if not low_memory:
        if embedding_space2 is None:
            dist = embedding_space1[:, np.newaxis, :] - embedding_space1[np.newaxis, :, :]
        else:
            dist = embedding_space2[np.newaxis, :, :] - embedding_space1[:, np.newaxis, :]

        if dist.shape[2] > 1:
            dist = np.sqrt(np.sum(dist ** 2, axis=2))
        else:
            dist = np.abs(dist)[:, :, 0]
        dist[np.isnan(dist)] = np.max(dist[~np.isnan(dist)])
    else:
        if embedding_space2 is None:

            dist = computeSingleDistance(embedding_space1)
            dist = lowDiagConstant(dist, dist.max())

        else:
            dist = np.zeros((embedding_space1.shape[0], embedding_space2.shape[0]), dtype=np.float32)
            for i in range(dist.shape[0]):
                for j in range(i, dist.shape[1]):
                    dist[i, j] = np.abs(embedding_space1[i, 0] - embedding_space2[j, 0])

    if get_mask:
        return dist < threshold * np.max(dist[~np.isnan(dist)])
    else:
        return dist


@njit
def computeSingleDistance(trace1):
    dist = np.zeros((trace1.shape[0], trace1.shape[0]), dtype=np.float32)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i, j] = np.abs(trace1[i, 0] - trace1[j, 0])

    return dist


@njit
def lowDiagConstant(mat, ct):
    for i in range(mat.shape[1]):
        for j in range(i, mat.shape[0]):
            mat[j, i] = ct
    return mat


def computeDistance(trace1, trace2):
    dist = np.zeros((trace1.shape[0], trace2.shape[0]), dtype=np.float32)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i, j] = np.abs(trace1[i, 0] - trace2[j, 0])

    return dist


def getCloseReturnsSegments(masked_reordered_returns, rt_len=50):
    cr_idxs = np.where(masked_reordered_returns == 1)
    cr_segments_dict = {}
    segment_count = []
    for p in np.sort(np.unique(cr_idxs[0])):

        times = cr_idxs[1][cr_idxs[0] == p]

        for group in mit.consecutive_groups(times):
            lg = list(group)
            if len(lg) > rt_len:
                try:
                    cr_segments_dict[p].append((lg[0], lg[-1]))
                except KeyError:
                    cr_segments_dict[p] = [(lg[0], lg[-1])]
        try:
            cr_segments_dict[p] = np.array(cr_segments_dict[p])
            segment_count.append((p, len(cr_segments_dict[p])))
        except KeyError:
            pass

    return cr_segments_dict, np.array(segment_count)


def getCloseReturnsSegmentsFromUnorderedMatrix1(masked_returns, single_file=False, min_dist=0, rt_len=20):
    """
    Is supposed to be working fine. Is for non-zero traces
    :param masked_returns:
    :param single_file:
    :param min_dist:
    :param rt_len:
    :return:
    """

    segments_dict = {}
    if not single_file:
        for k in range(-(masked_returns.shape[0] - rt_len), -min_dist):

            col_diag = np.where(masked_returns.diagonal(k))[0]
            if col_diag.size == 0:
                continue
            segments_dict[k] = []

            for group in mit.consecutive_groups(col_diag):

                col_idxs = np.fromiter(group, dtype=int)
                if col_idxs.size >= rt_len:
                    segments_dict[k].append([col_idxs[0] - k, col_idxs[-1] - k])

    for k in range(min_dist, masked_returns.shape[1] - rt_len):
        row_diag = np.where(masked_returns.diagonal(k))[0]

        if row_diag.size == 0:
            continue

        segments_dict[k] = []

        for group in mit.consecutive_groups(row_diag):

            row_idxs = np.fromiter(group, dtype=int)

            if row_idxs.size >= rt_len:
                segments_dict[k].append([row_idxs[0], row_idxs[-1]])

    new_dict = {}
    for k in segments_dict.keys():
        if segments_dict[k]:
            new_dict[k] = np.array(segments_dict[k])
    return new_dict


def getCloseReturnsSegmentsFromUnorderedMatrix(masked_returns, zeros=None, true_rt_pct=.2, rt_len=20, single_file=False,
                                               min_dist=0):
    """
    For sparse matrixes (plenty of false crs)
    :param masked_returns:
    :param zeros:
    :param true_rt_pct:
    :param rt_len:
    :param single_file:
    :param min_dist:
    :return:
    """
    segments_dict = {}
    if not single_file:
        for k in range(-(masked_returns.shape[0] - rt_len), -min_dist):

            col_diag = np.where(masked_returns.diagonal(k))[0]
            if col_diag.size == 0:
                continue
            segments_dict[k] = []
            if zeros is not None:
                col_zeros_diag = zeros.diagonal(k)
            else:
                col_zeros_diag = np.array([])

            for group in mit.consecutive_groups(col_diag):

                col_idxs = np.fromiter(group, dtype=int)
                row_idxs = col_idxs - k
                tot_zeros = col_zeros_diag[col_idxs[0]:col_idxs[-1]].sum()
                if tot_zeros <= ((1 - true_rt_pct) * col_idxs.shape[0]) and (col_idxs.shape[0] - tot_zeros) > rt_len:
                    segments_dict[k].append([row_idxs[0], row_idxs[-1]])

    for k in range(min_dist, masked_returns.shape[1] - rt_len):
        row_diag = np.where(masked_returns.diagonal(k))[0]
        if zeros is not None:
            row_zeros_diag = zeros.diagonal(k)
        else:
            row_zeros_diag = np.array([])

        if row_diag.size == 0:
            continue

        segments_dict[k] = []

        for group in mit.consecutive_groups(row_diag):

            row_idxs = np.fromiter(group, dtype=int)
            tot_zeros = row_zeros_diag[row_idxs[0]:row_idxs[-1]].sum()
            if tot_zeros <= ((1 - true_rt_pct) * row_idxs.shape[0]) and (row_idxs.shape[0] - tot_zeros) > rt_len:
                segments_dict[k].append([row_idxs[0], row_idxs[-1]])

    new_dict = {}
    for k in segments_dict.keys():
        if segments_dict[k]:
            new_dict[k] = np.array(segments_dict[k])
    return new_dict


#
# def getCloseReturnsSegmentsFromUnorderedMatrix(no_zeros_masked_returns, zeros, rt_len=50, single_file=False,
#                                                min_dist=0):
#     segments_dict = {}
#
#     if not single_file:
#         for k in range(-(no_zeros_masked_returns.shape[0] - rt_len), -min_dist):
#
#             col_diag = np.where(no_zeros_masked_returns.diagonal(k))[0]
#             col_zeros_diag = zeros.diagonal(k)
#
#             first = True
#             for group in mit.consecutive_groups(col_diag):
#                 if first:
#                     segments_dict[k] = []
#                     last = col_diag[0]
#
#
#
#                 col_idxs = np.fromiter(group, dtype=int)
#
#                 row_idxs = col_idxs - k
#
#                 if not first and (
#                         col_zeros_diag[last:col_idxs[0]].size == col_zeros_diag[last:col_idxs[0]].sum()):
#
#                     segments_dict[k][-1][-1] = row_idxs[-1]
#                 else:
#                     if col_zeros_diag[col_idxs[0]-1]:
#                         first_idx = getFirstIdxFromZeros(col_zeros_diag, col_idxs[0]-1)-k
#                     else:
#                         first_idx = row_idxs[0]
#                     try:
#                         if col_zeros_diag[col_idxs[-1]+1]:
#                             last_idx = getLastIdxFromZeros(col_zeros_diag, col_idxs[-1]+1)-k
#                         else:
#                             last_idx = row_idxs[-1]
#                     except IndexError:
#                         last_idx = row_idxs[-1]
#
#
#                     segments_dict[k].append([first_idx, last_idx, row_idxs[-1]-row_idxs[0]])
#
#                 if first:
#                     first = False
#                 last = col_idxs[-1]
#
#     for k in range(min_dist, no_zeros_masked_returns.shape[1] - rt_len):
#         row_diag = np.where(no_zeros_masked_returns.diagonal(k))[0]
#         row_zeros_diag = zeros.diagonal(k)
#
#
#         first = True
#         for group in mit.consecutive_groups(row_diag):
#             if first:
#                 last = row_diag[0]
#                 segments_dict[k] = []
#
#             row_idxs = np.fromiter(group, dtype=int)
#
#             if not first and (
#                     row_zeros_diag[last:row_idxs[0]].size == row_zeros_diag[last:row_idxs[0]].sum()):
#                 segments_dict[k][-1][-1] = row_idxs[-1]
#             else:
#                 if row_zeros_diag[row_idxs[0] - 1]:
#                     first_idx = getFirstIdxFromZeros(row_zeros_diag, row_idxs[0] - 1)
#                 else:
#                     first_idx = row_idxs[0]
#                 try:
#                     if row_zeros_diag[row_idxs[-1] + 1]:
#                         last_idx = getLastIdxFromZeros(row_zeros_diag, row_idxs[-1] + 1)
#                     else:
#                         last_idx = row_idxs[-1]
#                 except IndexError as e:
#                     last_idx = row_idxs[-1]
#
#
#                 segments_dict[k].append([first_idx, last_idx, row_idxs[-1]-row_idxs[0]])
#
#             if first:
#                 first = False
#     new_dict = {}
#     for k in segments_dict.keys():
#         l = [(int0, int1) for int0, int1, int2 in segments_dict[k] if (int2 > rt_len)]
#         if l:
#             new_dict[k] = np.array(l)
#     return new_dict


def getFirstIdxFromZeros(zeros_trace, first_idx):
    for group in mit.consecutive_groups(np.where(zeros_trace)[0]):
        lg = np.fromiter(group, dtype=int)
        if lg[-1] == first_idx:
            return lg[0]
        elif lg[-1] > first_idx:
            return first_idx + 1
    return first_idx + 1


def getLastIdxFromZeros(zeros_trace, last_idx):
    for group in mit.consecutive_groups(np.where(zeros_trace)[0]):
        lg = np.fromiter(group, dtype=int)
        if lg[0] == last_idx:
            return lg[-1]
        elif lg[0] > last_idx:
            return last_idx - 1
    return last_idx - 1


def getMainDiagonals(segment_dict):
    diags = list(segment_dict)
    main_diags = []
    for group in mit.consecutive_groups(diags):
        main_diags.append(int(np.median(list(group))))
    return main_diags


def getNearestNeighbor(burst_space, dim_size, min_time_step_dist=10):
    burst_space = burst_space[:, :dim_size]

    dist = burst_space[:, np.newaxis, :] - burst_space[np.newaxis, :, :]

    dist = np.sqrt(np.sum(dist ** 2, axis=2))

    idxs = np.arange(dist.shape[0])
    idx_mat = np.repeat(np.arange(dist.shape[0])[np.newaxis], dist.shape[0], axis=0)

    idx_mat = idx_mat - idxs.reshape(-1, 1)

    dist[np.abs(idx_mat) < min_time_step_dist] = dist.max()

    sorted_args = np.argsort(dist, axis=1)

    return sorted_args[:, 0]


def countFalseNeighbors(burst_space, nearest_neighbors, used_dim, threshold):
    distance = np.sqrt(
        np.sum(np.power(burst_space[:, :used_dim] - burst_space[nearest_neighbors, :used_dim], 2), axis=1))

    actual_distance = np.sqrt(np.sum(np.power(burst_space - burst_space[nearest_neighbors], 2), axis=1))

    false_neighbors = np.sum(distance / actual_distance < threshold)
    return false_neighbors, distance.shape[0] - false_neighbors


def getDerivativeEmbedding(trace, dt, emb_size):
    embedding = [trace]
    for i in range(1, emb_size):
        # embedding.append(np.gradient(embedding[-1], dt))
        df = findiff.FinDiff(0, dt, i)
        embedding.append(df(trace))
    return np.array(embedding).T


def getTraceEmbedding(trace, step, emb_size):
    imbedding = [trace[step * i:(i - emb_size) * step] for i in range(emb_size)]
    return np.stack(imbedding, axis=1)


def getCycles(trace, fs=.1, peak_height=0, distance=15):
    peaks_idxs = spsig.find_peaks(trace, height=peak_height, distance=int(distance / fs))[0]
    max_len = np.diff(peaks_idxs).max()

    rescaled_trace = np.array(())
    for i in range(peaks_idxs.shape[0] - 1):
        rescaled_trace = np.append(rescaled_trace, spsig.resample(trace[peaks_idxs[i]:peaks_idxs[i + 1]], num=max_len)
                                   )

    kernel = burstUtils.generateGaussianKernel(.5, 2, .1)
    rescaled_trace = spsig.fftconvolve(rescaled_trace, kernel, mode='same')

    for i in range(1, peaks_idxs.shape[0] - 2):
        try:
            cycle_array = np.vstack((cycle_array, rescaled_trace[i * max_len:(i + 1) * max_len]))
        except NameError:
            cycle_array = rescaled_trace[i * max_len:(i + 1) * max_len]
    return cycle_array


def getNCycles(trace, fs=.1, peak_height=0, distance=15, N=2):
    peaks_idxs = spsig.find_peaks(trace, height=peak_height, distance=int(distance / fs))[0]
    max_len = np.diff(peaks_idxs).max()

    rescaled_trace = np.array(())
    for i in range(peaks_idxs.shape[0] - 1):
        rescaled_trace = np.append(rescaled_trace, spsig.resample(trace[peaks_idxs[i]:peaks_idxs[i + 1]], num=max_len)
                                   )

    kernel = burstUtils.generateGaussianKernel(.5, 2, .1)
    rescaled_trace = spsig.fftconvolve(rescaled_trace, kernel, mode='same')

    for i in range(1, peaks_idxs.shape[0] - (N + 1)):
        try:
            cycle_array = np.vstack((cycle_array, rescaled_trace[i * max_len:(i + N) * max_len]))
        except NameError:
            cycle_array = rescaled_trace[i * max_len:(i + N) * max_len]
    return cycle_array


def compareCycles(cycle_mat):
    dist = cycle_mat[:, np.newaxis, :] - cycle_mat[np.newaxis, :, :]
    return np.sqrt(np.sum(dist ** 2, axis=2))


def getSimilarCycles(cycle_arr, num=10):
    cycle_dist = compareCycles(cycle_arr)
    np.fill_diagonal(cycle_dist, cycle_dist.max())
    return np.array(smallest_indices(cycle_dist, 2 * num)).T[::2]


def resampleByCycles(trace, fs=.1, peak_height=0, distance=15):
    peaks_idxs = spsig.find_peaks(trace, height=peak_height, distance=int(distance / fs))[0]
    max_len = np.diff(peaks_idxs).max()
    rescaled_trace = np.array(())

    for i in range(peaks_idxs.shape[0] - 1):
        rescaled_trace = np.append(rescaled_trace, spsig.resample(trace[peaks_idxs[i]:peaks_idxs[i + 1]], num=max_len)
                                   )
    return rescaled_trace


def plotCycles(trace, fs=.1, peak_height=0, distance=15, N=1):
    peaks_idxs = spsig.find_peaks(trace, height=peak_height, distance=int(distance / fs))[0]
    max_len = np.diff(peaks_idxs).max()
    fig, ax = plt.subplots()
    rescaled_trace = np.array(())
    for i in range(peaks_idxs.shape[0] - 1):
        rescaled_trace = np.append(rescaled_trace, spsig.resample(trace[peaks_idxs[i]:peaks_idxs[i + 1]], num=max_len)
                                   )

    kernel = burstUtils.generateGaussianKernel(.5, 2, .1)
    rescaled_trace = spsig.fftconvolve(rescaled_trace, kernel, mode='same')
    for i in range(1, peaks_idxs.shape[0] - (N + 1)):
        ax.plot(rescaled_trace[i * max_len:(i + N) * max_len], label=i)

    ax.legend()
    return fig, ax


def smallest_indices(ary, n):
    """Returns the n smallest indices from a numpy array."""
    flat = ary.flatten()
    if n < flat.shape[0]:
        indices = np.argpartition(flat, n)[:n]
    else:
        indices = np.argpartition(flat, flat.shape[0] - 1)
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, ary.shape)


def getPcaBase(array_mat, vect_num=3, plot_vectors=False):
    varcovmat = np.cov(array_mat)
    unitary, singular, v = np.linalg.svd(varcovmat)

    if plot_vectors:
        j = 0
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax = np.concatenate(ax)
        for i in range(vect_num):
            j += 1
            if j == 5:
                j = 1
                fig, ax = plt.subplots(2, 2, sharex=True)
                ax = np.concatenate(ax)

            ax[j - 1].plot(unitary[:, i])
            ax[j - 1].set_title('PC' + str(i) + ': ' + str(singular[i]))

    return unitary, singular, v


def plot3Dscatter(array3D, fig_ax_pair=None, label=None, colorbar=False, s=5, color=None):
    if fig_ax_pair is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        fig, ax = fig_ax_pair

    if label is not None:
        label = str(label)
    if color is None:
        N = 21
        cmap = plt.cm.jet
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.scatter(array3D[:, 0], array3D[:, 1], array3D[:, 2],
                   c=plt.cm.jet(np.linspace(0, 1, array3D.shape[0])), label=label, s=s)
        if colorbar:
            plt.colorbar(sm, ticks=np.linspace(0, 1, N),
                         boundaries=np.arange(-0.05, 1.1, .05))
    else:
        ax.scatter(array3D[:, 0], array3D[:, 1], array3D[:, 2],
                   c=color, label=label, s=s)
    return fig, ax


def plot3Dline(array3D, fig_ax_pair=None, label=None, colorbar=False, color='time', step=5, linewidth=2):
    if fig_ax_pair is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        fig, ax = fig_ax_pair

    if label is not None:
        label = str(label)

    N = 21
    cmap = plt.cm.jet
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if color == 'time':
        for i in np.arange(0, array3D.shape[0] - step, step - 1):
            ax.plot(array3D[:, 0][i:i + step], array3D[:, 1][i:i + step], array3D[:, 2][i:i + step],
                    color=plt.cm.jet(1 * i / array3D.shape[0]), label=label, linewidth=linewidth)

        if colorbar:
            plt.colorbar(sm, ticks=np.linspace(0, 1, N),
                         boundaries=np.arange(-0.05, 1.1, .05))
    elif type(color) is int:
        rescaled = (array3D[:, color] - array3D[:, color].min()) / (array3D[:, color].max() - array3D[:, color].min())
        for i in np.arange(0, array3D.shape[0] - step, step - 1):
            ax.plot(array3D[:, 0][i:i + step], array3D[:, 1][i:i + step], array3D[:, 2][i:i + step],
                    color=plt.cm.jet(rescaled[i:i + step].mean()), label=label, linewidth=linewidth)

        if colorbar:
            plt.colorbar(sm, ticks=np.linspace(0, 1, N),
                         boundaries=np.arange(-0.05, 1.1, .05))

    elif ((type(color) is str) and len(color) == 1) or (type(color) is tuple):
        ax.plot(array3D[:, 0], array3D[:, 1], array3D[:, 2], color=color, label=label, linewidth=linewidth)

    else:
        ax.plot(array3D[:, 0], array3D[:, 1], array3D[:, 2], color='k', label=label, linewidth=linewidth)

    return fig, ax


def plotCloseReturns(rt, thr=.05, masked=True, reorder=True, get_counts=True, return_reordered=False):
    returns = np.copy(rt)
    returns[np.isnan(returns)] = 0
    max_dist = returns.max()

    if get_counts or reorder:

        rt_reordered = np.zeros((returns.shape[1], returns.shape[0]), dtype=returns.dtype)
        for i in range(returns.shape[0]):
            for j in range(i, returns.shape[1]):
                rt_reordered[j - i, i] = returns[i, j]
        if rt_reordered.dtype == bool:
            reordered_mask = rt_reordered
        else:
            reordered_mask = (rt_reordered < max_dist * thr) & (rt_reordered > 0.)

    if get_counts:
        counts = reordered_mask.sum(axis=1)
        fig1, ax1 = plt.subplots()
        ax1.scatter(np.arange(1, reordered_mask.shape[0] + 1, 1), counts, s=2)

    if not reorder:
        rt_reordered = returns
        if rt_reordered.dtype == bool:
            reordered_mask = rt_reordered
        else:
            reordered_mask = (rt_reordered < max_dist * thr) & (rt_reordered > 0.)

    fig2, ax2 = plt.subplots()
    if masked or rt_reordered.dtype == bool:
        cax = ax2.matshow(reordered_mask.astype(int))
    else:
        rt_reordered[~reordered_mask] = max_dist * thr
        cax = ax2.matshow(rt_reordered, origin='lower')
    # if reorder:
    # cax.set_ylim(cax.get_ylim()[::-1])
    fig2.colorbar(cax)
    # plt.ylim([280, 260])
    fig2.gca().set_aspect('auto')
    if get_counts:
        if return_reordered:
            if masked:
                print("returning reordered mask")
                return reordered_mask.astype(int), fig1, ax1, fig2, ax2
            else:
                print("returning reordered matrix")
                return rt_reordered, fig1, ax1, fig2, ax2
        else:
            return fig1, ax1, fig2, ax2

    if return_reordered:
        if masked:
            print("returning reordered mask")
            return reordered_mask.astype(int), fig2, ax2
        else:
            print("returning reordered matrix")
            return rt_reordered, fig2, ax2

    return fig2, ax2


def generateColorbar(obj):
    N = 21
    cmap = plt.cm.jet
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    obj.colorbar(sm, ticks=np.linspace(0, 1, N),
                 boundaries=np.arange(-0.05, 1.1, .1))


def on_move(ax_list, event):
    for ax in ax_list:
        if event.inaxes == ax:
            for ax2 in ax_list:
                if ax2 != ax:
                    ax2.view_init(elev=ax.elev, azim=ax.azim)
    else:
        return
    fig.canvas.draw_idle()


@njit
def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


@njit
def rossler(x, y, z, a=.1, b=.1, c=14):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       a, b, c: parameters defining the Rössler attractor
    Returns:
       x_dot, y_dot, z_dot: values of the rössler attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = - y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return x_dot, y_dot, z_dot


@njit
def simDynamicalSistem(params, func=lorenz, num_steps=10000, x0=np.array([0., 1., 1.05]), dt=.01):

    ns_p1 = num_steps+1
    xs = np.zeros((ns_p1, 3), dtype=np.float_)


    xs[0] = x0

    if params.ndim == 1:
        for i in range(num_steps):
            x_dot, y_dot, z_dot = func(xs[i][0], xs[i][1], xs[i][2], params[0], params[1], params[2])
            xs[i + 1][0] = xs[i][0] + (x_dot * dt)
            xs[i + 1][1] = xs[i][1] + (y_dot * dt)
            xs[i + 1][2]= xs[i][2] + (z_dot * dt)

    else:
        for i in range(num_steps):
            x_dot, y_dot, z_dot = func(xs[i][0], xs[i][1], xs[i][2], params[i, 0], params[i, 1], params[i, 2])
            xs[i + 1][0] = xs[i][0] + (x_dot * dt)
            xs[i + 1][1] = xs[i][1] + (y_dot * dt)
            xs[i + 1][2] = xs[i][2] + (z_dot * dt)

    return xs

@njit
def simRossler(params, num_steps=10000, x0=np.array((0., 1., 1.05)), dt=.01):


    xs = np.zeros((num_steps+1, 3), dtype=np.float_)


    xs[0] = x0

    if params.ndim == 1:
        for i in range(num_steps):
            x_dot, y_dot, z_dot = rossler(xs[i][0], xs[i][1], xs[i][2], params[0], params[1], params[2])
            xs[i + 1][0] = xs[i][0] + (x_dot * dt)
            xs[i + 1][1] = xs[i][1] + (y_dot * dt)
            xs[i + 1][2]= xs[i][2] + (z_dot * dt)

    else:
        for i in range(num_steps):
            x_dot, y_dot, z_dot = rossler(xs[i][0], xs[i][1], xs[i][2], params[i, 0], params[i, 1], params[i, 2])
            xs[i + 1][0] = xs[i][0] + (x_dot * dt)
            xs[i + 1][1] = xs[i][1] + (y_dot * dt)
            xs[i + 1][2] = xs[i][2] + (z_dot * dt)

    return xs


if __name__ == "__main__":
    fn = "RegistrosDP_PP/2018_12_03_0005.pklspikes"
    burst_obj = bStorerLoader(fn, 'RegistrosDP_PP', mode='load')

    arr_dict, time, fs = abfe.getArraysFromAbfFiles(fn, ['Vm1'])
    NS = arr_dict['Vm1']
    del arr_dict
    cdb = CDU.loadCrawlingDatabase()

    good_neurons = cdb.loc[fn].index[cdb.loc[fn, 'neuron_is_good'].values.astype(bool)].values
    crawling_interval = [cdb.loc[fn].start_time.iloc[0], cdb.loc[fn].end_time.iloc[0]]

    dt_step = 0.1
    binned_sfd = burstUtils.digitizeSpikeFreqs(burst_obj.spike_freq_dict, dt_step, time[-1], counting=False)

    cut_binned_freq_array = burstUtils.binned_sfd_to_dict_array(binned_sfd, crawling_interval, good_neurons)

    sigma = .5
    rg = 5
    x_range = np.arange(-rg, rg, dt_step)
    gaussian = np.exp(-(x_range / sigma) ** 2)
    gaussian /= gaussian.sum()
    #    plt.figure()
    #    plt.plot(x_range, gaussian)

    smoothed_sfd = {}
    for key, items in cut_binned_freq_array.items():
        smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], gaussian, mode='same')])
    uncut_smoothed_sfd = {}
    for key, items in binned_sfd.items():
        uncut_smoothed_sfd[key] = np.array([items[0], spsig.fftconvolve(items[1], gaussian, mode='same')])

    sigma100 = .5
    rg = 5
    x_range100 = np.arange(-rg, rg, 1 / fs)
    gaussian100 = gaussian = np.exp(-(x_range100 / sigma100) ** 2)
    gaussian100 /= gaussian100.sum()
    conv_NS = spsig.fftconvolve(NS, gaussian100, mode='same')
    plt.figure()
    plt.plot(conv_NS)

    del NS

    cut_NS = conv_NS[np.where(time > crawling_interval[0])[0][0]: np.where(time < crawling_interval[1])[0][-1]]
    cut_time = time[np.where(time > crawling_interval[0])[0][0]: np.where(time < crawling_interval[1])[0][-1]]

    burstUtils.plotFreq(uncut_smoothed_sfd, color_dict=burst_obj.color_dict, optional_trace=[time[::2000], NS[::2000]],
                        template_dict=burst_obj.template_dict, scatter_plot=False,
                        outlier_thres=None, ms=4)

    burst_space = generateOutputSpace(smoothed_sfd, cut_NS.T, fs)

    sc = StandardScaler()
    scaled_burst_space = sc.fit_transform(burst_space)

    unitary, singular, v = getPcaBase(scaled_burst_space.T, vect_num=5, plot_vectors=True)

    proy = np.dot(scaled_burst_space, unitary)

    dim_size = 3
    fig, ax = plt.subplots(dim_size, 1, sharex=True)
    for i in range(dim_size):
        ax[i].plot(proy[:, i])
        ax[i].grid()

    N = 21
    cmap = plt.get_cmap('jet', N)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(proy[:, 0], proy[:, 1], proy[:, 2], c=plt.cm.jet(np.linspace(0, 2, proy.shape[0])),
               label='parametric curve', s=2)

    plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                 boundaries=np.arange(-0.05, 2.1, .1))
    #
    idxs = np.where(
        (
                (time > 400) & (time < 750)
        )
        # |
        #  (
        #  (time > 660) & (time < 820)
        #  )
    )

    # NS_embedding = getTraceEmbedding(conv_NS[::2000], int(.5 * fs), 20)
    NS_embedding = getDerivativeEmbedding(conv_NS[idxs][::100], 100 / fs, 3)
    dim_size = 3
    fig, ax = plt.subplots(dim_size, 1, sharex=True)
    for i in range(dim_size):
        ax[i].plot(NS_embedding[:, i])
        ax[i].grid()

    for thr in [.5]:
        false_neighbors = []
        for i in range(1, NS_embedding.shape[1]):
            NN = getNearestNeighbor(NS_embedding, i, 10)
            neighbors = countFalseNeighbors(NS_embedding, NN, i, thr)
            false_neighbors.append(neighbors[0])

        plt.figure()
        plt.title("Threshold = %f" % thr)
        plt.plot(np.arange(1, 1 + len(false_neighbors)), false_neighbors)

    dim_size = 3
    fig, ax = plt.subplots(dim_size, 1, sharex=True)
    for i in range(dim_size):
        ax[i].plot(NS_embedding[:, i])
        ax[i].grid()

    N = 21
    cmap = plt.get_cmap('jet', N)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure()
    ax = Axes3D(fig)
    NS_sh = NS_embedding[:, 0].shape[0]
    ax.scatter(NS_embedding[:int(NS_sh / 4), 0], NS_embedding[:int(NS_sh / 4), 1], NS_embedding[:int(NS_sh / 4), 2],
               c=plt.cm.jet(np.linspace(0, 2, int(NS_sh / 2))), s=1)

    plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                 boundaries=np.arange(-0.05, 2.1, .1))

    # df = pd.DataFrame(scaled_burst_space, columns=list(cut_binned_freq_array) + ['NS'])
    # df = pd.DataFrame(np.dot(burst_space, unitary[:, :4]))
    # df = pd.DataFrame(burst_space)
    # scatter_matrix(df, alpha=0.2, s=4, c='k', figsize=(6, 6),
    #                diagonal='kde', marker=".")
    # pc_proy = np.dot(scaled_burst_space, unitary[:, :4])
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(pc_proy[:,0], pc_proy[:,1],pc_proy[:,2], label='parametric curve')
    #
    #
    # fn = 'dimension_testing.csv'
    # f = open(fn, 'w')
    # w = csv.writer(f)
    # w.writerows(np.dot(burst_space, unitary))
    # f.close()

    ### Close returns

    idxs = np.where(
        (
                (time > crawling_interval[0]) & (time < 625)
        )
        |
        (
                (time > 660) & (time < 820)
        )
    )

    rt = getCloseReturns(burst_space[:, 0].reshape(-1, 1), get_mask=False)

    # rt[rt>50] = 0

    plt.matshow(rt)
    plt.colorbar()
    plt.gca().set_aspect('auto')

    rt_reordered = np.zeros(rt.shape)
    for i in range(rt.shape[0]):
        for j in range(i, rt.shape[0]):
            rt_reordered[j - i, i] = rt[i, j]

    reordered_mask = (rt_reordered < 1) & (rt_reordered > .1)
    plt.figure()
    plt.scatter(np.arange(1, reordered_mask.shape[0] + 1, 1), reordered_mask.sum(axis=1), s=2)

    plt.matshow(reordered_mask.astype(int))
    plt.colorbar()
    plt.gca().set_aspect('auto')

    # for k in range(3):
    k = 5
    rt = getCloseReturns(proy[:, :k], get_mask=False)

    rt_reordered = np.zeros(rt.shape)
    for i in range(rt.shape[0]):
        for j in range(i, rt.shape[0]):
            rt_reordered[j - i, i] = rt[i, j]

    # plt.matshow(rt_reordered)
    # plt.colorbar()
    # plt.gca().set_aspect('auto')

    reordered_mask = (rt_reordered < 1.5) & (rt_reordered > 0.)
    plt.figure()
    plt.scatter(np.arange(1, reordered_mask.shape[0] + 1, 1), reordered_mask.sum(axis=1), s=2, label="proy" + str(k))
    plt.legend()

    plt.matshow(reordered_mask.astype(int))
    plt.title("proy" + str(k))
    plt.colorbar()
    # plt.ylim([280, 260])
    plt.gca().set_aspect('auto')

    i = 746
    p = 600

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(proy[i:i + p, 0], proy[i:i + p, 1], proy[i:i + p, 2], label="n" + str(key) + " dt= " + str(i), lw=1)

    burstUtils.plotFreq(uncut_smoothed_sfd, color_dict=burst_obj.color_dict,
                        template_dict=burst_obj.template_dict, scatter_plot=False,
                        optional_trace=[time[::1000], NS[::1000]],
                        outlier_thres=None, ms=4)

    #### Imbeddings

    for key in good_neurons:
        imb, Dimb, DDimb = getDerivativeEmbedding(uncut_smoothed_sfd[key][1], dt=dt_step)

        fig = plt.figure()
        ax = Axes3D(fig)

        # ax.plot(new_NS[::100], DNS[::100], DDNS[::100], label='parametric curve', lw=1)
        # ax.scatter(new_NS, DNS, DDNS, label=key, s=10)

        ax.plot(imb, Dimb, DDimb, label="n" + str(key) + " dt= " + str(dt_step), lw=1)
        fig.legend()

    # ax.scatter(imb[0], Dimb[0], DDimb[0], marker='*', color='r', s=50)
    # ax.scatter(imb[-1], Dimb[-1], DDimb[-1], marker='*', color='r', s=50)

    step = 2000

    imb, Dimb, DDimb = getDerivativeEmbedding(conv_NS[idxs], dt=dt_step * fs)
    arr_imb = np.array(getDerivativeEmbedding(conv_NS[idxs], dt=dt_step * fs))[:, ::step]

    N = 21
    cmap = plt.get_cmap('jet', N)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                 boundaries=np.arange(-0.05, 2.1, .1))

    fig = plt.figure()
    ax = Axes3D(fig)
    i = 400
    p = 270
    ax.plot(arr_imb[0, i:i + p], arr_imb[1, i:i + p], arr_imb[2, i:i + p], label="n" + str(key) + " dt= " + str(i),
            lw=1)

    ax.scatter(imb[0], Dimb[0], DDimb[0], marker='*', color='r', s=50)
    ax.scatter(imb[-1], Dimb[-1], DDimb[-1], marker='*', color='r', s=50)

    fig.legend()

    #
    #
    # for i in range(burst_space.shape[1]):
    #     rate, Drate, DDrate = getDerivativeImbedding(burst_space[:, i], dt=.5)
    #
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.plot(rate, Drate, DDrate, label='parametric curve', lw=1)
    #
    # rate, Drate, DDrate = getDerivativeImbedding(conv_NS, dt=.5 / fs)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(rate[::100], Drate[::100], DDrate[::100], label='parametric curve', lw=1)

    intervals = [[200, 300],
                 [300, 445],
                 [472, 524],
                 [550, 575],
                 [600, 616],
                 [634, 681],
                 [700, 769],
                 [790, 850],
                 [879, 950],
                 [988, 1230],
                 ]
    N = 21
    cmap = plt.get_cmap('jet', N)
    ids = []
    for tau in [.1, .3, .5, .8, 1, 1.5, 2, 5, 10, 15, 20, 30]:
        for interval in intervals[:2]:
            idxs = np.where(
                (
                        (time > interval[0]) & (time < interval[1])
                )
            )

            NS_embedding = getTraceEmbedding(conv_NS[idxs][::2000], int(tau / .1), 3)

            step = 1
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(NS_embedding[::step, 0, ], NS_embedding[::step, 1], NS_embedding[::step, 2],
                       c=plt.cm.jet(np.linspace(0, 2, NS_embedding[::step, 0, ].shape[0])), label="NS embedding", s=1)
            fig.legend()
            norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                         boundaries=np.arange(-0.05, 2.1, .1))
            fig.suptitle("interval: " + str(interval[0]) + " - " + str(interval[1]) + "\n tau: " + str(tau))

    bin_times = uncut_smoothed_sfd[list(uncut_smoothed_sfd)[0]][0]

    cut_idxs = np.where(
        (
                (bin_times > 400) & (bin_times < 750)
        )
        # |
        # (
        #     (bin_times > 660) & (bin_times < 820)
        # )
    )

    for key in good_neurons:
        embedding = getDerivativeEmbedding(uncut_smoothed_sfd[key][1, cut_idxs[0]], 0.1, 3)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                   c=plt.cm.jet(np.linspace(0, 2, embedding.shape[0])), label=str(key), s=5)
        plt.colorbar(sm, ticks=np.linspace(0, 2, N),
                     boundaries=np.arange(-0.05, 2.1, .1))
        fig.legend()
