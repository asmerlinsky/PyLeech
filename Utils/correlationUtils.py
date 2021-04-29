import numpy as np
from numba import jit
import PyLeech.Utils.NLDUtils as NLD
import scipy.signal as spsig
from sklearn.decomposition import PCA
import PyLeech.Utils.burstUtils as burstUtils
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.cluster import SpectralCoclustering

# @jit(nopython=True)
def getFlattenedUpperTriangIdxs(n, k):
    mask = np.zeros((n, n), dtype=np.bool_)
    row, col = np.triu_indices(n, k)
    for i, j in zip(row, col):
        mask[i, j] = True
    return np.where(mask.flatten())[0]

def pairwiseSlidingWindow(trace1, trace2, func, step=1, window_size=100):
    """
    Returns the starting indexes and the corresponding list of r values

    Parameters
    ----------
    trace1 : array_like
        The first trace to correlate
    trace2 : array_like
        The second trace to correlate
    step : int, optional
        Iteration step
    window_size : int, optional
        window size to be used to run the correlation
    Returns
    -------
    idxs : ndarray
        Corresponding indexes of the correlation list
    corr : list
        list of correlation values
    """

    corr = []

    idxs = np.arange(trace1.shape[0] - window_size, step=step)
    for i in idxs:
        r = func(trace1[i:i + window_size],
                 trace2[i:i + window_size])
        corr.append(r)
    return idxs + int(round(window_size / 2)), corr


# @jit(nopython=True)
def multiUnitSlidingWindow(spike_freq_array, step=1, window_size=100):
    mat_size = spike_freq_array.shape[1]
    length = spike_freq_array.shape[0] - window_size
    idxs = np.arange(0, length, step)
    num_mats = idxs.shape[0]
    functional_connectivity_matrixes = np.zeros((num_mats, mat_size, mat_size))

    for i, j in zip(idxs, range(idxs.shape[0])):
        functional_connectivity_matrixes[j] = np.corrcoef(spike_freq_array[i:i + window_size], rowvar=False)

    return functional_connectivity_matrixes

def reorderSpikeFreqDict(correlation_matrix, spike_freq_dict, n_clusters=3, plot_reordered_mat=False):
    model = SpectralCoclustering(n_clusters=n_clusters)
    model.fit(correlation_matrix)

    keys = np.array(list(spike_freq_dict))
    new_order = np.argsort(model.row_labels_)
    ordered_keys = keys[new_order]

    new_dict = {}
    label_dict = {}
    for i in range(ordered_keys.shape[0]):
        new_dict[i] = spike_freq_dict[ordered_keys[i]]
        label_dict[i] = ordered_keys[i]
    if plot_reordered_mat:
        fit_data = correlation_matrix[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]


        plt.matshow(fit_data, vmin=-1, vmax=1, cmap='bwr')
        plt.colorbar()
        plt.gca().set_aspect('auto')
        plt.title("n_clusters = %i" % n_clusters)
    return new_dict, label_dict

def removeBaseline(trace, kernel_sigma=45, time_range=600, dt_step=2000):
    kernel = burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=time_range, dt_step=dt_step)
    bl = spsig.fftconvolve(trace, kernel, mode='same')

    return trace - bl


def processContinuousSignal(trace, substract_mean=True, cut_off_amp=30, baseline_sigma=45, baseline_time_range=600,
                            kernel_sigma=2, time_range=20,
                            dt_step=1 / 20000):
    if substract_mean:
        trace -= np.mean(trace)
    if cut_off_amp != 0:
        trace[(trace <= -cut_off_amp) | (trace >= cut_off_amp)] = 0

    trace = removeBaseline(trace, kernel_sigma=baseline_sigma, time_range=baseline_time_range, dt_step=dt_step)

    kernel = burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=time_range, dt_step=dt_step)

    return spsig.fftconvolve(trace, kernel, mode='same')


def processNerveSignal(trace, kernel_sigma=2, time_range=20, dt_step=1 / 20000):
    trace = np.abs(trace)
    kernel = burstUtils.generateGaussianKernel(kernel_sigma, time_range, dt_step)
    return spsig.fftconvolve(trace, kernel, mode='same')


def getDistMatrix(trace):
    return np.abs(trace[np.newaxis, :] - trace[:, np.newaxis])


def doubleCenterMatrix(matrix):
    return matrix - matrix.mean(axis=0)[:, np.newaxis] - matrix.mean(axis=1)[np.newaxis, :] + matrix.mean()


def getVariance(centered_matrix):
    return np.sqrt((centered_matrix ** 2).sum() / (centered_matrix.shape[0] ** 2))


def getCovariance(centered_matrix1, centered_matrix2):
    return np.sqrt((centered_matrix1 * centered_matrix2).sum() / (centered_matrix1.shape[0] ** 2))


def getDistanceCorrelation(trace1, trace2):
    dist_mat_1 = doubleCenterMatrix(getDistMatrix(trace1))
    dist_mat_2 = doubleCenterMatrix(getDistMatrix(trace2))
    dVar1 = getVariance(dist_mat_1)
    dVar2 = getVariance(dist_mat_2)
    dCov = getCovariance(dist_mat_1, dist_mat_2)
    return dCov / np.sqrt(dVar1 * dVar2)


def cosineSimilarity(trace1, trace2):
    return np.dot(trace1, trace2) / (np.linalg.norm(trace1) * np.linalg.norm(trace2))


def rasterPlot(spike_count_dict, generate_grid=True, color_dict=None, linelengths=1, linewidths=1):
    fig, ax = plt.subplots()
    i = 0
    for key, items in spike_count_dict.items():
        ax.eventplot(items[0][items[1].astype(bool)], colors=color_dict[key], linelengths=linelengths,
                     linewidths=linewidths, label=key, lineoffsets=i)
        i += 1

    fig.legend()
    if generate_grid:
        ax.grid(linestyle='dotted')

    return fig, ax


def plotSlidingWindowCorrelation(basename, De3, spike_count_dict, color_dict, time_vector, func=cosineSimilarity):
    fig, ax = plt.subplots()
    fig.suptitle("%s\nDE-3 is %i, %s" % (basename, De3, func.__name__))
    neurons = list(spike_count_dict)
    for j in range(len(neurons)):
        neuron_j = neurons[j]

        if neuron_j == De3:
            continue
        idxs, corr = pairwiseSlidingWindow(spike_count_dict[De3][1], spike_count_dict[neuron_j][1],
                                           func, step=40, window_size=300)

        if type(color_dict[neuron_j]) is list:
            c = color_dict[neuron_j][0]
        else:
            c = color_dict[neuron_j]
        ax.plot(time_vector[idxs], corr, marker='o', label=neuron_j, c=c)
        ax.set_ylim([-.1, 1])

    return fig, ax

def plotCrossCorrelation(binned_sfd, bin_step, De3, De3_counts, label_dict=None):
    fig, ax = plt.subplots(len(binned_sfd), 1, sharex=True)
    crosscorr_dict = {}
    i = 0
    size = len(binned_sfd[De3][1])
    mid_point = int(size / 2)

    excess = binned_sfd[list(binned_sfd)[0]][0].shape[0] - De3_counts.shape[0]
    time_shift = np.arange(0, (excess + 1) * bin_step, bin_step)
    time_shift -= np.median(time_shift)

    De3_counts -= De3_counts.mean()
    De3_counts /= np.std(De3_counts)

    for key, items in binned_sfd.items():
        if label_dict is not None:
            label = str(label_dict[key])
        elif key==De3:
            label = 'De3'
        else:
            label = str(key)

        time, counts = items
        counts -= counts.mean()
        counts /= np.std(counts)

        corr = correlate(counts, De3_counts, mode='valid')/De3_counts.shape[0]
        # corr = corr[int(size / 2) - 1:int(size * 1.5)]
        crosscorr_dict[key] = (time_shift, corr) #/ corr.max()

        idx_min = np.argmin(corr)
        idx_max = np.argmax(corr)

        shift_min = time_shift[idx_min]
        shift_max = time_shift[idx_max]

        # ax[i].scatter(time_shift, corr, s=1, c='k')
        # ax[i].plot(time_shift, corr / corr.max(), c='k', label=label)
        ax[i].plot(time_shift, corr , c='k', label=label)
        ax[i].axvline(0, c='g')
        ax[i].axvline(shift_min, c='b')
        ax[i].axvline(shift_max, c='r')

        ax[i].grid()
        ax[i].legend()
        # ax[i].set_ylim((-1, 1))
        burstUtils.removeTicksFromAxis(ax[i], 'x')

        # crosscorr_dict[key] = np.array((time[idxs], corr))
        i += 1

    burstUtils.showTicksFromAxis(ax[-1], 'x')

    # fig.subplots_adjust(wspace=0, hspace=0)

    return crosscorr_dict, fig, ax


def plotPCA(basename, data_array, bin_step, time_vector, window_size=300, n_components=3, joined_ax=None):
    pca_emb = PCA(n_components=3)
    rg = np.arange(data_array.shape[0] - window_size, step=5)
    explained_var_ratio = []
    for i in rg:
        pca_emb.fit(data_array[i:i + window_size])
        explained_var_ratio.append(pca_emb.explained_variance_ratio_)
    explained_var_ratio = np.cumsum(np.array(explained_var_ratio), axis=1).T

    fig, ax = plt.subplots()
    i = 1
    for row in explained_var_ratio:
        ax.plot((rg + int(window_size / 2)) * bin_step, row, label="%i comp" % i)
        i += 1
    if joined_ax is not None:
        ax.get_shared_x_axes().join(ax, joined_ax)
    fig.suptitle("%s PCA exaplined variance ratio,\n%i s window" % (basename, int(window_size * bin_step)))

    full_pca = pca_emb.fit_transform(data_array)
    fig1, ax1 = NLD.plot3Dscatter(full_pca)
    fig1.suptitle("%s full PCA" % basename)

    n_plots = n_components
    fig2, ax2 = plt.subplots(n_plots, 1, sharex=True)
    for i in range(n_plots):
        ax2[i].plot(time_vector[:-1], full_pca[:, i], color='k')
        ax2[i].set_xticks([])
        ax2[i].set_yticks([])

    ax2[0].get_shared_x_axes().join(ax2[0], ax)
    fig2.suptitle("%s full PCA" % basename)

    return (fig, ax), (fig1, ax1), (fig2, ax2)

# @jit
def getRandomCorrelationStats(trace1, trace2=None, time_step=None, corr_lengths=[60], num_samples=1000):
    if trace2 is None:
        trace2 = trace1

    avgd_corr = np.zeros(num_samples)
    corr = np.zeros(len(corr_lengths))
    min_idx = int(np.max(corr_lengths) / 2 / time_step)

    idxs = np.random.randint(min_idx, trace1.shape[0] - min_idx, size=(num_samples, 2))

    for i in range(idxs.shape[0]):
        idx1 = idxs[i, 0]
        idx2 = idxs[i, 1]
        j = 0
        for corr_time in corr_lengths:
            window_half_size = int(corr_time / time_step/2)

            corr[j] = np.corrcoef(trace1[int(idx1-window_half_size): int(idx1 + window_half_size)], trace2[int(idx2-window_half_size): int(idx2 + window_half_size)])[0, 1]
            j += 1
        avgd_corr[i] = np.mean(corr)


    return avgd_corr

# @jit(nopython=True)
def varCorrCoef(x, y=None):
    if y is None:
        return np.corrcoef(x, rowvar=False)

    return np.corrcoef(x,  rowvar=False)


# @jit(nopython=True)
def getPvalues(processed_spike_freq_array, corr_lengths, time_step, num_samples=1000, p_thres=(5, 95)):
    row_idx, col_idx = np.triu_indices(processed_spike_freq_array.shape[1], 1)
    i = 0
    p_vals = np.zeros((row_idx.shape[0], 2))
    for p1, p2 in zip(row_idx, col_idx):
        null_corrs = getRandomCorrelationStats(processed_spike_freq_array[:, p1], processed_spike_freq_array[:, p2], time_step, corr_lengths, num_samples)
        p_vals[i] = np.percentile(null_corrs[~np.isnan(null_corrs)], p_thres)
        i += 1
    return p_vals

# @jit(nopython=True)
def getFCM(processed_spike_freq_array, window_size, corr_step, corr_lengths, time_step, unweighted=True, num_samples=1000, p_thres=(5, 95)):
    fc_matrixes = multiUnitSlidingWindow(processed_spike_freq_array, window_size=window_size, step=corr_step)
    flat_idxs = getFlattenedUpperTriangIdxs(fc_matrixes.shape[1], 1)
    fc_matrixes = fc_matrixes.reshape(fc_matrixes.shape[0], fc_matrixes.shape[1] ** 2)[:, flat_idxs]

    fc_matrixes[np.isnan(fc_matrixes)] = 0.
    if unweighted:
        p_vals = getPvalues(processed_spike_freq_array, corr_lengths, time_step, num_samples, p_thres)
        for i in range(fc_matrixes.shape[1]):
            fc_matrixes[:, i][fc_matrixes[:,i] < p_vals[i,0]] = -1
            fc_matrixes[:, i][fc_matrixes[:, i] > p_vals[i, 1]] = 1
            fc_matrixes[:, i][(fc_matrixes[:, i] > p_vals[i, 0]) & (fc_matrixes[:, i] < p_vals[i, 1])] = 0

    return fc_matrixes

def getTraceCorrIdxs(trace_idx, num_traces):
    flat_mat_idxs  = np.arange(0, num_traces**2).reshape(num_traces, num_traces)
    corr_idxs =  np.append(flat_mat_idxs[:trace_idx, trace_idx], flat_mat_idxs[trace_idx, trace_idx+1:])
    flat_idxs = getFlattenedUpperTriangIdxs(num_traces, 1)
    return np.where(np.isin(flat_idxs, corr_idxs))[0]

def getSortedCorrelationIdxs(target_unit_idx, unit_list, processed_spike_freq_array, unweighted_FCM, comparison_interval):
    de3_idx = np.where(np.array(unit_list) == target_unit_idx)[0][0]
    corr_idxs = getTraceCorrIdxs(de3_idx, processed_spike_freq_array.shape[1])
    u_de3_corrs = unweighted_FCM[comparison_interval[0]:comparison_interval[1], corr_idxs]
    return corr_idxs, np.argsort(np.max(np.abs(u_de3_corrs), axis=0))[::-1]

def distanceClustering(FCM, diff_to_next, clust_dist, min_sample_time, time_step, avg_const_dist):
    '''

    :param FCM:
    :param diff_to_next:
    :param clust_dist:
    :param min_sample_time:
    :param time_step:
    :param comparison_interval:
    :return:
    '''

    i = int(diff_to_next / time_step)
    min_samples = int(min_sample_time / time_step)
    FC_dist = FCM[:-i] - FCM[i:]
    FC_dist = np.sqrt((FC_dist ** 2).sum(axis=1))

    close = FC_dist < clust_dist * avg_const_dist
    cluster_array = -np.ones(FCM.shape[0])
    j = 0
    for i in range(1, FC_dist.shape[0]):
        if not close[i - 1]:
            j += 1
        cluster_array[i] = j

    clusters, counts = np.unique(cluster_array, return_counts=True)
    bad_clusters = clusters[np.where(counts < min_samples)[0]]
    good_clusters = clusters[np.where(counts >= min_samples)[0]]

    j = 0
    new_cluster_array = np.zeros(cluster_array.shape[0], dtype=int)
    for cl in good_clusters:
        new_cluster_array[cluster_array == cl] = j
        j += 1

    new_cluster_array[np.in1d(cluster_array, bad_clusters)] = -1
    return new_cluster_array

