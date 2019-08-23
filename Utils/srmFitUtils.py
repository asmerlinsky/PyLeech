# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 23:29:39 2019

@author: Agustin Sanchez Merlinsky
"""

import os
import numpy as np
import PyLeech.Utils.srmSimUtils as srmSimUtils
from PyLeech.Utils.burstUtils import is_outlier
from scipy import linalg as linalg, optimize as optimize
import matplotlib.pyplot as plt
import winsound
import scipy.signal as spsig
import PyLeech.Utils.burstUtils as burstUtils
import time
import PyLeech.Utils.filterUtils as filterUtils


def getSpikeTimesDict(spike_freq_dict, neuron_list, outlier_threshold=3.5):
    times_dict = {}
    for key in neuron_list:
        times_dict[key] = spike_freq_dict[key][0][~is_outlier(spike_freq_dict[key][1], outlier_threshold)]
    return times_dict


def substractTimes(spike_times_dict, time):
    return {key: items - time for key, items in spike_times_dict.items()}


def beep():
    for i in range(4):
        winsound.Beep(1000, 200)


def generateNoiseFunction(amp, seed):
    def randomFunction(x):
        np.random.seed(seed)
        rd_vec = np.random.random(len(x))
        rd_vec = rd_vec - np.mean(rd_vec)
        return amp * (rd_vec / np.std(rd_vec))

    return randomFunction


#
# NegLogLikelihood(fit2.fit_array, fit2.input_mat, fit2.spike_count, fit2.dt, fit2.scaled_fit_size,
#                            fit2.k_length, int(fit2.refnconn_length / fit2.no_neurons), 0.5)
def getKFilterFromNeuronFilter(neuron_filter, fit_k_len):
    return neuron_filter[1:fit_k_len + 1]


def getConnFilterFromNeuronFilter(neuron_filter, fit_k_len, fit_ref_len, neuron):
    first_idx = 1 + fit_k_len
    return neuron_filter[first_idx + fit_ref_len * neuron: first_idx + fit_ref_len * (neuron + 1)]


def padArray(input_stim, filter_len):
    if len(input_stim.shape) > 1:
        return np.hstack((np.zeros((input_stim.shape[0], filter_len)), input_stim))
    else:
        return np.hstack((np.zeros(filter_len), input_stim))


def NegLogLikelihood(fit_array, input_stim, spike_matrix, dt, scaling_length, fit_k_len, full_k_len, fit_conn_len,
                     full_conn_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    NegLogLikelihood = 0

    for i in range(no_neurons):
        neuron_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))

        u = spsig.fftconvolve(input_stim * dt, getKFilterFromNeuronFilter(neuron_filter, full_k_len))[
            :input_stim.shape[0] - 1]

        for n in range(no_neurons):
            u += spsig.fftconvolve(spike_matrix[n],
                                   getConnFilterFromNeuronFilter(neuron_filter, full_k_len, full_conn_len, n))[
                 :input_stim.shape[0] - 1]

        u += neuron_filter[0]

        NegLogLikelihood -= np.dot(spike_matrix[i, 1:input_stim.shape[0]], u) - np.exp(u).sum() * dt
    return NegLogLikelihood


# NegLogLikelihood(fit1.fit_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size, fit1.k_length,
#                     fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
#                     fit1.single_conn_length * fit1.scaled_fit_size, 0)

def gradNegLogLikelihood(fit_array, input_stim, spike_matrix, dt, scaling_length, fit_k_len, full_k_len, fit_conn_len,
                         full_conn_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    grad_neg_log_likelihood = np.zeros(fit_array.shape[0])

    padded_stim = padArray(input_stim, full_k_len - 1)
    padded_spike_matrix = padArray(spike_matrix, full_conn_len - 1)
    for i in range(no_neurons):
        actual_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))

        u = spsig.fftconvolve(input_stim, getKFilterFromNeuronFilter(actual_filter, full_k_len))[
            :input_stim.shape[0] - 1] * dt

        for n in range(no_neurons):
            u += spsig.fftconvolve(spike_matrix[n],
                                   getConnFilterFromNeuronFilter(actual_filter, full_k_len, full_conn_len, n))[
                 :input_stim.shape[0] - 1]
        u += actual_filter[0]

        spikes_minus_exp = spike_matrix[i, 1:input_stim.shape[0]] - np.exp(u) * dt
        spikes_minus_exp = spikes_minus_exp[::-1]
        grad_neg_log_likelihood[i * neuron_filters[i].shape[0]] = -spikes_minus_exp.sum()

        k_grad_Neg_Log_Likelihood = spsig.correlate(padded_stim[::-1] * dt, spikes_minus_exp, mode='valid')[1:]

        k_start_idx = i * neuron_filters[i].shape[0] + 1

        grad_neg_log_likelihood[k_start_idx:k_start_idx + fit_k_len] = \
            - k_grad_Neg_Log_Likelihood.reshape(int(k_grad_Neg_Log_Likelihood.shape[0] / scaling_length),
                                                scaling_length).sum(axis=1)

        for n in range(no_neurons):
            conn_grad_Neg_Log_Likelihood = spsig.correlate(padded_spike_matrix[n][::-1], spikes_minus_exp,
                                                           mode='valid')[1:]

            conn_start_idx = k_start_idx + fit_k_len + n * fit_conn_len
            grad_neg_log_likelihood[conn_start_idx: conn_start_idx + fit_conn_len] = \
                - conn_grad_Neg_Log_Likelihood.reshape(int(conn_grad_Neg_Log_Likelihood.shape[0] / scaling_length),
                                                       scaling_length).sum(axis=1)

    return grad_neg_log_likelihood


#
# sec = gradNegLogLikelihood(fit1.fit_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size, fit1.k_length,
#                     fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
#                     fit1.single_conn_length * fit1.scaled_fit_size, 0)


def penalizedNegLogLikelihood(fit_array, input_stim, spike_matrix, dt, scaling_length, fit_k_len, full_k_len,
                              fit_conn_len,
                              full_conn_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    neg_log_likelihood = 0

    for i in range(no_neurons):
        neuron_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))

        penalty = np.sum(np.power(np.diff(neuron_filters[i][1:fit_k_len + 1]), 2))

        u = spsig.fftconvolve(input_stim * dt, getKFilterFromNeuronFilter(neuron_filter, full_k_len))[
            :input_stim.shape[0] - 1]

        for n in range(no_neurons):
            penalty += np.sum(
                np.power(
                    np.diff(getConnFilterFromNeuronFilter(neuron_filters[i], fit_k_len, fit_conn_len, n)
                            ), 2
                )
            )

            u += spsig.fftconvolve(spike_matrix[n],
                                   getConnFilterFromNeuronFilter(neuron_filter, full_k_len, full_conn_len, n))[
                 :input_stim.shape[0] - 1]

        u += neuron_filter[0]

        neg_log_likelihood -= np.dot(spike_matrix[i, 1:input_stim.shape[0]], u) - np.exp(
            u).sum() * dt - penalty_param * penalty
    return neg_log_likelihood


def gradPenalizedNegLogLikelihood(fit_array, input_stim, spike_matrix, dt, scaling_length, fit_k_len, full_k_len,
                                  fit_conn_len,
                                  full_conn_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    grad_Neg_Log_Likelihood = np.zeros(fit_array.shape[0])
    neuron_filter_length = neuron_filters[0].shape[0]
    padded_stim = padArray(input_stim, full_k_len - 1)

    padded_spike_matrix = padArray(spike_matrix, full_conn_len - 1)

    for i in range(no_neurons):
        actual_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))

        u = spsig.fftconvolve(input_stim, getKFilterFromNeuronFilter(actual_filter, full_k_len))[
            :input_stim.shape[0] - 1] * dt
        for n in range(no_neurons):
            u += spsig.fftconvolve(spike_matrix[n],
                                   getConnFilterFromNeuronFilter(actual_filter, full_k_len, full_conn_len, n))[
                 :input_stim.shape[0] - 1]

        u += actual_filter[0]

        penalty_grad = np.zeros(fit_k_len)

        penalty_grad[:-1] += 2 * penalty_param * np.diff(neuron_filters[i][1:fit_k_len + 1])
        penalty_grad[1:] -= 2 * penalty_param * np.diff(neuron_filters[i][1:fit_k_len + 1])

        grad_Neg_Log_Likelihood[1 + neuron_filter_length * i:1 + neuron_filter_length * i + fit_k_len] -= penalty_grad

        spikes_minus_exp = spike_matrix[i, 1:input_stim.shape[0]] - np.exp(u) * dt
        spikes_minus_exp = spikes_minus_exp[::-1]

        grad_Neg_Log_Likelihood[i * neuron_filters[i].shape[0]] -= spikes_minus_exp.sum()

        k_grad_Neg_Log_Likelihood = spsig.correlate(padded_stim[::-1] * dt, spikes_minus_exp, mode='valid')[1:]

        k_start_idx = i * neuron_filters[i].shape[0] + 1

        grad_Neg_Log_Likelihood[k_start_idx:k_start_idx + fit_k_len] -= \
            k_grad_Neg_Log_Likelihood.reshape(int(k_grad_Neg_Log_Likelihood.shape[0] / scaling_length),
                                              scaling_length).sum(axis=1)

        for n in range(no_neurons):
            conn_grad_Neg_Log_Likelihood = spsig.correlate(padded_spike_matrix[n][::-1], spikes_minus_exp,
                                                           mode='valid')[1:]

            conn_start_idx = k_start_idx + fit_k_len + n * fit_conn_len

            grad_Neg_Log_Likelihood[conn_start_idx: conn_start_idx + fit_conn_len] -= \
                conn_grad_Neg_Log_Likelihood.reshape(int(conn_grad_Neg_Log_Likelihood.shape[0] / scaling_length),
                                                     scaling_length).sum(axis=1)

            penalty_grad = np.zeros(fit_conn_len)
            diff = np.diff(getConnFilterFromNeuronFilter(neuron_filters[i], fit_k_len, fit_conn_len, n))

            penalty_grad[:-1] += 2 * penalty_param * diff
            penalty_grad[1:] -= 2 * penalty_param * diff

            grad_Neg_Log_Likelihood[
            1 + neuron_filter_length * i + fit_k_len + fit_conn_len * n:1 + neuron_filter_length * i + fit_k_len + fit_conn_len * (
                    n + 1)] -= penalty_grad

    return grad_Neg_Log_Likelihood


def MATNegLogLikelihood(fit_array, input_mat, spike_matrix, dt, scaling_length, k_len, ref_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    NegLogLikelihood = 0
    for i in range(no_neurons):
        actual_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))
        x_dot_k = np.sum(input_mat * actual_filter, axis=1)

        NegLogLikelihood -= np.dot(spike_matrix[i], x_dot_k) - np.exp(x_dot_k).sum() * dt

    return NegLogLikelihood


#
# NegLogLikelihood(fit2.fit_array, fit2.input_mat, fit2.spike_count, fit2.dt, fit2.scaled_fit_size,
#                            fit2.k_length, int(fit2.refnconn_length / fit2.no_neurons), 0.5)


def MATgradNegLogLikelihood(fit_array, input_mat, spike_matrix, dt, scaling_length, k_len, ref_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    gradNegLogLikelihood = np.zeros(fit_array.shape[0])
    neuron_filter_length = int(fit_array.shape[0] / no_neurons)

    for i in range(no_neurons):
        actual_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))
        x_dot_k = np.sum(input_mat * actual_filter, axis=1)

        partial_mat = spike_matrix[i] - np.exp(x_dot_k) * dt

        neuron_grad = np.sum(input_mat.T * partial_mat, axis=1)

        gradNegLogLikelihood[i * neuron_filter_length] = -neuron_grad[0]
        gradNegLogLikelihood[
        i * neuron_filter_length + 1: (i + 1) * neuron_filter_length] = -neuron_grad[1:].reshape(
            int(neuron_filter_length - 1 / scaling_length), scaling_length).sum(axis=1)

    return gradNegLogLikelihood


def runCrawlingFitter(time_delta, input_stim, spike_times_dict, k_duration, refnconn_duration, dim_red, filename,
                      first_idx, last_idx, func_grad_list, fs, dt, penalty, low_pass_threshold):

    input_stim = filterUtils.runButterFilter(input_stim, low_pass_threshold, butt_order=4, sampling_rate=fs)

    step = int(fs * dt)

    print("\n\ndt: %f, penalty: %f" % (dt, penalty))

    re_time = np.arange(0, round(time_delta, 4), dt)
    input_stim = input_stim[first_idx:last_idx - 1:step]

    fitter = SRMFitter(re_time, input_stim, spike_times_dict=spike_times_dict, k_duration=k_duration,
                       refnconn_duration=refnconn_duration, dim_red=dim_red, penalty=penalty,
                       filename=filename)

    if func_grad_list is None:
        func_grad_list = [penalizedNegLogLikelihood, gradPenalizedNegLogLikelihood]

    t_start = time.time()
    fitter.minimizeNegLogLikelihood(10 ** 5, func_grad_list, verbose=False, talk=False)
    timing = (time.time() - t_start) / 60
    print(timing)

    return fitter, timing, fitter.optimization.nit


class SRMFitter:
    """
    K_filter_size includes u_rest - V
    
    Changes must be made to NegLogLikelihood function if I wanted to change this
    ALso. the system has a single shared stimulus
    urest-V is included in k_filter as the first element
    spike_dict must be either neuron : list_of_spike_times or neuron : [list_of_spike_times, list_of_inst_freq]
    """

    def __init__(self, time_steps, input_stim, spike_times_dict, k_duration, refnconn_duration, dim_red=5, penalty=0,
                 filename=None):
        self.filename = os.path.basename(filename)
        self.dt = time_steps[1] - time_steps[0]
        self.scaled_fit_size = dim_red
        self.fit_length = len(time_steps)

        self.no_neurons = len(spike_times_dict)
        self.penalty_param = penalty
        self.time_steps = time_steps
        self.input_stim = input_stim

        fit_dt = self.dt * self.scaled_fit_size

        self.k_length = int(k_duration / fit_dt)
        self.single_conn_length = int(refnconn_duration / fit_dt)
        self.refnconn_length = self.no_neurons * self.single_conn_length
        self.spike_count = np.zeros((self.no_neurons, self.fit_length))

        i = 0
        for neuron, times in spike_times_dict.items():

            times = times[(times > 0) & (times < self.time_steps[-1])]
            # print(neuron, times)
            digitalization = np.digitize(times, self.time_steps + self.dt / 2)
            #            print(digitalization.shape, np.unique(digitalization).shape)
            # assert uid.shape[0] == digitalization.shape[0], " I found more one spike per bin in neuron %i" % neuron
            bin_count = np.bincount(digitalization, minlength=len(self.time_steps))
            if bin_count.shape[0] > 0:
                self.spike_count[i] = bin_count
            i += 1

        filters_to_fit_size = self.no_neurons * (self.k_length + self.refnconn_length + 1)

        self.fit_array = np.zeros(filters_to_fit_size)

    def minimizeNegLogLikelihood(self, n_iter=100, func_jac_list=[NegLogLikelihood, gradNegLogLikelihood],
                                 use_grad=True, verbose=False, penalty_param=None, talk=True):

        if penalty_param is None:
            penalty_param = self.penalty_param
        elif penalty_param != self.penalty_param:
            print("Warning. the assigned penalty is not the same as the one in self.penalty")
        b0_bound = [(-np.inf, 0)]
        bounds = b0_bound + [(-np.inf, np.inf)] * int(self.fit_array.shape[0] / self.no_neurons - 1)
        bounds *= self.no_neurons

        args = (self.input_stim, self.spike_count, self.dt, self.scaled_fit_size, self.k_length,
                self.k_length * self.scaled_fit_size, self.single_conn_length,
                self.single_conn_length * self.scaled_fit_size,
                penalty_param
                )
        if use_grad:
            jac = func_jac_list[1]
        else:
            jac = None
        self.optimization = optimize.minimize(func_jac_list[0], self.fit_array,
                                              args=args,
                                              jac=jac,
                                              method='L-BFGS-B',
                                              options={'maxiter': n_iter, 'gtol': 1e-6, 'maxfun': 10 ** 5,
                                                       'disp': verbose},
                                              bounds=bounds)
        self.fit_array = self.optimization.x
        if not verbose:
            print(self.optimization.message)

        if talk:
            beep()

    def plotFitArray(self, separate_plots=True):

        if separate_plots:
            fig, ax = plt.subplots(self.no_neurons, 1, sharex=True)
        else:
            fig, ax = plt.subplots(sharex=True)
        fig.suptitle(os.path.basename(self.filename).split('.')[0] + '\ndt: ' + str(self.dt) + ", penalty: " + str(
            self.penalty_param))
        single_conn_len = int(self.refnconn_length / self.no_neurons)
        neuron_filters = np.split(self.fit_array, self.no_neurons)
        for i in range(len(neuron_filters)):
            neuron_filters[i] = np.concatenate(
                ([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], self.scaled_fit_size)))
            if separate_plots:
                ax[i].plot(neuron_filters[i], color='k')
                ax[i].axvline(1 + self.k_length * self.scaled_fit_size, color='r')
                for j in range(1, self.no_neurons):
                    ax[i].axvline(1 + self.k_length * self.scaled_fit_size + single_conn_len * self.scaled_fit_size * j,
                                  color='b')
                ax[i].grid()
                burstUtils.removeTicksFromAxis(ax[i], 'x')

        if separate_plots:
            burstUtils.showTicksFromAxis(ax[-1], 'x')
            return

        neuron_filters = np.concatenate(neuron_filters)

        ax.plot(neuron_filters)

        single_neuron_arr_len = neuron_filters.shape[0] / self.no_neurons

        for i in np.arange(1, neuron_filters.shape[0], single_neuron_arr_len):
            ax.axvline(i, color='r')
        for j in range(self.no_neurons):

            for i in np.arange(1 + (self.k_length + single_conn_len * j) * self.scaled_fit_size,
                               neuron_filters.shape[0], single_neuron_arr_len):
                ax.axvline(i, color='b')


class MATSRMFitter:
    """
    K_filter_size includes u_rest - V
    The fitter assumes no more one spike per bin (per neuron)
    That is that dt should be shorter than the refractory tau
    Changes must be made to NegLogLikelihood function if I wanted to change this
    ALso. the system has a single shared stimulus
    urest-V is included in k_filter as the first element
    spike_dict must be either neuron : list_of_spike_times or neuron : [list_of_spike_times, list_of_inst_freq]
    """

    def __init__(self, time_steps, input_stim, spike_times_dict, k_duration, refnconn_duration, dim_red=5):
        self.dt = time_steps[1] - time_steps[0]
        self.scaled_fit_size = dim_red
        self.fit_length = len(time_steps)

        self.no_neurons = len(spike_times_dict)

        self.time_steps = time_steps

        fit_dt = self.dt * self.scaled_fit_size

        self.k_length = int(k_duration / fit_dt)
        self.single_conn_length = int(refnconn_duration / fit_dt)
        self.refnconn_length = self.no_neurons * self.single_conn_length

        self.input_mat = np.ones((self.fit_length - 1, 1))

        padding = np.zeros(self.k_length * self.scaled_fit_size - 1, input_stim.dtype)

        first_col = np.r_[input_stim, padding]
        first_row = np.r_[input_stim[0], padding]
        toep = linalg.toeplitz(first_col, first_row)[:self.fit_length - 1] * self.dt

        self.input_mat = np.hstack((self.input_mat, toep))

        padding = np.zeros(self.single_conn_length * self.scaled_fit_size - 1, input_stim.dtype)

        self.spike_count = np.zeros((self.no_neurons, self.fit_length))
        i = 0
        for neuron, times in spike_times_dict.items():
            digitalization = np.digitize(times, self.time_steps - self.dt)
            uid = np.unique(digitalization)

            assert uid.shape[0] == digitalization.shape[0], " I found more one spike per bin in neuron %i" % neuron
            self.spike_count[i, uid] = 1

            first_col = np.r_[self.spike_count[i], padding]
            first_row = np.r_[self.spike_count[i, 0], padding]
            spike_toeplitz = linalg.toeplitz(first_col, first_row)[:self.fit_length - 1]

            self.input_mat = np.hstack((self.input_mat, spike_toeplitz))
            i += 1

        self.spike_count = self.spike_count[:, 1:]
        filters_to_fit_size = self.no_neurons * (self.k_length + self.refnconn_length + 1)

        self.fit_array = np.zeros(filters_to_fit_size)

    def minimizeNegLogLikelihood(self, n_iter=100, func_jac_list=[NegLogLikelihood, gradNegLogLikelihood],
                                 use_grad=True, verbose=False, penalty_param=0, talk=True):
        args = (self.input_mat, self.spike_count, self.dt, self.scaled_fit_size, self.k_length,
                int(self.refnconn_length / self.no_neurons), penalty_param)

        # print(func_jac_list[0].__name__, func_jac_list[1].__name__)
        if use_grad:
            jac = func_jac_list[1]
        else:
            jac = None
        self.optimization = optimize.minimize(func_jac_list[0], self.fit_array,
                                              args=args,
                                              jac=jac,
                                              method='L-BFGS-B',
                                              options={'maxiter': n_iter, 'gtol': 1e-6, 'disp': verbose})
        self.fit_array = self.optimization.x
        if not verbose:
            print(self.optimization.message)
        if talk:
            beep()

    def plotFitArray(self, separate_plots=False):

        if separate_plots:
            fig, ax = plt.subplots(self.no_neurons, 1)
        else:
            fig, ax = plt.subplots()

        single_conn_len = int(self.refnconn_length / self.no_neurons)
        neuron_filters = np.split(self.fit_array, self.no_neurons)
        for i in range(len(neuron_filters)):
            neuron_filters[i] = np.concatenate(
                ([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], self.scaled_fit_size)))
            if separate_plots:
                ax[i].plot(neuron_filters[i], color='k')
                ax[i].axvline(1 + self.k_length * self.scaled_fit_size, color='r')
                for j in range(1, self.no_neurons):
                    ax[i].axvline(1 + self.k_length * self.scaled_fit_size + single_conn_len * self.scaled_fit_size * j,
                                  color='b')

        if separate_plots: return

        neuron_filters = np.concatenate(neuron_filters)

        ax.plot(neuron_filters)

        single_neuron_arr_len = neuron_filters.shape[0] / self.no_neurons

        for i in np.arange(1, neuron_filters.shape[0], single_neuron_arr_len):
            ax.axvline(i, color='r')
        for j in range(self.no_neurons):

            for i in np.arange(1 + (self.k_length + single_conn_len * j) * self.scaled_fit_size,
                               neuron_filters.shape[0], single_neuron_arr_len):
                ax.axvline(i, color='b')


if __name__ == '__main__':
    k_tau = .1
    k0 = 15
    k_filter1 = lambda x: k0 * np.exp(-x / k_tau)
    k_filter2 = lambda x: -k0 * np.exp(-x / k_tau)

    ref_tau = .01
    ref_0 = -80
    ref_filter = lambda x: ref_0 * np.exp(-x / ref_tau)

    ex_tau = .5
    conn_0 = 2
    exc_conn_filter = lambda x: conn_0 * np.exp(-x / ex_tau)

    inh_tau = .05
    conn_1 = -5
    inh_conn_filter = lambda x: conn_1 * np.exp(-x / inh_tau)
    # step1 = generateStepFunction(period=5, amplitude=2, duty_cycle=0.2)
    # step2 =
    # times = np.arange(0, 15, .1)
    # plt.plot(times, step(times))
    no_current = lambda x: np.array([0] * len(x))
    f = 1 / 20

    A = 3
    I0 = 0
    I1 = 1
    sincurrent1 = lambda x: A * np.power(np.sin(2 * np.pi * f * x), 2) - A / 2
    sincurrent2 = lambda x: A * np.power(np.sin(2 * np.pi * f * x + np.pi / 2), 2) + I0

    noise = generateNoiseFunction(4, 1)

    sinplusnoise1 = lambda x: sincurrent1(x) + generateNoiseFunction(.01, 1)(x)

    u_rest = -6
    neurons = {
        'n1': [u_rest, k_filter1, ref_filter, inh_conn_filter, sinplusnoise1],
        'n2': [u_rest, k_filter2, ref_filter, inh_conn_filter, sinplusnoise1]
    }
    # for i in range(1):
    #     neurons['n'+str(i)] = [u_rest, k_filter, ref_filter, inh_conn_filter, no_current]
    # for i in range(7,10):
    #     neurons['n' + str(i)] = [u_rest, k_filter, ref_filter, exc_conn_filter, no_current]
    #
    # neurons['n9'][-1] = step

    mn = srmSimUtils.SRMMultiNeuronSimulator(dt=1 / 100, sim_length=60 * 1, no_sim=1, k_length=1, ref_length=1,
                                             **neurons)
    mn.runSimulation()
    mn.plotSimulation()

    spike_freq_dict = mn.getTrialSpikeFreqDict()

    fit1 = SRMFitter(mn.time_steps, mn.input_stim[0], spike_freq_dict, k_duration=1, refnconn_duration=0.5, dim_red=5)
    penfit = SRMFitter(mn.time_steps, mn.input_stim[0], spike_freq_dict, k_duration=1, refnconn_duration=0.5,
                       dim_red=50)
    # fit2 = MATSRMFitter(mn.time_steps, mn.input_stim[0], spike_freq_dict, k_duration=1, refnconn_duration=0.5, dim_red=50)
    # fit2.minimizeNegLogLikelihood(10, use_grad=True,
    #                               func_jac_list=[MATNegLogLikelihood, MATgradNegLogLikelihood],
    #                               verbose=True, penalty_param=.1)
    fit1.minimizeNegLogLikelihood(4000, use_grad=True, func_jac_list=[NegLogLikelihood, gradNegLogLikelihood],
                                  verbose=False)
    penfit.minimizeNegLogLikelihood(4000, use_grad=True,
                                    func_jac_list=[penalizedNegLogLikelihood, gradPenalizedNegLogLikelihood],
                                    verbose=True, penalty_param=.1)

    # print(NegLogLikelihood(fit1.fit_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
    #                        fit1.k_length,
    #                        fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
    #                        fit1.single_conn_length * fit1.scaled_fit_size, 0),
    #       penalizedNegLogLikelihood(fit1.fit_array, penfit.input_stim, penfit.spike_count, penfit.dt, penfit.scaled_fit_size,
    #                        penfit.k_length,
    #                        penfit.k_length * penfit.scaled_fit_size, penfit.single_conn_length,
    #                        penfit.single_conn_length * penfit.scaled_fit_size, 0)
    #       )

    fit1.plotFitArray()
    penfit.plotFitArray()
    plt.figure()
    # plt.plot(fit1.fit_array, label='fit1')
    # plt.plot(penfit.fit_array, label='penfit')
    plt.plot(
        gradNegLogLikelihood(penfit.fit_array, penfit.input_stim, penfit.spike_count, penfit.dt, penfit.scaled_fit_size,
                             penfit.k_length,
                             penfit.k_length * penfit.scaled_fit_size, penfit.single_conn_length,
                             penfit.single_conn_length * penfit.scaled_fit_size, 0), label='gNLLH', linestyle='--')
    plt.plot(
        gradPenalizedNegLogLikelihood(penfit.fit_array, penfit.input_stim, penfit.spike_count, penfit.dt,
                                      penfit.scaled_fit_size,
                                      penfit.k_length,
                                      penfit.k_length * penfit.scaled_fit_size, penfit.single_conn_length,
                                      penfit.single_conn_length * penfit.scaled_fit_size, 100), label='PgNLLH')
    plt.legend()
