# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:45:12 2018

@author: Agustin Sanchez Merlinsky
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig


def generateStepFunction(period, amplitude=1, duty_cycle=0.5):
    return np.vectorize(lambda x: 0 if (((x % period) / period) < (1 - duty_cycle)) else amplitude)


class SRMSingleNeuronSimulator():

    def __init__(self, dt=0.0001, sim_length=15, no_sim=1, beta=2, u_rest_minus_V=-5, k_length=None, k_func=None,
                 ref_length=None, ref_func=None, current_function=None, test=False):

        if test:
            self.testSimulator()
            return
        self.dt = dt
        self.sim_length = sim_length
        self.time_steps = np.arange(0, self.sim_length, self.dt)

        self.no_sim = no_sim
        self.k_length = int(k_length / dt)
        self.generate_k_filter(k_func)

        self.ref_length = int(ref_length / dt)
        self.generate_ref_filter(ref_func)
        self.generate_input_current(current_function)

        self.u_sim = np.zeros(int(sim_length / dt))
        self.u_sim[:] = u_rest_minus_V
        self.firing_prob = np.zeros(int(sim_length / dt))
        self.spike_count = np.zeros((no_sim, int(sim_length / dt)), dtype=int)

        self.beta = beta

    def loadTestParams(self):
        k_tau = .5
        k0 = 20
        k_filter = lambda x: k0 * np.exp(-x / k_tau)

        ref_tau = .05
        ref_0 = 80
        ref_filter = lambda x: -ref_0 * np.exp(-x / ref_tau)

        stim = generateStepFunction(4, 5, 0.25)

        srm = self.__init__(dt=0.0001, sim_length=15, no_sim=1, beta=2, u_rest_minus_V=-5,
                            k_length=4, k_func=k_filter, ref_length=2, ref_func=ref_filter,
                            current_function=stim)

    def generate_k_filter(self, k_func):
        self.k_filter = k_func(self.time_steps[:self.k_length])

    def generate_ref_filter(self, ref_func):
        self.ref_filter = ref_func(self.time_steps[:self.ref_length])

    def generate_input_current(self, current_function):
        self.input_current = current_function(self.time_steps)

    def runSimulation(self):
        self.k_conv_i = spsig.fftconvolve(self.input_current, self.k_filter) * self.dt

        for trial in range(self.no_sim):
            self.u_sim[:] = self.u_sim[0]
            self.firing_prob[:] = 0
            for i in range(1, len(self.time_steps)):
                if i < self.ref_length:
                    S = self.spike_count[trial, :i][::-1]
                    u_s = np.dot(S, self.ref_filter[:i])
                else:
                    S = self.spike_count[trial, i - self.ref_length:i][::-1]
                    u_s = np.dot(S, self.ref_filter)

                self.u_sim[i] += self.k_conv_i[i] + u_s
                self.firing_prob[i] = np.exp(self.beta * self.u_sim[i])

                if self.firing_prob[i] >= 1:
                    self.spike_count[trial, i] = 1
                else:
                    self.spike_count[trial, i] = np.random.choice([1, 0],
                                                                  p=[self.firing_prob[i], 1 - self.firing_prob[i]])

    def testSimulator(self):
        k_tau = .5
        k0 = 20
        k_filter = lambda x: k0 * np.exp(-x / k_tau)

        ref_tau = .05
        ref_0 = 80
        ref_filter = lambda x: -ref_0 * np.exp(-x / ref_tau)
        rg = np.arange(0, 5, 0.1)
        plt.plot(rg, k_filter(rg))
        stim = generateStepFunction(4, 5, 0.25)

        self.__init__(dt=0.0001, sim_length=15, no_sim=1, beta=2, u_rest_minus_V=-5,
                      k_length=4, k_func=k_filter, ref_length=2, ref_func=ref_filter,
                      current_function=stim)
        self.runSimulation()

        fig, ax = plt.subplots(3, 1, sharex=True)

        ax[0].plot(self.time_steps, self.input_current)
        ax[0].set_title('Input Current')

        ax[1].set_title('potential')
        ax[1].plot(self.time_steps, self.u_sim, color='k')
        for i in self.time_steps[self.spike_count[-1, :].astype(bool)]:
            ax[1].axvline(i, color='r')
        ax[1].set_ylim([self.u_sim[0], 0])

        ax[2].set_title('firing prob')
        ax[2].plot(self.time_steps, self.firing_prob, color='k')

        plt.figure()
        plt.plot(self.time_steps, self.input_current)
        # plt.plot(self.time_steps, self.k_conv_i)


class SRMMultiNeuronSimulator():
    """
    kwargs has to be a dict pointing to a list with [u0, k_filt, ref_filt, connect_filt, input_stim]:
    n1: [u0, k_fun, ref_fun, connect_fun, input_fun]
    n2: [u0, k_fun, ref_fun, connect_fun, input_fun]
    .
    .
    .
    
    """

    def __init__(self, dt=0.0001, sim_length=15, no_sim=1, k_length=2, ref_length=2, **kwargs):

        self.dt = dt
        self.sim_length = sim_length
        self.time_steps = np.arange(0, sim_length, self.dt)
        self.sim_length = int(sim_length / dt)
        self.no_neurons = len(kwargs)

        self.no_sim = no_sim

        self.k_length = int(k_length / dt)
        self.ref_length = int(ref_length / dt)

        data_size = (self.no_neurons, int(sim_length / dt))
        self.data_size = data_size
        self.u_sim = np.zeros(data_size)

        self.input_stim = np.zeros(data_size)
        self.firing_prob = np.zeros(data_size)

        k_filter_size = (self.no_neurons, int(k_length / dt))
        self.k_filters = np.zeros(k_filter_size)

        ref_filter_size = (self.no_neurons, int(ref_length / dt))
        self.ref_filters = np.zeros(ref_filter_size)

        connectivity_matrix_size = (self.no_neurons, int(self.no_neurons * ref_length / dt))
        self.connectivity_matrix = np.zeros(connectivity_matrix_size)

        self.spike_count = np.zeros((no_sim, self.no_neurons, int(sim_length / dt)), dtype=int)
        i = 0
        connect_funcs = []
        ref_funcs = []
        for key, items in kwargs.items():
            self.u_sim[i, 0] = items[0]
            self.k_filters[i, :] = items[1](self.time_steps[:self.k_length])
            ref_funcs.append(items[2])
            connect_funcs.append(items[3])
            self.input_stim[i, :] = items[4](self.time_steps)
            i += 1

        for i in range(len(connect_funcs)):
            for j in range(len(connect_funcs)):
                if j != i:
                    self.connectivity_matrix[i, j * self.ref_length:(j + 1) * self.ref_length] = connect_funcs[j](
                        self.time_steps[:self.ref_length])
                else:
                    self.connectivity_matrix[i, j * self.ref_length:(j + 1) * self.ref_length] = ref_funcs[j](
                        self.time_steps[:self.ref_length])

    def runSimulation(self):
        self.k_conv_stim = np.zeros(self.data_size)
        for i in range(self.input_stim.shape[0]):
            self.k_conv_stim[i] = spsig.fftconvolve(self.input_stim[i], self.k_filters[i])[
                                  :len(self.time_steps)] * self.dt

        for trial in range(self.no_sim):

            self.u_sim[:] = np.repeat(self.u_sim[:, 0], self.sim_length).reshape(self.no_neurons, self.sim_length)
            self.firing_prob[:] = 0
            for i in range(1, len(self.time_steps)):
                if self.time_steps[i] % 60 == 0.:
                    print('Simulated %i min' % (self.time_steps[i]/60))
                for j in range(self.no_neurons):
                    neuron_ref_mat = self.connectivity_matrix[j].reshape(self.no_neurons, self.ref_length)

                    if i < self.ref_length:
                        S = self.spike_count[trial, :, :i][:, ::-1]

                        u_s = np.diag(S @ neuron_ref_mat[:, :i].T).sum()

                    else:
                        S = self.spike_count[trial, :, i - self.ref_length:i][:, ::-1]
                        u_s = np.diag(S @ neuron_ref_mat.T).sum()

                    self.u_sim[j, i] += self.k_conv_stim[j, i] + u_s
                    self.firing_prob[j, i] = np.exp(self.u_sim[j, i])

                    if self.firing_prob[j, i] >= 1:
                        self.spike_count[trial, j, i] = 1
                    else:
                        self.spike_count[trial, j, i] = np.random.choice([1, 0],
                                                                         p=[self.firing_prob[j, i],
                                                                            1 - self.firing_prob[j, i]])

    def plotSimulation(self):

        ax = None
        for j in range(self.no_neurons):
            fig = plt.figure()

            try:
                ax = plt.subplot(3, 1, 1, sharex=ax)
            except NameError as e:
                ax = plt.subplot(3, 1, 1)

            ax.plot(self.time_steps, self.input_stim[j])
            ax.set_title('Input Current')
            ax.grid()

            ax = plt.subplot(3, 1, 2, sharex=ax)
            ax.set_title('potential')
            ax.plot(self.time_steps, self.u_sim[j], color='k')
            for i in self.time_steps[self.spike_count[-1, j, :].astype(bool)]:
                ax.axvline(i, color='r')
            ax.set_ylim([4 * self.u_sim[j, 0], 0])
            ax.grid()

            ax = plt.subplot(3, 1, 3, sharex=ax)
            ax.set_title('firing prob')
            ax.plot(self.time_steps, self.firing_prob[j], color='k')
            ax.grid()

    def getTrialSpikeFreqDict(self, cut_time=None):
        sfd = {}
        for i in range(self.no_neurons):
            sfd[i] = self.time_steps[self.spike_count[0, i].astype(bool)]
            if cut_time is not None:
                sfd[i] = sfd[i][sfd[i]<cut_time]
        return sfd


if __name__ == "__main__":
    """
    [u0, k_filt, ref_filt, connect_filt, input_stim]
    """

    k_tau = .5
    k0 = 20
    k_filter1 = lambda x: k0 * np.exp(-x / k_tau)
    k_filter2 = lambda x: -k0 * np.exp(-x / k_tau)

    ref_tau = .05
    ref_0 = -80
    ref_filter = lambda x: ref_0 * np.exp(-x / ref_tau)


    ex_tau = .1
    conn_0 = 2
    exc_conn_filter = lambda x: conn_0 * np.exp(-x / ex_tau)

    inh_tau = .5
    conn_1 = -3
    inh_conn_filter = lambda x: conn_1 * np.exp(-x / inh_tau)
    step1 = generateStepFunction(period=5, amplitude=1, duty_cycle=0.2)
    # step2 =
    # times = np.arange(0, 15, .1)
    # plt.plot(times, step(times))
    no_current = lambda x: np.array([0] * len(x))
    f = .5
    A = 1
    I0 = 0
    sincurrent = lambda x : A * np.sin(2 * np.pi * f * x) + I0

    u_rest = -15
    neurons = {
        'n1': [u_rest, k_filter1, ref_filter, inh_conn_filter, step1],
        # 'n2': [u_rest, k_filter2, ref_filter, inh_conn_filter, sincurrent]
    }
    # for i in range(1):
    #     neurons['n'+str(i)] = [u_rest, k_filter, ref_filter, inh_conn_filter, no_current]
    # for i in range(7,10):
    #     neurons['n' + str(i)] = [u_rest, k_filter, ref_filter, exc_conn_filter, no_current]
    #
    # neurons['n9'][-1] = step

    mn = SRMMultiNeuronSimulator(dt=0.01, sim_length=(60*1), no_sim=1, k_length=2, ref_length=2, **neurons)
    mn.runSimulation()
    mn.plotSimulation()

