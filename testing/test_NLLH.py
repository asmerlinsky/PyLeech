from PyLeech.testing.fit_sin_matriz import *


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



def MATNegLogLikelihood(fit_array, input_mat, spike_matrix, dt, scaling_length, k_len, ref_len, penalty_param):
    no_neurons = spike_matrix.shape[0]
    neuron_filters = np.split(fit_array, no_neurons)
    NegLogLikelihood = 0
    for i in range(no_neurons):
        actual_filter = np.concatenate(([neuron_filters[i][0]], np.repeat(neuron_filters[i][1:], scaling_length)))
        x_dot_k = np.sum(input_mat * actual_filter, axis=1)

        NegLogLikelihood -= np.dot(spike_matrix[i], x_dot_k) - np.exp(x_dot_k).sum() * dt

    return NegLogLikelihood


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
    neuron_filter_length = (neuron_filters[0].shape[0] - 1)
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

        penalty_grad = np.zeros(fit_k_len)

        penalty_grad[:-1] += 2 * penalty_param * np.diff(neuron_filters[i][1:fit_k_len + 1])
        penalty_grad[1:] -= 2 * penalty_param * np.diff(neuron_filters[i][1:fit_k_len + 1])

        grad_Neg_Log_Likelihood[1 + neuron_filter_length * i:1 + neuron_filter_length * i + fit_k_len] -= penalty_grad

        u += actual_filter[0]
        #        print(u[41*scaling_length:41*scaling_length + scaling_length])
        spikes_minus_exp = spike_matrix[i, 1:input_stim.shape[0]] - np.exp(u) * dt
        spikes_minus_exp = spikes_minus_exp[::-1]

        grad_Neg_Log_Likelihood[i * neuron_filters[i].shape[0]] -= spikes_minus_exp.sum()

        #        print(grad_Neg_Log_Likelihood[41*scaling_length:41*scaling_length + scaling_length])
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
        sincurrent1 = lambda x: A * np.power(np.sin(2 * np.pi * f * x), 2) + I0
        sincurrent2 = lambda x: A * np.power(np.sin(2 * np.pi * f * x + np.pi / 2), 2) + I0

        noise = generateNoiseFunction(4, 1)

        u_rest = -15
        neurons = {
            'n1': [u_rest, k_filter1, ref_filter, inh_conn_filter, no_current],
            'n2': [u_rest, k_filter1, ref_filter, inh_conn_filter, no_current]
        }
        # for i in range(1):
        #     neurons['n'+str(i)] = [u_rest, k_filter, ref_filter, inh_conn_filter, no_current]
        # for i in range(7,10):
        #     neurons['n' + str(i)] = [u_rest, k_filter, ref_filter, exc_conn_filter, no_current]
        #
        # neurons['n9'][-1] = step

        mn = srmSimUtils.SRMMultiNeuronSimulator(dt=1 / 1000, sim_length=60 * 10, no_sim=1, k_length=1, ref_length=1,
                                                 **neurons)
        mn.runSimulation()
        mn.plotSimulation()
        spike_freq_dict = mn.getTrialSpikeFreqDict()

        fit1 = SRMFitter(mn.time_steps, mn.input_stim[0], spike_freq_dict, k_duration=1, refnconn_duration=0.5, dim_red=50)
        fit2 = MATSRMFitter(mn.time_steps, mn.input_stim[0], spike_freq_dict, k_duration=1, refnconn_duration=0.5,
                            dim_red=50)

        test_array = np.zeros(fit1.fit_array.shape[0])
        plt.figure()
        plt.plot(gradNegLogLikelihood(test_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
                                      fit1.k_length, fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
                                      fit1.single_conn_length * fit1.scaled_fit_size, 0), label='gNLLHwzeros',
                 linestyle='-.')


        test_array[2 + fit1.k_length + int(fit1.refnconn_length): 2 + 2* fit1.k_length + int(fit1.refnconn_length)] = np.random.random(10*2)
        # test_array[1 + fit1.k_length + int(fit1.refnconn_length/2): 1 + fit1.k_length + int(fit1.refnconn_length)] = np.random.random(10)*2

        # test_array = fit2.fit_array
        # test_array[1 : 1 + fit1.k_length] = np.random.random(fit1.k_length)

        print(test_array)
        print(NegLogLikelihood(test_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
                               fit1.k_length,
                               fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
                               fit1.single_conn_length * fit1.scaled_fit_size, 0),
              penalizedNegLogLikelihood(test_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
                                   fit1.k_length, fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
                                   fit1.single_conn_length * fit1.scaled_fit_size, 10))


        plt.plot(test_array*200, label='test_array')

        plt.plot(gradNegLogLikelihood(test_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
                                      fit1.k_length, fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
                                      fit1.single_conn_length * fit1.scaled_fit_size, 0), label='gNLLH', linestyle='--')
        plt.plot(
            gradPenalizedNegLogLikelihood(test_array, fit1.input_stim, fit1.spike_count, fit1.dt, fit1.scaled_fit_size,
                                          fit1.k_length, fit1.k_length * fit1.scaled_fit_size, fit1.single_conn_length,
                                          fit1.single_conn_length * fit1.scaled_fit_size, 1), label='PgNLLH')
        plt.legend()


