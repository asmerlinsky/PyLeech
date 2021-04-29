import numpy as np


def binaryNeuronEntropy(binarized_spike_train):
    if not isinstance(binarized_spike_train, np.ndarray):
        binarized_spike_train = np.array(binarized_spike_train, dtype=bool)

    T = binarized_spike_train.shape[0]
    prob_firing = []
    prob_firing.append(binarized_spike_train.sum()/T)
    prob_firing.append(1 - prob_firing[0])
    H = 0.
    for i in range(len(prob_firing)):
        if prob_firing[i] > .00001:
            H -= prob_firing[i] * np.log2(prob_firing[i])
    return H

def binaryMutualInformation(bin_spike_train1, bin_spike_train2):

    prob_firing1 = np.zeros(2)
    prob_firing2 = np.zeros(2)
    prob_firing12 = np.zeros((2,2))

    prob_firing1[0] = bin_spike_train1.sum() / bin_spike_train1.shape[0]
    prob_firing1[1] = 1 - prob_firing1[0]

    prob_firing2[0] = bin_spike_train2.sum() / bin_spike_train2.shape[0]
    prob_firing2[1] = 1 - prob_firing2[0]

    prob_firing12[0, 0] = (bin_spike_train1 & bin_spike_train2).sum()
    prob_firing12[0, 1] = (bin_spike_train1 & ~bin_spike_train2).sum()
    prob_firing12[1, 0] = (~bin_spike_train1 & bin_spike_train2).sum()
    prob_firing12[1, 1] = (~bin_spike_train1 & ~bin_spike_train2).sum()

    prob_firing12 /= prob_firing12.sum()

    H1 = binaryNeuronEntropy(bin_spike_train1)
    H2 = binaryNeuronEntropy(bin_spike_train2)
    MI = 0.
    for i in range(2):
        for j in range(2):
            if prob_firing12[i, j] > .000001:
                MI += prob_firing12[i, j] * np.log2(prob_firing12[i, j]/prob_firing1[i]/prob_firing2[j])

    return MI, prob_firing12



if __name__ == "__main__":

    train_size = 1000
    p_firing = .2
    train1 = np.random.choice([1, 0], size=train_size, p=[p_firing, 1-p_firing]).astype(bool)
    train2 = np.random.choice([1, 0], size=train_size, p=[p_firing, 1-p_firing]).astype(bool)

    H_train1 = binaryNeuronEntropy(train1)
    H_train2 = binaryNeuronEntropy(train2)
    MI, prob = binaryMutualInformation(train1, train2)
    print(H_train1)
    print(H_train2)
    print(MI)