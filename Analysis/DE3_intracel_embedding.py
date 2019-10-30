import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe
import PyLeech.Utils.burstStorerLoader as bStorerLoader
import PyLeech.Utils.burstUtils
import PyLeech.Utils.burstUtils as burstUtils
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


fn = "RegistrosDP_PP\\DE3_DP_DP.abf"

arr_dict, time_vector, fs = abfe.getArraysFromAbfFiles([fn], ['Vm1'])
DE3 = arr_dict['Vm1']
del arr_dict
#
fig, ax = plt.subplots()
ax.plot(time_vector, DE3, lw=1)
DE3_median = np.median(DE3)
# DE3 -= DE3_median

DE3[(DE3 > 20+DE3_median) | (DE3 < -20+DE3_median)] = np.median(DE3)


kernel_sigma = 10
kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=30, dt_step=1 / fs)
bl_DE3 = spsig.fftconvolve(DE3, kernel, mode='same')

kernel_sigma = 1
kernel = PyLeech.Utils.burstUtils.generateGaussianKernel(sigma=kernel_sigma, time_range=30, dt_step=1 / fs)

# conv_DE3 = spsig.fftconvolve(DE3-bl_DE3, kernel, mode='same')
conv_DE3 = spsig.fftconvolve(DE3, kernel, mode='same')


cut_DE3 = DE3[time_vector < 300]
conv_DE3 = conv_DE3[time_vector < 300]
cut_time_vector = time_vector[time_vector < 300]


fig, ax = plt.subplots()
ax.plot(cut_time_vector, cut_DE3, 'k')
ax.plot(cut_time_vector, conv_DE3, lw=1)

# fig, ax = plt.subplots()
# ax.plot(cut_time_vector, conv_DE3, 'k')

intervals = [[3, 86], [109, 145], [155, 190], [200, 236], [248, 274]]
# binning_dt = 0.1


neuron_idxs = []
# intervals = [[2, 87]]
for interval in intervals:
    neuron_idxs.append(np.where((cut_time_vector > interval[0]) & (cut_time_vector < interval[1]))[0])
neuron_idxs = np.hstack(neuron_idxs)

neuron_cut_time = cut_time_vector[neuron_idxs]


segmented_DE_3 = conv_DE3[neuron_idxs]
# plt.plot(neuron_cut_time, segmented_DE_3)

DE3_embedding = NLD.getDerivativeEmbedding(segmented_DE_3, 1 / fs, 3)

jumps = np.where(np.diff(neuron_idxs) != np.diff(neuron_idxs)[0])[0]

if len(jumps) > 0:
    DE3_embedding[jumps] = np.nan
    DE3_embedding[jumps + 1] = np.nan

# scaler = StandardScaler()
# DE3_embedding = scaler.fit_transform(DE3_embedding)

NLD.plot3Dline(DE3_embedding[::100])
subsampled_embedding = DE3_embedding[::100]
rt = NLD.getCloseReturns(subsampled_embedding, threshold=.1)
NLD.plotCloseReturns(rt, masked=True, thr=.03, reorder=True)

fig, ax = NLD.plot3Dline(subsampled_embedding[4200:4600], step=3, colorbar=False)
NLD.plot3Dline(subsampled_embedding[4200+950:4600+950], fig_ax_pair=[fig, ax], step=3)

fig, ax = NLD.plot3Dline(subsampled_embedding[2500:2800], step=3, colorbar=False)
NLD.plot3Dline(subsampled_embedding[2500+1200:2800+1200],fig_ax_pair=[fig, ax], step=3)

fig, ax = NLD.plot3Dline(subsampled_embedding[4200:4700], step=3)
NLD.plot3Dline(subsampled_embedding[4200+950:4700+950],fig_ax_pair=[fig, ax], step=3)
