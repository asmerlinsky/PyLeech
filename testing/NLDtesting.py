import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe
from functools import partial
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.unitInfo as bStorerLoader
import PyLeech.Utils.burstUtils as burstUtils
import os
import math
import numpy as np
import scipy.signal as spsig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import PyLeech.Utils.burstClasses as burstClasses
from sklearn.preprocessing import StandardScaler




if __name__ == "__main__":
    # fn = "RegistrosDP_PP/2018_12_03_0005.pklspikes"

    num_steps = 10000
    params = np.array((.1, .1, 4))
    x0 = np.array([4, 0, 0])
    data = NLD.simRossler(params, num_steps=num_steps, dt=0.01, x0=x0)
    x0 = data[-1]

    num_steps = 10000
    num_params = 30

    x = np.repeat(.1, num_steps)
    y = y = np.repeat(.1, num_steps)

    z = np.linspace(4, 8, num_params+1)
    z = np.repeat(z, math.ceil(num_steps/z.shape[0]))[:num_steps]

    params = np.array((x, y, z)).T
    del x, y, z


    data = NLD.simRossler(params, num_steps=num_steps, dt=0.01, x0=x0)
    trace = np.array([data[:, 0]]).T
    NLD.plot3Dline(data, linewidth=.5, color='k')
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(data.shape[1]):
        ax[i].plot(data[:,i])

    rt = NLD.getCloseReturns(trace, get_mask=True, threshold=.01)
    NLD.plotCloseReturns(rt, reorder=False)
    segments = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt, rt_len=1400, single_file=True, min_dist=200)
    print(len(segments))
    for diag, items in segments.items():
        for t0, t1 in items:
            fig, ax = plt.subplots(3, 1)
            for i in range(3):

                ax[i].plot(data[t0:t1,i])
                ax[i].plot(data[t0+diag: t1+diag,i])
            #
            # fig1 = plt.figure()
            # ax1 = Axes3D(fig1)
            # ax1.plot(data[t0:t1,0], data[t0:t1,1], data[t0:t1,2])
            # ax1.plot(data[t0+diag: t1+diag,0], data[t0+diag: t1+diag,1], data[t0+diag: t1+diag,2], linestyle='dashed')