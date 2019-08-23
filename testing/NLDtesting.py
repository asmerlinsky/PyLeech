import PyLeech.Utils.NLDUtils as NLD
import PyLeech.Utils.AbfExtension as abfe
from functools import partial
import PyLeech.Utils.CrawlingDatabaseUtils as CDU
import PyLeech.Utils.burstStorerLoader as bStorerLoader
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
    # fn = "RegistrosDP_PP\\2018_12_03_0005.pklspikes"

    data = NLD.simulateLorenz(0.01, num_steps=6000)
    trace = np.array([data[:, 0]]).T
    NLD.plot3Dline(data, linewidth=.5, color='k')
    rt = NLD.getCloseReturns(trace, get_mask=True, threshold=.02)[:, :, 0]
    NLD.plotCloseReturns(rt, reorder=False)
    segments = NLD.getCloseReturnsSegmentsFromUnorderedMatrix(rt, rt_len=200, single_file=True, min_dist=10)
    print(len(segments))
    for diag, items in segments.items():
        for t0, t1 in items:
            fig, ax = plt.subplots()
            ax.plot(data[t0:t1,0])
            ax.plot(data[t0+diag: t1+diag,0])

            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            ax1.plot(data[t0:t1,0], data[t0:t1,1], data[t0:t1,2])
            ax1.plot(data[t0+diag: t1+diag,0], data[t0+diag: t1+diag,1], data[t0+diag: t1+diag,2], linestyle='dashed')