import numpy as np
import PyLeech.Utils.NLDUtils as NLD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    rt = np.load("reduced.npy")
    NLD.plotCloseReturns(rt)
    # plt.matshow(rt`)