import PyLeech.Utils.NLDUtils as NLD








def getNSbands(NS, binning_dt, bl_cutoff_freq=.01, band_cutoff=.2):
    bl_kernel = NLD.generateGaussianKernel(sigma=1/bl_cutoff_freq, time_range=5/bl_cutoff_freq, binning_dt=binning_dt)
    band_kernel = NLD.generateGaussianKernel(sigma=1/bl_cutoff_freq, time_range=5/bl_cutoff_freq, binning_dt=binning_dt)