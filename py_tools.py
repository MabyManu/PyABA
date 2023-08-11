# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:26:16 2023

@author: manum
"""
import numpy as np
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def remove_multelements(List,elem): 
    for ele in sorted(elem, reverse = True):
        del List[ele]    
    return List 


def expNoOverflow(x):
    if (x>np.log(sys.float_info.max)):
        expx = np.exp(np.fix(np.log(sys.float_info.max)))
    else:
        if (x<np.log(sys.float_info.min)):
            expx = np.exp(np.fix(np.log(sys.float_info.min)))
        else:
            expx = np.exp(x)
        
    
    return expx


def SameSign(A,B):
    SameSign_bool = (((A>0)&(B>0)) | (A<0)&(B<0))
    return SameSign_bool

def DetectInflectionPointFromBaseline(X,Baseline):
    orderfilt = 7
    smoothed = gaussian_filter(np.abs(X), orderfilt)
    std = (X[Baseline[0 ]:Baseline[1]] ).std()
    m = smoothed > (3. * std)
    ixFlecPt = np.where(m)[0][0]
    return ixFlecPt


def DetectInflectionPointDerivative(X):
    smooth = gaussian_filter1d(X, 50)

    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))


    # find switching points
    infls, _ = find_peaks(np.abs(smooth_d2), height=0.5*np.max(np.abs(smooth_d2)))
    ixFlecPt = infls[0]
    return ixFlecPt



def SearchStartStopClusterFromIndex(IndexArray):
    Clusters = []
    if len(IndexArray)>0 :
        EndClusters = np.where(np.diff(IndexArray)>1)[0]
        NbClusters = len(EndClusters) + 1 
        
        for i_clust in range(NbClusters):
            if (i_clust == 0):
                start = IndexArray[0]
            else:
                start = IndexArray[EndClusters[i_clust-1]+1]
                
            if (i_clust == NbClusters-1):
                stop = IndexArray[-1]
            else:
                stop = IndexArray[EndClusters[i_clust]]           
            Clusters.append([start,stop])
            
    return Clusters


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise