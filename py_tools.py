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


