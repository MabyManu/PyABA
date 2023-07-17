# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:23:31 2023

@author: manum
"""
import numpy as np
import mne
from mne.viz import plot_compare_evokeds

from mne.channels import find_ch_adjacency,make_1020_channel_selections
from mne.stats import (spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test,permutation_cluster_test )
from mne.stats import bonferroni_correction, fdr_correction

from scipy import stats


from mne.viz import plot_evoked_topo
from mne.viz import plot_topomap
from mne.stats import permutation_t_test,f_threshold_mway_rm

import matplotlib.pyplot as plt

import py_tools


def RejectThresh(epochs,PercentageOfEpochsRejected):
    epochs.drop_bad()
    NbEpoch2Keep = np.fix(epochs.__len__() * (1.0-(PercentageOfEpochsRejected/100)))-1
    Mat_epoch =  epochs.get_data()
    MinWOI = Mat_epoch.min(axis=2)
    MaxWOI = Mat_epoch.max(axis=2)
    Peak2Peak = MaxWOI-MinWOI
    MaxPeak2PeakROI = Peak2Peak.max(axis=1)   
    ixSort = np.argsort(MaxPeak2PeakROI)
    MaxPeak2PeakROI.sort()
    Threshold = MaxPeak2PeakROI[int(NbEpoch2Keep)-1]
    ixRej = np.squeeze(np.where(Peak2Peak.max(axis=1)>Threshold))    
    MaxP2P = Peak2Peak.max(axis=1)
    
    return Threshold,MaxPeak2PeakROI,ixSort,ixRej, MaxP2P




def PermutCluster_plotCompare(X, colors_config , styles_config  ,evokeds,p_accept,n_permutations):
    n_conditions = 2
    n_replications = (X[0].shape[0])  // n_conditions
    factor_levels = [2]      #[2, 2]  # number of levels in each factor
    effects = 'A'  # this is the default signature for computing all effects
    # Other possible options are 'A' or 'B' for the corresponding main effects
    # or 'A:B' for the interaction effect only
        
    pthresh = 0.05  # set threshold rather high to save some time
    f_thresh = f_threshold_mway_rm(n_replications,
                                       factor_levels,
                                       effects,
                                       pthresh)
    del n_conditions, n_replications, factor_levels, effects, pthresh
        
    tail = 1  # f-test, so tail > 0
    
    threshold = f_thresh
    Nchans = X[0].shape[2]
    n_tests,n_samples,_ = X[0].shape
    Times = evokeds[list(evokeds.keys())[0]].times
  
    figtopocompare = plot_compare_evokeds(evokeds, picks='eeg', colors=colors_config,styles = styles_config, axes='topo',invert_y=True,split_legend=False,legend="lower center")
   
    for i_chan in range (Nchans):
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0][:,:,i_chan], X[1][:,:,i_chan]], n_permutations=n_permutations,
                             threshold=threshold, tail=tail, n_jobs=3,
                             out_type='mask')
        
        for i_cluster in range(len(cluster_p_values)):
            if (cluster_p_values[i_cluster]<p_accept):
                Clust_curr = clusters[i_cluster][0]
                figtopocompare[0].get_axes()[i_chan].axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
                

    figtopocompare[0].set_size_inches(8, 8)
    
    return figtopocompare












def PlotEvokedDeviationFrom0(X,evokeds,colors_config,styles_config,alpha):
    
    figtopo = plot_compare_evokeds(evokeds, picks='eeg', colors=colors_config,styles = styles_config, axes='topo',invert_y=True,split_legend=False,legend="lower center")
    Times =evokeds.times
    for ele in colors_config.values():
        colorspan = ele
    for i_chan in range (evokeds.info['nchan']):
            X_tmp = X[:, i_chan, :] 
            T, pval = stats.ttest_1samp(X_tmp, 0)
            
            n_samples, n_tests = X_tmp.shape
            
            threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
            
            reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
            threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
            
            
            reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
            # threshold_fdr = np.min(np.abs(T)[reject_fdr])
            
            Clusters = py_tools.SearchStartStopClusterFromIndex(np.where(reject_fdr)[0])
            
            for i_cluster in range(len(Clusters)):
                figtopo[0].get_axes()[i_chan].axvspan(Times[Clusters[i_cluster][0]], Times[Clusters[i_cluster][1]],facecolor=colorspan,alpha=0.3)
                    
    return figtopo
    
    
    # Emergence_clust_bool = np.zeros((evokeds.times.size,evokeds.info['nchan']),dtype='bool')
    # Emergence_clust = np.zeros((evokeds.times.size,evokeds.info['nchan']),dtype='float')
    # nbrow = np.int64(np.ceil(np.sqrt(evokeds.info['nchan'])))
    # nbcol = np.int64(np.ceil((evokeds.info['nchan']/nbrow)))
    # fig, axs = plt.subplots(nbrow,nbcol)
    # for i_chan in range (evokeds.info['nchan']):
    #     X_tmp = X[:, i_chan, :] 
    #     T, pval = stats.ttest_1samp(X_tmp, 0)
        
    #     n_samples, n_tests = X_tmp.shape
        
    #     reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
    #     threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
    #     times =evokeds.times
    #     Times_SigPos=times[np.where(T>threshold_bonferroni) ]
    #     Times_SigNeg=times[np.where(T<-threshold_bonferroni)]
        
    #     Emergence_clust[:,i_chan] = ((T>threshold_bonferroni)*T) + ((T<-threshold_bonferroni)*T)
    #     Emergence_clust_bool[:,i_chan] = ((T>threshold_bonferroni)) + ((T<-threshold_bonferroni))
        
    #     evok = evokeds.data[i_chan]*1e6
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].plot(times,evok)
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].plot(Times_SigPos,np.ones(Times_SigPos.size)*evok.max(),'*')
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].plot(Times_SigNeg,np.ones(Times_SigNeg.size)*evok.min(),'*')
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].invert_yaxis()   
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].set_title(evokeds.info['ch_names'][i_chan],fontsize='small')     
    #     axs[np.int64(np.fix(i_chan/nbcol)),i_chan%nbcol].axhline(y=0, color='k',linestyle='--',lw=0.5)
    #     fig.suptitle(evokeds.comment, fontsize=12)     
                                             
    # fig, ax = plt.subplots()                                                  
    # extend_tmp=np.min(times),np.max(times),evokeds.info['nchan']-1,0  
    
    # SignificantTemporalClustDetect,LatenciesClustDetect = tools.DetectTemporalCluster(Emergence_clust, times, int(SignTimeWinDuration*evokeds.info['sfreq']/1000.0))

    # # plt.imshow(SignificantTemporalClustDetect.T, cmap='RdBu',aspect='auto', interpolation='none',origin='upper', vmin=-np.max(np.abs(SignificantTemporalClustDetect)),vmax = np.max(np.abs(SignificantTemporalClustDetect)), extent=extend_tmp)
    # plt.imshow(SignificantTemporalClustDetect.T, cmap='seismic',aspect='auto', interpolation='none',origin='upper', vmin=-np.max(np.abs(SignificantTemporalClustDetect)),vmax = np.max(np.abs(SignificantTemporalClustDetect)), extent=extend_tmp)
    # ax.set_yticks(np.arange(0,evokeds.info['nchan']))
    # ax.set_yticklabels(evokeds.info['ch_names'])
    # fig.suptitle(evokeds.comment, fontsize=12)
    


    # return Emergence_clust_bool ,Emergence_clust #,   SignificantTemporalClustDetect , LatenciesClustDetect 





