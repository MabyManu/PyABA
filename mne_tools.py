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
from matplotlib.patches import Polygon
import py_tools

from mne.viz import iter_topography
import matplotlib.transforms as mtransforms
def ChangeMontage_epochs(Epochs_orig,NewSetElectrodes):
	
	Nsamples = len(Epochs_orig.times)
	Ntrials = len(Epochs_orig)
	NChan_Orig = Epochs_orig.info['nchan']
	ListChannels_orig = Epochs_orig.info['ch_names']
	
	NewChan2add = [x for x in NewSetElectrodes if not(x in ListChannels_orig)]
	NbNewChan2add = len(NewChan2add)
	NewData2add = np.random.randn(Ntrials, NbNewChan2add,Nsamples)*1e-6
	
	infoAddChan = mne.create_info(NewChan2add, Epochs_orig.info['sfreq'], NbNewChan2add*['eeg'])
	NewEpochArray  = mne.EpochsArray(NewData2add, infoAddChan)
	NewEpochArray.set_montage(mne.channels.make_standard_montage('standard_1020'))
	
	Epochs_curr = Epochs_orig.copy()
	Epochs_curr.add_channels([NewEpochArray], force_update_info=True)
	Epochs_curr.info['bads'] = NewChan2add
	Epochs_curr.interpolate_bads(reset_bads=False)
	Epochs_curr.info['bads'] = []
	
	
	NewEpoch = Epochs_curr.copy()
	OldChan2drop = [x for x in ListChannels_orig if not(x in NewSetElectrodes )]
	NewEpoch.drop_channels(OldChan2drop)
	NewEpoch.reorder_channels(NewSetElectrodes)
	
	return NewEpoch



def RejectThresh(epochs,PercentageOfEpochsRejected):
    epochs.drop_bad()
    NbEpoch2Keep = np.fix(epochs.__len__() * (1.0-(PercentageOfEpochsRejected/100)))-1
    Mat_epoch =  epochs.get_data(copy=True)
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




def PermutCluster_plotCompare(X, colors_config , styles_config  ,evokeds,p_accept,n_permutations,Title):
	# Compute comparison between  2 conditions for each channel
	# Display results with topographical way
	#
	# USAGE for one subject: 
	# X = [Epochs_Condition1.get_data().transpose(0, 2, 1), Epochs_Condition2.get_data().transpose(0, 2, 1)]
	# colors_config = {"Condition1": "magenta", "Condition2": 'steelblue'}
	# styles_config ={"Condition1": {"linewidth": 2.75},"Condition2": {"linewidth": 2.75}}
	# Evoked_Condition1   = Epochs_Condition1.average()
	# Evoked_Condition2   = Epochs_Condition2.average()
	# evokeds = {'Condition1':Evoked_Condition1,'Condition2':Evoked_Condition2}
	# p_accept = 0.05
	# n_permutations = 2000
	# fig = mne_tools.PermutCluster_plotCompare(X, colors_config, styles_config, evokeds,p_accept,n_permutations)
	
	n_conditions = 2
	n_replications = (X[0].shape[0])  // n_conditions
	factor_levels = [2]      #[2, 2]  # number of levels in each factor
	effects = 'A'  # this is the default signature for computing all effects
	# Other possible options are 'A' or 'B' for the corresponding main effects
	# or 'A:B' for the interaction effect only
	pthresh = 0.05  # set threshold rather high to save some time
	f_thresh = f_threshold_mway_rm(n_replications,factor_levels,effects,pthresh)
	del n_conditions, n_replications, factor_levels, effects, pthresh
	tail = 1  # f-test, so tail > 0
	threshold = f_thresh
	Nchans = X[0].shape[2]
	n_tests,n_samples,_ = X[0].shape
	Times = evokeds[list(evokeds.keys())[0]].times
	Info_ev = evokeds[list(evokeds.keys())[0]].info
	AmpMaxCond1 = np.max((np.max(evokeds[list(evokeds.keys())[0]].get_data()),np.abs(np.min(evokeds[list(evokeds.keys())[0]].get_data()))))
	AmpMaxCond2 = np.max((np.max(evokeds[list(evokeds.keys())[1]].get_data()),np.abs(np.min(evokeds[list(evokeds.keys())[1]].get_data()))))

	
	AmpMax = np.max((AmpMaxCond1,AmpMaxCond2))	
	Label1 = list(colors_config.items())[0][0]
	Color1 = list(colors_config.items())[0][1]
	Linewidth1 = list(styles_config[Label1].items())[0][1]
	Label2 = list(colors_config.items())[1][0]
	Color2 = list(colors_config.items())[1][1]
	Linewidth2 = list(styles_config[Label2].items())[0][1]	
	

	figtopocompare = plt.figure()

	for ax, idx in iter_topography(Info_ev,fig_facecolor="white",axis_facecolor="white",axis_spinecolor="white",layout_scale=1,fig=figtopocompare):
		
		l, b, w, h = ax.get_position().bounds
		newpos = [l, b-0.15, w, h]
		ax.set_position(pos=newpos,which='both')
		
		ax.plot(Times,evokeds[list(evokeds.keys())[0]].get_data()[idx,:],color=Color1,linewidth = Linewidth1,label=Label1)
		ax.plot(Times,evokeds[list(evokeds.keys())[1]].get_data()[idx,:],color=Color2,linewidth = Linewidth2,label=Label2)
		ax.set_title(Info_ev['ch_names'][idx],loc='left',fontdict={'fontsize':10})
		ax.axvline(x=0,linestyle=':',color='k')
		ax.axhline(y=0,color='k')
		ax.set_ylim(-AmpMax,AmpMax)
		ax.invert_yaxis()
		
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0][:,:,idx], X[1][:,:,idx]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')
		
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr = clusters[i_cluster][0]
				ax.axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
				
		if (idx == Info_ev['nchan']-1):
			ax.legend(loc=(1,0))
			
	legendax = figtopocompare.add_axes([0.97-w,0.05,w,h]) 
	legendax.set_xlabel('Time (s)',fontsize=10,labelpad=0)
	legendax.set_ylabel('µV',fontsize=10,labelpad=0)

	legendax.set_yticks([-AmpMax*1e6,AmpMax*1e6])
	legendax.set_yticklabels(np.round([-AmpMax*1e6,AmpMax*1e6],1), fontsize=10)
	legendax.set_xticks(np.arange(Times[0],Times[-1],0.4))
	legendax.set_xticklabels(np.round(np.arange(Times[0],Times[-1],0.4),1), fontsize=10)
	legendax.invert_yaxis()

	plt.gcf().suptitle(Title)
	plt.show()
	
	return figtopocompare







def PlotEvokedDeviationFrom0(X,evokeds,colors_config,styles_config,alpha,Title):
	Label,color = list(colors_config.items())[0]
	_,linewidth = list(styles_config[Label].items())[0]
	Times = evokeds.times
	AmpMax = np.max((np.max(evokeds.get_data()),np.abs(np.min(evokeds.get_data()))))

	figtopo = plt.figure()

	for ax, idx in iter_topography(evokeds.info,fig_facecolor="white",axis_facecolor="white",axis_spinecolor="white",layout_scale=1.0,fig=figtopo):
		
		l, b, w, h = ax.get_position().bounds
		newpos = [l, b-0.15, w, h]
		ax.set_position(pos=newpos,which='both')
		
		ax.plot(evokeds.times,evokeds.get_data()[idx,:],color=color,linewidth = linewidth,label=Label)
		ax.set_title(evokeds.info['ch_names'][idx],loc='left',fontdict={'fontsize':10})
		ax.axvline(x=0,linestyle=':',color='k')
		ax.axhline(y=0,color='k')
		ax.set_ylim(-AmpMax,AmpMax)
		ax.invert_yaxis()

		X_tmp = X[:, idx, :]
		T, pval = stats.ttest_1samp(X_tmp, 0)
		n_samples, n_tests = X_tmp.shape
		threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
		reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
		threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
		reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
		# threshold_fdr = np.min(np.abs(T)[reject_fdr])
		Clusters = py_tools.SearchStartStopClusterFromIndex(np.where(reject_fdr)[0])
		for i_cluster in range(len(Clusters)):
			ax.axvspan(Times[Clusters[i_cluster][0]], Times[Clusters[i_cluster][1]],facecolor=color,alpha=0.3)
			
		if (idx == evokeds.info['nchan']-1):
			ax.legend(loc=(1,0))
			
	legendax = figtopo.add_axes([0.97-w,0.05,w,h]) 
	legendax.set_xlabel('Time (s)',fontsize=10,labelpad=0)
	legendax.set_ylabel('µV',fontsize=10,labelpad=0)

	legendax.set_yticks([-AmpMax*1e6,AmpMax*1e6])
	legendax.set_yticklabels(np.round([-AmpMax*1e6,AmpMax*1e6],1), fontsize=10)
	legendax.set_xticks(np.arange(evokeds.times[0],evokeds.times[-1],0.4))
	legendax.set_xticklabels(np.round(np.arange(evokeds.times[0],evokeds.times[-1],0.4),1), fontsize=10)
	legendax.invert_yaxis()

	plt.gcf().suptitle(Title)
	plt.show()
	
	return figtopo







    
def SpatTemp_TFCE_plotCompare(X, colors_config, styles_config,evokeds,p_accept,n_permutations):
	# Compute Non-parametric cluster-level test for spatio-temporal data between  2 conditions
	# Display results with topographical way
	#
	# USAGE for one subject: 
	# X = [Epochs_Condition1.get_data().transpose(0, 2, 1), Epochs_Condition2.get_data().transpose(0, 2, 1)]
	# colors_config = {"Condition1": "magenta", "Condition2": 'steelblue'}
	# styles_config ={"Condition1": {"linewidth": 2.75},"Condition2": {"linewidth": 2.75}}
	# Evoked_Condition1   = Epochs_Condition1.average()
	# Evoked_Condition2   = Epochs_Condition2.average()
	# evokeds = {'Condition1':Evoked_Condition1,'Condition2':Evoked_Condition2}
	# p_accept = 0.05
	# n_permutations = 2000
	# fig = mne_tools.SpatTemp_TFCE_plotCompare(X, colors_config, styles_config, evokeds,p_accept,n_permutations)
	
	
	samplingFreq = evokeds[list(evokeds.keys())[0]].info['sfreq']
	adjacency, ch_names = find_ch_adjacency(evokeds[list(evokeds.keys())[0]].info, ch_type='eeg')

	
	tfce = dict(start=1, step=2)
	
	# Calculate statistical thresholds
	t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(X, tfce, adjacency=adjacency,n_permutations=1000)  # a more standard number would be 1000+
	significant_points = cluster_pv.reshape(t_obs.shape).T <= p_accept
	print(str(significant_points.sum()) + " points selected by TFCE ...")
    
	figtopocompare = plot_compare_evokeds(evokeds, picks='eeg', colors=colors_config,styles = styles_config,axes='topo',split_legend=False,legend="lower center")
	Nchans = t_obs.shape[1]
	
	for i_chan in range (Nchans):
		SignifiWin_curr = significant_points[i_chan,:]
		SampSign =np.where(SignifiWin_curr)[0]
		BoundWin_ix = np.where((np.diff(SampSign)>1))[0]
		
		if (len(BoundWin_ix)>0):
			Nb_Clust = len(BoundWin_ix)+1
			Clust_start = np.zeros(Nb_Clust,dtype='int64')
			Clust_stop = np.zeros(Nb_Clust,dtype='int64')
			for i_clustwin in range(Nb_Clust):
				if (i_clustwin==0):
					Clust_start[i_clustwin] = SampSign[0]
				else:
					Clust_start[i_clustwin] = SampSign[BoundWin_ix[i_clustwin-1]+1]
				
				if (i_clustwin==(Nb_Clust-1)):
					Clust_stop[i_clustwin] = SampSign[-1]
				else:
					Clust_stop[i_clustwin] = SampSign[BoundWin_ix[i_clustwin]]
				figtopocompare[0].get_axes()[i_chan].axvspan(Clust_start[i_clustwin]/samplingFreq,Clust_stop[i_clustwin]/samplingFreq,facecolor="crimson",alpha=0.3)
# 	figtopocompare[0].set_size_inches(8, 8)
	return figtopocompare
    
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





