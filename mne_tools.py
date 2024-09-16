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
from mne.preprocessing import ICA

import scipy

import os

from scipy import stats


from mne.viz import plot_evoked_topo
from mne.viz import plot_topomap
from mne.stats import permutation_t_test,f_threshold_mway_rm
from mne.preprocessing import create_eog_epochs

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
	pthresh = 0.01  # set threshold rather high to save some time
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
# 		newpos = [l, b-0.15, w, h]
		newpos = [l, b-0.02, w, h]
		ax.set_position(pos=newpos,which='both')
		
		ax.plot(Times,evokeds[list(evokeds.keys())[0]].get_data()[idx,:],color=Color1,linewidth = Linewidth1,label=Label1)
		ax.plot(Times,evokeds[list(evokeds.keys())[1]].get_data()[idx,:],color=Color2,linewidth = Linewidth2,label=Label2)

		ax.set_title(Info_ev['ch_names'][idx],loc='left',fontdict={'fontsize':10})
		ax.axvline(x=0,linestyle=':',color='k')
		ax.axhline(y=0,color='k')
		ax.set_xlim(Times[0]-0.01,np.round(Times[-1]*100)/100)
		ax.set_ylim(-AmpMax,AmpMax)
		ax.invert_yaxis()
		ax.set_xticks(np.arange(0,Times[-1],0.5))
		ax.set_xticklabels(np.round(np.arange(0,Times[-1],0.5),1), fontsize=6)
		
		
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([X[0][:,:,idx], X[1][:,:,idx]], n_permutations=n_permutations,threshold=threshold, tail=tail, n_jobs=3,out_type='mask',verbose='ERROR')
		
		for i_cluster in range(len(cluster_p_values)):
			if (cluster_p_values[i_cluster]<p_accept):
				Clust_curr = clusters[i_cluster][0]
				ax.axvspan(Times[Clust_curr.start], Times[Clust_curr.stop-1],facecolor="crimson",alpha=0.3)
				
		if (idx == Info_ev['nchan']-1):
			ax.legend(loc=(1,0))
			
# 	legendax = figtopocompare.add_axes([0.97-w,0.05,w,h]) 
	legendax = figtopocompare.add_axes([0.97-0.85*w,0.97-0.85*h,w*0.85,h*0.85])
	legendax.plot([Times[0],np.round(Times[-1]*100)/100],np.zeros(2),'w')
	legendax.set_xlim(Times[0],np.round(Times[-1]*100)/100)
	legendax.set_xlabel('Time (s)',fontsize=8,labelpad=0)
	legendax.set_ylabel('µV',fontsize=8,labelpad=0)

	legendax.set_yticks([-AmpMax*1e6,AmpMax*1e6])
	legendax.set_yticklabels(np.round([-AmpMax*1e6,AmpMax*1e6],1), fontsize=7)
	legendax.set_xticks(np.arange(Times[0],(np.round(Times[-1]*100)/100)+0.2,0.2))
	legendax.set_xticklabels(np.round(np.arange(Times[0],(np.round(Times[-1]*100)/100)+0.2,0.2),1), fontsize=7)
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
		

		ax.set_xticks(np.arange(0,evokeds.times[-1],0.5))
		ax.set_xticklabels(np.round(np.arange(0,evokeds.times[-1],0.5),1), fontsize=6)
		
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
	legendax.set_xlabel('Time (s)',fontsize=8.5,labelpad=0)
	legendax.set_ylabel('µV',fontsize=8.5,labelpad=0)

	legendax.set_yticks([-AmpMax*1e6,AmpMax*1e6])
	legendax.set_yticklabels(np.round([-AmpMax*1e6,AmpMax*1e6],1), fontsize=7)
	legendax.set_xticks(np.arange(evokeds.times[0],evokeds.times[-1],0.2))
	legendax.set_xticklabels(np.round(np.arange(evokeds.times[0],evokeds.times[-1],0.2),1), fontsize=7)
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






def FitIcaEpoch(raw,events,picks,tmin,tmax,PercentageOfEpochsRejected):
    # Filter raw data to remove mains and DC offset   
    iir_Butter_params = dict(order=2, ftype='butter', output='sos')
    
    raw_tmp =raw.copy()
    
    raw_tmp.filter(l_freq = 0.5, h_freq = 20.,method = 'iir', iir_params=iir_Butter_params,verbose='ERROR')
    
    epochs = mne.Epochs(raw_tmp, events=events, event_id=None, tmin=tmin,tmax=tmax, preload=True,proj=True,baseline=None, reject=None, picks=picks,verbose='ERROR')
    
    # PercentageOfEpochsRejected = 2.0
    ThresholdPeak2peak,_,_,_,_ = RejectThresh(epochs,PercentageOfEpochsRejected)
    
    reject = {'eeg': ThresholdPeak2peak}    
    epochs = mne.Epochs(raw_tmp, events=events, event_id=None, tmin=tmin,tmax=tmax, preload=True,proj=True,baseline=None, reject=reject, picks=picks,verbose='ERROR')
   
    # Fit ica on epochs without big artifact
    ica = ICA(n_components=epochs.info['nchan'], method='fastica',random_state = 30).fit(epochs, picks=picks, decim = 10)
    compo_ica_epoch = ica.get_sources(epochs)
    return ica,compo_ica_epoch
    
    
    
def FitIcaRaw(raw, picks,n_components):
    
    # Read raw data
    raw_tmp =raw.copy()

    # Filter raw data to remove mains and DC offset   
    iir_Butter_params = dict(order=2, ftype='butter', output='sos') 
    raw_tmp.filter(l_freq = 0.5, h_freq = 20.,method = 'iir', iir_params=iir_Butter_params)       
    
    # Fit ica on epochs without big artifact
    ica = ICA(n_components=n_components, method='fastica',random_state = 30).fit(raw_tmp, picks=picks, decim = 10)
    return ica



def ComputeVarICAWeights(ica):
    fast_dot = np.dot
    VarIcaWeigths = np.zeros(ica.n_components_, dtype='float')
    for icomp in range(ica.n_components_):
        maps = fast_dot(ica.mixing_matrix_[:, icomp].T,ica.pca_components_[:ica.n_components_])
        mapsNorm=maps/(np.max(np.abs([maps.max(),maps.min()])))
        VarIcaWeigths[icomp] = np.var(mapsNorm)
        
    return VarIcaWeigths


def AddVirtualEogChannels(raw,ChanName4VEOG,ChanName4HEOG_l,ChanName4HEOG_r):
    FlagVEOG = False
    FlagHEOG_L = False
    FlagHEOG_R = False
    FlagHEOG = True 
    
    if ChanName4VEOG is not None:               
        rawSelecChan4Veog  = raw.copy().pick(ChanName4VEOG)
        rawVEogData = np.zeros((1, rawSelecChan4Veog.n_times), dtype='float')
        rawVEogData[0,:] = (rawSelecChan4Veog.get_data(picks=range(len(ChanName4VEOG))).sum(axis=0))
        FlagVEOG = True

     # Create virtual Horizontal EOG
    if (ChanName4HEOG_l is not None) : 
        rawSelecChan4Heog_l = raw.copy().pick(ChanName4HEOG_l)
        rawHEogL_Data = np.zeros((1, rawSelecChan4Heog_l.n_times), dtype='float')
        rawHEogL_Data[0,:] = (rawSelecChan4Heog_l.get_data(picks=range(len(ChanName4HEOG_l))).sum(axis=0))
        FlagHEOG_L = True
        
        
    if (ChanName4HEOG_r is not None):
        rawSelecChan4Heog_r = raw.copy().pick(ChanName4HEOG_r)
        rawHEogR_Data = np.zeros((1, rawSelecChan4Heog_r.n_times), dtype='float')
        rawHEogR_Data[0,:] = (rawSelecChan4Heog_r.get_data(picks=range(len(ChanName4HEOG_r))).sum(axis=0))
        FlagHEOG_R = True
        
    if FlagHEOG_L:
        if FlagHEOG_R:
            rawHEogData = rawHEogL_Data - rawHEogR_Data
        else:
            rawHEogData = rawHEogL_Data
    else:
        if FlagHEOG_R:
            rawHEogData = rawHEogR_Data
        else:
            FlagHEOG = False
    
    rawWithVirtEOG = raw.copy()
    if FlagVEOG:
        infoVEog = mne.create_info(['VEOG'], rawWithVirtEOG.info['sfreq'], ['eog'])
        VEogRawArray  = mne.io.RawArray(rawVEogData, infoVEog)
        rawWithVirtEOG.add_channels([VEogRawArray], force_update_info=True)
            
    if FlagHEOG:
        infoHEog = mne.create_info(['HEOG'], rawWithVirtEOG.info['sfreq'], ['eog'])
        HEogRawArray  = mne.io.RawArray(rawHEogData, infoHEog)
        rawWithVirtEOG.add_channels([HEogRawArray], force_update_info=True)
    
    return rawWithVirtEOG,FlagVEOG,FlagHEOG


def VirtualEog(raw, ica, fig_directory, ChanName4VEOG, ChanName4HEOG_l,ChanName4HEOG_r,threshold):
	rawWithVirtEOG,FlagVEOG,FlagHEOG = AddVirtualEogChannels(raw,ChanName4VEOG,ChanName4HEOG_l,ChanName4HEOG_r)
	picks_eeg = mne.pick_types(rawWithVirtEOG.info, meg=False, eeg=True, eog=True,stim=True, exclude='bads')
	iir_Butter_params = dict(order=2, ftype='butter', output='sos')
	rawWithVirtEOG.filter(l_freq = 0.1, h_freq = 15.,method = 'iir', iir_params=iir_Butter_params,picks=picks_eeg)
	dict_scaling=dict(eeg=100e-6)
# 	rawWithVirtEOG.plot(duration=20,n_channels=rawWithVirtEOG.info['nchan'],scalings=dict_scaling)
	
	
	Veog_inds=[]
	Heog_inds=[]
	head_tail = os.path.split(raw.filenames[0])
	raw_f = os.path.splitext(head_tail[1])[0]
	if FlagVEOG:
		Veog_inds, Veog_scores = ica.find_bads_eog(rawWithVirtEOG,ch_name ='VEOG',stop = rawWithVirtEOG.times[-1]-1,threshold=threshold,measure='correlation')
	if FlagHEOG:
		Heog_inds, Heog_scores = ica.find_bads_eog(rawWithVirtEOG,ch_name ='HEOG',stop = rawWithVirtEOG.times[-1]-1,threshold=threshold,measure='correlation')
	if not(not(Veog_inds) and not(Heog_inds)):
		ica.plot_sources(rawWithVirtEOG,picks= Veog_inds+Heog_inds,stop  =30 )
		if len(fig_directory)>0:
			fnamesavefigICAcomponents = fig_directory + raw_f + "-icacomp.jpg"
			plt.savefig(fnamesavefigICAcomponents)
	
	if FlagVEOG:
		ica.plot_scores(Veog_scores, exclude=Veog_inds)  # look at r scores of components
	plt.show()
	if len(fig_directory)>0:
		fnamesavefigIcaScoreVEOG = fig_directory + raw_f + "-icascoreVeog.jpg"
		plt.savefig(fnamesavefigIcaScoreVEOG)
	
	if FlagHEOG:
		ica.plot_scores(Heog_scores, exclude=Heog_inds)  # look at r scores of components
	
	if len(fig_directory)>0:
		fnamesavefigIcaScoreHEOG = fig_directory + raw_f + "-icascoreHeog.jpg"
		plt.savefig(fnamesavefigIcaScoreHEOG)
		
	if not(not(Veog_inds) and not(Heog_inds)):
		eog_inds = Veog_inds + Heog_inds
		ica.plot_components(eog_inds,ch_type='eeg')
		
		if len(fig_directory)>0:
			fnamesavefigICATopocomponents = fig_directory + raw_f + "-icatopo.jpg"
			plt.savefig(fnamesavefigICATopocomponents)
	
	if FlagVEOG:
		if FlagHEOG:
			IcaScore2save = {'Veog_scores': Veog_scores[Veog_inds], 'Veog_inds' :Veog_inds,  'Heog_scores': Heog_scores[Heog_inds], 'Heog_inds' :Heog_inds}
		else:
			IcaScore2save = {'Veog_scores': Veog_scores[Veog_inds], 'Veog_inds' :Veog_inds}
	else:
		if FlagHEOG:
			IcaScore2save = {'Heog_scores': Heog_scores[Heog_inds], 'Heog_inds' :Heog_inds}
		else:
			IcaScore2save=[]
	IcaWeightsVar2save = {'LogVarIcWeights' : np.log(ComputeVarICAWeights(ica))}
	if not(not(Veog_inds) and not(Heog_inds)):
		ica.exclude.extend(eog_inds)
	return ica,IcaWeightsVar2save,IcaScore2save
    
   
