# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:40:18 2021

@author: iView
"""

import numpy as np
import mne
from scipy.linalg import toeplitz, qr, inv, det, svd
from pyriemann.utils.base import expm, invsqrtm, logm, sqrtm
from pyriemann.utils.covariance import covariances_EP
from pyriemann.utils.mean import mean_riemann

import pyABA_algorithms, py_tools,mne_tools
from scipy.linalg import eigvalsh

#_______________________________________________________________________________
def Xdawn(raw, events_id, tmin_time_window_s, tmax_time_window_s, nb_spatial_filters):
    """
    %% Compute the QR decompositions of the signal Y and D matrices
    %a QR decomposition (also called a QR factorization) of a matrix is a
    %decomposition of a matrix A into a product A=QR of an orthogonal matrix Q 
    %and an upper triangular matrix R. QR decomposition is often used to solve 
    %the linear least squares problem, and is the basis for a particular 
    %eigenvalue algorithm, the QR algorithm.
    """
    signal_data = raw._data
    nb_channels, nb_samples = signal_data.shape # nb_channels = signal_data.shape[0] # nb_samples = signal_data.shape[1]
    
    
    Time_windows_s = tmax_time_window_s - tmin_time_window_s
    erp_nb_samples = int(np.round(Time_windows_s*raw.info["sfreq"])) + 1
    Offset = int(np.round(tmin_time_window_s*raw.info["sfreq"]))
    events_from_annot,_ = mne.events_from_annotations(raw, event_id=events_id,verbose='ERROR')
    events_indexes =events_from_annot[:,0]-1 + Offset # indexes corresponding to the times of flashes ==> gotta convert them to integers
    
    events_from_annot_target,_ = mne.events_from_annotations(raw, event_id={list(events_id.keys())[0]:list(events_id.values())[0]},verbose='ERROR')
    events_target_indexes = events_from_annot_target[:,0]-1 + Offset# indexes target stimulievents_id
    
    
    
    ColD1 = np.zeros(nb_samples)
    ColD2 = np.zeros(nb_samples)

    ColD1[events_indexes] = 1
    ColD2[events_target_indexes] = 1
    if ColD1[0] == 1:
        RowD1 = np.ones(erp_nb_samples)
    else:
        RowD1 = np.zeros(erp_nb_samples)
    D1 = toeplitz(ColD1, RowD1)
    if ColD2[0] == 1:
        RowD2 = np.ones(erp_nb_samples)
    else:
        RowD2 = np.zeros(erp_nb_samples)
    D2 = toeplitz(ColD2, RowD2)
    D = np.hstack((D1, D2, np.ones((D1.shape[0], 1))))
#    custom_print_in_blue('DTD', D.shape, 'D1', 'D2', D1.shape, D2.shape)
    DTD = np.dot(D.T, D)
    iDTD = inv(DTD)
    DTY  = np.dot(D.T, signal_data.T)
    A = np.dot(iDTD, DTY)
    D = D2
    signal_data = signal_data.astype('float64')
    Qy, Ry = qr(signal_data.T, mode='economic')
    D = D.astype('float64')
    Qd, Rd = qr(D, mode='economic')
    Q = np.dot(Qd.T, Qy)
    U, S, V = svd(Q, full_matrices=True)

    #%% Compute the spatial filters
    W = np.zeros((nb_channels, nb_spatial_filters))
    Ts = np.zeros((nb_channels, nb_spatial_filters))

    for i in range(nb_spatial_filters):
        W[:, i] = np.dot(inv(Ry), V.T[:, i])
    return W







#_______________________________________________________________________________
def NBlearn(targets_features,no_targets_features):
	"""
	%% linear Naive BaStim1 classifier (learning phase)
	%
	% INPUTS:
	% - X       matrix of features ([P x N])
	% - L       vector of Labels   ([P x 1])
	% - opt     1:default prior probabilities (1/6 5/6) 0:computed
	%
	% OUTPUTS:
	% - m1      mean vector for class 1 ([N x 1])
	% - v1      variance vector for class 1
	% - t1      log-term for class 1 (depends on the prior probability and variance)
	% - m2
	% - v2
	% - t2
	%
	"""
	nb_target_features, nb_samples_target_features = targets_features.shape
	nb_NO_target_features, nb_samples_NO_target_features = no_targets_features.shape
	nb_total_features = nb_target_features + nb_NO_target_features
	
	m1 = np.nanmean(no_targets_features,axis=0)
	V1 = (no_targets_features - m1) ** 2
	p1 = float(nb_NO_target_features) / nb_total_features

	
	
	if (nb_NO_target_features>1):
		v1 = np.nansum(V1,axis=0) / float(nb_NO_target_features - 1)
		t1 = np.log(p1) - np.sum(np.log(np.sqrt(v1)))

	else:
		v1 = np.nansum(V1,axis=0)
		t1 = np.log(p1)

	
	m2 = np.nanmean(targets_features,axis=0)
	V2 = (targets_features - m2) ** 2
	p2  = 1 - p1

	if (nb_target_features>1):
		v2 = np.nansum(V2,axis=0)  / float(nb_target_features - 1)
		t2  = np.log(p2) - np.sum(np.log(np.sqrt(v2)))

	else:
		v2 = np.nansum(V2,axis=0)		
		t2  = np.log(p2)

	
	return {
		'm1': m1,
		'v1': v1,
		't1': t1,
		'm2': m2,
		'v2': v2,
		't2': t2
	}




#_______________________________________________________________________________
def NBapply(NB_parameters, feature):

    """
    %% linear Naive BaStim1 (classification phase)
    %
    % INPUTS:
    % - m1      mean vector for class 1 ([N x 1])
    % - v1      variance vector for class 1 ([N x 1])
    % - t1      log-term for class 1 (depends on the class variance and prior probabilities)
    % - m2
    % - v2
    % - t2
    % - feature       new data to be classified
    %
    % OUTPUTS:
    % - f       NB functional
    % - expf    estimated belonging probablities for each data point ([P x 2])
    %
    
    note: m2 is the target class
    """

    vec1 = (feature - NB_parameters['m1']) ** 2 / 2.
    vec1 = vec1.astype('float') / NB_parameters['v1']
    sum_vec1 = - np.nansum(vec1)
    vec2 = (feature - NB_parameters['m2']) ** 2 / 2.
    vec2 = vec2.astype('float') / NB_parameters['v2']
    sum_vec2 = - np.nansum(vec2)

    Delta = NB_parameters['t2'] - NB_parameters['t1'] + sum_vec2 - sum_vec1
    return Delta







def CrossValidationOnBlocks(Features_Stim1Std_AttStim1,
                            Features_Stim1Std_AttStim2,
                            Features_Stim2Std_AttStim2,
                            Features_Stim2Std_AttStim1,
                            Features_Stim1Dev_AttStim1,
                            Features_Stim1Dev_AttStim2,
                            Features_Stim2Dev_AttStim2,
                            Features_Stim2Dev_AttStim1):

    NbBlocksPerCond = Features_Stim1Std_AttStim2.shape[0]
    
    # Attention Stim1 BLOCK
    p_stds_and_devs_AttStim1 = np.zeros(NbBlocksPerCond)
    p_stds_AttStim1 = np.zeros(NbBlocksPerCond)
    p_devs_AttStim1 = np.zeros(NbBlocksPerCond)
    p_Stim1Stim_AttStim1 = np.zeros(NbBlocksPerCond)
    p_Stim2Stim_AttStim1 = np.zeros(NbBlocksPerCond)
    for i_Block in range(NbBlocksPerCond):
        ## Stim1 STD
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim1Std = Features_Stim1Std_AttStim1
        Feat_Train_Stim1Std = np.delete(Feat_Train_Stim1Std, obj=i_Block, axis=0)
        NB_Param_Stim1Std = pyABA_algorithms.NBlearn(Feat_Train_Stim1Std, Features_Stim1Std_AttStim2)
        # Apply Classifier
        Feat_Test_Stim1Std = Features_Stim1Std_AttStim1[i_Block,:]
        Delta_Stim1Std = pyABA_algorithms.NBapply(NB_Param_Stim1Std, Feat_Test_Stim1Std)
        
        ## Stim1 DEV
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim1Dev = Features_Stim1Dev_AttStim1
        Feat_Train_Stim1Dev = np.delete(Feat_Train_Stim1Dev, obj=i_Block, axis=0)
        NB_Param_Stim1Dev = pyABA_algorithms.NBlearn(Feat_Train_Stim1Dev, Features_Stim1Dev_AttStim2)
        # Apply Classifier
        Feat_Test_Stim1Dev = Features_Stim1Dev_AttStim1[i_Block,:]
        Delta_Stim1Dev = pyABA_algorithms.NBapply(NB_Param_Stim1Dev, Feat_Test_Stim1Dev)
        
        ## Stim2 STD
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim2Std = Features_Stim2Std_AttStim1
        Feat_Train_Stim2Std = np.delete(Feat_Train_Stim2Std, obj=i_Block, axis=0)
        NB_Param_Stim2Std = pyABA_algorithms.NBlearn(Features_Stim2Std_AttStim2, Feat_Train_Stim2Std)
        # Apply Classifier
        Feat_Test_Stim2Std = Features_Stim2Std_AttStim1[i_Block,:]
        Delta_Stim2Std = pyABA_algorithms.NBapply(NB_Param_Stim2Std, Feat_Test_Stim2Std)
        
        ## Stim2 DEV
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim2Dev = Features_Stim2Dev_AttStim1
        Feat_Train_Stim2Dev = np.delete(Feat_Train_Stim2Dev, obj=i_Block, axis=0)
        NB_Param_Stim2Dev = pyABA_algorithms.NBlearn(Features_Stim2Dev_AttStim2, Feat_Train_Stim2Dev)
        # Apply Classifier
        Feat_Test_Stim2Dev = Features_Stim2Dev_AttStim1[i_Block,:]
        Delta_Stim2Dev = pyABA_algorithms.NBapply(NB_Param_Stim2Dev, Feat_Test_Stim2Dev)
        

        
        sum_delta_stds_and_devs = Delta_Stim1Std + Delta_Stim1Dev -  Delta_Stim2Std -  Delta_Stim2Dev    
        p_stds_and_devs_AttStim1[i_Block] = 1. / (1 + py_tools.expNoOverflow(- sum_delta_stds_and_devs))
        
        p_stds_AttStim1[i_Block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_Stim1Std-Delta_Stim2Std)))
        p_devs_AttStim1[i_Block]      = 1. / (1 + py_tools.expNoOverflow(- (Delta_Stim1Dev-Delta_Stim2Dev)))
        p_Stim1Stim_AttStim1[i_Block] = 1. / (1 + py_tools.expNoOverflow(- (Delta_Stim1Std + Delta_Stim1Dev)))
        p_Stim2Stim_AttStim1[i_Block] = 1. / (1 + py_tools.expNoOverflow(- (-  Delta_Stim2Std -  Delta_Stim2Dev)))
        

    

    # Attention Stim2 BLOCK
    p_stds_and_devs_AttStim2 = np.zeros(NbBlocksPerCond)
    p_stds_AttStim2 = np.zeros(NbBlocksPerCond)
    p_devs_AttStim2 = np.zeros(NbBlocksPerCond)
    p_Stim1Stim_AttStim2 = np.zeros(NbBlocksPerCond)
    p_Stim2Stim_AttStim2 = np.zeros(NbBlocksPerCond)
    for i_Block in range(NbBlocksPerCond):
        ## Stim1 STD
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim1Std = Features_Stim1Std_AttStim2
        Feat_Train_Stim1Std = np.delete(Feat_Train_Stim1Std, obj=i_Block, axis=0)
        NB_Param_Stim1Std = pyABA_algorithms.NBlearn(Features_Stim1Std_AttStim1, Feat_Train_Stim1Std)
        # Apply Classifier
        Feat_Test_Stim1Std = Features_Stim1Std_AttStim2[i_Block,:]
        Delta_Stim1Std = pyABA_algorithms.NBapply(NB_Param_Stim1Std, Feat_Test_Stim1Std)
        
        ## Stim1 DEV
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim1Dev = Features_Stim1Dev_AttStim2
        Feat_Train_Stim1Dev = np.delete(Feat_Train_Stim1Dev, obj=i_Block, axis=0)
        NB_Param_Stim1Dev = pyABA_algorithms.NBlearn( Features_Stim1Dev_AttStim1,Feat_Train_Stim1Dev)
        # Apply Classifier
        Feat_Test_Stim1Dev = Features_Stim1Dev_AttStim2[i_Block,:]
        Delta_Stim1Dev = pyABA_algorithms.NBapply(NB_Param_Stim1Dev, Feat_Test_Stim1Dev)
        
        ## Stim2 STD
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim2Std = Features_Stim2Std_AttStim2
        Feat_Train_Stim2Std = np.delete(Feat_Train_Stim2Std, obj=i_Block, axis=0)
        NB_Param_Stim2Std = pyABA_algorithms.NBlearn(Feat_Train_Stim2Std,Features_Stim2Std_AttStim1)
        # Apply Classifier
        Feat_Test_Stim2Std = Features_Stim2Std_AttStim2[i_Block,:]
        Delta_Stim2Std = pyABA_algorithms.NBapply(NB_Param_Stim2Std, Feat_Test_Stim2Std)
        
        ## Stim2 DEV
        # Compute Naive BaStim1 Parameters
        Feat_Train_Stim2Dev = Features_Stim2Dev_AttStim2
        Feat_Train_Stim2Dev = np.delete(Feat_Train_Stim2Dev, obj=i_Block, axis=0)
        NB_Param_Stim2Dev = pyABA_algorithms.NBlearn(Feat_Train_Stim2Dev,Features_Stim2Dev_AttStim1)
        # Apply Classifier
        Feat_Test_Stim2Dev = Features_Stim2Dev_AttStim2[i_Block,:]
        Delta_Stim2Dev = pyABA_algorithms.NBapply(NB_Param_Stim2Dev, Feat_Test_Stim2Dev)
        

        sum_delta_stds_and_devs = Delta_Stim1Std + Delta_Stim1Dev -  Delta_Stim2Std -  Delta_Stim2Dev    
        p_stds_and_devs_AttStim2[i_Block] = 1. / (1 + py_tools.expNoOverflow(- sum_delta_stds_and_devs))
        
         
        p_stds_AttStim2[i_Block] = 1.    / (1 + py_tools.expNoOverflow(- (Delta_Stim1Std-Delta_Stim2Std)))
        p_devs_AttStim2[i_Block] = 1.    / (1 + py_tools.expNoOverflow(- (Delta_Stim1Dev-Delta_Stim2Dev)))
        p_Stim1Stim_AttStim2[i_Block] = 1. / (1 + py_tools.expNoOverflow(- (Delta_Stim1Std + Delta_Stim1Dev)))
        p_Stim2Stim_AttStim2[i_Block] = 1.  / (1 + py_tools.expNoOverflow(- (-  Delta_Stim2Std -  Delta_Stim2Dev)))
      
        
    accuracy_stds_devs = (np.sum(p_stds_and_devs_AttStim1 > .5) + np.sum(p_stds_and_devs_AttStim2 <  .5)) / float(NbBlocksPerCond*2)
    accuracy_stds    = (np.sum(p_stds_AttStim1 > .5) + np.sum(p_stds_AttStim2 <  .5)) / float(NbBlocksPerCond*2)
    accuracy_devs    = (np.sum(p_devs_AttStim1 > .5) + np.sum(p_devs_AttStim2 <  .5)) / float(NbBlocksPerCond*2)
    accuracy_Stim1Stim = (np.sum(p_Stim1Stim_AttStim1 > .5) + np.sum(p_Stim1Stim_AttStim2 <  .5)) / float(NbBlocksPerCond*2)
    accuracy_Stim2Stim  = (np.sum(p_Stim2Stim_AttStim1 > .5) + np.sum(p_Stim2Stim_AttStim2 <  .5)) / float(NbBlocksPerCond*2)
    return accuracy_stds_devs, accuracy_stds, accuracy_devs, accuracy_Stim1Stim, accuracy_Stim2Stim




def Ave_Epochs_FeatComp(epochs,SpatialFiler,TabNbStimPerBlock,rejection_rate):
	_,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(epochs,int(rejection_rate*100))
	NbBlocksPerCond = len(TabNbStimPerBlock)  
	
	DataMat= epochs.get_data(copy=True)
	_,nb_channels,nb_samples  = DataMat.shape
	_,nb_virtual_channels = SpatialFiler.shape
	AllFeatures = np.zeros((NbBlocksPerCond,nb_samples * nb_virtual_channels))
	ix = 0
	for i_block in range(NbBlocksPerCond):
		slice_ix = np.arange(ix,ix + TabNbStimPerBlock[i_block],dtype=np.int64)
		slice_ix_rej = np.setdiff1d(slice_ix, np.intersect1d(ixEpochs2Remove,slice_ix))
		if (len(slice_ix_rej)>0):
			MeanCurr = np.squeeze(np.nanmean(DataMat[slice_ix_rej,:,:],axis=0))
			VirtMean = np.transpose(np.dot(MeanCurr.T,SpatialFiler))
			Feature = VirtMean.reshape(1,( nb_samples * nb_virtual_channels))
		else:
			Feature = np.zeros(nb_samples * nb_virtual_channels)
		AllFeatures[i_block,:] = Feature
		ix = ix + TabNbStimPerBlock[i_block]
	return AllFeatures




def matCov(MatEpoch, ERP_Template_Target):
    All_MatCov = []
    # Concatenate each epoch with mean of targets epochs
    for i, epoch in enumerate(MatEpoch):
        # MATRICE DE COV
        MatCov = covariances_EP(epoch, ERP_Template_Target)

        # Rempli deux listes avec les matrices
        All_MatCov.append(MatCov)

    # Convert la liste en matrice
    MatCovAll = np.array(All_MatCov)
    return MatCovAll


#_______________________________________________________________________________

# def covariances_EP(X, P):
#     Ne, Ns = X.shape
#     Np, Ns = P.shape

#     covmats = np.cov(np.concatenate((X, P), axis=0))
#     return covmats
	
#_______________________________________________________________________________
def predict_R_TNT(X, mu_MatCov_T,mu_MatCov_NT):
    """
    Predict the r_TNT for a new set of trials.
    """
    
    dist_0 = distance_riemann(X, mu_MatCov_T)
    dist_1 = distance_riemann(X, mu_MatCov_NT)
    
     

    
    r_TNT = np.log(dist_0 / dist_1)
    
    return r_TNT

#_______________________________________________________________________________
def compute_rTNT(MatCov_Trial, mean_MatCov_Target, mean_MatCov_NoTarget):
    All_rTNT = []
    for i, epoch in enumerate(MatCov_Trial):
        # dT = distance_riemann(epoch, mean_MatCov_Target)
        # dNT = distance_riemann(epoch, mean_MatCov_NoTarget)
        # All_rTNT.append(np.log(dT/dNT))
        All_rTNT.append(predict_R_TNT(epoch, mean_MatCov_Target,mean_MatCov_NoTarget))

    All_rTNT = np.array(All_rTNT)

    # MOYENNES des rTNT
    Mu_rTNT = np.mean(All_rTNT)

    # Variance des rTNT
    Var_rTNT = np.var(All_rTNT)

    return Mu_rTNT, Var_rTNT, All_rTNT


#_______________________________________________________________________________
			


	
#_______________________________________________________________________________
def distance_riemann(A, B):
    """Riemannian distance between two covariance matrices A and B.
    .. math::
    d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}
		
    where :math:`\lambda_i` are the joint eigenvalues of A and B
		
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B
		
    """
    l_logsquare = np.sum(np.log(eigvalsh(A, B))**2)

    return np.sqrt(l_logsquare)
    
		
		
#_______________________________________________________________________________
def compute_likelihood( l_r_TNT,  l_mu_TNT_T, l_mu_TNT_NT, l_sigma_TNT_T, l_sigma_TNT_NT):
		# 0 is target, 1 is nontarget
		
    Vec0 = (l_r_TNT - l_mu_TNT_T) ** 2
    Vec0 = Vec0 / l_sigma_TNT_T
		
    Vec1 = (l_r_TNT - l_mu_TNT_NT) ** 2
    Vec1 = Vec1 / l_sigma_TNT_NT
		
		
    ld0 = np.log( 2 *np. pi *l_sigma_TNT_T)
    ld1 = np.log(2 * np.pi * l_sigma_TNT_NT)
		
		
    lf0 = - 0.5 * (Vec0 + ld0)
    lf1 = - 0.5 * (Vec1 + ld1)
		

    return np.array([lf0 , lf1])               
                


#_______________________________________________________________________________
def ComputePostProba_BayesInference(Likelihood,NbItems,NbFlashs,PaternsItemsFlashed):
    """
    Compute posterior probabilities for a item to be the target

    %% ATTENTION
    % class 1: non-targets
    % class 2: targets
    """
    
    items_bayes_priors_array = np.ones(NbItems) / NbItems
    flashed_item_posteriors_array = np.zeros(NbItems)
    
    for i_flashs in range(NbFlashs):
        PaternsItemsFlashed_Curr = PaternsItemsFlashed[i_flashs]
        Likelihood_Curr = Likelihood[i_flashs]
        for j_items in range(NbItems):
            if np.isin((j_items+1),PaternsItemsFlashed_Curr):
                flashed_item_posteriors_array[j_items] = Likelihood_Curr[0] - Likelihood_Curr[1] + np.log(items_bayes_priors_array[j_items])
                # flashed_item_posteriors_array[j_items] = Likelihood_Curr[0] + np.log(items_bayes_priors_array[j_items])
            else:
                flashed_item_posteriors_array[j_items] = np.log(items_bayes_priors_array[j_items])
                # flashed_item_posteriors_array[j_items] = Likelihood_Curr[1] + np.log(items_bayes_priors_array[j_items])

  
        #normalize posteriors array
        max_flashed_item_posteriors_array = flashed_item_posteriors_array.max()
        flashed_item_posteriors_array = flashed_item_posteriors_array - max_flashed_item_posteriors_array + 1
        exp_flashed_item_posteriors_array = np.exp(flashed_item_posteriors_array)
        posteriors = np.divide(exp_flashed_item_posteriors_array, exp_flashed_item_posteriors_array.sum())
        items_bayes_priors_array = posteriors
        # print('post vector', posteriors)

    return posteriors
################################################################################                


def riemann_template_learn(EpochsTargets, EpochsNoTargets, rejection_rate, Gain):
    """This function is generating a riemann template.

    It take one .vhdr path in argument and return
    a dict containing all calibration data.

    Riemann template is use to calibrate the MYB game.

    :param EpochsTargets: Targets Epochs for Calibration.
    :type  EpochsTargets: mne epochs object
    
    :param EpochsNoTargets: No Targets Epochs for Calibration.
    :type  EpochsNoTargets: mne epochs object
    
    :param rejection_rate: Determined the rate of epochs rejection. 
    :type rejection_rate: float
    

    """
    Epo_Targ = EpochsTargets.copy()
    _,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epo_Targ,rejection_rate*100)
     # removing epochs and events to get thresholded data
    Epo_Targ.drop(ixEpochs2Remove,verbose=False)
    
    Epo_NoTarg = EpochsNoTargets.copy()    
    _,_,_,ixEpochs2Remove,_ = mne_tools.RejectThresh(Epo_NoTarg,rejection_rate*100)
     # removing epochs and events to get thresholded data
    Epo_NoTarg.drop(ixEpochs2Remove,verbose=False)
    
    
    epochs_T  = Epo_Targ.get_data(copy=True)*Gain
    epochs_NT = Epo_NoTarg.get_data(copy=True)*Gain
    ERP_Template_Target = np.mean(epochs_T, axis=0)
    ERP_Template_NoTarget = np.mean(epochs_NT, axis=0)

    varERP_Template_Target = np.var(epochs_T, axis=0)
    varERP_Template_NoTarget = np.var(epochs_NT, axis=0)
    
    matCov_TrialTarget = covariances_EP(epochs_T, ERP_Template_Target, 'oas')
    matCov_TrialNoTarget = covariances_EP(epochs_NT, ERP_Template_Target, 'oas')

    # matCov_TrialTarget = tools_Riemann.matCov(epochs_T, ERP_Template_Target)
    # matCov_TrialNoTarget = tools_Riemann.matCov(epochs_NT, ERP_Template_Target)

    matCov_TrialTarget = np.array(matCov_TrialTarget)
    matCov_TrialNoTarget = np.array(matCov_TrialNoTarget)
   
    # TODO check `logm` stability
    
   
    
    mean_MatCov_Target = mean_riemann(matCov_TrialTarget)
    
    mean_MatCov_NoTarget = mean_riemann(matCov_TrialNoTarget)
    
    


    mu_rTNT_TrialTarget,  var_rTNT_TtrialTarget, all_rTNT_TrialTarget = compute_rTNT(matCov_TrialTarget, mean_MatCov_Target, mean_MatCov_NoTarget)
    mu_rTNT_TrialNoTarget, Var_rTNT_TrialNoTarget, all_rTNT_TrialNoTarget = compute_rTNT(matCov_TrialNoTarget, mean_MatCov_Target, mean_MatCov_NoTarget)

    NbGoodTarget = float(np.sum(all_rTNT_TrialTarget < .0))
    NbGoodNoTarget = float(np.sum(all_rTNT_TrialNoTarget > .0))
    NbTotTrials = float(all_rTNT_TrialTarget.shape[0] + all_rTNT_TrialNoTarget.shape[0])

    accP300 = np.float64((NbGoodTarget)*100 / len(all_rTNT_TrialTarget))

    riemann_template = {}
    riemann_template['mu_Epoch_T'] = ERP_Template_Target
    riemann_template['mu_Epoch_NT'] = ERP_Template_NoTarget
    riemann_template['var_Epoch_T'] = varERP_Template_Target
    riemann_template['var_Epoch_NT'] = varERP_Template_NoTarget
    riemann_template['mu_MatCov_T'] = mean_MatCov_Target
    riemann_template['mu_MatCov_NT'] = mean_MatCov_NoTarget
    riemann_template['mu_rTNT_T'] = mu_rTNT_TrialTarget
    riemann_template['mu_rTNT_NT'] = mu_rTNT_TrialNoTarget
    riemann_template['sigma_rTNT_T'] = var_rTNT_TtrialTarget
    riemann_template['sigma_rTNT_NT'] = Var_rTNT_TrialNoTarget
    riemann_template['accP300'] = accP300

    return riemann_template






def riemannOneBlockApply(riemann_template,Epoch_Test,Gain):
     
    # Extract Riamman Template settings
    ERP_Template_Target = riemann_template['mu_Epoch_T'][...]
    MatCov_T            = riemann_template['mu_MatCov_T'][...]
    MatCov_NT           = riemann_template['mu_MatCov_NT'][...]
    mu_rTNT_T           = riemann_template['mu_rTNT_T']
    mu_rTNT_NT          = riemann_template['mu_rTNT_NT']
    sigma_rTNT_T        = riemann_template['sigma_rTNT_T']
    sigma_rTNT_NT       = riemann_template['sigma_rTNT_NT']
    
       
    l_nbFlashsPerBlock = len(Epoch_Test)
    epo_tmp = Epoch_Test.copy()
    LabelEvt = epo_tmp.events[:,2]
    ColTarget = np.unique(LabelEvt[ np.where(LabelEvt>10)] - 10)[0]
    LabelEvt[ np.where(LabelEvt>10)] = LabelEvt[ np.where(LabelEvt>10)] - 10
    Curr_LikelihoodFunction =np.zeros((l_nbFlashsPerBlock,2))
    R_TNT_tot =np.zeros(l_nbFlashsPerBlock)

    v_NbItemsPerPart = len(np.unique(LabelEvt))
    SingleTrial_curr = np.zeros((1,Epoch_Test.get_data(copy=True).shape[1],Epoch_Test.get_data(copy=True).shape[2]))
    for j_flashs in range(l_nbFlashsPerBlock):
        SingleTrial_curr[0,:,:] = Epoch_Test.get_data(copy=True)[j_flashs,:,:]*Gain
        # Curr_Cov = tools_Riemann.covariances_EP(SingleTrial_curr, ERP_Template_Target)
        Curr_Cov = np.squeeze(covariances_EP(SingleTrial_curr, ERP_Template_Target, estimator='oas'))
        Curr_r_TNT = np.squeeze(predict_R_TNT(Curr_Cov, MatCov_T,MatCov_NT))
        R_TNT_tot[j_flashs]=Curr_r_TNT
        if hasattr(mu_rTNT_T, "__len__"):
            Curr_LikelihoodFunction[j_flashs,:] = compute_likelihood(Curr_r_TNT, mu_rTNT_T[0], mu_rTNT_NT[0],sigma_rTNT_T[0],sigma_rTNT_NT[0])
        else:
            Curr_LikelihoodFunction[j_flashs,:] = compute_likelihood(Curr_r_TNT, mu_rTNT_T, mu_rTNT_NT,sigma_rTNT_T,sigma_rTNT_NT)

    NbGoodTarget = float(np.sum((R_TNT_tot < .0) * Epoch_Test.events[:,2]>10))
    NbGoodNoTarget = float(np.sum((R_TNT_tot > .0) * Epoch_Test.events[:,2]<10))
    NbTotTrials = len(R_TNT_tot)

    accP300 = np.float64((NbGoodTarget)*100 / np.sum(Epoch_Test.events[:,2]>10))
    
    BoolGoodSelect = (np.argmax(ComputePostProba_BayesInference(Curr_LikelihoodFunction, v_NbItemsPerPart, l_nbFlashsPerBlock, LabelEvt))+1) == ColTarget
    ItemSelected = (np.argmax(ComputePostProba_BayesInference(Curr_LikelihoodFunction, v_NbItemsPerPart, l_nbFlashsPerBlock, LabelEvt))+1)
     
    return BoolGoodSelect,accP300,ItemSelected,ColTarget

