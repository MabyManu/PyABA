# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:40:18 2021

@author: iView
"""

import numpy as np
import mne
from scipy.linalg import toeplitz, qr, inv, det, svd

import pyABA_algorithms, py_tools,mne_tools

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
    events_from_annot,_ = mne.events_from_annotations(raw, event_id=events_id)
    events_indexes =events_from_annot[:,0]-1 + Offset # indexes corresponding to the times of flashes ==> gotta convert them to integers
    
    events_from_annot_target,_ = mne.events_from_annotations(raw, event_id={list(events_id.keys())[0]:list(events_id.values())[0]})
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
    v1 = np.nansum(V1,axis=0) / float(nb_NO_target_features - 1)
    p1 = float(nb_NO_target_features) / nb_total_features
    t1 = np.log(p1) - np.sum(np.log(np.sqrt(v1)))

    m2 = np.nanmean(targets_features,axis=0)
    V2 = (targets_features - m2) ** 2
    v2 = np.nansum(V2,axis=0)  / float(nb_target_features - 1)
    p2  = 1 - p1
    t2  = np.log(p2) - np.sum(np.log(np.sqrt(v2)))

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
    
    DataMat= epochs.get_data()
    _,nb_channels,nb_samples  = DataMat.shape
    _,nb_virtual_channels = SpatialFiler.shape
    
    AllFeatures = np.zeros((NbBlocksPerCond,nb_samples * nb_virtual_channels))
    ix = 0
    for i_block in range(NbBlocksPerCond):
        slice_ix = np.arange(ix,ix + TabNbStimPerBlock[i_block],dtype=np.int64)
        slice_ix_rej = np.setdiff1d(slice_ix, np.intersect1d(ixEpochs2Remove,slice_ix))
        
        MeanCurr = np.squeeze(np.mean(DataMat[slice_ix_rej,:,:],axis=0))
        VirtMean = np.transpose(np.dot(MeanCurr.T,SpatialFiler))
        Feature = VirtMean.reshape(1,( nb_samples * nb_virtual_channels))
        
        AllFeatures[i_block,:] = Feature
        
        ix = ix + TabNbStimPerBlock[i_block]
    return AllFeatures






