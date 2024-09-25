# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:34:53 2023

@author: manum
"""
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path + '/PyGazeAnalyser')
from pygazeanalyser import detectors
from pygazeanalyser.gazeplotter import draw_heatmap


sys.path.append(PyABA_path)
import py_tools

def computeVelocity(Gaze_X,Gaze_Y,Times_ms):
    intdist = (np.diff(Gaze_X)**2 + np.diff(Gaze_Y)**2)**0.5
    # get inter-sample times
    inttime = np.diff(Times_ms)
    # recalculate inter-sample times to seconds
    inttime = inttime / 1000.0
	
    # VELOCITY AND ACCELERATION
    # the velocity between samples is the inter-sample distance
    # divided by the inter-sample time
    vel = intdist / inttime
    return vel



def Resamp_EyeTrackerData(Orig_Times,Orig_Data,New_Times):
    ix = np.where(Orig_Data==0.0)
    Orig_Data[ix]=np.NaN    
    f = interpolate.interp1d(Orig_Times,Orig_Data)   
    Resamp_Data = f(New_Times)   # use interpolation function returned by `interp1d`  
    
    return Resamp_Data



def ComputeFeature_GazeEpoch(Gaze_data,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg):
	Latency_InitSacc=np.NaN
	Fix_Baseline = np.empty(5)
	Fix_Baseline[:] = np.nan
	Baseline_Latencies = np.empty(2)
	Baseline_Latencies[:] = np.nan
	Fix_OnTarget_LatStart = []
	Fix_OnTarget_LatEnd = []
	FixDurationOnTarget = 0
	NbNan = 0
	NbSampFix = 0
	PercentNanInFixPeriod = np.nan
	FixPeriod_GazePosition = np.nan
	FixPeriod_GazePosition_Std = np.nan
	FixPeriod_GainAmp = np.nan
	FixPeriod_GainAUC = np.nan
	GainAUC_TargetDisplayPeriod = np.nan
	
	ix_ShowTarget = np.where((Times_epoch==0.0))[0][0]
	ix_HideTarget  = np.where((Times_epoch==TargetFixationDuration))[0][0]
	    
	Ssac, Esac = detectors.saccade_detection(Gaze_data, Gaze_data, Times_epoch*SampFreq, minlen = 20,  maxvel=MaxVelocitySaccade)
	Sacc2Target_PosStart = []
	Sacc2Target_PosEnd = []
	Sacc2Target_LatStart = []
	Sacc2Target_LatEnd = []
	Sacc2Target_Amp = []
	
	for isac in range(np.shape(Esac)[0]):
		Amp_Sacc_Curr = Esac[isac][5]-Esac[isac][3]
		if ((Esac[isac][0] < TargetFixationDuration*SampFreq) & (Esac[isac][0] > -500.0)):
			if (np.abs(Amp_Sacc_Curr)>(SaccAmp_Min_Deg/Pix2DegCoeff)):
				if (py_tools.SameSign(Amp_Sacc_Curr,Target_Pix_RefCross)):
					Sacc2Target_PosStart = np.append(Sacc2Target_PosStart,Esac[isac][3])
					Sacc2Target_PosEnd = np.append(Sacc2Target_PosEnd,Esac[isac][4])
					Sacc2Target_LatStart = np.append(Sacc2Target_LatStart,Esac[isac][0])
					Sacc2Target_LatEnd = np.append(Sacc2Target_LatEnd,Esac[isac][1])
					Sacc2Target_Amp = np.append(Sacc2Target_Amp,Esac[isac][5]-Esac[isac][3])
	
	if (len(Sacc2Target_LatStart)>0):
		Latency_InitSacc=Sacc2Target_LatStart[np.argmax(np.abs(Sacc2Target_Amp))]
		PosStart_InitSacc=Sacc2Target_PosStart[np.argmax(np.abs(Sacc2Target_Amp))]
	
	Fix_LatEnd = []
	Fix_LatStart = []
	Fix_Pos = []
	Fix_ReturnBL = []
	
	Sfix, Efix = detectors.fixation_detection(Gaze_data, Gaze_data, Times_epoch*SampFreq, missing=np.NaN, maxdist=MaxDistFixation_deg/Pix2DegCoeff, mindur=75)
	
	# Define Baseline Period
	for ifix in range(np.shape(Efix)[0]):
		if (Efix[ifix][0] <  50.0):
			Fix_LatEnd = np.append(Fix_LatEnd,Efix[ifix][1])
			Fix_LatStart = np.append(Fix_LatStart,Efix[ifix][0])
			Fix_Pos = np.append(Fix_Pos,Efix[ifix][3])
	
	if (len(Fix_LatEnd)>0):
		if not(np.isnan( Latency_InitSacc)): # case : Init Saccade
			if len(np.where(Fix_LatStart<Latency_InitSacc)[0])>0:
# 				Fix_Baseline = Efix[np.argmin((Latency_InitSacc- Fix_LatStart[np.where(Fix_LatStart<Latency_InitSacc)[0]]))]
				Fix_Baseline = Efix[np.argmin((Cross_pos- Fix_Pos[np.where(Fix_LatStart<Latency_InitSacc)[0]]))]
			else:  
				Fix_Baseline = [Times_epoch[0]*SampFreq , Latency_InitSacc, Latency_InitSacc-Times_epoch[0]*SampFreq, PosStart_InitSacc]
		else: # case : No Init Saccade
			if (len(Efix)>0): 
				Fix_Baseline = Efix[np.argmin(np.abs(Fix_LatEnd))]
		
		diffAmpBL = 1e9
		for ifix in range(np.shape(Efix)[0]):
			if ((Efix[ifix][1] >  TargetFixationDuration*SampFreq) ):
				if ((Efix[ifix][3] != Fix_Baseline[3]) & (Efix[ifix][0]>Fix_Baseline[1])):
					if(np.abs((Efix[ifix][3] - Cross_pos))<diffAmpBL):						
						diffAmpBL = np.abs(Efix[ifix][3] - Cross_pos)
						Fix_ReturnBL =  Efix[ifix]
		AllFixData = []
		if (len(Fix_ReturnBL)>0):
			AllFixData = []
			for ifix in range(np.shape(Efix)[0]):
				if (Efix[ifix][0] <Fix_ReturnBL[0]): # Fixation begins before start return to cross Period
					cond1 = (Efix[ifix][0] > Fix_Baseline[1]) # Fixations stops after the end of the Baseline
					cond4 = py_tools.SameSign(Efix[ifix][3]-Fix_Baseline[3],Target_Pix_RefCross) # Fixation on same side of target
					cond2 = (np.abs(Efix[ifix][3]-Fix_ReturnBL[3])>np.abs((Target_Pos_Px-Cross_pos)/3)) # Fixation amplitude -/- Return 2 Cross Period > (distance Target-Cross)/3
					cond3 = (np.abs(Efix[ifix][3]-Fix_Baseline[3])>np.abs((Target_Pos_Px-Cross_pos)/3)) # Fixation amplitude -/- Baseline Period > (distance Target-Cross)/3
					if (((cond2| cond3))*cond1*cond4):
						Fix_OnTarget_LatStart = np.append(Fix_OnTarget_LatStart,Efix[ifix][0])
						Fix_OnTarget_LatEnd = np.append(Fix_OnTarget_LatEnd,Efix[ifix][1])
						FixDurationOnTarget =  FixDurationOnTarget + (Efix[ifix][1]-Efix[ifix][0])
						ix_start_Fix = np.where((Times_epoch*SampFreq==Efix[ifix][0]))[0][0]
						ix_stop_Fix  = np.where((Times_epoch*SampFreq==Efix[ifix][1]))[0][0]
						FixData_Curr = Gaze_data[ix_start_Fix:ix_stop_Fix]
						AllFixData = np.append(AllFixData,FixData_Curr)
						NbNan = NbNan + len(np.where(np.isnan(Gaze_data[ix_start_Fix:ix_stop_Fix]))[0])
						NbSampFix = NbSampFix + len(Gaze_data[ix_start_Fix:ix_stop_Fix])
		else:
			for ifix in range(np.shape(Efix)[0]):
				if ((Efix[ifix][0] > Fix_Baseline[1]) & (np.abs(Efix[ifix][3]-Fix_Baseline[3])>np.abs((Target_Pos_Px-Cross_pos)/3))):
					Fix_OnTarget_LatStart = np.append(Fix_OnTarget_LatStart,Efix[ifix][0])
					Fix_OnTarget_LatEnd = np.append(Fix_OnTarget_LatEnd,Efix[ifix][1])
					FixDurationOnTarget =  FixDurationOnTarget + (Efix[ifix][1]-Efix[ifix][0])
					ix_start_Fix = np.where((Times_epoch*SampFreq==Efix[ifix][0]))[0][0]
					ix_stop_Fix  = np.where((Times_epoch*SampFreq==Efix[ifix][1]))[0][0]
					FixData_Curr = Gaze_data[ix_start_Fix:ix_stop_Fix]
					AllFixData = np.append(AllFixData,FixData_Curr)
					NbNan = NbNan + len(np.where(np.isnan(Gaze_data[ix_start_Fix:ix_stop_Fix]))[0])
					NbSampFix = NbSampFix + len(Gaze_data[ix_start_Fix:ix_stop_Fix])

	if (NbSampFix>0):
		PercentNanInFixPeriod = NbNan/NbSampFix
		FixPeriod_GazePosition = np.nanmedian(AllFixData)
		FixPeriod_GazePosition_Std = np.nanstd(AllFixData)
		FixPeriod_GainAmp = (FixPeriod_GazePosition-Fix_Baseline[3])/Target_Pix_RefCross
		
		AUC = np.nansum(AllFixData-Fix_Baseline[3])
		FixPeriod_GainAUC = AUC / (Target_Pix_RefCross*((TargetFixationDuration*SampFreq)-NbNan))
		GainAUC_TargetDisplayPeriod = np.nansum(Gaze_data[ix_ShowTarget:ix_HideTarget]-Fix_Baseline[3])/(Target_Pix_RefCross*((TargetFixationDuration*SampFreq)-NbNan))
		
		
	Baseline_Latencies = [Fix_Baseline[0] , Fix_Baseline[1]]
	
	return {'InitSaccade_Latency': Latency_InitSacc,
               'GazeBaseline_Latencies' : Baseline_Latencies,
               'FixationOnTarget_StartLatency' : Fix_OnTarget_LatStart,
               'FixationOnTarget_EndLatency' : Fix_OnTarget_LatEnd,
               'FixationOnTarget_Duration' : FixDurationOnTarget,
               'FixationOnTarget_PercentOfNan': PercentNanInFixPeriod,
			   'FixationOnTarget_GazePosition': FixPeriod_GazePosition,
			   'FixationOnTarget_AmplitudeGain' : FixPeriod_GainAmp,	
			   'FixationOnTarget_FixationStability' : FixPeriod_GazePosition_Std*Pix2DegCoeff,
			   'FixationOnTarget_AUCGain' : FixPeriod_GainAUC,
			   'TargetDisplayPeriod_AUCGain' : GainAUC_TargetDisplayPeriod}	
    
    
    



def PlotFixationGaze_STEP(DATA, Times_epoch, SampFreq, TargetName, Target_PixPosition,TargetFixationDuration, Cross_PixPosition,Pix2DegCoeff,SaccAmp_Min_Deg):
	MaxVelocitySaccade = 400
	MaxDistFixation_deg = 2
	NbTrials = np.shape(DATA)[0]
	
	Latency_InitSacc_LeftEye    = np.empty(NbTrials)
	Latency_InitSacc_LeftEye[:] = np.nan
	
	PosStart_InitSacc_LeftEye    = np.empty(NbTrials)
	PosStart_InitSacc_LeftEye[:] = np.nan
	
	FixDurationOnTarget_LeftEye    = np.empty(NbTrials)
	FixDurationOnTarget_LeftEye[:] = np.nan
	
	EpochMissingDataPerCent_LeftEye   = np.empty(NbTrials)
	EpochMissingDataPerCent_LeftEye[:] = np.nan
	
	FixationOnTarget_AmplitudeGain_LeftEye   = np.empty(NbTrials)
	FixationOnTarget_AmplitudeGain_LeftEye[:] = np.nan
	
	Latency_InitSacc_RightEye    = np.empty(NbTrials)
	Latency_InitSacc_RightEye[:] = np.nan
	
	PosStart_InitSacc_RightEye    = np.empty(NbTrials)
	PosStart_InitSacc_RightEye[:] = np.nan
	
	FixDurationOnTarget_RightEye    = np.empty(NbTrials)
	FixDurationOnTarget_RightEye[:] = np.nan
	
	EpochMissingDataPerCent_RightEye   = np.empty(NbTrials)
	EpochMissingDataPerCent_RightEye[:] = np.nan
	
	FixationOnTarget_AmplitudeGain_RightEye   = np.empty(NbTrials)
	FixationOnTarget_AmplitudeGain_RightEye[:] = np.nan	
	
	FixationOnTarget_VariabilityOfFixation_LeftEye   = np.empty(NbTrials)
	FixationOnTarget_VariabilityOfFixation_LeftEye[:] = np.nan		
	
	FixationOnTarget_VariabilityOfFixation_RightEye   = np.empty(NbTrials)
	FixationOnTarget_VariabilityOfFixation_RightEye[:] = np.nan		
	
	LE_X = DATA[:,0,:]
	LE_Y = DATA[:,1,:]
	RE_X = DATA[:,2,:]
	RE_Y = DATA[:,3,:]
	
	NbCol = int(np.ceil(np.sqrt(NbTrials)))
	NbRow = int(np.ceil(NbTrials/NbCol))
	
	if (Target_PixPosition[1] == 0):
		Target_Pix_RefCross = Target_PixPosition[0]
		LE_Data = LE_X
		RE_Data = RE_X
		Cross_pos = Cross_PixPosition[0]
	else:
		Target_Pix_RefCross = Target_PixPosition[1]
		LE_Data = LE_Y
		RE_Data = RE_Y
		Cross_pos = Cross_PixPosition[1]
	
	Target_Sig = np.zeros(len(Times_epoch))
	Target_Sig[np.where((Times_epoch>0) & (Times_epoch<TargetFixationDuration))[0]] = Target_Pix_RefCross*Pix2DegCoeff
	SignTarget = Target_Pix_RefCross/(np.abs(Target_Pix_RefCross))
	Target_Pos_Px = Cross_pos + Target_Pix_RefCross
	
	fig_curr  =plt.figure()
	for i_trials in range(NbTrials):   # LOOP ON TRIALS
		LE_Data_curr = LE_Data[i_trials,:]
		RE_Data_curr = RE_Data[i_trials,:]
		
		ax = plt.subplot(NbRow, NbCol, i_trials + 1)
		ax.plot(Times_epoch,(LE_Data_curr-Cross_pos)*Pix2DegCoeff,'r',linewidth=1)
		ax.plot(Times_epoch,(RE_Data_curr-Cross_pos)*Pix2DegCoeff,'g',linewidth=1)
		ax.plot(Times_epoch,Target_Sig,'b',linewidth=1.5,linestyle='dotted')
		ax.set_ylim(bottom=-25, top=25)
		
		
		
		# Process Left Eye
		Results_LeftEye = ComputeFeature_GazeEpoch(LE_Data_curr,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg)
		
		Latency_InitSacc_LeftEye[i_trials] = Results_LeftEye['InitSaccade_Latency']
		FixDurationOnTarget_LeftEye[i_trials] = Results_LeftEye['FixationOnTarget_Duration']
		Fix_OnTarget_LatStart = Results_LeftEye['FixationOnTarget_StartLatency']
		Fix_OnTarget_LatEnd = Results_LeftEye['FixationOnTarget_EndLatency']
		Baseline_Latencies_LeftEye =  Results_LeftEye['GazeBaseline_Latencies']
		FixationOnTarget_AmplitudeGain_LeftEye[i_trials] =  np.log(Results_LeftEye['FixationOnTarget_AmplitudeGain'])
		EpochMissingDataPerCent_LeftEye[i_trials] = len(np.where(np.isnan(LE_Data_curr))[0])/len(LE_Data_curr)
		FixationOnTarget_VariabilityOfFixation_LeftEye[i_trials] = Results_LeftEye['FixationOnTarget_FixationStability']
		
		
		# Display Left Eye
		
		ax.axvline(Latency_InitSacc_LeftEye[i_trials]/SampFreq,0,1,linestyle='dotted',color = 'firebrick',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_LeftEye[0]/SampFreq,Baseline_Latencies_LeftEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*4,'Init Lat LE : ' + str(Latency_InitSacc_LeftEye[i_trials]) + ' ms',fontsize=6,color='firebrick')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='r',alpha=0.2)
		ax.text(0.5,-1*SignTarget*10,'log(AmpGain) : ' + f"{FixationOnTarget_AmplitudeGain_LeftEye[i_trials]:.3f}" ,fontsize=6,color = 'firebrick')
		ax.text(0.5,-1*SignTarget*16,'Targ Fix Dur : ' + f"{FixDurationOnTarget_LeftEye[i_trials]:.0f}" + ' ms',fontsize=6,color = 'firebrick')
		ax.text(0.5,-1*SignTarget*22,'Var Fix  : ' + f"{FixationOnTarget_VariabilityOfFixation_LeftEye[i_trials]:.3f}" + ' °',fontsize=6,color = 'firebrick')
		
		
		# Process Right Eye
		Results_RightEye = ComputeFeature_GazeEpoch(RE_Data_curr,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg)
		
		Latency_InitSacc_RightEye[i_trials] = Results_RightEye['InitSaccade_Latency']
		FixDurationOnTarget_RightEye[i_trials] = Results_RightEye['FixationOnTarget_Duration']
		Fix_OnTarget_LatStart = Results_RightEye['FixationOnTarget_StartLatency']
		Fix_OnTarget_LatEnd = Results_RightEye['FixationOnTarget_EndLatency']
		Baseline_Latencies_RightEye =  Results_RightEye['GazeBaseline_Latencies']
		FixationOnTarget_AmplitudeGain_RightEye[i_trials] = np.log(Results_RightEye['FixationOnTarget_AmplitudeGain'])
		EpochMissingDataPerCent_RightEye[i_trials] = len(np.where(np.isnan(RE_Data_curr))[0])/len(RE_Data_curr)
		FixationOnTarget_VariabilityOfFixation_RightEye[i_trials] = Results_RightEye['FixationOnTarget_FixationStability']

		
		
		# Display Right Eye
		
		ax.axvline(Latency_InitSacc_RightEye[i_trials]/SampFreq,0,1,linestyle='dotted',color = 'darkgreen',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_RightEye[0]/SampFreq,Baseline_Latencies_RightEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*1,'Init Lat : ' + str(Latency_InitSacc_RightEye[i_trials]) + ' ms',fontsize=6,color='darkgreen')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='g',alpha=0.2)
		ax.text(0.5,-1*SignTarget*7,'log(AmpGain) : ' + f"{FixationOnTarget_AmplitudeGain_RightEye[i_trials]:.3f}" ,fontsize=6,color = 'darkgreen')
		ax.text(0.5,-1*SignTarget*13,'Targ Fix Dur : ' + f"{FixDurationOnTarget_RightEye[i_trials]:.0f}" + ' ms',fontsize=6,color = 'darkgreen')
		ax.text(0.5,-1*SignTarget*19,'Var Fix : ' + f"{FixationOnTarget_VariabilityOfFixation_RightEye[i_trials]:.3f}" + ' °',fontsize=6,color = 'darkgreen')
		ax.set_xlabel('Time (s)',fontsize=6)            
		ax.set_ylabel('Eye Position (°)',fontsize=6)  
		ax.yaxis.set_tick_params(labelsize=6)
		ax.xaxis.set_tick_params(labelsize=6)
		if (SignTarget<0):
			ax.legend(['Left Eye', 'Right Eye','Target'],fontsize=6,loc='lower left')
		else:
		   ax.legend(['Left Eye', 'Right Eye','Target'],fontsize=6,loc='upper left')
   
   
	   
            
	plt.subplots_adjust(left=0.03, bottom=0.04, right=0.98, top=0.95, wspace=0.2, hspace=0.2)
	plt.suptitle(TargetName)
	plt.show()
	
	
	return { 'FigureObject' : fig_curr,
			 'Latency_InitSacc_LeftEye':Latency_InitSacc_LeftEye,
			 'Latency_InitSacc_RightEye':Latency_InitSacc_RightEye,
			 'LogAmpGain_LeftEye':FixationOnTarget_AmplitudeGain_LeftEye,
			 'LogAmpGain_RightEye':FixationOnTarget_AmplitudeGain_RightEye,
			 'FixationDurationOnTarget_LeftEye':FixDurationOnTarget_LeftEye,
			 'FixationDurationOnTarget_RightEye':FixDurationOnTarget_RightEye,
 			 'VariabilityOfFixation_LeftEye':FixationOnTarget_VariabilityOfFixation_LeftEye,
			 'VariabilityOfFixation_RightEye':FixationOnTarget_VariabilityOfFixation_RightEye,
			 'MissingDataPercent_LeftEye' : EpochMissingDataPerCent_LeftEye,
			 'MissingDataPercent_RightEye' : EpochMissingDataPerCent_RightEye}



def Plot_MeanGaze_STEP(List_Epoch,List_Target_PixPosition,TargetFixationDuration, Cross_PixPosition,Pix2DegCoeff,SaccAmp_Min_Deg):
	MaxVelocitySaccade = 400
	MaxDistFixation_deg = 2
	
	NbCond = len(List_Epoch)
	
	Latency_InitSacc_LeftEye   = np.empty(NbCond)
	Latency_InitSacc_LeftEye[:] = np.nan
	Latency_InitSacc_RightEye  = np.empty(NbCond)
	Latency_InitSacc_RightEye[:] = np.nan
	
	LogAmpGain_LeftEye   = np.empty(NbCond)
	LogAmpGain_LeftEye[:] = np.nan
	LogAmpGain_RightEye   = np.empty(NbCond)
	LogAmpGain_RightEye[:] = np.nan
	
	FixationDurationOnTarget_LeftEye   = np.empty(NbCond)
	FixationDurationOnTarget_LeftEye[:] = np.nan
	FixationDurationOnTarget_RightEye   = np.empty(NbCond)
	FixationDurationOnTarget_RightEye[:] = np.nan
	
	VariabilityOfFixation_LeftEye   = np.empty(NbCond)
	VariabilityOfFixation_LeftEye[:] = np.nan
	VariabilityOfFixation_RightEye   = np.empty(NbCond)
	VariabilityOfFixation_RightEye[:] = np.nan
	
	EpochMissingDataPerCent_LeftEye  = np.empty(NbCond)
	EpochMissingDataPerCent_LeftEye[:] = np.nan
	EpochMissingDataPerCent_RightEye  = np.empty(NbCond)
	EpochMissingDataPerCent_RightEye[:] = np.nan
	
	NbCol = int(np.ceil(np.sqrt(NbCond)))
	NbRow = int(np.ceil(NbCond/NbCol))
	fig_MeanGaze  = plt.figure()

	for i_cond in range(NbCond):
		Epoch_curr = List_Epoch[i_cond]
		Target_PixPosition = List_Target_PixPosition[i_cond]
		Times_epoch = Epoch_curr.times
		SampFreq = Epoch_curr.info['sfreq']
		LE_X = np.nanmean(Epoch_curr.get_data(copy=True)[:,0,:],axis=0)
		LE_Y = np.nanmean(Epoch_curr.get_data(copy=True)[:,1,:],axis=0)
		RE_X = np.nanmean(Epoch_curr.get_data(copy=True)[:,2,:],axis=0)
		RE_Y = np.nanmean(Epoch_curr.get_data(copy=True)[:,3,:],axis=0)
		
		if (Target_PixPosition[1] == 0):
			Target_Pix_RefCross = Target_PixPosition[0]
			LE_Data = LE_X
			RE_Data = RE_X
			Cross_pos = Cross_PixPosition[0]
		else:
			Target_Pix_RefCross = Target_PixPosition[1]
			LE_Data = LE_Y
			RE_Data = RE_Y
			Cross_pos = Cross_PixPosition[1]
		
		Target_Sig = np.zeros(len(Times_epoch))
		Target_Sig[np.where((Times_epoch>0) & (Times_epoch<TargetFixationDuration))[0]] = Target_Pix_RefCross*Pix2DegCoeff
		SignTarget = Target_Pix_RefCross/(np.abs(Target_Pix_RefCross))
		Target_Pos_Px = Cross_pos + Target_Pix_RefCross
		
		ax = plt.subplot(NbRow, NbCol, i_cond + 1)
		ax.plot(Times_epoch,(LE_Data-Cross_pos)*Pix2DegCoeff,'r',linewidth=1)
		ax.plot(Times_epoch,(RE_Data-Cross_pos)*Pix2DegCoeff,'g',linewidth=1)
		ax.plot(Times_epoch,Target_Sig,'b',linewidth=1.5,linestyle='dotted')
		ax.set_ylim(bottom=-25, top=25)
		
		# Process Left Eye
		Results_LeftEye = ComputeFeature_GazeEpoch(LE_Data,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg)
		
		Latency_InitSacc_LeftEye[i_cond] = Results_LeftEye['InitSaccade_Latency']
		FixationDurationOnTarget_LeftEye[i_cond]= Results_LeftEye['FixationOnTarget_Duration']
		Fix_OnTarget_LatStart = Results_LeftEye['FixationOnTarget_StartLatency']
		Fix_OnTarget_LatEnd = Results_LeftEye['FixationOnTarget_EndLatency']
		Baseline_Latencies_LeftEye =  Results_LeftEye['GazeBaseline_Latencies']
		LogAmpGain_LeftEye[i_cond]=  np.log(Results_LeftEye['FixationOnTarget_AmplitudeGain'])
		EpochMissingDataPerCent_LeftEye[i_cond]= len(np.where(np.isnan(LE_Data))[0])/len(LE_Data)
		VariabilityOfFixation_LeftEye[i_cond] = Results_LeftEye['FixationOnTarget_FixationStability']
		
		
		# Display Left Eye		
		ax.axvline(Latency_InitSacc_LeftEye[i_cond]/SampFreq,0,1,linestyle='dotted',color = 'firebrick',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_LeftEye[0]/SampFreq,Baseline_Latencies_LeftEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*4,'Init Lat LE : ' + str(Latency_InitSacc_LeftEye[i_cond]) + ' ms',fontsize=6,color='firebrick')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='r',alpha=0.2)
		ax.text(0.5,-1*SignTarget*10,'log(AmpGain) : ' + f"{LogAmpGain_LeftEye[i_cond]:.3f}" ,fontsize=6,color = 'firebrick')
		ax.text(0.5,-1*SignTarget*16,'Targ Fix Dur : ' + f"{FixationDurationOnTarget_LeftEye[i_cond]:.0f}" + ' ms',fontsize=6,color = 'firebrick')
		ax.text(0.5,-1*SignTarget*22,'Var Fix  : ' + f"{VariabilityOfFixation_LeftEye[i_cond]:.3f}" + ' °',fontsize=6,color = 'firebrick')

		
		# Process Right Eye
		Results_RightEye = ComputeFeature_GazeEpoch(RE_Data,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg)
		
		Latency_InitSacc_RightEye[i_cond] = Results_RightEye['InitSaccade_Latency']
		FixationDurationOnTarget_RightEye[i_cond] = Results_RightEye['FixationOnTarget_Duration']
		Fix_OnTarget_LatStart = Results_RightEye['FixationOnTarget_StartLatency']
		Fix_OnTarget_LatEnd = Results_RightEye['FixationOnTarget_EndLatency']
		Baseline_Latencies_RightEye =  Results_RightEye['GazeBaseline_Latencies']
		LogAmpGain_RightEye[i_cond] = np.log(Results_RightEye['FixationOnTarget_AmplitudeGain'])
		EpochMissingDataPerCent_RightEye[i_cond] = len(np.where(np.isnan(RE_Data))[0])/len(RE_Data)
		VariabilityOfFixation_RightEye[i_cond] = Results_RightEye['FixationOnTarget_FixationStability']
		
		
		# Display Right Eye
		
		ax.axvline(Latency_InitSacc_RightEye[i_cond]/SampFreq,0,1,linestyle='dotted',color = 'darkgreen',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_RightEye[0]/SampFreq,Baseline_Latencies_RightEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*1,'Init Lat : ' + str(Latency_InitSacc_RightEye[i_cond]) + ' ms',fontsize=6,color='darkgreen')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='g',alpha=0.2)
		ax.text(0.5,-1*SignTarget*7,'log(AmpGain) : ' + f"{LogAmpGain_RightEye[i_cond]:.3f}" ,fontsize=6,color = 'darkgreen')
		ax.text(0.5,-1*SignTarget*13,'Targ Fix Dur : ' + f"{FixationDurationOnTarget_RightEye[i_cond]:.0f}" + ' ms',fontsize=6,color = 'darkgreen')
		ax.text(0.5,-1*SignTarget*19,'Var Fix : ' + f"{VariabilityOfFixation_RightEye[i_cond]:.3f}" + ' °',fontsize=6,color = 'darkgreen')

		ax.set_xlabel('Time (s)',fontsize=6)            
		ax.set_ylabel('Eye Position (°)',fontsize=6)  
		ax.yaxis.set_tick_params(labelsize=6)
		ax.xaxis.set_tick_params(labelsize=6)
		if (SignTarget<0):
			ax.legend(['Left Eye', 'Right Eye','Target'],fontsize=6,loc='lower left')
		else:
		   ax.legend(['Left Eye', 'Right Eye','Target'],fontsize=6,loc='upper left')
		TargetName = list(Epoch_curr.event_id.keys())[0]
		
		ax.set_title(TargetName)
	plt.show()
		
	return { 'FigureObject' : fig_MeanGaze,
				 'Latency_InitSacc_LeftEye':Latency_InitSacc_LeftEye,
				 'Latency_InitSacc_RightEye':Latency_InitSacc_RightEye,
				 'LogAmpGain_LeftEye':LogAmpGain_LeftEye,
				 'LogAmpGain_RightEye':LogAmpGain_RightEye,
				 'FixationDurationOnTarget_LeftEye':FixationDurationOnTarget_LeftEye,
				 'FixationDurationOnTarget_RightEye':FixationDurationOnTarget_RightEye,
				 'VariabilityOfFixation_LeftEye':VariabilityOfFixation_LeftEye,
				 'VariabilityOfFixation_RightEye':VariabilityOfFixation_RightEye,
				 'MissingDataPercent_LeftEye' : EpochMissingDataPerCent_LeftEye,
				 'MissingDataPercent_RightEye' : EpochMissingDataPerCent_RightEye}
		
		
		
		
	
	
	

def PlotFixationOnCross(Kind_VisAtt, EyeName, TabAttSide,NbBlocks,ListGaze_X,ListGaze_Y,SampFreq,ScreenResolution_Width,ScreenResolution_Height,Cross_X,Cross_Y,Cross_Area_X,Cross_Area_Y):
	Percentage_FixationCross = np.zeros(NbBlocks)

	NbRow = int(np.ceil(np.sqrt(NbBlocks)))
	NbCol = int(np.ceil(NbBlocks/NbRow))
	fig, axs = plt.subplots(NbRow,NbCol,figsize=(12,10))
	fig.suptitle(Kind_VisAtt + ' : ' + EyeName + 'Eye Gaze Fixation')
	axs = axs.ravel()
	
	for i_block in range(NbBlocks):
		AttSide = TabAttSide[i_block]
		
		Gaze_X = ListGaze_X[i_block]
		Gaze_Y = ListGaze_Y[i_block]
		
		time_Block = np.array(range(len(Gaze_X)))/SampFreq
		axs[i_block].set_facecolor((0.0, 0.0, 0.5))
		
		# Detect Fixation on Left Eye
		Sfix, Efix = detectors.fixation_detection(Gaze_X, Gaze_Y, time_Block*SampFreq, missing=np.NaN, maxdist=25, mindur=50)
		for i_fix in range(len(Efix)):
			gazex_curr = Gaze_X[range(int(Efix[i_fix][0]) , int(Efix[i_fix][1]))]
			gazey_curr = Gaze_Y[range(int(Efix[i_fix][0]) , int(Efix[i_fix][1]))]
			if (np.sum(np.array(np.isnan(gazex_curr), dtype=int)) > 0 ):
				gazex_curr = gazex_curr[np.invert(np.isnan(gazex_curr))]
				gazey_curr = gazey_curr[np.invert(np.isnan(gazey_curr))]
				Efix[i_fix][1] = Efix[i_fix][0] + float(len(gazex_curr))
				Efix[i_fix][2] = float(len(gazex_curr))
				Efix[i_fix][3] = np.mean(gazex_curr)
				Efix[i_fix][4] = np.mean(gazey_curr)
		draw_heatmap(Efix, (ScreenResolution_Width,ScreenResolution_Height), imagefile=None, durationweight=True, alpha=1.0,savefilename=None,ax=axs[i_block])    # Detect Fixation 
		axs[i_block].vlines(Cross_X,0,ScreenResolution_Height,'w',linestyle ='dotted',linewidth=2)
		axs[i_block].hlines(Cross_Y,0,ScreenResolution_Width,'w',linestyle ='dotted',linewidth=2)
		axs[i_block].set_xlim(0,ScreenResolution_Width)
		axs[i_block].set_ylim(0,ScreenResolution_Height)
		
		# Plot Cross Area
		axs[i_block].vlines(Cross_Area_X[0],Cross_Area_Y[0],Cross_Area_Y[1],'w',linestyle ='dotted')
		axs[i_block].vlines(Cross_Area_X[1],Cross_Area_Y[0],Cross_Area_Y[1],'w',linestyle ='dotted')
		axs[i_block].hlines(Cross_Area_Y[0],Cross_Area_X[0],Cross_Area_X[1],'w',linestyle ='dotted')
		axs[i_block].hlines(Cross_Area_Y[1],Cross_Area_X[0],Cross_Area_X[1],'w',linestyle ='dotted')
		axs[i_block].xaxis.set_ticklabels([])
		axs[i_block].yaxis.set_ticklabels([])
	
		# Compute Percentage of Cross fixation
		DurationFix_Cross = 0
		TotalDuration = np.sum(np.array(np.invert(np.isnan(Gaze_X)), dtype=int))
		for i_fix in range(len(Efix)):
			Condition_X = (Efix[i_fix][3]>Cross_Area_X[0]) & (Efix[i_fix][3]<Cross_Area_X[1])
			Condition_Y = (Efix[i_fix][4]>Cross_Area_Y[0]) & (Efix[i_fix][4]<Cross_Area_Y[1])
			if (Condition_X & Condition_Y):
				DurationFix_Cross = DurationFix_Cross + Efix[i_fix][2]
				
		if (TotalDuration>0):
			Percentage_FixationCross[i_block] = DurationFix_Cross*100/TotalDuration
		else:
			Percentage_FixationCross[i_block] = np.NaN
			
		axs[i_block].set_title('Trial #' + str(i_block+1) + ' Attented side : ' + r"$\bf{" + AttSide + "}$"  + ' - Cross Fixation : ' + f'{Percentage_FixationCross[i_block]:.2f}' + '%',fontsize=8)
	return Percentage_FixationCross



def PlotDetectSaccade(Kind_VisAtt,TabAttSide,NbBlocks,ListGaze_LEye_X,ListGaze_LEye_Y,ListGaze_REye_X,ListGaze_REye_Y,SampFreq,Pix2DegCoeff,Cross_X,Cross_Y,Excentricity,ScreenResolution_Width,ScreenResolution_Height):
	NbSaccades_LEye =np.zeros(NbBlocks)
	NbSaccades_REye =np.zeros(NbBlocks)
	NbRow = int(np.ceil(np.sqrt(NbBlocks)))
	NbCol = int(np.ceil(NbBlocks/NbRow))
	fig, axs = plt.subplots(NbRow,NbCol,figsize=(12,10))
	fig.suptitle(Kind_VisAtt + ' : Gaze - Detect Saccades')
	axs = axs.ravel()
	for i_block in range(NbBlocks):
		AttSide = TabAttSide[i_block]

		Gaze_LEye_X = ListGaze_LEye_X[i_block]
		Gaze_LEye_Y = ListGaze_LEye_Y[i_block]
		Gaze_REye_X = ListGaze_REye_X[i_block]
		Gaze_REye_Y = ListGaze_REye_Y[i_block]
		
		time_Block = np.array(range(len(Gaze_LEye_X)))/SampFreq
		
		# Detect saccades
		Ssac_LE, Esac_LE = detectors.saccade_detection(Gaze_LEye_X, Gaze_LEye_Y, time_Block*SampFreq, minlen = 20,  maxvel=400)
		Ssac_RE, Esac_RE = detectors.saccade_detection(Gaze_REye_X, Gaze_REye_Y, time_Block*SampFreq, minlen = 20,  maxvel=400)
		
		axs[i_block].plot(time_Block,(Gaze_LEye_X-Cross_X)*Pix2DegCoeff,'r')
		axs[i_block].plot(time_Block,(Gaze_REye_X-Cross_X)*Pix2DegCoeff,'g')
		
		SaccAmp_Min_Deg = (Excentricity/2)*Pix2DegCoeff
		nbsacc_LE =0
		
		for i_sac in range(len(Esac_LE)):
			Amp_Sacc_Curr = Esac_LE[i_sac][5]-Esac_LE[i_sac][3]
			ix_start_Fix = np.where((time_Block*SampFreq==Esac_LE[i_sac][0]))[0][0]
			ix_stop_Fix  = np.where((time_Block*SampFreq==Esac_LE[i_sac][1]))[0][0]
			SaccData_Curr = Gaze_LEye_X[ix_start_Fix:ix_stop_Fix]
			
		
			Condition1 = np.abs(Amp_Sacc_Curr)>(SaccAmp_Min_Deg/Pix2DegCoeff)
			Condition2 = len(np.where(np.isnan(SaccData_Curr))[0])/len(SaccData_Curr) < 0.2

			if (Condition1&Condition2):
				axs[i_block].axvspan(Esac_LE[i_sac][0]/SampFreq,Esac_LE[i_sac][1]/SampFreq,color='r',alpha=0.3)
				nbsacc_LE = nbsacc_LE + 1
			
		nbsacc_RE =0
		for i_sac in range(len(Esac_RE)):
			Amp_Sacc_Curr = Esac_RE[i_sac][5]-Esac_RE[i_sac][3]
			ix_start_Fix = np.where((time_Block*SampFreq==Esac_RE[i_sac][0]))[0][0]
			ix_stop_Fix  = np.where((time_Block*SampFreq==Esac_RE[i_sac][1]))[0][0]
			SaccData_Curr = Gaze_REye_X[ix_start_Fix:ix_stop_Fix]
	
			Condition1 = np.abs(Amp_Sacc_Curr)>(SaccAmp_Min_Deg/Pix2DegCoeff)
			Condition2 = len(np.where(np.isnan(SaccData_Curr))[0])/len(SaccData_Curr) < 0.2
	
			if (Condition1&Condition2):
				axs[i_block].axvspan(Esac_RE[i_sac][0]/SampFreq,Esac_RE[i_sac][1]/SampFreq,color='g',alpha=0.3)
				nbsacc_RE = nbsacc_RE + 1			
		axs[i_block].set_title('Trial #' + str(i_block+1) + ' Attented side : ' + r"$\bf{" + AttSide + "}$"  + ' - Number of saccades : ' + str(np.max([nbsacc_LE,nbsacc_RE])) ,fontsize=8)
		axs[i_block].set_ylim(bottom=-Cross_X*Pix2DegCoeff, top=(ScreenResolution_Width-Cross_X)*Pix2DegCoeff)
		axs[i_block].tick_params(axis='x', labelsize=6)
		NbSaccades_LEye[i_block] = nbsacc_LE
		NbSaccades_REye[i_block] = nbsacc_RE
	
	return NbSaccades_LEye,NbSaccades_REye


def parse_fixations(fixations):
	
	"""Returns all relevant data from a list of fixation ending events
	
	arguments
	
	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']

	returns
	
	fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
				a numpy array) for the x and y coordinates and duration of
				each fixation
	"""
	
	# empty arrays to contain fixation coordinates
	fix = {	'x':np.zeros(len(fixations)),
			'y':np.zeros(len(fixations)),
			'dur':np.zeros(len(fixations))}
	# get all fixation coordinates
	for fixnr in range(len( fixations)):
		stime, etime, dur, ex, ey = fixations[fixnr]
		fix['x'][fixnr] = ex
		fix['y'][fixnr] = ey
		fix['dur'][fixnr] = dur
	
	return fix



def compute_heatmap(fixations, dispsize, imagefile=None, durationweight=True, alpha=0.5, savefilename=None, ax = None):
	
	"""Draws a heatmap of the provided fixations, optionally drawn over an
	image, and optionally allocating more weight to fixations with a higher
	duration.
	
	arguments
	
	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']
	dispsize		-	tuple or list indicating the size of the display,
					e.g. (1024,768)
	
	keyword arguments
	
	imagefile		-	full path to an image file over which the heatmap
					is to be laid, or None for no image; NOTE: the image
					may be smaller than the display size, the function
					assumes that the image was presented at the centre of
					the display (default = None)
	durationweight	-	Boolean indicating whether the fixation duration is
					to be taken into account as a weight for the heatmap
					intensity; longer duration = hotter (default = True)
	alpha		-	float between 0 and 1, indicating the transparancy of
					the heatmap, where 0 is completely transparant and 1
					is completely untransparant (default = 0.5)
	savefilename	-	full path to the file in which the heatmap should be
					saved, or None to not save the file (default = None)
	
	returns
	
	fig			-	a matplotlib.pyplot Figure instance, containing the
					heatmap
	"""

	# FIXATIONS
	fix = parse_fixations(fixations)

	
	# HEATMAP
	# Gaussian
	gwh = 200
	gsdwh = int(gwh/6)
	gaus = py_tools.gaussian(gwh,gsdwh)
	# matrix of zeroes
	strt = int(gwh/2)
	heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
	heatmap = np.zeros(heatmapsize, dtype=float)
	# create heatmap
	for i in range(0,len(fix['dur'])):
		# get x and y coordinates
		#x and y - indexes of heatmap array. must be integers
		if (np.isnan(fix['x'][i])):
			x=np.NaN
		else:
			x = strt + int(fix['x'][i]) - int(gwh/2)
            
		if (np.isnan(fix['y'][i])):
 			y=np.NaN
		else:
 			y = strt + int(fix['y'][i]) - int(gwh/2)
             
		# correct Gaussian size if either coordinate falls outside of
		# display boundaries
		if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
			hadj=[0,gwh];vadj=[0,gwh]
			if 0 > x:
				hadj[0] = abs(x)
				x = 0
			elif dispsize[0] < x:
				hadj[1] = gwh - int(x-dispsize[0])
			if 0 > y:
				vadj[0] = abs(y)
				y = 0
			elif dispsize[1] < y:
				vadj[1] = gwh - int(y-dispsize[1])
			# add adjusted Gaussian to the current heatmap
			try:
				heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
			except:
				# fixation was probably outside of display
				pass
		else:				
			# add Gaussian to the current heatmap
			heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i]
	# resize heatmap
	heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
	# remove zeros
	lowbound = np.nanmean(heatmap[heatmap>0])
	heatmap[heatmap<lowbound] = np.NaN 
	
	return heatmap





