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
	FixPeriod_GainAmp = np.nan
	FixPeriod_GainAUC = np.nan
	GainAUC_TargetDisplayPeriod = np.nan
	
	ix_ShowTarget = np.int(np.where((Times_epoch==0.0))[0])
	ix_HideTarget  = np.int(np.where((Times_epoch==TargetFixationDuration))[0])
	    
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
						ix_start_Fix = np.int(np.where((Times_epoch*SampFreq==Efix[ifix][0]))[0])
						ix_stop_Fix  = np.int(np.where((Times_epoch*SampFreq==Efix[ifix][1]))[0])
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
					ix_start_Fix = np.int(np.where((Times_epoch*SampFreq==Efix[ifix][0]))[0])
					ix_stop_Fix  = np.int(np.where((Times_epoch*SampFreq==Efix[ifix][1]))[0])
					FixData_Curr = Gaze_data[ix_start_Fix:ix_stop_Fix]
					AllFixData = np.append(AllFixData,FixData_Curr)
					NbNan = NbNan + len(np.where(np.isnan(Gaze_data[ix_start_Fix:ix_stop_Fix]))[0])
					NbSampFix = NbSampFix + len(Gaze_data[ix_start_Fix:ix_stop_Fix])

	if (NbSampFix>0):
		PercentNanInFixPeriod = NbNan/NbSampFix
		FixPeriod_GazePosition = np.nanmedian(AllFixData)
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
# 	fig_curr.set_size_inches(19, 9)
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
		
		(Results_LeftEye['FixationOnTarget_GazePosition']-Cross_pos)*Pix2DegCoeff
		
		# Display Left Eye
		
		ax.axvline(Latency_InitSacc_LeftEye[i_trials]/SampFreq,0,1,linestyle='dotted',color = 'firebrick',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_LeftEye[0]/SampFreq,Baseline_Latencies_LeftEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*4.5,'Init Lat LE : ' + str(Latency_InitSacc_LeftEye[i_trials]) + ' ms',fontsize=6,color='firebrick')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='r',alpha=0.2)
		ax.text(0.5,-1*SignTarget*18.5,'Targ Fix Dur : ' + f"{FixDurationOnTarget_LeftEye[i_trials]:.0f}" + ' ms',fontsize=6,color = 'firebrick')
		ax.text(0.5,-1*SignTarget*11.5,'log(AmpGain) : ' + f"{FixationOnTarget_AmplitudeGain_LeftEye[i_trials]:.3f}" ,fontsize=6,color = 'firebrick')
		
		
		# Process Right Eye
		Results_RightEye = ComputeFeature_GazeEpoch(RE_Data_curr,Times_epoch, SampFreq, Target_Pix_RefCross,TargetFixationDuration, Target_Pos_Px, Cross_pos, Pix2DegCoeff,SaccAmp_Min_Deg,MaxVelocitySaccade,MaxDistFixation_deg)
		
		Latency_InitSacc_RightEye[i_trials] = Results_RightEye['InitSaccade_Latency']
		FixDurationOnTarget_RightEye[i_trials] = Results_RightEye['FixationOnTarget_Duration']
		Fix_OnTarget_LatStart = Results_RightEye['FixationOnTarget_StartLatency']
		Fix_OnTarget_LatEnd = Results_RightEye['FixationOnTarget_EndLatency']
		Baseline_Latencies_RightEye =  Results_RightEye['GazeBaseline_Latencies']
		FixationOnTarget_AmplitudeGain_RightEye[i_trials] = np.log(Results_RightEye['FixationOnTarget_AmplitudeGain'])
		EpochMissingDataPerCent_RightEye[i_trials] = len(np.where(np.isnan(RE_Data_curr))[0])/len(RE_Data_curr)
		
		
		
		# Display Right Eye
		
		ax.axvline(Latency_InitSacc_RightEye[i_trials]/SampFreq,0,1,linestyle='dotted',color = 'darkgreen',linewidth=2.5)
		ax.axvspan(Baseline_Latencies_RightEye[0]/SampFreq,Baseline_Latencies_RightEye[1]/SampFreq,color='k',alpha=0.3)
		ax.text(0.5,-1*SignTarget*1,'Init Lat : ' + str(Latency_InitSacc_RightEye[i_trials]) + ' ms',fontsize=6,color='darkgreen')
		
		for ifix in range(len(Fix_OnTarget_LatStart)):
			ax.axvspan(Fix_OnTarget_LatStart[ifix]/SampFreq,Fix_OnTarget_LatEnd[ifix]/SampFreq,color='g',alpha=0.2)
		ax.text(0.5,-1*SignTarget*15,'Targ Fix Dur : ' + f"{FixDurationOnTarget_RightEye[i_trials]:.0f}" + ' ms',fontsize=6,color = 'darkgreen')
		ax.text(0.5,-1*SignTarget*8,'log(AmpGain) : ' + f"{FixationOnTarget_AmplitudeGain_RightEye[i_trials]:.3f}" ,fontsize=6,color = 'darkgreen')
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
			 'FixationDurationOnTarget_RightEye':FixDurationOnTarget_RightEye}
