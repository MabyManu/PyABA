# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:46:45 2023

@author: manum
"""

import numpy as np
import mne
import json

from AddPyABA_Path import PyABA_path
import sys
sys.path.append(PyABA_path)
import py_tools


def ReadCalib_TemplateBrainReality(JsonFileName):
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
	
	AccP300 = data_dict['AccP300']
	TargetMean = np.array(json.loads((data_dict['targetEpochMu'])))
	NonTargetMean = np.array(json.loads((data_dict['nonTargetEpochMu'])))
	return TargetMean,NonTargetMean
		
def ReadJsonData_BR(JsonFileName):
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
		
	Settings = data_dict['Settings']['Values'][0]
	SamplingFrequency = Settings['samplingRate']
	ChannelsName = Settings['channelNameList']
	NbChans = len(ChannelsName)
	NumberOfRepetitions = Settings['stimulationSettings']['calibrationStimuliCount']
	
	NbSampleWin = Settings['nbSampleWin']
	
	
	NbItems = len(data_dict['Visibility']['Values'])
	# EPOCH
	
	Nb_Epochs_TOT = len(data_dict['Epochs']['Values'])
	ListEvt_Target = []
	ListEvt_NoTarget = []
	
	Data_epo_Target = []
	Data_epo_NoTarget = []
	
	Data_epo_Raw_Target = []	
	Data_epo_Raw_NoTarget = []
	
	for i_epoch in range(Nb_Epochs_TOT):
		Data_Epoch_curr = data_dict['Epochs']['Values'][i_epoch]
		Epoch_Filt = np.array(Data_Epoch_curr['filteredData'])
		Epoch_raw = np.array(Data_Epoch_curr['eegData'])
		Stim_Param_curr = Data_Epoch_curr['stimulation']
		
		if (i_epoch ==0):
			PosixT0 = Stim_Param_curr['PosXTime']
		latency = int((Stim_Param_curr['PosXTime'] - PosixT0)*SamplingFrequency/1000)
		NameStim = Stim_Param_curr['StimulusNameList'][0]
		numcol = int(NameStim[6:])
		Target = Stim_Param_curr['IsTarget']
		if Target:
			ListEvt_Target.append([latency,0,numcol + Target * 10]) 
			Data_epo_Target.append(Epoch_Filt)
			Data_epo_Raw_Target.append(Epoch_raw)
		else:
			ListEvt_NoTarget.append([latency,0,numcol]) 
			Data_epo_NoTarget.append(Epoch_Filt)
			Data_epo_Raw_NoTarget.append(Epoch_raw)
			
				
	Data_epo_Target = np.array(Data_epo_Target)
	Data_epo_NoTarget = np.array(Data_epo_NoTarget)
	Data_epo_Raw_Target = np.array(Data_epo_Raw_Target)
	Data_epo_Raw_NoTarget = np.array(Data_epo_Raw_NoTarget)
	ListEvt_Target = np.array(ListEvt_Target) 	
	ListEvt_NoTarget = np.array(ListEvt_NoTarget) 	
	return Data_epo_Target,	Data_epo_NoTarget, Data_epo_Raw_Target,Data_epo_Raw_NoTarget,SamplingFrequency,ListEvt_Target,ListEvt_NoTarget

def ReadCalib_JsonFile_BrainReality(JsonFileName):
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
		
	Settings = data_dict['Settings']['Values'][0]
	SamplingFrequency = Settings['samplingRate']
	ChannelsName = Settings['channelNameList']
	NbChans = len(ChannelsName)
	NumberOfRepetitions = Settings['stimulationSettings']['calibrationStimuliCount']
	
	NbSampleWin = Settings['nbSampleWin']
	
	
	NbItems = len(data_dict['Visibility']['Values'])
	# EPOCH
	
	Nb_Epochs_TOT = len(data_dict['Epochs']['Values'])
	ListEvt_Target = []
	ListEvt_NoTarget = []
	
	Data_epo_Target = []
	Data_epo_NoTarget = []
	
	Data_epo_Raw_Target = []	
	Data_epo_Raw_NoTarget = []
	
	for i_epoch in range(Nb_Epochs_TOT):
		Data_Epoch_curr = data_dict['Epochs']['Values'][i_epoch]
		Epoch_Filt = np.array(Data_Epoch_curr['filteredData'])
		Epoch_raw = np.array(Data_Epoch_curr['eegData'])
		Stim_Param_curr = Data_Epoch_curr['stimulation']
		
		if (i_epoch ==0):
			PosixT0 = Stim_Param_curr['PosXTime']
		latency = int((Stim_Param_curr['PosXTime'] - PosixT0)*SamplingFrequency/1000)
		NameStim = Stim_Param_curr['StimulusNameList'][0]
		numcol = int(NameStim[6:])
		Target = Stim_Param_curr['IsTarget']
		if Target:
			ListEvt_Target.append([latency,0,numcol + Target * 10]) 
			Data_epo_Target.append(Epoch_Filt)
			Data_epo_Raw_Target.append(Epoch_raw)
		else:
			ListEvt_NoTarget.append([latency,0,numcol]) 
			Data_epo_NoTarget.append(Epoch_Filt)
			Data_epo_Raw_NoTarget.append(Epoch_raw)
			
			
		
			
			
	Data_epo_Target = np.array(Data_epo_Target)
	Data_epo_NoTarget = np.array(Data_epo_NoTarget)
			
	Nb_Evt_Targets = len(ListEvt_Target)
	Nb_Evt_NoTargets = len(ListEvt_NoTarget)
	
	
	MatEvt_Target = np.array(ListEvt_Target)
	MatEvt_NoTarget = np.array(ListEvt_NoTarget)
	
	ch_types_Chan = []
	for i_chan in range(NbChans):
		ch_types_Chan.append('eeg')
	
	NbTargets  = int((Nb_Evt_Targets + Nb_Evt_NoTargets)/(NumberOfRepetitions*NbItems))
	
			
			
	info = mne.create_info(ChannelsName, SamplingFrequency, ch_types=ch_types_Chan)
	ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
	
	Evtcurr = np.unique(MatEvt_Target[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]-10) + '/Target'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr] 
	Epochs_Target=mne.EpochsArray(Data_epo_Target*1e-6,info=info,events = MatEvt_Target, event_id=event_id)
	Epochs_Target.set_montage(ten_twenty_montage)
	
	
	
	Evtcurr = np.unique(MatEvt_NoTarget[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]) + '/NoTarget'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr]
	
	Epochs_NoTarget=mne.EpochsArray(Data_epo_NoTarget*1e-6,info=info,events = MatEvt_NoTarget, event_id=event_id)
	Epochs_NoTarget.set_montage(ten_twenty_montage)
	
	return Epochs_Target,Epochs_NoTarget,NumberOfRepetitions,NbItems,NbTargets




















def ReadCalib_JsonFile_Mentalink4(JsonFileName,FlagEpochFlex=False):
	Flag_ForcingChannelsName = False
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
		
	Settings = data_dict['Settings']
	SamplingFrequency = Settings['SampleRate']
	
	if (FlagEpochFlex & (SamplingFrequency == 128)):
		Flag_ForcingChannelsName = True
	
	if 'FlashCount:' in Settings:
		NumberOfRepetitions = int(Settings['FlashCount:'])
	else:
		NumberOfRepetitions = int(Settings['FlashCount'])
		
	if 'NbSampleWin' in Settings:
		NbSampleWin = Settings['NbSampleWin']
		
	

		
	
	# EVENT
	if 'SequenceList' in data_dict:
		i_Flash = 0
		Seqlist = data_dict["SequenceList"]["Values"][0]
		
		if 'Tile' in (Seqlist[0]['StimulationList'][0]['StimulusNameList'][0]):
			Game = 'Battleship'
		else:
			Game = 'Connect4'
			
		
			
			
		
		
		NbItems = len(Seqlist[0]['StimulationList'])
		NbSeq = len(Seqlist)
		MatEvt = np.zeros((NbItems*NbSeq,3),dtype=int)

		for i_seq in range(NbSeq):
			Seq_curr = Seqlist[i_seq]
			for i_item in range(NbItems):
				pos_tmp = Seq_curr["StimulationList"][i_item]["PosXTime"]
				Target = Seq_curr["StimulationList"][i_item]['IsTarget']
				NameStim =  Seq_curr["StimulationList"][i_item]["StimulusNameList"][0]
				if "Stimulus (" in NameStim:
					numcol = int(NameStim[10:NameStim.find(")")]) + 1
				else:
					numcol = int(NameStim[int(np.fix(float(NameStim.find("Column"))))+6:])
				if (i_Flash==0):
					PosixT0 = pos_tmp
				latency = pos_tmp - PosixT0
				MatEvt[i_Flash,0] = latency
				MatEvt[i_Flash,2] = numcol + Target * 10
				i_Flash = i_Flash + 1
		
	if 'StimulusSequence' in data_dict:
		FlashSequence = data_dict['StimulusSequence']
		
		NbFlashs =  len(FlashSequence["Values"][0]['FlashList'])
		NbItems = len(data_dict['Visibility']["Values"][0])
		
		
		MatEvt = np.zeros((NbFlashs,3),dtype=int)
		for i_Flash in range(NbFlashs):
			Flash_curr=FlashSequence["Values"][0]['FlashList'][i_Flash]
			numcol = int(Flash_curr['Name'][int(np.fix(float(Flash_curr['Name'].find("Column"))))+6:])
			Target = Flash_curr['IsTarget']
			pos_tmp = Flash_curr['PosXTime']
			if (i_Flash==0):
				 PosixT0 = pos_tmp
			latency = pos_tmp - PosixT0
			MatEvt[i_Flash,0] = latency
			MatEvt[i_Flash,2] = numcol + Target * 10
	
	NbTargets  = int(len(MatEvt[MatEvt[:,2]>10,2])/NumberOfRepetitions)
	
	
	# EPOCHS 
	Data_Target = data_dict['EEGRawData']['Target']
	Data_NoTarget = data_dict['EEGRawData']['NonTarget']
	
	
	if Flag_ForcingChannelsName:
		NbChans = np.shape(Data_Target)[1]
		ChannelsName = ["Timestamp","Counter","Interpolate","Cz","T7","FC1","FC5","F9","F3","F7","Fp1","Pz","C3","O1","P9","P3","P7","CP1","CP5","Fz","T8","FC2","FC6","F10","F4","F8","Fp2","Oz","C4","O2","P10","P4","P8","CP2","CP6", "HardwareMarker","Markers"]
		ch_types_Chan = ['misc'] * 3 + ["eeg"] * (NbChans-5) + ['misc'] * 2
	else:	
		NbChans = len(Settings['Electrodes'])
		ChannelsName=[]
		ch_types_Chan = []
		for i_chan in range(NbChans):
			ChannelsName.append(Settings['Electrodes'][i_chan]['name'])
			if '_dir' in Settings['Electrodes'][i_chan]['name']:
				ch_types_Chan.append('misc')
			else:
				ch_types_Chan.append('eeg')
	
	
	
	
	
	
	NbEpochs_Target = len(Data_Target)
	Data_epo_Target = np.zeros((NbEpochs_Target,NbChans,NbSampleWin))
	for i_epoch in range(NbEpochs_Target):
		for j_chan in range(NbChans):
			Data_epo_Target[i_epoch,j_chan,:] = Data_Target[i_epoch][j_chan]
			
	NbEpochs_NoTarget = len(Data_NoTarget)
	Data_epo_NoTarget = np.zeros((NbEpochs_NoTarget,NbChans,NbSampleWin))
	for i_epoch in range(NbEpochs_NoTarget):
		for j_chan in range(NbChans):
			Data_epo_NoTarget[i_epoch,j_chan,:] = Data_NoTarget[i_epoch][j_chan]
			
			
			
			
			
			
	
			
			
	info = mne.create_info(ChannelsName, SamplingFrequency, ch_types=ch_types_Chan)
	ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
	
	MatEvt_Target = np.squeeze(MatEvt[np.where(MatEvt[:,2]>10),:])
	Evtcurr = np.unique(MatEvt_Target[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]-10) + '/Target'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr] 
	Epochs_Target=mne.EpochsArray(Data_epo_Target*1e-6,info=info,events = MatEvt_Target, event_id=event_id)
	Epochs_Target.set_montage(ten_twenty_montage)
	
	
	
	MatEvt_NoTarget = np.squeeze(MatEvt[np.where(MatEvt[:,2]<10),:])
	Evtcurr = np.unique(MatEvt_NoTarget[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]) + '/NoTarget'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr]
	
	Epochs_NoTarget=mne.EpochsArray(Data_epo_NoTarget*1e-6,info=info,events = MatEvt_NoTarget, event_id=event_id)
	Epochs_NoTarget.set_montage(ten_twenty_montage)
	
	return Epochs_Target,Epochs_NoTarget,NumberOfRepetitions,NbItems,NbTargets












def ReadTest_JsonFile_Mentalink4(JsonFileName):
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
		
	EEG_item_Results = data_dict['EEGFocus']
		
	Settings = data_dict['Setting']
	SamplingFrequency = Settings['SampleRate']
	if 'FlashCount:' in Settings:
		NumberOfRepetitions = int(Settings['FlashCount:'])
	else:
		NumberOfRepetitions = int(Settings['FlashCount'])
		
	if 'NbSampleWin' in Settings:
		NbSampleWin = Settings['NbSampleWin']
	else:
		NbSampleWin = len(data_dict['EEGRawData']['Test'][0][0])
			
		
	NbChans = len(Settings['Electrodes'])
	ChannelsName=[]
	ch_types_Chan=[]
	for i_chan in range(NbChans):
		ChannelsName.append(Settings['Electrodes'][i_chan]['name'])
		if '_dir' in Settings['Electrodes'][i_chan]['name']:
			ch_types_Chan.append('misc')
		else:
			ch_types_Chan.append('eeg')
		
	# EVENT
	Gaze = data_dict['EyeFocus']['Values']
	if 'SequenceList' in data_dict:
		i_Flash = 0
		Seqlist = data_dict["SequenceList"]["Values"]
	
		NbItems = len(Seqlist[0][0]['StimulationList'])
		NbTurns = len(Seqlist)
		NbTotFlashs = NbTurns*NumberOfRepetitions*NbItems

		MatEvt = np.zeros((NbTotFlashs,3),dtype=int)
		IsTarget = np.zeros((NbTotFlashs),dtype=bool)

	
		for i_seq in range(NbTurns):
			Gaze_Seq_Turncurr=Gaze[i_seq]
			if (np.argmax(list(Gaze_Seq_Turncurr.values())) + 1>NbItems):
				Gaze_Seq_Turncurr_b = Gaze_Seq_Turncurr
				Gaze_Seq_Turncurr = {}
				for key, value in Gaze_Seq_Turncurr_b.items():
					if key.startswith("Column"):
						Gaze_Seq_Turncurr[key] = Gaze_Seq_Turncurr_b[key]			
			
			Target_curr =np.argmax(list(Gaze_Seq_Turncurr.values())) + 1
			
			Seq_curr = Seqlist[i_seq]
			for i_rep in range(NumberOfRepetitions):
				Stim_Rep = Seq_curr[i_rep]
				for i_item in range(NbItems):
					pos_tmp = Stim_Rep["StimulationList"][i_item]["PosXTime"]
					NameStim =  Stim_Rep["StimulationList"][i_item]["StimulusNameList"][0]
					if "Stimulus (" in NameStim:
						numcol = int(NameStim[10:NameStim.find(")")]) + 1
					else:
						numcol = int(NameStim[int(np.fix(float(NameStim.find("Column"))))+6:])
					Target = (numcol == Target_curr)
					if (i_Flash==0):
						PosixT0 = pos_tmp
					latency = pos_tmp - PosixT0
					MatEvt[i_Flash,0] = latency
					MatEvt[i_Flash,2] = numcol + Target * 10
					IsTarget[i_Flash] = Target
					i_Flash = i_Flash + 1
				
				
	if 'StimulusSequence' in data_dict:
		FlashSequence = data_dict['StimulusSequence']['Values']
		NbTurns = len(FlashSequence)
		NbItems = len(data_dict['Visibility']["Values"][0])
	
		NbTotFlashs = NbTurns*NumberOfRepetitions*NbItems
		
		MatEvt = np.zeros((NbTotFlashs,3),dtype=int)
		IsTarget = np.zeros((NbTotFlashs),dtype=bool)
		i_f = 0
		for i_turn in range(NbTurns):
			Flash_Seq_Turncurr=FlashSequence[i_turn]['FlashList']
			Gaze_Seq_Turncurr=Gaze[i_turn]
			
			
			if (np.argmax(list(Gaze_Seq_Turncurr.values())) + 1>NbItems):
				Gaze_Seq_Turncurr_b = Gaze_Seq_Turncurr
				Gaze_Seq_Turncurr = {}
				for key, value in Gaze_Seq_Turncurr_b.items():
					if key.startswith("Column"):
						Gaze_Seq_Turncurr[key] = Gaze_Seq_Turncurr_b[key]			
			
			Target_curr =np.argmax(list(Gaze_Seq_Turncurr.values())) + 1
	
			for i_flash in range(len(Flash_Seq_Turncurr)):
				pos_tmp = Flash_Seq_Turncurr[i_flash]['PosXTime']
				numcol = int(Flash_Seq_Turncurr[i_flash]['Name'][int(np.fix(float(Flash_Seq_Turncurr[i_flash]['Name'].find("Column"))))+6:])
				Target = (numcol == Target_curr)
				if (i_f==0):
					 PosixT0 = pos_tmp
				latency = pos_tmp - PosixT0
				MatEvt[i_f,0] = latency
				MatEvt[i_f,2] = numcol + Target * 10
				IsTarget[i_f] = Target
				i_f = i_f + 1
	
	# EPOCHS
	DataEpoch = data_dict['EEGRawData']['Test']
	NbEpochs_Target = NbTurns * NumberOfRepetitions
	NbEpochs_NoTarget = NbTurns * NumberOfRepetitions * (NbItems-1)
	
	
	
	Data_epo_Target = np.zeros((NbEpochs_Target,NbChans,NbSampleWin))
	Data_epo_NoTarget = np.zeros((NbEpochs_NoTarget,NbChans,NbSampleWin))
	i_epoch_Targ = 0
	i_epoch_NoTarg = 0
	for i_flash in range(NbTotFlashs):
		List_epoch_curr = DataEpoch[i_flash]
		if (IsTarget[i_flash]):
			for j_chan in range(NbChans):
				Data_epo_Target[i_epoch_Targ,j_chan,:] = List_epoch_curr[j_chan]
			i_epoch_Targ = i_epoch_Targ + 1
		else:
			for j_chan in range(NbChans):
				Data_epo_NoTarget[i_epoch_NoTarg,j_chan,:] = List_epoch_curr[j_chan]
			i_epoch_NoTarg = i_epoch_NoTarg + 1
				
				
				
	info = mne.create_info(ChannelsName, SamplingFrequency, ch_types=ch_types_Chan)
	ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
	
	MatEvt_Target = np.squeeze(MatEvt[np.where(MatEvt[:,2]>10),:])
	
	Evtcurr = np.unique(MatEvt_Target[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]-10) + '/Target'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr]
	
	Epochs_Target=mne.EpochsArray(Data_epo_Target*1e-6,info=info,events = MatEvt_Target, event_id=event_id)
	Epochs_Target.set_montage(ten_twenty_montage)
	
	MatEvt_NoTarget = np.squeeze(MatEvt[np.where(MatEvt[:,2]<10),:])
	Evtcurr = np.unique(MatEvt_NoTarget[:,2])
	event_id = {}
	for ievtcurr in range(len(Evtcurr)):
		Labelevtcurr = 'Col' + str(Evtcurr[ievtcurr]) + '/NoTarget'
		event_id[Labelevtcurr] = Evtcurr[ievtcurr]
	
	Epochs_NoTarget=mne.EpochsArray(Data_epo_NoTarget*1e-6,info=info,events = MatEvt_NoTarget, event_id=event_id)
	Epochs_NoTarget.set_montage(ten_twenty_montage)
	
	return Epochs_Target,Epochs_NoTarget,NumberOfRepetitions,NbItems,NbTurns,EEG_item_Results



#{"name": "Cz"}, {"name": "T7"}, {"name": "FC1"}, {"name": "FC5"}, {"name": "F9"}, {"name": "F3"}, {"name": "F7"}, {"name": "Fp1"}, {"name": "Pz"}, {"name": "C3"}, {"name": "O1"}, {"name": "P9"}, {"name": "P3"}, {"name": "P7"}, {"name": "CP1"}, {"name": "CP5"}, {"name": "Fz"}, {"name": "T8"}, {"name": "FC2"}, {"name": "FC6"}, {"name": "F10"}, {"name": "F4"}, {"name": "F8"}, {"name": "Fp2"}, {"name": "Oz"}, {"name": "C4"}, {"name": "O2"}, {"name": "P10"}, {"name": "P4"}, {"name": "P8"}, {"name": "CP2"}, {"name": "CP6"}


