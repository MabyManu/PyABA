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

def ReadCalib_JsonFile_Mentalink4(JsonFileName):
	with open(JsonFileName) as json_data:
		print(type(json_data))
		data_dict = json.load(json_data)
		
	Settings = data_dict['Setting']
	SamplingFrequency = Settings['SampleRate']
	if 'FlashCount:' in Settings:
		NumberOfRepetitions = int(Settings['FlashCount:'])
	else:
		NumberOfRepetitions = int(Settings['FlashCount'])
		
	if 'NbSampleWin' in Settings:
		NbSampleWin = Settings['NbSampleWin']
		
	NbChans = len(Settings['Electrodes'])
	ChannelsName=[]
	ch_types_Chan = []
	for i_chan in range(NbChans):
		ChannelsName.append(Settings['Electrodes'][i_chan]['name'])
		if '_dir' in Settings['Electrodes'][i_chan]['name']:
			ch_types_Chan.append('misc')
		else:
			ch_types_Chan.append('eeg')

		
	
	# EVENT
	if 'SequenceList' in data_dict:
		i_Flash = 0
		Seqlist = data_dict["SequenceList"]["Values"][0]
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
	NbEpochs_Target = len(Data_Target)
	Data_epo_Target = np.zeros((NbEpochs_Target,NbChans,NbSampleWin))
	for i_epoch in range(NbEpochs_Target):
		for j_chan in range(NbChans):
			Data_epo_Target[i_epoch,j_chan,:] = Data_Target[i_epoch][j_chan]
			
	Data_NoTarget = data_dict['EEGRawData']['NonTarget']
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
	
	return Epochs_Target,Epochs_NoTarget,NumberOfRepetitions,NbItems,NbTurns