# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:26:16 2023

@author: manum
"""
import numpy as np
from scipy.signal import fftconvolve
import scipy.signal as signal
from scipy.signal import hilbert
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy import interpolate

import h5py

from pyriemann.utils.base import expm, invsqrtm, logm, sqrtm
from pyriemann.utils.covariance import covariances_EP
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import eigvalsh
import mne_tools
import os
import json
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QFileDialog,QListView,QAbstractItemView,QTreeView


def RemplaceContentAndCopy(OrigFile, NewFile, TargetText, RemplacementText):
	# Opening our text file in read only 
	# mode using the open() function 
	with open(OrigFile, 'r') as file: 
	  
	    # Reading the content of the file 
	    # using the read() function and storing 
	    # them in a new variable 
	    data = file.read() 
	  
	    # Searching and replacing the text 
	    # using the replace() function 
	    data = data.replace(TargetText, RemplacementText) 
	  
	# Opening our text file in write only 
	# mode to write the replaced content 
	with open(NewFile, 'w') as file: 
	  
	    # Writing the replaced data in our 
	    # text file 
	    file.write(data) 

def select_folders(RootDirectory):
			
	file_dialog = QFileDialog()
	file_dialog.setDirectory(RootDirectory)
	file_dialog.setFileMode(QFileDialog.DirectoryOnly)
	file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
	file_view = file_dialog.findChild(QListView, 'listView')
	
	# to make it possible to select multiple directories:
	if file_view:
	    file_view.setSelectionMode(QAbstractItemView.MultiSelection)
	f_tree_view = file_dialog.findChild(QTreeView)
	if f_tree_view:
	    f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
	
	if file_dialog.exec():
	    paths = file_dialog.selectedFiles()
		
	return paths

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


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise




def superlet(
    data_arr,
    samplerate,
    scales,
    order_max,
    order_min=1,
    c_1=3,
    adaptive=False,
):

    """
       Performs Superlet Transform (SLT) according to Moca et al. [1]_
       Both multiplicative SLT and fractional adaptive SLT are available.
       The former is recommended for a narrow frequency band of interest,
       whereas the  is better suited for the analysis of a broad range
       of frequencies.

       A superlet (SL) is a set of Morlet wavelets with increasing number
       of cycles within the Gaussian envelope. Hence the bandwith
       is constrained more and more with more cycles yielding a sharper
       frequency resolution. Complementary the low cycle numbers will give a
       high time resolution. The SLT then is the geometric mean
       of the set of individual wavelet transforms, combining both wide
       and narrow-bandwidth wavelets into a super-resolution estimate.

       Parameters
       ----------
       data_arr : nD :class:`numpy.ndarray`
           Uniformly sampled time-series data
           The 1st dimension is interpreted as the time axis
       samplerate : float
           Samplerate of the time-series in Hz
       scales : 1D :class:`numpy.ndarray`
           Set of scales to use in wavelet transform.
           Note that for the SL Morlet the relationship
           between scale and frequency simply is s(f) = 1/(2*pi*f)
           Need to be ordered high to low for `adaptive=True`
       order_max : int
           Maximal order of the superlet set. Controls the maximum
           number of cycles within a SL together
           with the `c_1` parameter: c_max = c_1 * order_max
       order_min : Minimal order of the superlet set. Controls
           the minimal number of cycles within a SL together
           with the `c_1` parameter: c_min = c_1 * order_min
           Note that for admissability reasons c_min should be at least 3!
       c_1 : int
           Number of cycles of the base Morlet wavelet. If set to lower
           than 3 increase `order_min` as to never have less than 3 cycles
           in a wavelet!
       adaptive : bool
           Wether to perform multiplicative SLT or fractional adaptive SLT.
           If set to True, the order of the wavelet set will increase
           linearly with the frequencies of interest from `order_min`
           to `order_max`. If set to False the same SL will be used for
           all frequencies.

       Returns
       -------
       gmean_spec : :class:`numpy.ndarray`
           Com Moca, Vasile V., et al. "Time-frequency super-resolution with superlets."
          Nature communications 12.1 (2021): 1-18.

    plex time-frequency representation of the input data.
           Shape is (len(scales), data_arr.shape[0], data_arr.shape[1]).

       Notes
       -----
       .. [1]
    """

    # adaptive SLT
    if adaptive:

        gmean_spec = FASLT(data_arr, samplerate, scales, order_max, order_min, c_1)

    # multiplicative SLT
    else:

        gmean_spec = multiplicativeSLT(
            data_arr, samplerate, scales, order_max, order_min, c_1
        )

    return gmean_spec


def multiplicativeSLT(data_arr, samplerate, scales, order_max, order_min=1, c_1=3):

    dt = 1 / samplerate
    # create the complete multiplicative set spanning
    # order_min - order_max
    cycles = c_1 * np.arange(order_min, order_max + 1)
    order_num = order_max + 1 - order_min  # number of different orders
    SL = [MorletSL(c) for c in cycles]

    # lowest order
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)
    gmean_spec = np.power(gmean_spec, 1 / order_num)

    for wavelet in SL[1:]:

        spec = cwtSL(data_arr, wavelet, scales, dt)
        gmean_spec *= np.power(spec, 1 / order_num)

    return gmean_spec


def FASLT(data_arr, samplerate, scales, order_max, order_min=1, c_1=3):

    """Fractional adaptive SL transform

    For non-integer orders fractional SLTs are
    calculated in the interval [order, order+1) via:

    R(o_f) = R_1 * R_2 * ... * R_i * R_i+1 ** alpha
    with o_f = o_i + alpha
    """

    dt = 1 / samplerate
    # frequencies of interest
    # from the scales for the SL Morlet
    fois = 1 / (2 * np.pi * scales)
    orders = compute_adaptive_order(fois, order_min, order_max)

    # create the complete superlet set from
    # all enclosed integer orders
    orders_int = np.int32(np.floor(orders))
    cycles = c_1 * np.unique(orders_int)
    SL = [MorletSL(c) for c in cycles]

    # every scale needs a different exponent
    # for the geometric mean
    exponents = 1 / (orders - order_min + 1)

    # which frequencies/scales use the same integer orders SL
    order_jumps = np.where(np.diff(orders_int))[0]
    # each frequency/scale will have its own multiplicative SL
    # which overlap -> higher orders have all the lower orders

    # the fractions
    alphas = orders % orders_int

    # 1st order
    # lowest order is needed for all scales/frequencies
    gmean_spec = cwtSL(data_arr, SL[0], scales, dt)  # 1st order <-> order_min
    # Geometric normalization according to scale dependent order
    gmean_spec = np.power(gmean_spec.T, exponents).T

    # we go to the next scale and order in any case..
    # but for order_max == 1 for which order_jumps is empty
    last_jump = 1

    for i, jump in enumerate(order_jumps):

        # relevant scales for the next order
        scales_o = scales[last_jump:]
        # order + 1 spec
        next_spec = cwtSL(data_arr, SL[i + 1], scales_o, dt)

        # which fractions for the current next_spec
        # in the interval [order, order+1)
        scale_span = slice(last_jump, jump + 1)
        gmean_spec[scale_span, :] *= np.power(
            next_spec[: jump - last_jump + 1].T,
            alphas[scale_span] * exponents[scale_span],
        ).T

        # multiply non-fractional next_spec for
        # all remaining scales/frequencies
        gmean_spec[jump + 1 :] *= np.power(
            next_spec[jump - last_jump + 1 :].T, exponents[jump + 1 :]
        ).T

        # go to the next [order, order+1) interval
        last_jump = jump + 1

    return gmean_spec


class MorletSL:
    def __init__(self, c_i=3, k_sd=5):

        """The Morlet formulation according to
        Moca et al. shifts the admissability criterion from
        the central frequency to the number of cycles c_i
        within the Gaussian envelope which has a constant
        standard deviation of k_sd.
        """

        self.c_i = c_i
        self.k_sd = k_sd

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):

        """
        Complext Morlet wavelet in the SL formulation.

        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time

        """

        ts = t / s
        # scaled time spread parameter
        # also includes scale normalisation!
        B_c = self.k_sd / (s * self.c_i * (2 * np.pi) ** 1.5)

        output = B_c * np.exp(1j * ts)
        output *= np.exp(-0.5 * (self.k_sd * ts / (2 * np.pi * self.c_i)) ** 2)

        return output


def fourier_period(scale):

    """
    This is the approximate Morlet fourier period
    as used in the source publication of Moca et al. 2021

    Note that w0 (central frequency) is always 1 in this
    Morlet formulation, hence the scales are not compatible
    to the standard Wavelet definitions!
    """

    return 2 * np.pi * scale


def scale_from_period(period):

    return period / (2 * np.pi)


def cwtSL(data, wavelet, scales, dt):

    """
    The continuous Wavelet transform specifically
    for Morlets with the Superlet formulation
    of Moca et al. 2021.

    - Morlet support gets adjusted by number of cycles
    - normalisation is with 1/(scale * 4pi)
    - this way the norm of the spectrum (modulus)
      at the corresponding harmonic frequency is the
      harmonic signal's amplitude

    Notes
    -----

    The time axis is expected to be along the 1st dimension.
    """

    # wavelets can be complex so output is complex
    output = np.zeros((len(scales),) + data.shape, dtype=np.complex64)

    # this checks if really a Superlet Wavelet is being used
    if not isinstance(wavelet, MorletSL):
        raise ValueError("Wavelet is not of MorletSL type!")

    # 1st axis is time
    slices = [None for _ in data.shape]
    slices[0] = slice(None)

    # compute in time
    for ind, scale in enumerate(scales):

        t = _get_superlet_support(scale, dt, wavelet.c_i)
        # sample wavelet and normalise
        norm = dt**0.5 / (4 * np.pi)
        wavelet_data = norm * wavelet(t, scale)  # this is an 1d array for sure!
        output[ind, :] = fftconvolve(data, wavelet_data[tuple(slices)], mode="same")

    return output


def _get_superlet_support(scale, dt, cycles):

    """
    Effective support for the convolution is here not only
    scale but also cycle dependent.
    """

    # number of points needed to capture wavelet
    M = 10 * scale * cycles / dt
    # times to use, centred at zero
    t = np.arange((-M + 1) / 2.0, (M + 1) / 2.0) * dt

    return t


def compute_adaptive_order(freq, order_min, order_max):

    """
    Computes the superlet order for a given frequency of interest
    for the fractional adaptive SLT (FASLT) according to
    equation 7 of Moca et al. 2021.

    This is a simple linear mapping between the minimal
    and maximal order onto the respective minimal and maximal
    frequencies.

    Note that `freq` should be ordered low to high.
    """

    f_min, f_max = freq[0], freq[-1]

    assert f_min < f_max

    order = (order_max - order_min) * (freq - f_min) / (f_max - f_min)

    # return np.int32(order_min + np.rint(order))
    return order_min + order


# ---------------------------------------------------------
# Some test data akin to figure 3 of the source publication
# ---------------------------------------------------------


def gen_superlet_testdata(freqs=[20, 40, 60], cycles=11, fs=1000, eps=0):

    """
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 3 in Moca et al. 2021 NatComm
    """

    signal = []
    for freq in freqs:

        # 10 cycles of f1
        tvec = np.arange(cycles / freq, step=1 / fs)

        harmonic = np.cos(2 * np.pi * freq * tvec)
        f_neighbor = np.cos(2 * np.pi * (freq + 10) * tvec)
        packet = harmonic + f_neighbor

        # 2 cycles time neighbor
        delta_t = np.zeros(int(2 / freq * fs))

        # 5 cycles break
        pad = np.zeros(int(5 / freq * fs))

        signal.extend([pad, packet, delta_t, harmonic])

    # stack the packets together with some padding
    signal.append(pad)
    signal = np.concatenate(signal)

    # additive white noise
    if eps > 0:
        signal = np.random.randn(len(signal)) * eps + signal

    return signal


def fill_nan(A):
	'''
	interpolate to fill nan values
	'''
	inds = np.arange(A.shape[0])
	good = np.where(np.isfinite(A))
	
	if (len(good[0])>0):
		if (np.isnan(A[0])):
			A[0] = A[good[0][0]]
			
		if (np.isnan(A[-1])):
			A[-1] = A[good[0][-1]]	
		good = np.where(np.isfinite(A))
		f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
		B = np.where(np.isfinite(A),A,f(inds))
	else:
		B=[]
	return B


def AutoReject(Data,PercentageOfEpochsRejected):
	NbEpoch2Keep = int(np.fix(Data.shape[0] * (1.0-(PercentageOfEpochsRejected/100)))-1)
	Mat_epoch =  Data
	MinWOI = Mat_epoch.min(axis=1)
	MaxWOI = Mat_epoch.max(axis=1)
	Peak2Peak = MaxWOI-MinWOI
	Peak2Peak.sort()
	Threshold = Peak2Peak[NbEpoch2Keep-1]
	ixRej = np.squeeze(np.where(Peak2Peak>Threshold))
	KeptData = np.delete(Data, ixRej, 0)
	return KeptData




def append_to_json_file(file_path, new_data):
    # Vérifier si le fichier existe
    if not os.path.isfile(file_path):
        # Si le fichier n'existe pas, initialiser un tableau vide
        data = []
    else:
        # Lire le contenu existant du fichier JSON
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = [json.load(file)]
            except json.JSONDecodeError:
                # Si le fichier est vide ou corrompu, initialiser un tableau vide
                data = []

    # Ajouter les nouvelles données au contenu existant
    if isinstance(data, list):
        data.append(new_data)
    else:
        # Si le fichier JSON n'est pas un tableau, lever une exception
        raise ValueError("Le fichier JSON doit contenir un tableau à la racine.")

    # Écrire les données mises à jour dans le fichier JSON
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
		
		
		
		
		
def linearly_interpolate_nans(y):
	# Fit a linear regression to the non-nan y values
	
	# Create X matrix for linreg with an intercept and an index
	X = np.vstack((np.ones(len(y)), np.arange(len(y))))
	
	# Get the non-NaN values of X and y
	X_fit = X[:, ~np.isnan(y)]
	y_fit = y[~np.isnan(y)].reshape(-1, 1)
	
	# Estimate the coefficients of the linear regression
	beta = np.linalg.lstsq(X_fit.T, y_fit,rcond=-1)[0]
	
	# Fill in all the nan values using the predicted coefficients
	y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
	return y



def calculer_spectre(signal, frequence_echantillonnage):
	"""
	Calcule le spectre d'un signal en utilisant la transformation de Fourier rapide (FFT).
	
	:param signal: Le signal temporel (array-like).
	:param frequence_echantillonnage: La fréquence d'échantillonnage du signal en Hz.
	:return: Les fréquences et le spectre (amplitude) du signal.
	"""
	# Calcul de la FFT
	spectre = np.fft.fft(signal)
	
	# Calculer la phase
	phase = np.angle(spectre)
	# Nombre de points dans le signal
	n  = len(signal)
	    
	# Fréquences associées aux composants de la FFT
	freqs = np.fft.fftfreq(n, 1 / frequence_echantillonnage)
	    
	# Sélectionner la moitié du spectre (partie positive des fréquences)
	freqs = freqs[:n // 2]
	spectre = np.abs(spectre[:n // 2]) / n
	phase = phase[:n // 2]
	return freqs, spectre,phase



def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y



def highpass_filter(data, cutoff_freq, sample_rate, order=2):
    """
    Applique un filtre passe-haut à un signal.

    Parameters:
    data (array-like): Le signal à filtrer.
    cutoff_freq (float): La fréquence de coupure du filtre (en Hz).
    sample_rate (float): La fréquence d'échantillonnage du signal (en Hz).
    order (int): L'ordre du filtre. Plus l'ordre est élevé, plus le filtrage est précis. (par défaut 5)

    Returns:
    array-like: Le signal filtré.
    """
    nyquist = 0.5 * sample_rate  # Fréquence de Nyquist
    normal_cutoff = cutoff_freq / nyquist  # Fréquence de coupure normalisée

    # Conception du filtre passe-haut Butterworth
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    # Application du filtre
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data




def phase_atFreqTarget(signal,FreqTarget, fs, bandwidth=0.5):
	filtered_signal = bandpass_filter(signal, FreqTarget - bandwidth / 2, FreqTarget + bandwidth / 2, fs)
	analytic_signal = hilbert(filtered_signal)
	instantaneous_phase = np.angle(analytic_signal)
	return instantaneous_phase



def phase_synchrony_index(signal,Deltaf,Freqmin,Freqmax, fs, bandwidth=1.0):
	"""
	Calcule la synchronie de phase entre toutes fréquences pour un signal donné.
	:param signal: Le signal d'entrée (array)
 	:param Deltaf: précison fréquentielle (Hz)
 	:param Freqmin: Fréquence min (Hz)
 	:param Freqmax: Fréquence max (Hz)
 	:param bandwidth: Largeur de bande pour le filtrage passe-bande
 	:return: Indice de synchronie de phase entre f1 et f2
 	"""
	 
	 
	freqs = np.arange(Freqmin,Freqmax,Deltaf)
	CoSpectral_Sync = np.zeros((len(freqs),len(freqs)))
	for i_freq1,freq_curr1 in enumerate(freqs):
		filtered_signal1 = bandpass_filter(signal, freq_curr1 - bandwidth / 2, freq_curr1 + bandwidth / 2, fs)
		# Calculer les phases instantanées
		analytic_signal1 = hilbert(filtered_signal1)
		instantaneous_phase1 = np.angle(analytic_signal1)
		for i_freq2,freq_curr2 in enumerate(freqs):
			filtered_signal2 = bandpass_filter(signal, freq_curr2 - bandwidth / 2, freq_curr2 + bandwidth / 2, fs)
			analytic_signal2 = hilbert(filtered_signal2)
			instantaneous_phase2 = np.angle(analytic_signal2)
			
			# Calculer les différences de phase
			phase_diff = instantaneous_phase1 - instantaneous_phase2
			phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi  # Ramener à [-π, π]
			
			# Calculer l'indice de synchronie de phase
			CoSpectral_Sync[i_freq1,i_freq2] = np.abs(np.mean(np.exp(1j * phase_diff)))
			
	return CoSpectral_Sync,freqs











def ReadTemplate_H5(Template_H5Filename):
    f = h5py.File(Template_H5Filename, 'r')
    
    print('## Lecture du fichier {}'.format(Template_H5Filename))
    
    TemplateParams = {}
    for element in f:
        groupe = f[element]
                    
        for element in groupe:
            TemplateParams[element] = groupe[element][:]
            
    return TemplateParams
        


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
	
	
	
	
	
def remove_outliers(data,Pct_Percentile,Factor_bound):
    # Calcul des quartiles
    Q1 = np.percentile(data, Pct_Percentile)
    Q3 = np.percentile(data, 100-Pct_Percentile)
    IQR = Q3 - Q1
    
    # Définir les bornes inférieure et supérieure
    lower_bound = Q1 - Factor_bound * IQR
    upper_bound = Q3 + Factor_bound * IQR
    
    # Filtrer les outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data	
	


def plot_phases_on_circle(phases):
	"""
	Trace les phases respiratoires sur un cercle.
	Args:
		phases (array): Phases respiratoires en termes d'angle.
    """
	# Convertir les phases en radians pour l'affichage en coordonnées polaires
	radians = np.deg2rad(phases)
	# Tracer les phases sur un cercle
	plt.figure(figsize=(6, 6))
	ax = plt.subplot(111, polar=True)
	ax.scatter(radians, np.ones_like(radians), s=7, color='blue')
	ax.set_yticklabels([])  # Masquer les labels du rayon
	ax.set_theta_zero_location("E")  # 0° en haut (au nord)
	ax.set_theta_direction(1)  # Sens des aiguilles d'une montre 
	plt.title('Respiration phases synchronized with stimuli')
	plt.show()