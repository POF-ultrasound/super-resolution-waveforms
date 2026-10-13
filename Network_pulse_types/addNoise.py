# -*- coding: utf-8 -*-
"""
Add noise to the RF lines

Author: Rienk Zorgdrager, University of Twente, 2024
"""

import torch
import numpy as np
from scipy import signal

#%% NOISE

V_ref = 9.65                                # Reference value noise
noiselevels_p = np.array([4,8,16,32,64,128,256],dtype='int32')       # Noise level percentage
noiselevels   = noiselevels_p*V_ref/100       # Noise level

# Filter parameters; frequencies are in MHz to match fs.
fs = 62.5          # Sampling frequency (MHz)
n = 2              # Butterworth prototype order; bandpass transfer function is 4th order
f_low = 0.5         # Lower cutoff (MHz), consistent with experimental processing
f_high = 1.7 * 3    # Upper cutoff (MHz), preserving the current 5.1 MHz limit

# Design a bandpass filter using normalized cutoff frequencies.
filt_b, filt_a = signal.butter(
    n,
    [2 * f_low / fs, 2 * f_high / fs],
    btype='bandpass'
)

def add_noise(V,sigma,b,a):
    # Add noise to the signals and convert to torch cuda tensor
    
    mu = 0  # Mean value of the random distribution
    
    # Add noise to the signal and apply low-pass filter
    V_noise = V + np.random.normal(mu,sigma,size = V.shape)
    V_noise_filt  = signal.filtfilt(b,a,V_noise,axis=-1)
    
    # Convert the RF signals to torch cuda tensor
    V_noise_filt = torch.from_numpy(V_noise_filt.copy())
    V_noise_filt = V_noise_filt.type(torch.FloatTensor)
    V_noise_filt = V_noise_filt.cuda()
    
    return V_noise_filt