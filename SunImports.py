### Imports ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.constants import c
###############


### Reading In Data ###
def read_in_data(file) :
    """
    IN : file as ordner/filename
    OUT: 2D-array data, time, frequency
    """
    path = '/Users/juliusadolff/Desktop/Interferometer 2024/' + file
    f = fits.open(path)
    data = np.array(f[0].data, dtype=float)
    t = f[1].data['TIME'][0]
    freq = f[1].data['FREQUENCY'][0]
    return data, t, freq

def glue_together(files) :
    N = len(files)
    Data = np.zeros((200, N * 3600), dtype=float)
    t_tot = np.linspace(0, N * 900, 3600 * N)
    for k, file in enumerate(files) :
        datai, _, freq = read_in_data(file)
        Data[:, k*3600:((1+k)*3600)] = datai
    return Data, t_tot, freq
#######################

### Useful Definitions ###
omega = 2 * np.pi / ( 24.0 * 60 * 60 )

# Normal Distribution
def normal(t, tmax, sigma) :
    # sigma = 0.065
    return np.exp(- omega**2 * (t - tmax)**2 / (2 * sigma**2) )

# Noise Offset
N = lambda t, a, b : a*t + b

# Model Filter 
def model_filter(t, S0, tmax, sigma, a, b) :
    return S0 * normal(t, tmax, sigma) + N(t, a, b)

# Model Source
def model_source(t, V0, Beff, )
return 1 + V0 * np.cos(2 * np.pi * B * omega * t / lamda + phi)
###################

### Power Distribution 1st Step ###
def data_filter(data, t, freq, freq_pos, p0) :
    """
    data, t, freq are standart notation.
    freq_pos = (pos_freq_min, pos_freq_max) is the band of frequencies considered
    p0 = (S0, tmax, sigma, a, b), inital guess of fit parameters for freq_min
    
    returns the filtered data and optimal fit parameters for further evaluation
    """
    # Generate Arrays For Return
    data_filtered = np.zeros(data[freq_pos[0] : freq_pos[1], :].shape, dtype=float)
    opt_params_tot = np.zeros((freq_pos[1] - freq_pos[0], 5), dtype=float)
    cov_tot = np.zeros((freq_pos[1] - freq_pos[0], 5, 5), dtype=float)
    
    
    
    # Loop over all frequencies, use opt_params from previous as p0 
    our_freqs = freq[freq_pos[0] : freq_pos[1]]
    for k, f in enumerate(our_freqs) :
        