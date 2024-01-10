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
def model_source(t, V0, Beff, phi, lamda) :
    return 1 + V0 * np.cos(2 * np.pi * Beff * omega * t / lamda + phi)
###################

### Power Distribution 1st Step ###
def data_filter(data, t, freq_pos, p0) :
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
    
    # First Iteration to find Optimal tmax and sigma (should be constant across λ)
    our_freqs = np.arange(freq_pos[0], freq_pos[1])
    chisq = np.zeros(len(our_freqs), dtype=float)
    for k, f in enumerate(our_freqs) :
        # Get p0
        if k == 0 :
            p = p0
        else :
            p = opt_params_tot[k-1, :]
            
        # Fit data
        params, cov = curve_fit(model_filter, t, data[f, :], p, maxfev=5000)
        chi2 = 1.0 * np.sum( (model_filter(t, *params) - data[f, :])**2 ) 
        chisq[k] = chi2
        opt_params_tot[k, :] = params
        cov_tot[k, :, :] = cov
     
    # Find best tmax & sigma via optimal chisq
    chimin = min(chisq)
    for k in range(len(chisq)) :
        if chisq[k] == chimin : 
            tmax_opt = opt_params_tot[k, 1]
            sigma_opt = opt_params_tot[k, 2]
            sig_t = np.sqrt(cov_tot[k, 1, 1])
            sig_sigma = np.sqrt(cov_tot[k, 2, 2])
    
    print("Optimal Parameters of Normaldistribution")
    print("----------------------------------------")
    print(f"tmax  = {tmax_opt:.3} ± {sig_t:.3} s")
    print(f"sigma = {sigma_opt:.3} ± {sig_sigma:.3}")
    print("")
    
    # Def Model
    model_fit = lambda t, S0, a, b : S0 * normal(t, tmax_opt, sigma_opt) + N(t, a, b)
    
    # Loop over all frequencies, use opt_params from previous as p0 
    our_freqs = np.arange(freq_pos[0], freq_pos[1])
    opt_params_tot = np.zeros((freq_pos[1] - freq_pos[0], 3), dtype=float)
    cov_tot = np.zeros((freq_pos[1] - freq_pos[0], 3, 3), dtype=float)
    for k, f in enumerate(our_freqs) :
        # Get p0
        if k == 0 :
            p = np.array([p0[0], p0[3], p0[4]])
        else :
            p = opt_params_tot[k-1, :]
        
        # Fit data
        params, cov = curve_fit(model_fit, t, data[f, :], p, maxfev=5000)
        chi2 = 1.0 * np.sum( (model_fit(t, *params) - data[f, :])**2 ) 
        print(f"chisq in fit of {k}th f: r = {chi2:.3}")
        
        # Plot Fitted Data
        if k % 10 == 0:
            plt.figure()
            plt.plot(t, data[f, :], label=f"Raw Data of {k}th f")
            plt.plot(t, model_fit(t, *params), label="Fit of Offset and Exponential")
            plt.xlabel("t [s]")
            plt.ylabel("Spectral Density [W/Hz]")
            plt.legend()
            plt.show()
        
        # Update
        opt_params_tot[k, :] = params
        cov_tot[k, :, :] = cov
        
        # Filter Data
        S0, a, b = params
        data_filtered[k, :] = (data[f, :] - (a*t+b))/(S0 * normal(t, tmax_opt, sigma_opt))
    
    # Return
    return data_filtered, opt_params_tot, cov_tot, tmax_opt, sigma_opt
##################

### Power Distribution 2nd Step ###
def data_fitter(data_filtered, t, freqs, freq_pos, tmax, sigma, p0) :
    """
    data_filtered is result of previous function. 
    t, freqs are standart notation.
    freqs_pos are as in data_filter.
    p0 = (V0, Beff, phi)
    
    returns V0 and Beff/lambda.
    """
    # Concentrate on one sigma neighboorhoud tmax
    eps = 0.1
    a = np.where(eps > abs(tmax - t))[0][0]
    d = np.where(eps > abs(tmax + sigma/omega - t))[0][0] - np.where(eps > abs(tmax - t))[0][0]
    t_range = t[a-d : a+d]
    
    # Generate Arrays for Return
    V0 = np.zeros((freq_pos[1] - freq_pos[0]), dtype=float)
    lamda = c/freqs[req_pos[0]: freq_pos[1]]
    
    # Find optimal Beff
    
    # Loop over all frequencies, use opt_params from previous as p0 
    our_freqs = np.arange(freq_pos[0], freq_pos[1])
    opt_params_tot = np.zeros((freq_pos[1] - freq_pos[0], 3), dtype=float)
    for k, f in enumerate(our_freqs) :
        # Get p0
        if k == 0 :
            p = p0
        else :
            p = opt_params_tot[k-1, :]
        
        # Model For Fit (lambda dependet)
        l = c/freqs[f]
        model_fit = lambda t, V0, Beff, phi : 1 + V0 * np.cos(2 * np.pi * Beff * omega * t / l + phi)
        
        # Fit
        params, cov = curve_fit(model_fit, t_range, data_filtered[k, a-d : a+d], p, maxfev=5000)
        chi2 = 1.0 * np.sum( (model_fit(t_range, *params) - data_filtered[k, a-d : a+d])**2 ) 
        print(f"chisq in fit of {k}th f: r = {chi2:.3}")
        
        # Update
        V0[k] = params[0]
        
        # Plot Fitted Data
        if k % 10 == 0:
            plt.figure()
            plt.plot(t_range, data_filtered[k, a-d : a+d], label=f"Filtered Data of {k}th f")
            plt.plot(t_range, model_fit(t_range, *params), label="Fit of Wavelength Dependence")
            plt.xlabel("t [s]")
            plt.ylabel("Spectral Density [W/Hz]")
            plt.legend()
            plt.show()
    
    # Return Dependence
    return V0, Beff/lamda