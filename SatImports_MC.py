### Imports ###
import numpy as np
import os
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
    path = os.path.join(os.getcwd(),file)
    f = fits.open(path)
    data = np.array(f[0].data, dtype=float)
    t = f[1].data['TIME'][0]
    freq = f[1].data['FREQUENCY'][0]
    return data, t, freq
###############

### General Fit Functions ###
B = 6.4

def Ptheo(t, V0, lamda, omega, phi) :
    return 1 + V0 * np.cos(2 * np.pi * B * omega * t / lamda + phi)


N = lambda t, a, b : a*t + b




###############

### Satelite Specfic Functions ###
def max_neigh(data, range_1, range_2) :
    """
    IN : range_1 is y bounds as 2d tupel, range_2 is x bound as 2D tupel
    OUT: position of first max as 2d tupel
    """
    box = data[range_1[0]:range_1[1], range_2[0]:range_2[1]]
    argmax = np.where( box == max(box.copy().flatten()) )
    return argmax[0][0] + range_1[0], argmax[1][0] + range_2[0]


rng = np.random.default_rng()

def monte_carlo_filter(model,data, t, mean, cov) :
    
    # Random Variables
    N_mc = 10000
    paras = rng.multivariate_normal(mean, cov, N_mc)
    S0 = paras[:, 0]; omega = paras[:, 1]
    a = paras[:, 2]; b = paras[:, 3]; 
    # Data
    mc_data = np.zeros((N_mc, *data.shape), dtype=float)
    mc_mean = data.copy(); mc_sig = data.copy()
    
    for i in range(N_mc) :
        mc_data[i, :] = ( data - N(t, a[i], b[i]) ) / model(t, S0[i], omega[i])
    
    for j in range(len(data)) :
        mc_mean[j] = np.mean(mc_data[:,j])
        mc_sig[j] = np.std(mc_data[:, j])
    
    # Return
    return mc_mean, mc_sig


def monte_carlo_omega(data, data_sig, t, l, p02):
    Ptheo_fit = lambda t, omega, phi : 1 + np.cos(2 * np.pi * B * omega * t / l + phi)

    N_mc = 10000
    mean = np.array(data)
    cov = np.diag(data_sig**2) 
    data_paras = rng.multivariate_normal(mean, cov, N_mc)
    omega_mc = np.zeros(N_mc, dtype=float)
    phi_mc = np.zeros(N_mc, dtype=float)

    
    for i in range(N_mc) :
        omega_phi_params, cov = curve_fit(Ptheo_fit, t, data_paras[i,:], p02) 
        omega_mc[i] = omega_phi_params[0]
        phi_mc[i] = omega_phi_params[1]
    

    return np.median(omega_mc), np.std(omega_mc), np.mean(phi_mc)



def fit_sat(data, pos_sat, t, freq, p01, p02) :
    """
    IN : Standart Notation. pos_sat = (range_1, range_2). p0 = inital guess of params, tmax eval by function
    p01 = (S0, omega, sigma, a, b)
    p02 = (omega, phi)
    OUT: params, cov
    """
    fpos, tpos = max_neigh(data, pos_sat[0], pos_sat[1])
    
    tmax = t[tpos]; fmax = freq[fpos] * 1e6
    sigma = 0.065 
    l = c/fmax;
    t_range = t[tpos-80 : tpos+80]

    def Pexp(t, S0, omega, a, b) :
        return S0 * np.exp(- omega**2 * (t - tmax)**2 / (2 * sigma**2) ) + (a*t+b)
    
    Pexp_fit = lambda t, S0, omega: S0 * np.exp(- omega**2 * (t - tmax)**2 / (2 * sigma**2) ) 

    
    params, cov = curve_fit(Pexp, t, data[fpos, :], p01, absolute_sigma=True) #t, S0, omega, a, b
    
    data_pure, data_pure_sig = monte_carlo_filter(Pexp_fit,data[fpos,:], t, params, cov)#model,data, t, mean, cov

   
    
    plt.figure()
    plt.plot(t, data[fpos, :], label="Raw Data")
    plt.plot(t, Pexp(t, *params), label="Fit of Offset and Exponential")
    plt.xlabel("t [s]")
    plt.ylabel("Spectral Density [W/Hz]")
    plt.legend()
    plt.show()
    
    data_pure_range = data_pure[tpos-80 : tpos+80]
    data_pure_sig_range = data_pure_sig[tpos-80:tpos+80]
   

    omega, omega_sig, phi = monte_carlo_omega(data_pure_range, data_pure_sig_range, t_range, l, p02)
    
    plt.figure()
    plt.plot(t_range, data_pure_range, label="Filtered Data")
    plt.plot(t_range, Ptheo(t_range, 1, l, omega, phi), label="Fit of Wavelength Dependence")
    plt.xlabel("t [s]")
    plt.ylabel("Spectral Density [W/Hz]")
    plt.legend()
    plt.show()
        
    Ptot = lambda t : params[0] * (1 + np.cos(2 * np.pi * B * omega * t / l + phi)) * np.exp(-params[1]**2 * (t - tmax)**2 / (2 * sigma**2) ) + (params[2] * t + params[3])
    
    plt.figure()
    plt.plot(t[tpos-180 : tpos+180], data[fpos, tpos-180 : tpos+180], label="Raw Data")
    plt.plot(t[tpos-180 : tpos+180], Ptot(t[tpos-180 : tpos+180]), label="Total Fit")
    plt.xlabel("t [s]")
    plt.ylabel("Spectral Density [W/Hz]")
    plt.legend()
    plt.show()

    # Determining the height of the satelite
    from scipy.optimize import fsolve
    from scipy.constants import G
    func = lambda h : omega**2 * (6371e3 + h)*h**2 - G * 5.972e24
    h = fsolve(func, 2000e3)[0]
    
    print(f"Angular Velocity of Satelite from our PoV: omega ~ {omega:.3} ± {omega_sig:.3} 1/s")
    print(f"Velocity of Satelite: vr ~ {omega*h:.3} m/s")
    omega_prime = omega * h / (6371e3 + h)
    print(f"Period of Satelite: T ~ {2 * np.pi / omega_prime:.3} s = {2 * np.pi / (60 * omega_prime):.3} min")
    print(f"Height of Salelite: h ~ {h:.3} m")
###############