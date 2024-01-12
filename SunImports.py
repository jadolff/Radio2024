### Imports ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from scipy.constants import c
import os
from uncertainties import ufloat

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["figure.figsize"] = (16,8)
plt.rcParams['font.size'] = 35
plt.style.use('ggplot')
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

def glue_together(files, d) :
    # Integration Period: d
    # Glueing
    N = len(files) 
    n = int(3600/d)
    Data = np.zeros((200, N * n), dtype=float)
    t_tot = np.linspace(0, N * 900, n * N)
    for k, file in enumerate(files) :
        datak, _, freq = read_in_data(file)
        for i in range( 200 ) :
            for j in range(n*k, n*(1+k)) :
                Data[i, j] = np.mean( datak[i, (j - n*k)*10:((j+1 - n*k)*10)] )
    return Data, t_tot, freq * 1e6
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

# Sun
def sinc(Bl, alpha) :
    return np.sin(np.pi * Bl * alpha) / (np.pi * Bl * alpha)
###################


### Monte Carlo ###
rng = np.random.default_rng()

def monte_carlo(data, t, mean, cov) :
    
    # Random Variables
    N_mc = 1000
    paras = rng.multivariate_normal(mean, cov, N_mc)
    tmax = paras[:, 0]; sig = paras[:, 1]
    S0 = paras[:, 2]; a = paras[:, 3]; b = paras[:, 4];
    
    # Data
    mc_data = np.zeros((N_mc, *data.shape), dtype=float)
    mc_mean = data.copy(); mc_sig = data.copy()
    
    for i in range(N_mc) :
        mc_data[i, :] = ( data - N(t, a[i], b[i]) ) / (S0[i] * normal(t, tmax[i], sig[i]))
    
    for j in range(len(data)) :
        #mc_mean[j] = ( data[j] - N(t[j], mean[3], mean[4]) ) / (mean[2] * normal(t[j], mean[0], mean[1]))
        mc_mean[j] = np.mean(mc_data[:, j])
        mc_sig[j] = np.std(mc_data[:, j])
    
    # Return
    return mc_mean, mc_sig

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
    data_filtered_error = np.zeros(data[freq_pos[0] : freq_pos[1], :].shape, dtype=float)
    
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
        params, cov = curve_fit(model_filter, t, data[f, :], p, maxfev=20000)
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
            fmin = our_freqs[k]
    
    print("Optimal Parameters of Normaldistribution")
    print("----------------------------------------")
    print(f"tmax  = {tmax_opt:.3} ± {sig_t:.3} s")
    print(f"sigma = {sigma_opt:.3} ± {sig_sigma:.3}")
    print("")
    
    # sigma_opt = 0.065
    tmax_opt, sigma_opt = 2000, 0.07
    
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
        
        # Plot Fitted Data
        #print(f"chisq in fit of {k}th f: r = {chi2:.3}")
        if f == fmin :
            print(params)
            
            plt.figure()
            plt.plot(t, data[f, :], label=f"Raw Data of {k}th f", color="red")
            plt.plot(t, model_fit(t, *params), label="Fit of Offset and Exponential", color="blue")
            plt.xlabel("Time")
            plt.ylabel("Spectral Density")
            plt.legend()
            plt.show()
        
        # Get Optimal Parameters
        if f == fmin :
            params_opt = np.array([params[0], tmax_opt, sigma_opt, params[1], params[2]])
        
        # Update
        opt_params_tot[k, :] = params
        cov_tot[k, :, :] = cov
        
        # Filter Data
        para_mc = np.array([tmax_opt, sigma_opt, params[0], params[1], params[2]])
        cov_mc = np.diag([sig_t**2, sig_sigma**2, cov[0, 0], cov[1, 1], cov[2, 2]])
        data_filtered[k, :], data_filtered_error[k, :] = monte_carlo(data[f, :], t, para_mc, cov_mc)
        
        """
        cov_mc_cross = np.zeros((cov.shape[0]+2, cov.shape[1]+2))
        cov_mc_cross[0,0] = sig_t**2
        cov_mc_cross[1,1] = sig_sigma**2
        cov_mc_cross[2:,2:] = cov
        data_filtered[k, :], data_filtered_error[k, :] = monte_carlo(data[f, :], t, para_mc, cov_mc_cross)
        """
    
    # Return
    return data_filtered, data_filtered_error, opt_params_tot, cov_tot, tmax_opt, sigma_opt, params_opt, fmin
##################

### Power Distribution 2nd Step ###
def data_fitter(data_filtered, data_filtered_error, t, freqs, freq_pos, tmax, sigma, p0, fmin) :
    """
    data_filtered is result of previous function. 
    t, freqs are standart notation.
    freqs_pos are as in data_filter.
    p0 = (V0, Beff, phi)
    
    returns V0 and Beff/lambda.
    """
    # Concentrate on sigma/2 neighboorhoud tmax
    eps = 5.0
    a = np.where(eps > abs(tmax - t))[0][0]
    d = np.where(eps > abs(tmax + sigma/(2 * omega) - t))[0][0] - np.where(eps > abs(tmax - t))[0][0]
    t_range = t[a-d : a+d]
    
    # Generate Arrays for Return
    V0 = np.zeros((freq_pos[1] - freq_pos[0]), dtype=float)
    V0_uncertainties = np.zeros((freq_pos[1] - freq_pos[0]), dtype=float)
    lamda = c/freqs[freq_pos[0] : freq_pos[1]]
    
    # Find optimal Beff
    chisq = np.zeros(freq_pos[1] - freq_pos[0], dtype=float)
    our_freqs = np.arange(freq_pos[0], freq_pos[1])
    opt_params_tot = np.zeros((freq_pos[1] - freq_pos[0], 3), dtype=float)
    opt_cov = np.zeros((freq_pos[1] - freq_pos[0], 3, 3), dtype=float)
    
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
        params, cov = curve_fit(model_fit, t_range, data_filtered[k, a-d : a+d], p, maxfev=5000, sigma=100*data_filtered_error[k, a-d : a+d])
        chi2 = 1.0 * np.sum( (model_fit(t_range, *params) - data_filtered[k, a-d : a+d])**2 ) 
        opt_params_tot[k, :] = params
        opt_cov[k, :, :] = cov
        chisq[k] = chi2
        
    chimin = min(chisq)
    for k in range(len(chisq)) :
        if chisq[k] == chimin : 
            Beff_opt = abs(opt_params_tot[k, 1])
            Beff_sig = np.std(opt_params_tot[:, 1]) + np.mean( np.sqrt(opt_cov[:, 1, 1]) )
    
    print("Optimal Value of Beff")
    print("----------------------------------------")
    print(f"Beff  = {Beff_opt:.3} ± {Beff_sig:.3} m")
    print("")
    
    # Loop over all frequencies, use opt_params from previous as p0 
    our_freqs = np.arange(freq_pos[0], freq_pos[1])
    for k, f in enumerate(our_freqs) :
        # Model For Fit (lambda dependet)
        l = c/freqs[f]
        p = np.array([ sinc(Beff_opt/l, np.deg2rad(0.5)), 0.0 ])
        
        # Fit
        N_mc = 100
        mean = np.concatenate( (np.array([Beff_opt]), data_filtered[k, a-d : a+d]) )
        cov = np.diag( np.concatenate( (np.array([Beff_sig**2]), data_filtered_error[k, a-d : a+d]**2) ) )
        paras = rng.multivariate_normal(mean, cov, N_mc)
        V0_mc = np.zeros(N_mc, dtype=float)
    
        for i in range(N_mc) :
            Beff_mc = paras[i, 0]
            data_range_filtered = paras[i, 1:]
            model_fit = lambda t, V0, phi : 1 + V0 * np.cos(2 * np.pi * Beff_mc * omega * t / l + phi)
            params, cov = curve_fit(model_fit, t_range, data_range_filtered, p, maxfev=5000, sigma=data_filtered_error[k, a-d : a+d])
            V0_mc[i] = params[0]
        
        
        # Update
        model_fit = lambda t, V0, phi : 1 + V0 * np.cos(2 * np.pi * Beff_opt * omega * t / l + phi)
        params, cov = curve_fit(model_fit, t_range, data_filtered[k, a-d : a+d], p, maxfev=5000, sigma=data_filtered_error[k, a-d : a+d])
        V0[k] = np.median(V0_mc)
        V0_uncertainties[k] = np.std(V0_mc)
        
        # Optimal Paramters
        if f == fmin :
            params_opt = np.array([params[0], Beff_opt, params[1]])
            
        # Plot Fitted Data
        #chi2 = 1.0 * np.sum( (model_fit(t_range, *params) - data_filtered[k, a-d : a+d])**2 ) 
        #print(f"chisq in fit of {k}th f: r = {chi2:.3}")
        if f == fmin:
            plt.figure()
            plt.errorbar(t_range, data_filtered[k, a-d : a+d], yerr=data_filtered_error[k, a-d : a+d], fmt=".", label=f"Filtered Data of {k}th f", color="red")
            plt.plot(t_range, model_fit(t_range, *params), label="Fit of Wavelength Dependence", color="blue")
            plt.xlabel("Time")
            plt.ylabel("Spectral Density")
            plt.legend()
            plt.show()
        
    # Return Dependence
    return V0, V0_uncertainties, Beff_opt/lamda, Beff_sig/lamda, params_opt
##################

### Nice Plot ###
def nice_plot(data, t, freq, params_opt1, params_opt2, fmin) :
    
    def model(t, params_opt1, params_opt2) :
        S0, tmax, sigma, a, b = params_opt1
        V0, Beff, phi = params_opt2
        l = c/freq[fmin]
        return S0 * (1 + V0 * np.cos(2 * np.pi * Beff * omega * t / l + phi)) * normal(t, tmax, sigma) + N(t, a, b)
    
    plt.figure()
    plt.plot(t, (data[fmin, :]), label=f"Raw Data", color="red")
    plt.plot(t, (model(t, params_opt1, params_opt2)), label="Total Fit", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Spectral Density")
    plt.legend()
    plt.show()
    
    
#################

### Final Fit ###
def sun_diameter(V0, V0_uncert, Bl, Bl_sig, p) :
    
    N_mc = 1000
    mean = np.concatenate( (V0, Bl) )
    cov = np.diag( np.concatenate( (V0_uncert**2, Bl_sig**2) ) )
    paras = rng.multivariate_normal(mean, cov, N_mc)
    alpha_mc = np.zeros(N_mc, dtype=float)
    
    for i in range(N_mc) :
        V0_mc = paras[i, :len(V0)]
        Bl_mc = paras[i, len(V0):]
        p, _ = curve_fit(sinc, Bl_mc, abs(V0_mc), p0=p, sigma=V0_uncert)
        alpha_mc[i] = p
    
    m = np.mean(alpha_mc)
    s = np.std(alpha_mc)
    
    print(f"alpha = {ufloat(m, s):.P} rad")
    return ufloat(m, s)
##################

### Final Glueing ###
def sun_diameter_final(V0s, V0_uncerts, Bls, Bl_sigs, time, alphas, p) :
    
    V0 = np.concatenate(V0s); V0_uncert = np.concatenate(V0_uncerts)
    Bl = np.concatenate(Bls); Bl_sig = np.concatenate(Bl_sigs)
    
    """
    # Monte Carlo with Uniform distribution --> Baseline
    N_mc = 10000; n = 3
    alpha_mc = np.zeros(N_mc, dtype=float)
    alpha_sig = np.zeros(N_mc, dtype=float)
    Bl_mc = np.array([ rng.uniform(low=(Bl[i] - n * Bl_sig[i]), high=(Bl[i] + n * Bl_sig[i]), size=N_mc) for i in range(len(Bl)) ])
    
    # Monte Carlo with normal distribution --> V0
    V0_mc = rng.multivariate_normal(V0, np.diag(V0_uncert**2), N_mc)
    
    # Calculation of Alpha
    alpha_mc = np.zeros(N_mc, dtype=float)
    for i in range(N_mc) :
        p, cov_fit = curve_fit(sinc, Bl_mc[:, i], abs(V0_mc[i, :]), p0=p, sigma=V0_uncert)
        alpha_sig[i] = np.sqrt(cov_fit[0, 0])
        alpha_mc[i] = p
    
    m = np.median(alpha_mc)
    #m, _ = curve_fit(sinc, Bl, abs(V0), p0=p)
    s = ( np.std(alpha_mc) + np.mean(alpha_sig) )
    
    plt.figure()
    plt.hist(alpha_mc, bins=100)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Frequency")
    plt.show()

    # Use Analytic Solution for alpha
    from scipy.optimize import fsolve
    N_mc = 500
    n = len(V0)
    alpha_cloud = []
    for i in range(n) :
        Bl_mc = rng.uniform(low=(Bl[i] - 3 * Bl_sig[i]), high=(Bl[i] + 3 * Bl_sig[i]), size=N_mc)
        V0_mc = rng.multivariate_normal(np.array([V0[i]]), np.diag([V0_uncert[i]**2]), N_mc)
        for j in range(N_mc) :
            bl_mc, v0_mc = Bl_mc[j], V0_mc[j]
            if v0_mc > 1 or v0_mc < 0 :
                continue
            sinc_fsolve = lambda alpha : v0_mc - sinc(bl_mc, alpha)
            jac_fsolve = lambda alpha : np.cos(np.pi * alpha * bl_mc) / alpha - np.sin(np.pi * alpha * bl_mc) / (np.pi * alpha**2 * bl_mc)
            alpha_cloud.append( fsolve(sinc_fsolve, 0.01, xtol=1e-14, fprime=jac_fsolve)[0] )
    
    alpha_mc = np.sort(np.array(alpha_cloud))
    #m = np.median(alpha_mc)
    m, _ = curve_fit(sinc, Bl, abs(V0), p0=p, sigma=V0_uncert)
    s = np.std(alpha_mc)/np.sqrt(n)
    
    plt.figure()
    plt.hist(alpha_mc, bins=100)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Frequency")
    plt.show()
    """
    
    alpha = ufloat(np.average( np.array([alphai.n for alphai in alphas]), weights=1/np.array([alphai.s for alphai in alphas])**2 ), 
                   np.std([alphai.n for alphai in alphas]) / np.sqrt( len(alphas) ) )
    m = alpha.n
    s = alpha.s
    
    plt.figure( figsize=(8, 6) )
    plt.errorbar(time, np.array([alphai.n for alphai in alphas]), np.array([alphai.s for alphai in alphas]), fmt=".")
    plt.xlabel("Data Set")
    plt.ylabel(r"$\alpha$")
    plt.show()
    
    plt.figure()
    for bl, bl_sig, v0, v0_uncert, t in zip(Bls, Bl_sigs, V0s, V0_uncerts, time) :
        lbl = "Data Set " + t
        plt.errorbar(bl, abs(v0), v0_uncert, label=lbl, xerr=bl_sig, fmt=".")
    b = np.linspace(0.1, 100, 1000)
    plt.plot(b, sinc(b, m), label="Fit", color="red")
    plt.plot(b, sinc(b, m + s), label="Fit ± Sigma", color="red", linestyle="dashed")
    plt.plot(b, sinc(b, m - s), color="red", linestyle="dashed")
    plt.xlabel(r"$B_{eff}/\lambda$")
    plt.ylabel(r"$|V_0(B_\lambda)|$")
    plt.legend()
    plt.show()

    plt.figure()
    for bl, bl_sig, v0, v0_uncert, t in zip(Bls, Bl_sigs, V0s, V0_uncerts, time) :
        lbl = "Data Set " + t
        plt.errorbar(bl, abs(v0), v0_uncert, label=lbl, xerr=bl_sig, fmt=".")
    plt.plot(Bl, sinc(Bl, m), label="Fit", color="red")
    plt.plot(Bl, sinc(Bl, m + s), label="Fit ± Sigma", color="red", linestyle="dashed")
    plt.plot(Bl, sinc(Bl, m - s), color="red", linestyle="dashed")
    plt.xlabel(r"$B_{eff}/\lambda$")
    plt.ylabel(r"$|V_0(B_\lambda)|$")
    plt.legend()
    plt.show()
    
    print("Alpha Commuted as Weigthed Sum of Fits of Indivdual Data Sets")
    print("-------------------------------------------------------------")
    print(f"d = {alpha * (149597870700 - 6370e3):.P} m")
    R0 = 696243e3;
    print(f"d_exakt = {2 * R0 * 1.2} m")
    d = alpha * (149597870700 - 6370e3)
    print(f"R/R0 = {d/(2*R0):.P}")
    print(f"alpha = {alpha:.P} rad")
    print(f"Number of Datapoints = {len(V0)}")
#####################