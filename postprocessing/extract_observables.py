"""
Feature Extraction and Observables Postprocessing Module for Paper 1.

Processes time-series histories (Cd, Cl, Cp, velocity fields) to extract
all four feature suites mandated by the research supervisor:
    1. Load features: mean_Cd, rms_Cd, mean_Cl, rms_Cl, max_Cl
    2. Frequency features: St, dominant_freq, harmonic_ratio_2f, harmonic_ratio_3f, spectral_bandwidth
    3. Statistical features: skewness_Cl, kurtosis_Cl, crest_factor_Cl, spectral_entropy_Cl
    4. Wake & Sensor features: base_pressure_Cpb, recirculation_length_Lr, sensor_Cp_means, sensor_Cp_rms
"""

import numpy as np
from scipy.signal import find_peaks, welch


def extract_time_series_features(time, cd, cl, D=1.0, U_inf=1.0, start_fraction=0.3):
    """
    Extract scalar load, frequency, and statistical features from settled signals.
    
    Parameters:
        time           : 1D array of time stamps.
        cd, cl         : 1D arrays of drag and lift coefficients.
        D              : Cylinder diameter.
        U_inf          : Free stream reference velocity.
        start_fraction : Fraction of initial transient signal to discard (default 30%).
        
    Returns:
        dict: Extracted scalar features.
    """
    time = np.asarray(time)
    cd = np.asarray(cd)
    cl = np.asarray(cl)
    
    n_total = len(time)
    idx_start = int(n_total * start_fraction)
    t_win = time[idx_start:]
    cd_win = cd[idx_start:]
    cl_win = cl[idx_start:]
    
    dt_eff = np.median(np.diff(t_win))
    fs = 1.0 / dt_eff
    
    # 1. Load Features
    mean_cd = float(np.mean(cd_win))
    cd_fluc = cd_win - mean_cd
    rms_cd = float(np.sqrt(np.mean(cd_fluc**2)))
    
    mean_cl = float(np.mean(cl_win))
    cl_fluc = cl_win - mean_cl
    rms_cl = float(np.sqrt(np.mean(cl_fluc**2)))
    max_cl = float(np.max(cl_win))
    
    # 2. Frequency & Spectral Features (via Welch PSD)
    nperseg = min(len(cl_win), 2048)
    freqs, psd = welch(cl_fluc, fs=fs, nperseg=nperseg, scaling='spectrum')
    
    # Find dominant peak in lift spectrum (shedding frequency)
    # Ignore DC component (freq == 0)
    pos_mask = freqs > (0.05 * U_inf / D)
    if np.any(pos_mask):
        f_pos = freqs[pos_mask]
        psd_pos = psd[pos_mask]
        peak_idx = np.argmax(psd_pos)
        f_shedding = float(f_pos[peak_idx])
        strouhal = float(f_shedding * D / U_inf)
        peak_power = float(psd_pos[peak_idx])
        
        # Harmonics at 2*f_s and 3*f_s
        f_2f = 2.0 * f_shedding
        f_3f = 3.0 * f_shedding
        
        # Find closest frequency bins
        idx_2f = np.argmin(np.abs(freqs - f_2f))
        idx_3f = np.argmin(np.abs(freqs - f_3f))
        
        power_2f = float(psd[idx_2f])
        power_3f = float(psd[idx_3f])
        
        # Amplitude ratio is sqrt of power ratio
        ratio_2f = float(np.sqrt(power_2f / np.maximum(peak_power, 1e-15)))
        ratio_3f = float(np.sqrt(power_3f / np.maximum(peak_power, 1e-15)))
        
        # Half-power spectral bandwidth (FWHM)
        half_power = peak_power * 0.5
        above_half = np.where(psd_pos >= half_power)[0]
        if len(above_half) > 1:
            bandwidth = float((f_pos[above_half[-1]] - f_pos[above_half[0]]) / np.maximum(f_shedding, 1e-15))
        else:
            bandwidth = float((freqs[1] - freqs[0]) / np.maximum(f_shedding, 1e-15))
    else:
        f_shedding = 0.0
        strouhal = 0.0
        ratio_2f = 0.0
        ratio_3f = 0.0
        bandwidth = 0.0
        
    # 3. Statistical Features
    var_cl = np.var(cl_win)
    std_cl = np.sqrt(var_cl) if var_cl > 1e-15 else 1.0
    
    skewness = float(np.mean(((cl_win - mean_cl) / std_cl)**3))
    kurtosis = float(np.mean(((cl_win - mean_cl) / std_cl)**4))
    crest_factor = float(np.max(np.abs(cl_fluc)) / np.maximum(rms_cl, 1e-15))
    
    # Spectral Entropy of lift
    psd_norm = psd / np.maximum(np.sum(psd), 1e-15)
    psd_norm = psd_norm[psd_norm > 1e-15]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))
    
    return {
        "mean_Cd": mean_cd,
        "rms_Cd": rms_cd,
        "mean_Cl": mean_cl,
        "rms_Cl": rms_cl,
        "max_Cl": max_cl,
        "f_shedding": f_shedding,
        "St": strouhal,
        "harmonic_ratio_2f": ratio_2f,
        "harmonic_ratio_3f": ratio_3f,
        "spectral_bandwidth": bandwidth,
        "skewness_Cl": skewness,
        "kurtosis_Cl": kurtosis,
        "crest_factor_Cl": crest_factor,
        "spectral_entropy_Cl": spectral_entropy
    }


def extract_wake_recirculation_length(x_coords, y_coords, u_mean, xc, yc, D=1.0):
    """
    Extract recirculation bubble length Lr / D from time-averaged centerline u-velocity.
    
    Finds the downstream zero-crossing x_zero where u_mean(x, yc) = 0.
    Lr = (x_zero - (xc + D/2)) / D.
    """
    # Locate centerline index in y
    j_center = np.argmin(np.abs(y_coords - yc))
    u_centerline = u_mean[:, j_center]
    
    # Restrict to downstream region: x > xc + D/2
    x_base = xc + 0.5 * D
    mask_downstream = x_coords > x_base
    x_down = x_coords[mask_downstream]
    u_down = u_centerline[mask_downstream]
    
    # Check for sign change from negative to positive
    recirc_length = 0.0
    for i in range(len(u_down) - 1):
        if u_down[i] <= 0.0 and u_down[i+1] > 0.0:
            # Linear interpolation for zero crossing
            x0 = x_down[i] - u_down[i] * (x_down[i+1] - x_down[i]) / (u_down[i+1] - u_down[i])
            recirc_length = float((x0 - x_base) / D)
            break
            
    return recirc_length


def extract_surface_pressure_sensors(theta_angles_deg, cp_time_series):
    """
    Extract statistics for candidate surface pressure taps (24 angles: 0, 15, ..., 345 deg).
    
    Parameters:
        theta_angles_deg : 1D array of sensor angles in degrees.
        cp_time_series   : 2D array of shape (n_time_steps, n_sensors).
        
    Returns:
        dict: mean_Cp and rms_Cp for each sensor.
    """
    cp_time_series = np.asarray(cp_time_series)
    means = np.mean(cp_time_series, axis=0)
    stds = np.std(cp_time_series, axis=0)
    
    res = {}
    for i, ang in enumerate(theta_angles_deg):
        key = f"sensor_{int(round(ang)):03d}"
        res[f"{key}_mean"] = float(means[i])
        res[f"{key}_rms"] = float(stds[i])
    return res
