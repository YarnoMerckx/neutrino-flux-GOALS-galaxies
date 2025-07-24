import numpy as np
from scipy.interpolate import UnivariateSpline
import os

def read_out_diffuse_HESE_data(filename="diffuse_HESE_7.5_year_differential.txt"):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, filename)
    f = open(file_path,"r")
    lines = f.readlines() 
    
    energy, energy_lower_err, energy_upper_err = np.array([]), np.array([]), np.array([]) # [GeV]
    flux,   flux_lower_err,   flux_upper_err   = np.array([]), np.array([]), np.array([]) # [GeV cm^-2 s^-1 sr^-1]
    
    for line in lines:
        if line[0] == "#":
            continue
        columns = [column.replace(' ','').replace('\n','') for column \
                   in line.split("\t") if column.replace(' ','').replace('\n','') != ""]
        
        energy = np.append(energy,float(columns[0]))
        energy_lower_err = np.append(energy_lower_err,float(columns[0]) - float(columns[1]))
        energy_upper_err = np.append(energy_upper_err,float(columns[2]) - float(columns[0]))
                      
        flux = np.append(flux,float(columns[3]))
        flux_lower_err = np.append(flux_lower_err,float(columns[3]) - float(columns[4]))
        flux_upper_err = np.append(flux_upper_err,float(columns[5]) - float(columns[3]))
        
    # [GeV], [GeV], [GeV], [GeV cm^-2 s^-1 sr^-1], [GeV cm^-2 s^-1 sr^-1], [GeV cm^-2 s^-1 sr^-1]                  
    return energy, energy_lower_err, energy_upper_err, flux, flux_lower_err, flux_upper_err

# Get the HESE differential data
# [GeV], [GeV cm^-2 s^-1 sr^-1]
energy, energy_lower_err, energy_upper_err,\
flux,   flux_lower_err,   flux_upper_err = read_out_diffuse_HESE_data(filename="diffuse_HESE_7.5_year_differential.txt")

# Check for upper limits
upper_limits   = np.array([error == 0 for error in flux_lower_err])

# Set a length for the arrow of the upper limit
for i in range(len(upper_limits)):
    if upper_limits[i]:
        flux_lower_err[i] = 0.25*flux[i]
    

def read_out_diffuse_data(filename,
                          E_min, # [GeV]
                          E_max, # [GeV]
                          steps=1e3):
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, filename)
    f = open(file_path,"r")
    lines = f.readlines()
    
    energy, flux = np.array([]), np.array([])

    for line in lines:
        if line[0] == "#":
            continue
        columns = line.split("\t")
        energy = np.append(energy,float(columns[0]))
        flux = np.append(flux,float(columns[1].replace("\n","")))
    
    spline = UnivariateSpline(np.log10(energy),np.log10(flux),k=2)
    
    energy_range = np.logspace(np.log10(E_min),np.log10(E_max),int(1e3))
    flux = 10**spline(np.log10(energy_range))
    
    # [GeV], [GeV cm^-2 s^-1 sr^-1]
    return energy_range, flux

#Get the diffuse nu_mu data from ApJ 928 50 (2022)
norm  = {'best_fit':1.44e-18,
         'plus_1sigma':1.44e-18+0.25e-18,
         'minus_1sigma':1.44e-18-0.26e-18} # [GeV^-1 cm^-2 s^-1 sr^-1]
gamma = {'best_fit':2.37,
         'plus_1sigma':2.37+0.09,
         'minus_1sigma':2.37-0.09}
E_0   = 1e5 # [GeV]

energy_range_numu = np.logspace(np.log10(1.5e4),np.log10(5e6),100) # [TeV]




def get_flux(energy,
             gamma=2.0,
             E_0=1000., # [GeV]
             norm=1.):
    '''
    Description
    -----------
    Calculates the neutrino flux at a certain energy
    according to a power law by specifying a spectral
    index, the normalization energy and the
    normalization flux.
    
    
    Arguments
    ---------
    `energy`
    type        : float
    description : Energy for which to calculate the flux.
                  Units: GeV.
                  
    `gamma`
    type        : float
    description : Option to specify the spectral index of the power law.
                  
    `E_0`
    type        : float
    description : Option to specify the normalization energy.
                  Units: GeV.
                         
    `norm`
    type        : float
    description : Option to specify the normalization flux.
                  Units of choice.
    
    Returns
    -------
    `flux`
    type        : float
    description : The flux at the specified energy.
                  Units of choice.
    '''

    flux = norm*(energy/E_0)**(-gamma)
    
    return flux



energy_numu, flux_lower = read_out_diffuse_data("diffuse_numu_9.5_year_lower_band.txt",
                                                 energy_range_numu.min(),
                                                 energy_range_numu.max(),
                                                 steps=1e3)

energy_numu, flux_upper = read_out_diffuse_data("diffuse_numu_9.5_year_upper_band.txt",
                                                 energy_range_numu.min(),
                                                 energy_range_numu.max(),
                                                 steps=1e3)
flux_numu = get_flux(energy=energy_range_numu,
                               E_0=E_0,
                               norm=norm['best_fit'],
                               gamma=gamma['best_fit']) # [TeV^-1 cm^-2 s^-1 sr^-1]