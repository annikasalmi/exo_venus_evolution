import numpy as np

from exo_venus_predict import VenusParameters, run_prediction
from postprocess import create_predictions
from plot_outputs_full import plot_outputs_full

params = VenusParameters()

# number of runs
params.num_runs = 1

# Planet parameters
params.RE = 0.9499
params.ME = 0.8149
params.planet_sep = 0.7
params.rc = 3.11e6 # (metallic) core radius (m) 
params.pm = 3500.0 # average (silicate) mantle density (kg/m^3)
params.total_Fe_mol_fraction = 0.06

# Stellar evolution parameters
# CANNOT CHANGE ATM
params.Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),params.num_runs)
params.tsat_sun_ar = (2.9*params.Omega_sun_ar**1.14)/1000
params.fsat_sun = 10**(-3.13)
params.beta_sun_ar = 1.0/(0.35*np.log10(params.Omega_sun_ar) - 0.98)
params.beta_sun_ar = 0.86*params.beta_sun_ar 
params.stellar_mass = 1.0

output_files, input_files = run_prediction(params)

inputs, MCinputs, plotting_outs, post_process_out = create_predictions(output_files, input_files)

plot_outputs_full(inputs, MCinputs, post_process_out, plotting_outs)
