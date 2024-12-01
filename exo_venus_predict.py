'''
Run MC calculations on the whole thing.
'''
import multiprocessing as mp
import numpy as np
from typing import Tuple
import os
from joblib import Parallel, delayed
import shutil

RUNTIME_WARNING=False #suppress runtime warnings if set to false 
if RUNTIME_WARNING is False:
    import warnings
    warnings.filterwarnings("ignore")   

from venus_evolution.classes import * 
from venus_evolution.main import forward_model
from venus_evolution.user_tools.tools import VENUS_ROOT

class VenusParameters():
    '''
    Hold Venus Parameters in one location.
    '''
    def __init__(self, num_runs: float = 10):
        self.num_runs = num_runs
        self.num_cores = mp.cpu_count()

        if os.path.exists(os.path.join(VENUS_ROOT,'switch_garbage3')):
            shutil.rmtree(os.path.join(VENUS_ROOT,'switch_garbage3'))
        os.mkdir(os.path.join(VENUS_ROOT,'switch_garbage3'))
        self.output_path = os.path.join(VENUS_ROOT,'switch_garbage3')

        self.VenusInputs = SwitchInputs(print_switch = "n", speedup_flag = "n", start_speed=15e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=30e6)   
        self.VenusNumerics = Numerics(total_steps = 2 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=-999, step4=-999, tfin0=self.VenusInputs.Start_time+10000, tfin1=self.VenusInputs.Start_time+30e6, tfin2=4.5e9, tfin3=-999, tfin4 = -999)

        ## PARAMETER RANGES ##
        #initial volatile inventories
        self.init_water = 10**np.random.uniform(19.5,21.5,self.num_runs)  
        self.init_CO2 = 10**np.random.uniform(20.4,21.5,self.num_runs)
        self.init_O = np.random.uniform(2e21,6e21,self.num_runs)

        #Weathering and ocean chemistry parameters
        self.Tefold = np.random.uniform(5,30,self.num_runs)
        self.alphaexp = np.random.uniform(0.1,0.5,self.num_runs)
        self.suplim_ar = 10**np.random.uniform(5,7,self.num_runs)
        self.ocean_Ca_ar = 10**np.random.uniform(-4,np.log10(0.3),self.num_runs)
        self.ocean_Omega_ar = np.random.uniform(1.0,10.0,self.num_runs) 

        #escape parameters
        self.mult_ar = 10**np.random.uniform(-2,2,self.num_runs)
        self.mix_epsilon_ar = np.random.uniform(0.0,1.0,self.num_runs)
        self.Epsilon_ar = np.random.uniform(0.01,0.3,self.num_runs)
        self.Tstrat_array = np.random.uniform(180.5,219.5,self.num_runs)

        #Albedo parameters
        self.Albedo_C_range = np.random.uniform(0.2,0.7,self.num_runs)
        self.Albedo_H_range = np.random.uniform(0.0001,0.3,self.num_runs)
        for k in range(0,len(self.Albedo_C_range)):
            if self.Albedo_C_range[k] < self.Albedo_H_range[k]:
                self.Albedo_H_range[k] = self.Albedo_C_range[k]-1e-5   

        #Stellar evolution parameters
        self.Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),self.num_runs)
        self.tsat_sun_ar = (2.9*self.Omega_sun_ar**1.14)/1000
        self.fsat_sun = 10**(-3.13)
        self.beta_sun_ar = 1.0/(0.35*np.log10(self.Omega_sun_ar) - 0.98)
        self.beta_sun_ar = 0.86*self.beta_sun_ar 
        self.stellar_mass = 1.0

        # Interior parameters
        self.offset_range = 10**np.random.uniform(1.0,3.0,self.num_runs)
        self.heatscale_ar = np.random.uniform(0.5,2.0,self.num_runs)
        self.K_over_U_ar = np.random.uniform(6000.0,8440,self.num_runs) 
        self.Stag_trans_ar = np.random.uniform(50e6,4e9,self.num_runs)

        #Oxidation parameters
        self.MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),self.num_runs) 
        self.dry_ox_frac_ac = 10**np.random.uniform(-4,-1,self.num_runs)
        self.wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,self.num_runs)
        self.Mantle_H2O_max_ar = 10**np.random.uniform(np.log10(0.5),np.log10(15.0),self.num_runs) 
        self.surface_magma_frac_array = 10**np.random.uniform(-4,0,self.num_runs)  

        #impact parameters (not used)
        self.imp_coef = 10**np.random.uniform(11,14.5,num_runs)
        self.tdc = np.random.uniform(0.06,0.14,num_runs)

        self.RE = 0.9499
        self.ME = 0.8149
        self.planet_sep = 0.7
        self.rc = 3.11e6
        self.pm = 3500.0
        self.total_Fe_mol_fraction = 0.06
        self.Albedo_H_range = np.random.uniform(0.0001,0.3,self.num_runs)

    def create_input_files(self):

        ##Output arrays and parameter inputs to be filled:
        inputs = range(0,self.num_runs)

        for ii in inputs:

            Venus_PlanetInputs = PlanetInputs(RE = self.RE, ME = self.ME, rc=self.rc, pm=self.pm, 
                                            Total_Fe_mol_fraction = self.total_Fe_mol_fraction, 
                                            Planet_sep=self.planet_sep,
                                            albedoC=self.Albedo_C_range[ii], albedoH=self.Albedo_H_range[ii])   

            Venus_InitConditions = InitConditions(Init_solid_H2O=0.0, Init_fluid_H2O=self.init_water[ii] , 
                                                  Init_solid_O=0.0,Init_fluid_O=self.init_O[ii],Init_solid_FeO1_5 = 0.0,
                                                  Init_solid_FeO=0.0, Init_solid_CO2=0.0, 
                                                  Init_fluid_CO2 = self.init_CO2[ii])   

            Sun_StellarInputs = StellarInputs(tsat_XUV=self.tsat_sun_ar[ii], Stellar_Mass=self.stellar_mass, 
                                              fsat=self.fsat_sun, beta0=self.beta_sun_ar[ii], 
                                              epsilon=self.Epsilon_ar[ii] )

            MCInputs_ar = MCInputs(esc_a=self.imp_coef[ii], esc_b=self.tdc[ii],  esc_c = self.mult_ar[ii], 
                                   esc_d = self.mix_epsilon_ar[ii],ccycle_a=self.Tefold[ii], ccycle_b=self.alphaexp[ii],
                                   supp_lim = self.suplim_ar[ii], interiora =self.offset_range[ii], 
                                   interiorb=self.MFrac_hydrated_ar[ii],interiorc=self.dry_ox_frac_ac[ii],
                                   interiord = self.wet_oxid_eff_ar[ii],interiore = self.heatscale_ar[ii], 
                                   interiorf = self.Mantle_H2O_max_ar[ii], interiorg = self.Stag_trans_ar[ii], 
                                   ocean_a=self.ocean_Ca_ar[ii],ocean_b=self.ocean_Omega_ar[ii],
                                   K_over_U = self.K_over_U_ar[ii],Tstrat=self.Tstrat_array[ii],
                                   surface_magma_frac=self.surface_magma_frac_array[ii])

            inputs_for_later = [self.VenusInputs,Venus_PlanetInputs,Venus_InitConditions,self.VenusNumerics,
                                Sun_StellarInputs,MCInputs_ar]
        
            sve_name = 'switch_garbage3/inputs4L%d' %ii
            np.save(sve_name,inputs_for_later)

        return inputs

    def run_input_files(self, i):
        load_name = 'switch_garbage3/inputs4L%d.npy' %i
        max_time_attempt = 1.5
        [Venus_inputs,Venus_PlanetInputs,Venus_InitConditions,Venus_Numerics,Sun_StellarInputs,MCInputs_ar] = np.load(load_name,allow_pickle=True)

        outs = forward_model(Venus_inputs,Venus_PlanetInputs,Venus_InitConditions,Venus_Numerics,Sun_StellarInputs,MCInputs_ar,max_time_attempt, runtime_warning=RUNTIME_WARNING) 
        print('success')

        return outs

def run_prediction(parameters: VenusParameters, 
                   output_files: str = 'Venus_ouputs_revisions', 
                   input_files: str = 'Venus_inputs_revisions') -> Tuple[list, list]:
    '''
    Run the prediction file
    '''

    inputs = VenusParameters().create_input_files()

    for i in inputs:
        parameters.run_input_files(i)

    # out = Parallel(n_jobs=parameters.num_cores)(delayed(parameters.run_input_files)(i) for i in inputs)

    input_mega=[] # Collect input parameters for saving
    everything_to_drop = []
    for kj in range(0,len(inputs)):
        # print ('saving garbage',kj)
        if type(out[kj]) == list:
            everything_to_drop.append(kj)
        else:
            load_name = 'switch_garbage3/inputs4L%d.npy' %kj
            input_mega.append(np.load(load_name,allow_pickle=True))

    for drop in everything_to_drop:
        del out[drop]

    np.save(output_files,out) 
    np.save(input_files,input_mega) 

    shutil.rmtree('switch_garbage3')

    return output_files, input_files