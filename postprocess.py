import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize

RUNTIME_WARNING=False #suppress runtime warnings if set to false 
if RUNTIME_WARNING is False:
    import warnings
    warnings.filterwarnings("ignore")  

from venus_evolution.classes import *
from venus_evolution.models.other_functions import *
from postprocess_stats import post_process
from postprocess_class import PlottingOutputs

def use_one_output(inputs: list, MCinputs: list) -> PlottingOutputs:
    '''
    Create an easy way to interpret the outputs file.
    '''
    k= 0
    out = PlottingOutputs()

    while k<len(inputs):
        if (np.size(inputs[k])==1)and (inputs[k].total_time[-1] > 4.4e9):        
            top = (inputs[k].total_y[13][0] + inputs[k].total_y[12][0]) 
            bottom =  (inputs[k].total_y[1][0] + inputs[k].total_y[0][0] )
            CO2_H2O_ratio = top/bottom

            out.rp = 6371000*MCinputs[k][1].RE
            out.mp = 5.972e24*MCinputs[k][1].ME
            out.g = 6.67e-11*out.mp/(out.rp**2)
            Pbase = (inputs[k].Pressre_H2O[-1] + inputs[k].CO2_Pressure_array[-1] + inputs[k].total_y[22][-1])
            Mvolatiles = (Pbase*4*np.pi*out.rp**2)/out.g 

            out.new_inputs.append(inputs[k])
            out.inputs_for_MC.append(MCinputs[k])
            out.Total_Fe_array.append(MCinputs[k][1].Total_Fe_mol_fraction)
            out.rp = 6371000*MCinputs[k][1].RE
            out.mp = 5.972e24*MCinputs[k][1].ME
            out.g = 6.67e-11*out.mp/(out.rp**2)

        k= k+1 
    return out

def interpolate_class(saved_outputs):
    outs=[]
    for i, _ in enumerate(saved_outputs):

        time_starts = np.min([np.where(saved_outputs[i].total_time>1)])-1
        time = saved_outputs[i].total_time[time_starts:]
        num_t_elements = 1000 
        new_time = np.logspace(np.log10(np.min(time[:])),np.max([np.log10(time[:-1])]),num_t_elements)
        num_y = 56
        new_total_y = np.zeros(shape=(num_y,len(new_time)))
        for k in range(0,num_y):
            f1 = interp1d(time,saved_outputs[i].total_y[k][time_starts:])
            new_total_y[k] = f1(new_time)

        f1 = interp1d(time,saved_outputs[i].FH2O_array[time_starts:])
        new_FH2O_array = f1(new_time)

        f1 = interp1d(time,saved_outputs[i].FCO2_array[time_starts:])
        new_FCO2_array = f1(new_time)
        
        f1 = interp1d(time,saved_outputs[i].MH2O_liq[time_starts:])
        new_MH2O_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].MH2O_crystal[time_starts:])
        new_MH2O_crystal = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].MCO2_liq[time_starts:])
        new_MCO2_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].Pressre_H2O[time_starts:])
        new_Pressre_H2O = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].CO2_Pressure_array[time_starts:])
        new_CO2_Pressure_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].fO2_array[time_starts:])
        new_fO2_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_dissolved[time_starts:])
        new_Mass_O_dissolved = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].water_frac[time_starts:])
        new_water_frac = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_depth[time_starts:])
        new_ocean_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Max_depth[time_starts:])
        new_max_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_fraction[time_starts:])
        new_ocean_fraction = f1(new_time) 

        output_class = ModelOutputs(new_time,new_total_y,new_FH2O_array,new_FCO2_array,new_MH2O_liq,new_MH2O_crystal,
                                    new_MCO2_liq,new_Pressre_H2O,new_CO2_Pressure_array,new_fO2_array,new_Mass_O_atm,
                                    new_Mass_O_dissolved,new_water_frac,new_ocean_depth,new_max_depth,new_ocean_fraction)
        outs.append(output_class)
    return outs
    
def create_predictions(output_file: str = 'Venus_ouputs_revisions_gliese_1.npy', 
                       input_file: str = 'Venus_inputs_revisions_gliese_1.npy'):
    '''
    Create predicted results
    '''

    ### Load outputs and inputs. Note it is possible to load multiple output files and process them all at once
    inputs = np.load(output_file,allow_pickle = True)
    MCinputs = np.load(input_file,allow_pickle = True)
    plotting_outs = use_one_output(inputs,MCinputs)

    #Pause here to check number of successful model runs etc. Type 'c' to continue.
    if len(inputs) == 0:
        raise ValueError('No new inputs found for some reason during modelling.')
    inputs = np.array(plotting_outs.new_inputs)

    post_process_out = post_process(inputs, MCinputs, plotting_outs.g, plotting_outs.rp)

    interp_outputs = interpolate_class(inputs)
    inputs = interp_outputs 

    return inputs, MCinputs, plotting_outs, post_process_out