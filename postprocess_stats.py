
import scipy.stats

from venus_evolution.models.outgassing_module import *
from venus_evolution.classes import *
from venus_evolution.models.other_functions import *
from postprocess_class import PostProcessOutput, PlottingOutputs

def buffer_fO2(T,Press,redox_buffer): 
    '''
    T in K, P in bar

    Oxygen fugacity and mantle redox functions (for post-processing)
    '''
    if redox_buffer == 'FMQ':
        [A,B,C] = [25738.0, 9.0, 0.092]
    elif redox_buffer == 'IW':
        [A,B,C] = [27215 ,6.57 ,0.0552]
    elif redox_buffer == 'MH':
        [A,B,C] = [25700.6,14.558,0.019] 
    else:
        print ('error, no such redox buffer')
        return -999
    return 10**(-A/T + B + C*(Press-1)/T)

def get_fO2(XFe2O3_over_XFeO,P,T,Total_Fe): 
    '''
    Total_Fe is a mole fraction of iron minerals 
    XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
    '''
    XAl2O3 = 0.022423 
    XCaO = 0.0335 
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    terms1 =  11492.0/T - 6.675 - 2.243*XAl2O3
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    fO2 =  np.exp( (np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe -(terms1+terms2+terms3+terms4) )/0.196)
    return fO2  

def post_process(inputs, MCinputs, plot_outs: PlottingOutputs):
    out = PostProcessOutput()

    rc = MCinputs[0][1].rc
    mantle_mass = 0.0

    rp = 6371000*MCinputs[0][1].RE
    alpha = 2e-5
    k = 4.2 

    N2_Pressure = 1e5 # Do not change
    x = 0.01550152865954013
    M_H2O = 18.01528
    M_CO2 = 44.01
    Aoc = 4*np.pi*rp**2

    for k, _ in enumerate(inputs):
        out.total_time.append( inputs[k].total_time )
        out.total_y.append(inputs[k].total_y)
        out.FH2O_array.append(inputs[k].FH2O_array )
        out.FCO2_array.append(inputs[k].FCO2_array )
        out.MH2O_liq.append(inputs[k].MH2O_liq )
        out.MCO2_liq.append(inputs[k].MCO2_liq )
        out.Pressre_H2O.append(inputs[k].Pressre_H2O) 
        out.CO2_Pressure_array.append(inputs[k].CO2_Pressure_array )
        out.fO2_array.append(inputs[k].fO2_array )
        out.Mass_O_atm.append(inputs[k].Mass_O_atm )
        out.Mass_O_dissolved.append(inputs[k].Mass_O_dissolved )
        out.water_frac.append(inputs[k].water_frac )
        out.Ocean_depth.append(inputs[k].Ocean_depth )
        out.Max_depth.append(inputs[k].Max_depth )
        out.Ocean_fraction.append(inputs[k].Ocean_fraction )

    for i in np.arange(0,10):
        print(len(out.FH2O_array[i]))

    [int1,int2,int3] = [2.5,50,97.5]
    out.confidence_y=scipy.stats.scoreatpercentile(out.total_y, [int1,int2,int3], 
                                                   interpolation_method='fraction',axis=0)
    out.confidence_FH2O = scipy.stats.scoreatpercentile(out.FH2O_array, [int1,int2,int3], 
                                                        interpolation_method='fraction',axis=0)
    out.confidence_FCO2 = scipy.stats.scoreatpercentile(out.FCO2_array ,[int1,int2,int3], 
                                                        interpolation_method='fraction',axis=0)
    out.confidence_MH2O_liq = scipy.stats.scoreatpercentile(out.MH2O_liq ,[int1,int2,int3], 
                                                            interpolation_method='fraction',axis=0)
    out.confidence_MCO2_liq = scipy.stats.scoreatpercentile(out.MCO2_liq ,[int1,int2,int3], 
                                                            interpolation_method='fraction',axis=0)
    out.confidence_Pressre_H2O = scipy.stats.scoreatpercentile(out.Pressre_H2O ,[int1,int2,int3], 
                                                               interpolation_method='fraction',axis=0)
    out.confidence_CO2_Pressure_array = scipy.stats.scoreatpercentile(out.CO2_Pressure_array, 
                                                                      [int1,int2,int3], 
                                                                      interpolation_method='fraction',axis=0)
    out.confidence_fO2_array = scipy.stats.scoreatpercentile(out.fO2_array ,[int1,int2,int3], 
                                                             interpolation_method='fraction',axis=0)
    out.confidence_Mass_O_atm = scipy.stats.scoreatpercentile(out.Mass_O_atm ,[int1,int2,int3], 
                                                              interpolation_method='fraction',axis=0)
    out.confidence_Mass_O_dissolved = scipy.stats.scoreatpercentile(out.Mass_O_dissolved ,[int1,int2,int3], 
                                                                    interpolation_method='fraction',axis=0)
    out.confidence_water_frac = scipy.stats.scoreatpercentile(out.water_frac ,[int1,int2,int3], 
                                                              interpolation_method='fraction',axis=0)
    out.confidence_Ocean_depth = scipy.stats.scoreatpercentile(out.Ocean_depth ,[5,50,95], 
                                                               interpolation_method='fraction',axis=0)
    out.confidence_Max_depth = scipy.stats.scoreatpercentile(out.Max_depth ,[int1,int2,int3], 
                                                             interpolation_method='fraction',axis=0)
    out.confidence_Ocean_fraction = scipy.stats.scoreatpercentile(out.Ocean_fraction ,[int1,int2,int3], 
                                                                  interpolation_method='fraction',axis=0)

    out.Melt_volume = np.copy(out.total_time)
    out.Plate_velocity = np.copy(out.total_time)
    out.total_Ar40  = np.copy(out.total_time) 
    out.total_K40  = np.copy(out.total_time)

    out.DH_atmo =  np.copy(out.total_time)
    out.DH_solid =  np.copy(out.total_time)

    for k, _ in enumerate(inputs):
        out.f_O2_FMQ.append(inputs[k].Ocean_fraction*0 )
        out.f_O2_IW.append(inputs[k].Ocean_fraction*0 )
        out.f_O2_MH.append(inputs[k].Ocean_fraction*0 )
        out.f_O2_mantle.append(inputs[k].Ocean_fraction*0 )
        out.iron_ratio.append(inputs[k].Ocean_fraction*0 )
        out.total_iron.append(inputs[k].Ocean_fraction*0 )
        out.iron2_array.append(inputs[k].Ocean_fraction*0 )
        out.iron3_array.append(inputs[k].Ocean_fraction*0 )
        out.actual_phi_surf_melt_ar.append(inputs[k].Ocean_fraction*0 )
        out.XH2O_melt.append(inputs[k].Ocean_fraction*0 )
        out.XCO2_melt.append(inputs[k].Ocean_fraction*0 )
        out.F_H2O_new.append(inputs[k].Ocean_fraction*0 )
        out.F_CO2_new.append(inputs[k].Ocean_fraction*0 )
        out.F_CO_new.append(inputs[k].Ocean_fraction*0 )
        out.F_H2_new.append(inputs[k].Ocean_fraction*0 )
        out.F_CH4_new.append(inputs[k].Ocean_fraction*0 )
        out.F_SO2_new.append(inputs[k].Ocean_fraction*0 )
        out.F_H2S_new.append(inputs[k].Ocean_fraction*0 )
        out.F_S2_new.append(inputs[k].Ocean_fraction*0 )
        out.O2_consumption_new.append(inputs[k].Ocean_fraction*0 )

        out.mantle_CO2_fraction.append(inputs[k].Ocean_fraction*0 ) 
        out.mantle_H2O_fraction.append(inputs[k].Ocean_fraction*0 )

        ocean_depth_locs = np.where(inputs[k].Ocean_depth>0)
        if len(ocean_depth_locs) > 1:
            ocean_start_index = np.min(np.where(inputs[k].Ocean_depth>0))
            max_T_runaway = np.max(out.total_y[k][8][ocean_start_index:])
            out.Max_runaway_temperatures.append(max_T_runaway)
        else:
            out.Max_runaway_temperatures.append(np.nan)

        for i in range(0,len(out.total_time[k])):
            out.mantle_H2O_fraction[k][i] = out.total_y[k][0][i]/(out.total_y[k][0][i]+out.total_y[k][1][i])
            out.mantle_CO2_fraction[k][i] = out.total_y[k][13][i]/(out.total_y[k][13][i]+out.total_y[k][12][i])
            Pressure_surface =out.fO2_array[k][i] + out.Pressre_H2O[k][i]*out.water_frac[k][i] + \
                        out.CO2_Pressure_array[k][i] + N2_Pressure  
            
            out.f_O2_FMQ[k][i] = buffer_fO2(out.total_y[k][7][i],Pressure_surface/1e5,'FMQ')
            out.f_O2_IW[k][i] = buffer_fO2(out.total_y[k][7][i],Pressure_surface/1e5,'IW')
            out.f_O2_MH[k][i] =  buffer_fO2(out.total_y[k][7][i],Pressure_surface/1e5,'MH')
            iron3 = out.total_y[k][5][i]*56/(56.0+1.5*16.0)
            iron2 = out.total_y[k][6][i]*56/(56.0+16.0)
            out.iron2_array[k][i] = iron2
            out.iron3_array[k][i] = iron3
            out.total_iron[k][i] = iron3+iron2
            out.iron_ratio[k][i] = iron3/iron2
            out.f_O2_mantle[k][i] = get_fO2(0.5*iron3/iron2,Pressure_surface,
                                            out.total_y[k][7][i],plot_outs.Total_Fe_array[k])
            T_for_melting = float(out.total_y[k][7][i])
            Poverburd = out.fO2_array[k][i] + out.Pressre_H2O[k][i] + out.CO2_Pressure_array[k][i] + N2_Pressure  
            alpha = 2e-5
            cp = 1.2e3 
            pm = 4000.0
            rdck = optimize.minimize(find_r,x0=float(out.total_y[k][2][i]),args = (T_for_melting,alpha,
                                                                                   plot_outs.g,cp,pm,rp,
                                                                                   float(Poverburd),0,0.0))
            rad_check = float(rdck.x[0])
            if rad_check>rp:
                rad_check = rp
            rlid = rp -out.total_y[k][26][i]
            [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,
                                                                  plot_outs.g,Poverburd,0,rlid)
            out.actual_phi_surf_melt_ar[k][i]= actual_phi_surf_melt
            F = out.actual_phi_surf_melt_ar[k][i]
            mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))
            XH2O_melt_max = x*M_H2O*0.499 # half of mol fraction allowed to be H2O
            XCO2_melt_max = x*M_CO2*0.499 # half of mol fraction allowed to be CO2
            if F >0:
                out.XH2O_melt[k][i] = np.min([0.99*XH2O_melt_max,(1- (1-F)**(1/0.01)) * \
                                                (out.total_y[k][0][i]/mantle_mass)/F ]) \
                                                    # mass frac, ensures mass frac never implies all moles volatile!
                out.XCO2_melt[k][i] =  np.min([0.99*XCO2_melt_max,(1- (1-F)**(1/2e-3)) * \
                                               (out.total_y[k][13][i]/mantle_mass)/F ])# mass frac
            else:
                out.XH2O_melt[k][i] = 0.0 
                out.XCO2_melt[k][i] =  0.0

        out.Late_melt_production.append(np.mean(out.total_y[k][25][994:]))
        

        Q = out.total_y[k][11]
        for i in range(0,len(Q)):
            if out.total_y[k][16][i]/1000 < 1e-11:
               out.total_y[k][16][i] = 0.0
                    
        out.Melt_volume[k] =out.total_y[k][25]  
        out.Plate_velocity[k] = 365*24*60*60*out.Melt_volume[k]/(out.total_y[k][16]  * 3 * np.pi*rp)

        min_HabTime = 0
        hab_counter = 0
        max_HabTime = 0
        for i in range(0,len(Q)):
            if out.total_y[k][16][i]/1000 < 1e-11:
                out.Plate_velocity[k][i] = 0  
            out.total_Ar40[k][i] =out.total_y[k][38][i]/(out.total_y[k][38][i] +out.total_y[k][36][i])
            out.total_K40[k][i] =out.total_y[k][35][i] +out.total_y[k][37][i]

            out.DH_atmo[k][i] = 0.5*out.total_y[k][41][i]/out.total_y[k][1][i]
            out.DH_solid[k][i] = 0.5*out.total_y[k][42][i]/out.total_y[k][0][i]
            if (inputs[k].Ocean_depth[i]>0.0):
                if hab_counter == 0:
                    min_HabTime = out.total_time[k][i]
                    hab_counter = 1.0
                max_HabTime = out.total_time[k][i]

            #### outgassing aside 
            Pressure_surface = out.fO2_array[k][i] + out.Pressre_H2O[k][i]*out.water_frac[k][i] + \
                                            out.CO2_Pressure_array[k][i] + N2_Pressure  
            melt_mass =out.total_y[k][25][i]*pm*1000 
            Tsolidus = sol_liq(rp,plot_outs.g,pm,rp,0.0,0.0)
            if (0.5*out.iron_ratio[k][i]>0)and(melt_mass>0)and(out.actual_phi_surf_melt_ar[k][i])>0:
                try:
                    [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = \
                                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                except:
                    [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = \
                                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            else:
                [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
            out.F_H2O_new[k][i] = F_H2O*365*24*60*60/(1e12)
            out.F_CO2_new[k][i] = F_CO2*365*24*60*60/(1e12)
            out.F_CO_new[k][i] = F_CO*365*24*60*60/(1e12)
            out.F_H2_new[k][i] = F_H2*365*24*60*60/(1e12)
            out.F_CH4_new[k][i] = F_CH4*365*24*60*60/(1e12)
            out.F_SO2_new[k][i] = F_SO2*365*24*60*60/(1e12)
            out.F_H2S_new[k][i] = F_H2S*365*24*60*60/(1e12)
            out.F_S2_new[k][i] = F_S2*365*24*60*60/(1e12)
            out.O2_consumption_new[k][i] = O2_consumption*365*24*60*60/(1e12)
        
        out.HTmin.append(min_HabTime)
        out.HTmax.append(max_HabTime)
        out.HT_duration.append(max_HabTime - min_HabTime)
        
    out.HTmin= np.array(out.HTmin)
    out.HTmax = np.array(out.HTmax)
    out.HT_duration = np.array(out.HT_duration)
    out.Late_melt_production = np.array(out.Late_melt_production)

    out.conf_DH_atmo = scipy.stats.scoreatpercentile(out.DH_atmo ,[int1,int2,int3], 
                                                     interpolation_method='fraction',axis=0)
    out.conf_DH_solid = scipy.stats.scoreatpercentile(out.DH_solid ,[int1,int2,int3], 
                                                      interpolation_method='fraction',axis=0)

    out.conf_HT_duration = scipy.stats.scoreatpercentile(out.HT_duration ,[int1,int2,int3], 
                                                         interpolation_method='fraction',axis=0)
    out.conf_HTmax = scipy.stats.scoreatpercentile(out.HTmax ,[int1,int2,int3], 
                                                   interpolation_method='fraction',axis=0)
    out.conf_HTmin = scipy.stats.scoreatpercentile(out.HTmin ,[int1,int2,int3], 
                                                   interpolation_method='fraction',axis=0)

    out.conf_Late_melt_production = scipy.stats.scoreatpercentile(365*24*60*60*out.Late_melt_production/1e9,
                                                                  [int1,int2,int3], interpolation_method='fraction',
                                                                  axis=0)

    out.four_percentilesa = scipy.stats.percentileofscore(out.Max_runaway_temperatures ,700)
    out.four_percentilesb = scipy.stats.percentileofscore(out.Max_runaway_temperatures ,800)
    out.four_percentilesc = scipy.stats.percentileofscore(out.Max_runaway_temperatures ,900)
    out.four_percentilesd = scipy.stats.percentileofscore(out.Max_runaway_temperatures,1000)

    ## Mantle and magma ocean redox relative to FMQ:
    out.confidence_mantle_CO2_fraction = scipy.stats.scoreatpercentile(out.mantle_CO2_fraction,
                                                                       [int1,int2,int3],
                                                                       interpolation_method='fraction',axis=0)
    out.confidence_mantle_H2O_fraction = scipy.stats.scoreatpercentile(out.mantle_H2O_fraction,
                                                                       [int1,int2,int3],
                                                                       interpolation_method='fraction',axis=0)
    out.confidence_f_O2_FMQ = scipy.stats.scoreatpercentile(out.f_O2_FMQ ,[int1,int2,int3],
                                                            interpolation_method='fraction',axis=0)
    out.confidence_f_O2_IW = scipy.stats.scoreatpercentile(out.f_O2_IW ,[int1,int2,int3],
                                                           interpolation_method='fraction',axis=0)
    out.confidence_f_O2_MH = scipy.stats.scoreatpercentile(out.f_O2_MH ,[int1,int2,int3],
                                                           interpolation_method='fraction',axis=0)
    out.confidence_f_O2_mantle = scipy.stats.scoreatpercentile(out.f_O2_mantle ,[int1,int2,int3],
                                                               interpolation_method='fraction',axis=0)
    out.confidence_iron_ratio = scipy.stats.scoreatpercentile(out.iron_ratio ,[int1,int2,int3],
                                                              interpolation_method='fraction',axis=0)
    out.confidence_f_O2_relative_FMQ = scipy.stats.scoreatpercentile(np.log10(out.f_O2_mantle) - np.log10(out.f_O2_FMQ),
                                                                     [int1,int2,int3], 
                                                                     interpolation_method='fraction', axis=0)
                        
    
    out.Melt_volume = 365*24*60*60*out.Melt_volume/1e9
    out.Melt_volumeCOPY = np.copy(out.Melt_volume)
    out.confidence_melt=scipy.stats.scoreatpercentile(out.Melt_volume,[int1,int2,int3], interpolation_method='fraction',axis=0)
    out.confidence_velocity =  scipy.stats.scoreatpercentile(out.Plate_velocity ,[int1,int2,int3], interpolation_method='fraction',axis=0)

    return out