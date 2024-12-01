import numpy as np
import pylab
import scipy
from scipy.integrate import cumtrapz

from postprocess_class import PostProcessOutput, PlottingOutputs
from venus_evolution.models.other_functions import sol_liq

def plot_outputs_full(inputs, MCInputs, out: PostProcessOutput, plot_outs: PlottingOutputs):
    '''
    Plot all possible outputs
    '''
    x_low = 1.0 #Start time for plotting (years)
    x_high =np.max(out.total_time[0])  #Finish time for plotting (years)
    
    rc = MCInputs[0][1].rc
    
    [int1,int2,int3] = [2.5,50,97.5]

    mantle_mass = 0.0

    pylab.figure()
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][0]+out.total_y[k][1],'b')
        pylab.semilogx(out.total_time[k],out.total_y[k][1],'r--')
    pylab.ylabel("Total water (kg)")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.figure()
    pylab.subplot(2,1,1)
    for k, _ in enumerate(inputs):
        pylab.semilogx(out.total_time[k],out.total_y[k][52],'b')
    pylab.ylabel("depletion_fraction")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)
    pylab.subplot(2,1,2)
    Vmantle0 = ((4.0*np.pi/3.0) * (plot_outs.rp**3 - rc**3))/1e9
    for k in range(0,len(inputs)):
        for i in range(0,len(out.Melt_volume[k])): 
            if (out.total_time[k][i]<plot_outs.inputs_for_MC[k][5].interiorg):
                out.Melt_volume[k][i] = 0.0
        pylab.semilogx(out.total_time[k],cumtrapz(out.Melt_volume[k],x=out.total_time[k],initial=0)/Vmantle0,'b')
    pylab.ylabel("cumulative melt")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])

    pylab.figure()
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][24],'b')
    pylab.ylabel("MMW")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.figure(figsize=(20,10))
    pylab.subplot(4,3,1)
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][7],'b', label="Mantle" if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.total_y[k][8],'r', label="Surface" if k == 0 else "")
    pylab.ylabel("Temperature (K)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,2)
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][2]/1000.0)
    pylab.ylabel("Radius of solidification (km)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])

    pylab.subplot(4,3,3)
    pylab.ylabel("Pressure (bar)")
    for k in range(0,len(inputs)):
        pylab.loglog(out.total_time[k],out.Mass_O_atm[k]*g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
                     'g',label='O2'if k == 0 else "")
        pylab.loglog(out.total_time[k],out.water_frac[k]*out.Pressre_H2O[k]/1e5,
                     'b',label='H2O'if k == 0 else "")
        pylab.loglog(out.total_time[k],out.total_y[k][23]/1e5,
                     'r',label='CO2'if k == 0 else "")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,4)
    pylab.ylabel("Liquid water depth (km)")
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.Max_depth[k]/1000.0,'k--',label='Max elevation land' if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.Ocean_depth[k]/1000.0,'b',label='Ocean depth' if k == 0 else "")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,5)
    for k in range(0,len(inputs)):
        pylab.loglog(out.total_time[k],out.total_y[k][9] , 'b' ,label = 'OLR' if k == 0 else "")
        pylab.loglog(out.total_time[k],out.total_y[k][10] , 'r' ,label = 'ASR'  if k == 0 else "")
        pylab.loglog(out.total_time[k],out.total_y[k][11] , 'g' ,label = 'q_m'  if k == 0 else "")
        pylab.loglog(out.total_time[k],280+0*out.total_y[k][9] , 'k--' ,label = 'Runaway limit' if k == 0 else "")
    pylab.ylabel("Heat flux (W/m2)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)


    pylab.subplot(4,3,6)
    pylab.ylabel('Carbon fluxes (Tmol/yr)')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][14],'k' ,label = 'Weathering'  if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.total_y[k][15],'r' ,label = 'Outgassing' if k == 0 else "") 
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,7)
    pylab.ylabel('Melt production, MP (km$^3$/yr)')
    for k in range(0,len(inputs)):
        pylab.loglog(out.total_time[k],out.Melt_volumeCOPY[k],'r')
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])

    pylab.subplot(4,3,8)
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],np.log10(out.f_O2_mantle[k]) - \
                            np.log10(out.f_O2_FMQ[k]),'r', label = 'Mantle fO2'  if k == 0 else "")
    pylab.ylabel("Mantle oxygen fugacity ($\Delta$QFM)")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)


    pylab.subplot(4,3,9)
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][18]*365*24*60*60/(0.032*1e12),'g' ,
                       label = 'Dry crustal' if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.total_y[k][19]*365*24*60*60/(0.032*1e12),'k' ,
                       label = 'Escape' if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.total_y[k][20]*365*24*60*60/(0.032*1e12),'b' ,
                       label = 'Wet crustal' if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.total_y[k][21]*365*24*60*60/(0.032*1e12),'r' ,
                       label = 'Outgassing' if k == 0 else "")
        pylab.semilogx(out.total_time[k],(out.total_y[k][18]+out.total_y[k][19]+out.total_y[k][20]+\
                                          out.total_y[k][21])*365*24*60*60/(0.032*1e12),'c--' ,
                                          label = 'Net' if k == 0 else "")
    pylab.ylabel("O2 flux (Tmol/yr)")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([1.0, np.max(out.total_time)])
    pylab.yscale('symlog',linthresh = 0.01)
    pylab.legend(frameon=False,loc = 3,ncol=2)

    pylab.figure()
    pylab.subplot(4,1,1)
    pylab.ylabel("Solid mantle Fe3+/Fe2+")
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.iron_ratio[k])#,'k')
    pylab.xlabel('Time (yrs)')

    pylab.subplot(4,1,2)
    pylab.ylabel("total solid iron (kg)")
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_iron[k])#,'k')
    pylab.xlabel('Time (yrs)')

    pylab.subplot(4,1,3)
    pylab.ylabel("Fe3+ (kg)")
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.iron3_array[k])#,'k')
    pylab.xlabel('Time (yrs)')

    pylab.subplot(4,1,4)
    pylab.ylabel("Fe2+ (kg)")
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.iron2_array[k])#,'k')
    pylab.xlabel('Time (yrs)')
    #### End plot individual model runs

    pylab.figure()
    pylab.semilogx(out.total_time[0],out.confidence_y[1][1]+out.confidence_y[1][0],'b', label='Tstrat')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][0]+out.confidence_y[0][1], 
                       out.confidence_y[2][0]+out.confidence_y[2][1], color='blue', alpha=0.4)  
    pylab.ylabel("Total water (kg)")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.figure()
    pylab.ylabel("Tstrat")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)


    #######################################################################
    #######################################################################
    ##### 95% confidence interval plots 
    pylab.figure(figsize=(20,10))
    pylab.subplot(4,3,1)
    Mantlelabel="Mantle, T$_p$"
    pylab.semilogx(out.total_time[0],out.confidence_y[1][7],'b', label=Mantlelabel)
    pylab.fill_between(out.total_time[0],out.confidence_y[0][7], out.confidence_y[2][7], color='blue', alpha=0.4)  
    surflabel="Surface, T$_{surf}$"
    pylab.semilogx(out.total_time[0],out.confidence_y[1][8],'r', label=surflabel)
    pylab.fill_between(out.total_time[0],out.confidence_y[0][8], out.confidence_y[2][8], color='red', alpha=0.4)  
    sol_val = sol_liq(plot_outs.rp,g,4000,plot_outs.rp,0.0,0.0)
    sol_val2 = sol_liq(plot_outs.rp,g,4000,plot_outs.rp,3e9,0.0)
    modlabel="Modern T$_{surf}$"
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][8]+737,'r--', label=modlabel)
    pylab.ylabel("Temperature (K)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.ylim([250, 4170])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,2)
    pylab.semilogx(out.total_time[0],out.confidence_y[1][2]/1000.0,'k')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][2]/1000.0, 
                       out.confidence_y[2][2]/1000.0, color='grey', alpha=0.4)  
    pylab.ylabel("Radius of solidification, r$_s$ (km)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])

    pylab.subplot(4,3,3)
    pylab.ylabel("Pressure (bar)")
    O2_label = 'O$_2$'
    pylab.loglog(out.total_time[0], 
                 out.confidence_Mass_O_atm[1]*g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
                 'g',label=O2_label)
    pylab.fill_between(out.total_time[0], 
                       out.confidence_Mass_O_atm[0]*g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
                       out.confidence_Mass_O_atm[2]*g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5), color='green', alpha=0.4)  
    H2O_label = 'H$_2$O'
    pylab.loglog(out.total_time[0],
                 out.confidence_water_frac[1]*out.confidence_Pressre_H2O[1]/1e5,'b',label=H2O_label)
    pylab.fill_between(out.total_time[0],
                       out.confidence_water_frac[0]*out.confidence_Pressre_H2O[0]/1e5, 
                       out.confidence_water_frac[2]*out.confidence_Pressre_H2O[2]/1e5, color='blue', alpha=0.4)  
    CO2_label = 'CO$_2$'
    pylab.loglog(out.total_time[0],
                 out.confidence_y[1][23]/1e5,'r',label=CO2_label)
    pylab.fill_between(out.total_time[0],
                       out.confidence_y[0][23]/1e5, out.confidence_y[2][23]/1e5, color='red', alpha=0.4) 
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,4)
    pylab.ylabel("Liquid water depth (km)")
    pylab.semilogx(out.total_time[0],out.confidence_Ocean_depth[1]/1000.0,'b',label='Ocean depth')
    pylab.fill_between(out.total_time[0],out.confidence_Ocean_depth[0]/1000.0, 
                       out.confidence_Ocean_depth[2]/1000.0, color='blue', alpha=0.4)  
    pylab.yscale('symlog',linthresh = 0.001)
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.ylim([-1e-4, 3.5])
    pylab.legend(frameon=False)

    pylab.subplot(4,3,5)
    pylab.loglog(out.total_time[0],280+0*out.confidence_y[1][9] , 'k:' ,label = 'Runaway limit')
    pylab.loglog(out.total_time[0],out.confidence_y[1][9] , 'b' ,label = 'OLR' )
    pylab.fill_between(out.total_time[0],out.confidence_y[0][9],out.confidence_y[2][9], color='blue', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][10] , 'r' ,label = 'ASR')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][10],out.confidence_y[2][10], color='red', alpha=0.4)  
    q_interior_label = 'q$_m$'
    pylab.loglog(out.total_time[0],out.confidence_y[1][11] , 'g' ,label = q_interior_label)
    pylab.fill_between(out.total_time[0],out.confidence_y[0][11],out.confidence_y[2][11], color='green', alpha=0.4)  
    modlabel = 'Modern q$_m$'
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][11]+0.02,'g--', label=modlabel)
    pylab.ylabel("Heat flux (W/m2)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False,ncol=2)

    pylab.subplot(4,3,6)
    pylab.ylabel('Volatile fluxes (Tmol/yr)')

    nextlabel= 'H$_2$O Escape'
    pylab.loglog(out.total_time[0],-out.confidence_y[1][53]*365*24*60*60/(0.018*1e12),'b' ,label = nextlabel )
    pylab.fill_between(out.total_time[0],-out.confidence_y[2][53]*365*24*60*60/(0.018*1e12),-out.confidence_y[0][53]*365*24*60*60/(0.018*1e12), color='blue', alpha=0.4)  
    nextlabel= 'H$_2$O Ingassing'
    pylab.loglog(out.total_time[0],-out.confidence_y[1][54]*365*24*60*60/(0.018*1e12),'m' ,label = nextlabel) 
    pylab.fill_between(out.total_time[0],-out.confidence_y[2][54]*365*24*60*60/(0.018*1e12),-out.confidence_y[0][54]*365*24*60*60/(0.018*1e12), color='magenta', alpha=0.4)  
    nextlabel= 'H$_2$O Outgassing'
    pylab.loglog(out.total_time[0],out.confidence_y[1][55]*365*24*60*60/(0.018*1e12),'k' ,label = nextlabel) 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][55]*365*24*60*60/(0.018*1e12),out.confidence_y[2][55]*365*24*60*60/(0.018*1e12), color='grey', alpha=0.4) 

    nextlabel= 'CO$_2$ Weathering'
    pylab.loglog(out.total_time[0],-out.confidence_y[1][14],'g:' ,label = nextlabel )
    pylab.fill_between(out.total_time[0],-out.confidence_y[2][14],-out.confidence_y[0][14], color='green', alpha=0.4)  
    nextlabel= 'CO$_2$ Outgassing'
    pylab.loglog(out.total_time[0],out.confidence_y[1][15],'r:' ,label = nextlabel ) 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][15],out.confidence_y[2][15], color='red', alpha=0.4)  

    pylab.yscale('symlog',linthresh = 0.001)
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.legend(frameon=False,ncol=2)

    pylab.subplot(4,3,7)
    pylab.ylabel('Melt production, MP (km$^3$/yr)')
    pylab.loglog(out.total_time[0],out.confidence_melt[1],'r')
    pylab.fill_between(out.total_time[0],out.confidence_melt[0],out.confidence_melt[2], color='red', alpha=0.4)  
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])


    pylab.subplot(4,3,8)
    pylab.semilogx(out.total_time[0],out.confidence_f_O2_relative_FMQ[1],'r')#,label = 'Mantle fO2' )
    pylab.fill_between(out.total_time[0],out.confidence_f_O2_relative_FMQ[0],out.confidence_f_O2_relative_FMQ[2], color='red', alpha=0.4) 

    ypts = np.array([ -1.7352245862884175,1.827423167848699,2.425531914893617,3.5177304964539005,0.9172576832151291])
    xpts = 4.5e9 - np.array([4.027378964941569, 4.162604340567613, 4.176627712854758, 4.345909849749583,4.363939899833055])*1e9
    ypts = 0*xpts + 2.3

    pylab.ylabel("Mantle oxygen fugacity ($\Delta$QFM)")
    pylab.xlabel("Time (yrs)")
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.ylim([-2.1, 2.1])
    pylab.legend(frameon=False,ncol=2)

    pylab.subplot(4,3,9)
    pylab.semilogx(out.total_time[0],out.confidence_y[1][18]*365*24*60*60/(0.032*1e12),'g' ,label = 'Dry crustal')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][18]*365*24*60*60/(0.032*1e12),out.confidence_y[2][18]*365*24*60*60/(0.032*1e12), color='green', alpha=0.4)  
    pylab.semilogx(out.total_time[0],out.confidence_y[1][19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Escape')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][19]*365*24*60*60/(0.032*1e12),out.confidence_y[2][19]*365*24*60*60/(0.032*1e12), color='grey', alpha=0.4)  
    pylab.semilogx(out.total_time[0],out.confidence_y[1][20]*365*24*60*60/(0.032*1e12),'b' ,label = 'Wet crustal')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][20]*365*24*60*60/(0.032*1e12),out.confidence_y[2][20]*365*24*60*60/(0.032*1e12), color='blue', alpha=0.4)  
    pylab.semilogx(out.total_time[0],out.confidence_y[1][21]*365*24*60*60/(0.032*1e12),'r' ,label = 'Outgassing')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][21]*365*24*60*60/(0.032*1e12),out.confidence_y[2][21]*365*24*60*60/(0.032*1e12), color='red', alpha=0.4)  

    pylab.semilogx(out.total_time[0],(out.confidence_y[1][18]+out.confidence_y[1][19]+out.confidence_y[1][20]+out.confidence_y[1][21])*365*24*60*60/(0.032*1e12),'c:' ,label = 'Net')
    O2_label = 'O$_2$ flux (Tmol/yr)'
    pylab.ylabel(O2_label)
    pylab.xlabel("Time (yrs)")
    pylab.yscale('symlog',linthresh = 0.01)
    pylab.legend(frameon=False,loc = 3,ncol=2)
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.minorticks_on()

    pylab.subplot(4,3,10)
    pylab.ylabel('Crustal thickness (km)')
    pylab.semilogx(out.total_time[0],out.confidence_y[1][27]/(4*np.pi*plot_outs.rp**2*1000),'r')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][27]/(4*np.pi*plot_outs.rp**2*1000),out.confidence_y[2][27]/(4*np.pi*plot_outs.rp**2*1000), color='red', alpha=0.4)  
    modlabel = 'Modern range'
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][27]+20.0,'r--')
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][27]+60.0,'r--', label=modlabel) 
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.ylim([-1, 125])
    pylab.minorticks_on()
    pylab.legend(frameon=False)

    pylab.subplot(4,3,11)
    pylab.ylabel('Atmospheric $^{40}$Ar (kg)')
    pylab.loglog(out.total_time[0],out.confidence_y[1][38],'g')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][38],out.confidence_y[2][38], color='green', alpha=0.4)  
    modlabel = 'Modern $^{40}$Ar'
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][11]+1.61e16,'g--', label=modlabel)
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.minorticks_on()
    pylab.legend(frameon=False)


    pylab.ylabel('Atmospheric $^{4}$He (kg)')
    pylab.ylabel('Trace gas mass (kg)')
    pylab.loglog(out.total_time[0],out.confidence_y[1][50],'b')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][50],out.confidence_y[2][50], color='blue', alpha=0.4) 
    modlabel = 'Modern $^{4}$He range'
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][50]+1.3e14,'b--')
    pylab.semilogx(out.total_time[0],0*out.confidence_y[1][50]+6.5e14,'b--', label=modlabel) 
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.minorticks_on()
    pylab.legend(frameon=False)

    pylab.subplot(4,3,12)
    CO2label = 'CO$_2$'
    pylab.semilogx(out.total_time[0],out.confidence_mantle_CO2_fraction[1],'g',label=CO2label)
    pylab.fill_between(out.total_time[0],out.confidence_mantle_CO2_fraction[0],out.confidence_mantle_CO2_fraction[2], color='green', alpha=0.4)  
    pylab.ylabel('Fraction solid')
    H2Olabel = 'H$_2$O'
    pylab.semilogx(out.total_time[0],out.confidence_mantle_H2O_fraction[1],'b',label=H2Olabel)
    pylab.fill_between(out.total_time[0],out.confidence_mantle_H2O_fraction[0],out.confidence_mantle_H2O_fraction[2], color='blue', alpha=0.4) 
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
    pylab.xlim([x_low, x_high])
    pylab.minorticks_on()
    pylab.legend(frameon=False)

    ### waterworlds comparison
    pylab.figure()
    pylab.subplot(5,1,1)
    pylab.semilogx(out.total_time[0],out.confidence_y[1][13],'g')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[0],out.total_y[k][13],'g')
    pylab.xlabel('Time (yrs)')
    pylab.ylabel('Mantle CO2')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

    pylab.subplot(5,1,2)
    pylab.semilogx(out.total_time[0],out.confidence_y[1][15],'r' ,label = 'Outgassing' ) 
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[0],out.total_y[k][15],'g')
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

    pylab.subplot(5,1,3)
    pylab.ylabel('Crustal thickness (km)')
    pylab.semilogx(out.total_time[0],out.confidence_y[1][16]/1000,'r')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[0],out.total_y[k][16]/1000,'g')
        pylab.semilogx(out.total_time[0],out.total_y[k][27]/(4*np.pi*plot_outs.rp**2*1000),'k')
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

    pylab.subplot(5,1,4)
    pylab.ylabel('Outgassing  melt frac')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[0],out.actual_phi_surf_melt_ar[k],'g')
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]) 

    pylab.subplot(5,1,5)
    pylab.ylabel('Outgassing  melt frac')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[0],out.XCO2_melt[k],'r')
        pylab.semilogx(out.total_time[0],out.XH2O_melt[k],'b')
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]) 


    pylab.figure()
    pylab.subplot(3,1,1)
    pylab.loglog(out.total_time[0],out.confidence_y[1][35],'r',label='40K Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][35],out.confidence_y[2][35], color='red', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][36],'b',label='40Ar Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][36],out.confidence_y[2][36], color='blue', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][37],'g',label='40K lid') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][37],out.confidence_y[2][37], color='green', alpha=0.4)  

    con_total_Ar40 = scipy.stats.scoreatpercentile(out.total_Ar40 ,[int1,int2,int3], interpolation_method='fraction',axis=0)
    con_total_K40 = scipy.stats.scoreatpercentile(out.total_K40 ,[int1,int2,int3], interpolation_method='fraction',axis=0)

    pylab.legend()       
    pylab.subplot(3,1,2)
    pylab.ylabel('40Ar atmo')
    pylab.loglog(out.total_time[0],out.confidence_y[1][38],'g')#
    pylab.fill_between(out.total_time[0],out.confidence_y[0][38],out.confidence_y[2][38], color='green', alpha=0.4)  

    pylab.subplot(3,1,3)
    pylab.ylabel('40Ar atmo / Ar40 total')
    pylab.semilogx(out.total_time[0],con_total_Ar40[1],'g')#
    pylab.fill_between(out.total_time[0],con_total_Ar40[0],con_total_Ar40[2], color='green', alpha=0.4)  

    pylab.figure()
    pylab.subplot(3,1,1)
    pylab.loglog(out.total_time[0],out.confidence_y[1][43],'r',label='238U Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][43],out.confidence_y[2][43], color='red', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][44],'b',label='235U Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][44],out.confidence_y[2][44], color='blue', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][45],'g',label='Th Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][45],out.confidence_y[2][45], color='green', alpha=0.4)  
    pylab.legend()

    pylab.subplot(3,1,2)
    pylab.loglog(out.total_time[0],out.confidence_y[1][46],'r',label='238U Lid') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][46],out.confidence_y[2][46], color='red', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][47],'b',label='235U Lid') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][47],out.confidence_y[2][47], color='blue', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][48],'g',label='Th Lid') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][48],out.confidence_y[2][48], color='green', alpha=0.4)  
    pylab.legend()

    pylab.subplot(3,1,3)
    pylab.loglog(out.total_time[0],out.confidence_y[1][49],'r',label='4He Mantle') 
    pylab.fill_between(out.total_time[0],out.confidence_y[0][49],out.confidence_y[2][49], color='red', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.confidence_y[1][50],'b',label='4He Atmo')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][50],out.confidence_y[2][50], color='blue', alpha=0.4)  
    pylab.legend()

    pylab.figure()
    pylab.subplot(3,1,1)
    pylab.ylabel('Power, TW')
    pylab.loglog(out.total_time[0],out.confidence_y[1][32]/1e12,'g',label='Convective loss') 
    pylab.loglog(out.total_time[0],out.confidence_y[1][33]/1e12,'k',label='Volanic loss') 
    pylab.loglog(out.total_time[0],out.confidence_y[1][30]/1e12,'r',label='Mantle heat prod')
    pylab.loglog(out.total_time[0],out.confidence_y[1][32]/1e12+out.confidence_y[1][33]/1e12,'b--',label='Total loss')
    pylab.legend()

    pylab.subplot(3,1,2)
    pylab.ylabel('Equivalent surface flux mW/m2')
    pylab.loglog(out.total_time[0],1000*out.confidence_y[1][32]/(4*np.pi*plot_outs.rp**2),'g',label='Convective loss') 
    pylab.fill_between(out.total_time[0],1000*out.confidence_y[0][32]/(4*np.pi*plot_outs.rp**2),1000*out.confidence_y[2][32]/(4*np.pi*plot_outs.rp**2), color='green', alpha=0.4)  
    pylab.loglog(out.total_time[0],1000*out.confidence_y[1][33]/(4*np.pi*plot_outs.rp**2),'k',label='Volanic loss') 
    pylab.fill_between(out.total_time[0],1000*out.confidence_y[0][33]/(4*np.pi*plot_outs.rp**2),1000*out.confidence_y[2][33]/(4*np.pi*plot_outs.rp**2), color='grey', alpha=0.4) 
    pylab.loglog(out.total_time[0],1000*out.confidence_y[1][30]/(4*np.pi*plot_outs.rp**2),'r',label='Mantle heat prod')
    pylab.fill_between(out.total_time[0],1000*out.confidence_y[0][30]/(4*np.pi*plot_outs.rp**2),1000*out.confidence_y[2][30]/(4*np.pi*plot_outs.rp**2), color='red', alpha=0.4) 
    pylab.loglog(out.total_time[0],1000*out.confidence_y[1][32]/(4*np.pi*plot_outs.rp**2)+1000*out.confidence_y[1][33]/(4*np.pi*plot_outs.rp**2),'b--',label='Total loss')
    pylab.fill_between(out.total_time[0],1000*out.confidence_y[0][32]/(4*np.pi*plot_outs.rp**2)+1000*out.confidence_y[0][33]/(4*np.pi*plot_outs.rp**2),1000*out.confidence_y[2][32]/(4*np.pi*plot_outs.rp**2)+1000*out.confidence_y[2][33]/(4*np.pi*plot_outs.rp**2), color='blue', alpha=0.4) 
    pylab.legend()

    pylab.subplot(3,1,3)
    pylab.ylabel('Crustal thickness (km)')
    pylab.semilogx(out.total_time[0],out.confidence_y[1][27]/(4*np.pi*plot_outs.rp**2*1000),'r')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][27]/(4*np.pi*plot_outs.rp**2*1000),out.confidence_y[2][27]/(4*np.pi*plot_outs.rp**2*1000), color='red', alpha=0.4)  
    pylab.xlabel('Time (yrs)')
    pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

    pylab.figure()
    pylab.subplot(2,2,1)
    pylab.loglog(out.total_time[0],out.conf_DH_atmo[1],'g',label='D/H surface')
    pylab.fill_between(out.total_time[0],out.conf_DH_atmo[0],out.conf_DH_atmo[2], color='green', alpha=0.4)  
    pylab.loglog(out.total_time[0],out.conf_DH_solid[1],'r',label='D/H interior')
    pylab.fill_between(out.total_time[0],out.conf_DH_solid[0],out.conf_DH_solid[2], color='red', alpha=0.4)
    pylab.legend()
    
    pylab.subplot(2,2,3)
    pylab.loglog(out.total_time[0],0.5*out.confidence_y[1][41],'g',label='D surface')
    pylab.fill_between(out.total_time[0],0.5*out.confidence_y[0][41],0.5*out.confidence_y[2][41], color='green', alpha=0.4)  
    pylab.loglog(out.total_time[0],0.5*out.confidence_y[1][42],'r',label='D interior')
    pylab.fill_between(out.total_time[0],0.5*out.confidence_y[0][42],0.5*out.confidence_y[2][42], color='red', alpha=0.4)
    pylab.legend()

    pylab.subplot(2,2,2)
    for k in range(0,len(inputs)):
        pylab.loglog(out.total_time[0],0.5*out.total_y[k][41]/out.total_y[k][1],'g')
        pylab.loglog(out.total_time[0],0.5*out.total_y[k][42]/out.total_y[k][0],'r')
    pylab.legend()
    
    pylab.subplot(2,2,4)
    for k in range(0,len(inputs)):
        pylab.loglog(out.total_time[0],out.total_y[k][41],'g')
        pylab.loglog(out.total_time[0],out.total_y[k][42],'r')

    pylab.legend()

    pylab.figure()
    for k in range(0,len(plot_outs.new_inputs)):
        pylab.plot(out.total_y[k][16][-50]/1000,out.total_y[k][15][-50],'.')
    pylab.title(out.total_time[0][-50])
    pylab.xlabel('Crust thickness')
    pylab.ylabel('CO2 outgas (Tmol/yr)')

    pylab.figure()
    for k in range(0,len(plot_outs.new_inputs)):
        pylab.semilogx(out.total_y[k][13][-50]/mantle_mass,out.total_y[k][15][-50],'.')
    pylab.title(out.total_time[0][-50])
    pylab.xlabel('Mantle CO2')
    pylab.ylabel('CO2 outgas (Tmol/yr)')

    pylab.figure()
    pylab.ylabel("MMW")
    pylab.semilogx(out.total_time[0],out.confidence_y[1][24],'g')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][24],out.confidence_y[2][24], color='green', alpha=0.4)  
    pylab.xlabel('Time (yrs)')

    pylab.figure()
    pylab.ylabel("Solid mantle Fe3+/Fe2+")
    pylab.semilogx(out.total_time[0],out.confidence_iron_ratio[1],'k')
    pylab.fill_between(out.total_time[0],out.confidence_iron_ratio[0],out.confidence_iron_ratio[2], color='grey', alpha=0.4)  
    pylab.xlabel('Time (yrs)')

    pylab.tight_layout()

    #######################################################################
    #######################################################################
    ### Begin parameter space plots 
    ######## First, need to fill input arrays from input files
    #######################################################################

    pylab.figure()
    pylab.subplot(4,1,1)
    O2_final_ar = []
    H2O_final_ar = []
    CO2_final_ar = []
    Total_P_ar = []
    atmo_H2O_ar = []


    for k in range(0,len(inputs)):
        addO2 = out.Mass_O_atm[k][-1]*g/(4*np.pi*(0.032/out.total_y[k][24][-1])*plot_outs.rp**2*1e5)
        if (addO2 <= 0 ) or np.isnan(addO2):
            addO2 = 1e-8
        addH2O = out.Pressre_H2O[k][-1]/1e5
        atmo_H2O = out.water_frac[k][-1]
        if (addH2O <= 0 ) or np.isnan(addH2O):
            addH2O = 1e-8
        if (atmo_H2O<=0) or np.isnan(atmo_H2O):
            atmo_H2O = 0
        addCO2 = out.CO2_Pressure_array[k][-1]/1e5
        if (addCO2 <= 0 ) or np.isnan(addCO2):
            addCO2 = 1e-8
        O2_final_ar.append(np.log10(addO2))
        H2O_final_ar.append(np.log10(addH2O))
        CO2_final_ar.append(np.log10(addCO2))
        atmo_H2O_ar.append(atmo_H2O)
        Total_P_ar.append(np.log10(addO2+addH2O+addCO2))

    pylab.hist(O2_final_ar,bins = 50,color = 'g')
    pylab.xlabel('log(pO2)')
    pylab.subplot(4,1,2)
    pylab.hist(H2O_final_ar,color = 'b',bins = 50)
    pylab.xlabel('log(pH2O)')
    pylab.subplot(4,1,3)
    pylab.hist(CO2_final_ar,color = 'r',bins = 50)
    pylab.xlabel('log(pCO2)')
    pylab.subplot(4,1,4)
    pylab.hist(Total_P_ar,color = 'c',bins = 50)
    pylab.xlabel('Log10(Pressure (bar))')

    init_CO2_H2O = []
    init_CO2_ar =[]
    Final_O2 = []
    Final_CO2 = []
    Final_H2O = []
    Surface_T_ar = []
    H2O_upper_ar = []
    Weathering_limit = []
    tsat_array = []
    epsilon_array = []

    Ca_array = []
    Omega_ar = []
    init_H2O_ar=[]
    offset_ar = []
    Te_ar =[]
    expCO2_ar = []
    Mfrac_hydrated_ar= []
    dry_frac_ar = []
    wet_OxFrac_ar = []
    Radiogenic= []
    Init_fluid_O_ar = []
    albedoH_ar = []
    albedoC_ar = []
    MaxMantleH2O=[]
    imp_coef_ar = []
    imp_slope_ar = []
    hist_total_imp_mass=[]
    mult_ar= []
    mix_epsilon_ar=[]
    Transition_time = []
    Final_40Ar = []
    Ar40_ratio = []
    beta_array = []
    completion_time = []
    surface_magma_frac_array = []
    Tstrat_ar = []

    for k in range(0,len(inputs)):
        Tstrat_ar.append(plot_outs.inputs_for_MC[k][5].Tstrat)
        init_CO2 = out.total_y[k][12][0]+out.total_y[k][13][0]
        init_H2O = out.total_y[k][0][0]+out.total_y[k][1][0]
        surface_magma_frac_array.append(plot_outs.inputs_for_MC[k][5].surface_magma_frac)
        init_H2O_ar.append(plot_outs.inputs_for_MC[k][2].Init_fluid_H2O)
        Init_fluid_O_ar.append(plot_outs.inputs_for_MC[k][2].Init_fluid_O)
        albedoC_ar.append(plot_outs.inputs_for_MC[k][1].albedoC)
        albedoH_ar.append(plot_outs.inputs_for_MC[k][1].albedoH)
        init_CO2_H2O.append(init_CO2/init_H2O)
        init_CO2_ar.append(plot_outs.inputs_for_MC[k][2].Init_fluid_CO2)
        Final_O2.append(out.total_y[k][22][-1]/1e5)
        Final_CO2.append(out.CO2_Pressure_array[k][-1]/1e5) 
        Final_H2O.append(out.water_frac[k][-1]*out.Pressre_H2O[k][-1]/1e5)
        H2O_upper_ar.append(out.total_y[k][14][-1])
        Surface_T_ar.append(out.total_y[k][8][-1])
        Weathering_limit.append(plot_outs.inputs_for_MC[k][5].supp_lim)
        tsat_array.append(plot_outs.inputs_for_MC[k][4].tsat_XUV)
        epsilon_array.append(plot_outs.inputs_for_MC[k][4].epsilon)
        beta_array.append(plot_outs.inputs_for_MC[k][4].beta0)
        Ca_array.append(plot_outs.inputs_for_MC[k][5].ocean_a)
        Omega_ar.append(plot_outs.inputs_for_MC[k][5].ocean_b)
        offset_ar.append(plot_outs.inputs_for_MC[k][5].interiora)
        Mfrac_hydrated_ar.append(plot_outs.inputs_for_MC[k][5].interiorb)
        Te_ar.append(plot_outs.inputs_for_MC[k][5].ccycle_a)
        expCO2_ar.append(plot_outs.inputs_for_MC[k][5].ccycle_b) 
        dry_frac_ar.append(plot_outs.inputs_for_MC[k][5].interiorc)
        wet_OxFrac_ar.append(plot_outs.inputs_for_MC[k][5].interiord)
        Radiogenic.append(plot_outs.inputs_for_MC[k][5].interiore)
        MaxMantleH2O.append(plot_outs.inputs_for_MC[k][5].interiorf)
        Transition_time.append(plot_outs.inputs_for_MC[k][5].interiorg)
        Final_40Ar.append(out.total_y[k][38][-1])
        Ar40_ratio.append(out.total_Ar40[k][-1])
        imp_coef_ar.append(plot_outs.inputs_for_MC[k][5].esc_a)
        imp_slope_ar.append(plot_outs.inputs_for_MC[k][5].esc_b)
        mult_ar.append(plot_outs.inputs_for_MC[k][5].esc_c)
        mix_epsilon_ar.append(plot_outs.inputs_for_MC[k][5].esc_d)
        t_ar = np.linspace(0,1,1000)
        y = np.copy(t_ar)
        for i in range(0,len(t_ar)):
            y[i] = plot_outs.inputs_for_MC[k][5].esc_a*np.exp(-t_ar[i]/plot_outs.inputs_for_MC[k][5].esc_b)
        hist_total_imp_mass.append(np.trapz(y,t_ar*1e9))
        completion_time.append(out.total_y[k][41][-1])


    Ca_array = np.array(Ca_array)
    Omega_ar = np.array(Omega_ar)


    pylab.figure()
    pylab.subplot(2,2,1)
    pylab.loglog(dry_frac_ar,Final_O2,'.')
    pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
    pylab.ylabel('Final O$_2$ (bar)')
    pylab.xlim([1e-4,1e-1])

    pylab.subplot(2,2,2)
    pylab.semilogy(epsilon_array,Final_O2,'.')
    pylab.xlabel('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$ ')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.subplot(2,2,3)
    pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*plot_outs.rp**2),'.')
    pylab.xlabel('Initial H$_2$O inventory (Earth oceans)')
    pylab.ylabel('Initial CO$_2$ inventory (bar)')

    pylab.subplot(2,2,4)
    pylab.loglog(surface_magma_frac_array,Final_O2,'.')
    pylab.xlabel('Extrusive lava fraction, $f_{lava}$')
    pylab.ylabel('Final O$_2$ (bar)')
    pylab.xlim([1e-4,1])

    pylab.figure()
    pylab.semilogy(Tstrat_ar,Final_O2,'b.') 
    pylab.xlabel('T$_{stratosphere}$ (K)')
    pylab.ylabel('Final O$_2$ (bar)')
    pylab.xlim([150,250])

    pylab.figure()
    pylab.title('completion time (min)')

    pylab.subplot(2,2,1)
    pylab.plot(Radiogenic,completion_time,'x')
    pylab.xlabel('Radiongenic')

    pylab.subplot(2,2,2)
    pylab.semilogx(offset_ar,completion_time,'x')
    pylab.xlabel('offset_ar')

    pylab.subplot(2,2,3)
    pylab.semilogx(init_CO2_ar,completion_time,'x')
    pylab.xlabel('init_CO2_ar')

    pylab.subplot(2,2,4)
    pylab.semilogx(init_H2O_ar,completion_time,'x')
    pylab.xlabel('init_H2O_ar')

    pylab.figure()
    pylab.semilogy(np.log10(hist_total_imp_mass),Final_O2,'.')
    pylab.xlabel('Total impactor mass, log$_{10}$(kg)')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.figure()
    pylab.subplot(2,1,1)
    pylab.plot(Transition_time,Final_40Ar,'.')
    pylab.xlabel('Transition_time (yrs)')
    pylab.ylabel('Final_40Ar')
    pylab.subplot(2,1,2)
    pylab.plot(Transition_time,Ar40_ratio,'.')
    pylab.xlabel('Transition_time (yrs)')
    pylab.ylabel('Final_40Ar_ratio')

    pylab.figure()
    pylab.subplot(2,3,1)
    pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*plot_outs.rp**2),'.')
    pylab.xlabel('Initial H2O inventory (Earth oceans)')
    pylab.ylabel('Initial CO2 inventory (bar)')

    pylab.subplot(2,3,2)
    pylab.loglog(init_H2O_ar,init_CO2_ar,'.')
    pylab.xlabel('Initial H2O inventory')
    pylab.ylabel('Initial CO2 inventory')

    pylab.subplot(2,3,3)
    pylab.loglog(Radiogenic,offset_ar,'.')
    pylab.xlabel('Radiogenic')
    pylab.ylabel('offset_ar')

    pylab.subplot(2,3,4)
    pylab.loglog(Init_fluid_O_ar,wet_OxFrac_ar,'.')
    pylab.xlabel('Init_fluid_O')
    pylab.ylabel('wet_OxFrac_ar')

    pylab.subplot(2,3,6)
    pylab.loglog(MaxMantleH2O,Final_O2,'.')
    pylab.xlabel('MaxMantleH2O')
    pylab.ylabel('Final_O2')

    pylab.subplot(2,3,5)
    pylab.loglog(albedoC_ar,albedoH_ar,'.')
    pylab.xlabel('AlbedoC_ar')
    pylab.ylabel('AlbedoH_ar')

    pylab.figure()
    pylab.subplot(2,3,1)
    pylab.semilogy(Te_ar,Final_O2,'.')
    pylab.xlabel('Te_ar (K)')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,2)
    pylab.semilogy(expCO2_ar,Final_O2,'.')
    pylab.xlabel('expCO2_ar')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,3)
    pylab.loglog(Mfrac_hydrated_ar,Final_O2,'.')
    pylab.xlabel('Mfrac_hydrated_ar')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,4)
    pylab.loglog(dry_frac_ar,Final_O2,'.')
    pylab.xlabel('dry_frac_ar')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,5)
    pylab.loglog(wet_OxFrac_ar,Final_O2,'.')
    pylab.xlabel('wet_OxFrac_ar')
    pylab.ylabel('Final O2 (bar)')


    pylab.subplot(2,3,6)
    pylab.loglog(Radiogenic,Final_O2,'.')
    pylab.xlabel('Radiogenic')
    pylab.ylabel('Final O2 (bar)')

    pylab.figure()
    pylab.subplot(1,2,1)
    pylab.loglog(dry_frac_ar,Final_O2,'.')
    pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.subplot(1,2,2)
    pylab.loglog(init_CO2_H2O,Final_O2,'.')
    pylab.xlabel('Initial CO$_2$:H$_2$O')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.figure()
    pylab.loglog(init_H2O_ar,Final_O2,'.')
    pylab.xlabel('Initial H$_2$O (kg)')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.figure()
    pylab.loglog(np.array(init_H2O_ar)/1.4e21,Final_O2,'.')
    pylab.xlabel('Initial H$_2$O (Earth oceans)')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.figure()
    pylab.subplot(1,3,1)
    pylab.loglog(dry_frac_ar,Final_O2,'.')
    pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.subplot(1,3,2)
    pylab.semilogy(epsilon_array,Final_O2,'.')
    pylab.xlabel('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$ ')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.subplot(1,3,3)
    pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*plot_outs.rp**2),'.')
    pylab.xlabel('Initial H$_2$O inventory (Earth oceans)')
    pylab.ylabel('Initial CO$_2$ inventory (bar)')

    pylab.figure()
    pylab.subplot(2,3,1)
    pylab.semilogy(epsilon_array,Final_O2,'.')
    pylab.xlabel('epsilon (for XUV)')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,2)
    pylab.loglog(Omega_ar/Ca_array,Final_O2,'.')
    pylab.xlabel('omega/Ca ~ CO3')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,3)
    pylab.loglog(offset_ar,Final_O2,'.')
    pylab.xlabel('offset')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,4)
    pylab.loglog(init_H2O_ar,Final_O2,'.')
    pylab.xlabel('init_H2O')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,5)
    pylab.loglog(mult_ar,Final_O2,'.')
    pylab.xlabel('mult_ar')
    pylab.ylabel('Final O2 (bar)')

    pylab.subplot(2,3,6)
    pylab.semilogy(mix_epsilon_ar,Final_O2,'.')
    pylab.xlabel('mix_epsilon_ar')
    pylab.ylabel('Final O2 (bar)')

    pylab.figure()
    pylab.subplot(2,2,1)
    pylab.plot(Surface_T_ar,H2O_upper_ar,'.')
    pylab.xlabel('SurfT')
    pylab.ylabel('y14 H2O upper atmo frac')

    pylab.subplot(2,2,2)
    pylab.semilogy(Surface_T_ar,Final_H2O,'.')
    pylab.xlabel('SurfT')
    pylab.ylabel('atmo H2o pressure (bar)')

    pylab.subplot(2,2,3)
    pylab.loglog(init_CO2_H2O,Final_O2,'.')
    pylab.xlabel('CO2/H2O init')
    pylab.ylabel('final pO2 (bar)')

    pylab.subplot(2,2,4)
    pylab.loglog(init_CO2_ar,Final_O2,'.')
    pylab.xlabel('CO2 init')
    pylab.ylabel('final pO2 (bar)')

    pylab.figure()
    pylab.subplot(3,2,1)
    pylab.loglog(Final_H2O,Final_O2,'.')
    pylab.xlabel('Final_H2O (bar)')
    pylab.ylabel('final pO2 (bar)')

    pylab.subplot(3,2,2)
    pylab.loglog(Final_CO2,Final_O2,'.')
    pylab.xlabel('final CO2 (bar)')
    pylab.ylabel('final pO2 (bar)')


    pylab.subplot(3,2,3)
    pylab.loglog(Weathering_limit,Final_O2,'.')
    pylab.xlabel('Weathering limit (kg/s)')
    pylab.ylabel('final pO2 (bar)')

    pylab.subplot(3,2,4)
    pylab.loglog(tsat_array,Final_O2,'.')
    pylab.xlabel('tsat XUV (Gyr)')
    pylab.ylabel('final pO2 (bar)')

    pylab.subplot(3,2,5)
    pylab.semilogy(beta_array,Final_O2,'.')
    pylab.xlabel('beta0')
    pylab.ylabel('final pO2 (bar)')

    pylab.subplot(3,2,6)
    pylab.loglog(Final_H2O,Final_CO2,'.')
    pylab.xlabel('Final_H2O (bar)')
    pylab.ylabel('final CO2 (bar)')

    pylab.figure()
    pylab.subplot(6,1,1)
    pylab.ylabel('Mass H2O solid, kg')

    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][0])
    pylab.subplot(6,1,2)
    pylab.ylabel('H2O reservoir (kg)')
    for k in range(0,len(inputs)):
        pylab.semilogx(out.total_time[k],out.total_y[k][1],'r',label='Mass H2O, magma ocean + atmo'  if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.MH2O_liq[k],'b',label='Mass H2O, magma ocean'  if k == 0 else "")
        pylab.semilogx(out.total_time[k],out.Pressre_H2O[k] *4 * np.pi * (0.018/out.total_y[k][24])* (plot_outs.rp**2/g) ,label='Mass H2O atmosphere'  if k == 0 else "")
    pylab.subplot(6,1,3)
    for k in range(0,len(inputs)):
        pylab.ylabel('H2O fraction solid')
        pylab.semilogx(out.total_time[k],out.total_y[k][0]/(out.total_y[k][0]+out.total_y[k][1]))
    pylab.subplot(6,1,4)
    for k in range(0,len(inputs)):
        pylab.ylabel('CO2 fraction solid')
        pylab.semilogx(out.total_time[k],out.total_y[k][13]/(out.total_y[k][13]+out.total_y[k][12]))
    pylab.subplot(6,1,5)
    pylab.semilogx(out.total_time[0],out.confidence_mantle_CO2_fraction[1],'k')
    pylab.fill_between(out.total_time[0],out.confidence_mantle_CO2_fraction[0],out.confidence_mantle_CO2_fraction[2], color='grey', alpha=0.4)  
    pylab.ylabel('CO2 fraction solid')
    pylab.xlabel('Time (yrs)')
    pylab.subplot(6,1,6)
    pylab.semilogx(out.total_time[0],out.confidence_mantle_H2O_fraction[1],'k')
    pylab.fill_between(out.total_time[0],out.confidence_mantle_H2O_fraction[0],out.confidence_mantle_H2O_fraction[2], color='grey', alpha=0.4)  
    pylab.ylabel('H2O fraction solid')
    pylab.xlabel('Time (yrs)')
    pylab.show()

