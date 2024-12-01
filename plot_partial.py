import numpy as np
import pylab

from postprocess_class import PostProcessOutput, PlottingOutputs
from venus_evolution.models.other_functions import sol_liq

def plot_partial(inputs, MCInputs, out: PostProcessOutput, plot_outs: PlottingOutputs):
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
        pylab.loglog(out.total_time[k],out.Mass_O_atm[k]*plot_outs.g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
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
    pylab.semilogx(out.total_time[0],out.confidence_y[1][1]+out.confidence_y[1][0],'b', label='Tstrat')
    pylab.fill_between(out.total_time[0],out.confidence_y[0][0]+out.confidence_y[0][1], 
                       out.confidence_y[2][0]+out.confidence_y[2][1], color='blue', alpha=0.4)  
    pylab.ylabel("Total water (kg)")
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
    sol_val = sol_liq(plot_outs.rp,plot_outs.g,4000,plot_outs.rp,0.0,0.0)
    sol_val2 = sol_liq(plot_outs.rp,plot_outs.g,4000,plot_outs.rp,3e9,0.0)
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
                 out.confidence_Mass_O_atm[1]*plot_outs.g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
                 'g',label=O2_label)
    pylab.fill_between(out.total_time[0], 
                       out.confidence_Mass_O_atm[0]*plot_outs.g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5),
                       out.confidence_Mass_O_atm[2]*plot_outs.g/(4*np.pi*(0.032/out.total_y[k][24])*plot_outs.rp**2*1e5), color='green', alpha=0.4)  
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

    #######################################################################
    #######################################################################
    ### Begin parameter space plots 
    ######## First, need to fill input arrays from input files
    #######################################################################

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

    pylab.subplot(2,2,4)
    pylab.semilogx(init_H2O_ar,completion_time,'x')
    pylab.xlabel('init_H2O_ar')

    pylab.figure()
    pylab.loglog(init_H2O_ar,Final_O2,'.')
    pylab.xlabel('Initial H$_2$O (kg)')
    pylab.ylabel('Final O$_2$ (bar)')

    pylab.subplot(2,3,4)
    pylab.loglog(init_H2O_ar,Final_O2,'.')
    pylab.xlabel('init_H2O')
    pylab.ylabel('Final O2 (bar)')

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
        pylab.semilogx(out.total_time[k],out.Pressre_H2O[k] *4 * np.pi * (0.018/out.total_y[k][24])* (plot_outs.rp**2/plot_outs.g) ,label='Mass H2O atmosphere'  if k == 0 else "")
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
    a=1
