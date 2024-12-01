
import scipy.stats

from venus_evolution.models.outgassing_module import *
from venus_evolution.classes import *
from venus_evolution.models.other_functions import *

class PlottingOutputs():
    '''
    Contain information from the outputs file.
    '''
    def __init__(self):
        self.new_inputs = []
        self.inputs_for_MC=[]
        self.Total_Fe_array = []
        self.mp = None
        self.rp = None
        self.g = None

class PostProcessOutput():
    def __init__(self):
        self.total_time = []
        self.total_y = []
        self.FH2O_array=  []
        self.FCO2_array= []
        self.MH2O_liq =  []
        self.MCO2_liq =  []
        self.Pressre_H2O = []
        self.CO2_Pressure_array =  []
        self.fO2_array =  []
        self.Mass_O_atm =  []
        self.Mass_O_dissolved =  []
        self.water_frac =  []
        self.Ocean_depth =  []
        self.Max_depth = []
        self.Ocean_fraction =  []

        self.confidence_y = np.array()
        self.confidence_FH2O = np.array()
        self.confidence_FCO2 = np.array()
        self.confidence_MH2O_liq = np.array()
        self.confidence_MCO2_liq = np.array()
        self.confidence_Pressre_H2O = np.array()
        self.confidence_CO2_Pressure_array = np.array()
        self.confidence_fO2_array = np.array()
        self.confidence_Mass_O_atm = np.array()
        self.confidence_Mass_O_dissolved = np.array()
        self.confidence_water_frac = np.array()
        self.confidence_Ocean_depth = np.array()
        self.confidence_Max_depth = np.array()
        self.confidence_Ocean_fraction = np.array()

        self.f_O2_FMQ = []
        self.f_O2_IW = []
        self.f_O2_MH = []
        self.f_O2_mantle = []
        self.iron_ratio = []
        self.total_iron = []
        self.iron2_array = []
        self.iron3_array = []
        self.actual_phi_surf_melt_ar = []
        self.XH2O_melt = []
        self.XCO2_melt = []
        self.Max_runaway_temperatures = []

        self.F_H2O_new = []
        self.F_CO2_new = []
        self.F_CO_new = []
        self.F_H2_new = []
        self.F_CH4_new = []
        self.F_SO2_new = []
        self.F_H2S_new = []
        self.F_S2_new = []
        self.O2_consumption_new = []
        self.Late_melt_production = []
        
        self.mantle_CO2_fraction = []
        self.mantle_H2O_fraction=[]

        self.Total_Fe_array = []

        self.HTmin = []
        self.HTmax = []
        self.HT_duration = []


        self.Melt_volume = []
        self.Plate_velocity = []
        self.total_Ar40  = [] 
        self.total_K40  = []

        self.DH_atmo =  []
        self.DH_solid =  []
        self.DH_solid =  []

        self.conf_DH_atmo = np.array()
        self.conf_DH_solid = np.array()
        
        self.conf_HT_duration = np.array()
        self.conf_HTmax = np.array()
        self.conf_HTmin = np.array()

        self.conf_Late_melt_production = np.array()
        self.four_percentilesa = np.array()
        self.four_percentilesb = np.array()
        self.four_percentilesc = np.array()
        self.four_percentilesd = np.array()
        
        ## Mantle and magma ocean redox relative to FMQ:
        self.confidence_mantle_CO2_fraction = np.array()
        self.confidence_mantle_H2O_fraction = np.array()
        self.confidence_f_O2_FMQ = np.array()
        self.confidence_f_O2_IW = np.array()
        self.confidence_f_O2_MH = np.array()
        self.confidence_f_O2_mantle = np.array()
        self.confidence_iron_ratio = np.array()
        self.confidence_f_O2_relative_FMQ = np.array()
        
        self.Melt_volume = np.array()
        self.Melt_volumeCOPY = np.array()
        self.confidence_melt=np.array()
        self.confidence_velocity =np.array()
