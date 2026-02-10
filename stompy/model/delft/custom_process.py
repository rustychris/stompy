import shutil
import sys
import os
from collections import defaultdict
from stompy import utils
from . import dflow_model, waq_scenario


# When including this as a 'mixin', it needs to go first, before
# DFlowModel.

# Making this work with WaqModel:
#   .write_inp() hook, and caller should use self.waq_proc_def.

class CustomProcesses:
    # Ideally this would get integrated in waq_scenario. Since this model
    # is online coupled it's awkward to have it here.

    # path to clean set of csv files, though potentially wrong DWAQ version
    @property
    def proc_table_src_dir(self):
        return os.environ['PROC_TABLE_SRC_DIR']
    
    # path to where edits, import/export will happen
    @property
    def proc_table_dst_dir(self):
        if isinstance(self,dflow_model.DFlowModel):
            run_dir = self.run_dir
        else:
            run_dir = self.base_path
            
        return os.path.join(run_dir,"proc_tables")

    @property
    def bin_dir(self):
        if isinstance(self,dflow_model.DFlowModel):
            return self.dfm_bin_dir
        else:
            return self.delft_bin
        
    _waqpbexport=None
    @property
    def waqpbexport(self):
        return self._waqpbexport or os.path.join(self.bin_dir,'waqpb_export')
    @waqpbexport.setter
    def waqpbexport(self,value):
        self._waqpbexport = value
        
    _waqpbimport=None
    @property
    def waqpbimport(self):
        return self._waqpbimport or os.path.join(self.bin_dir,'waqpb_import')
    @waqpbimport.setter
    def waqpbimport(self,value):
        self._waqpbimport = value
        
    def __init__(self,*a,**k):
        # Map process names like 'CART' to the number of processes
        # defined so far. 
        self.copy_count=defaultdict(lambda: 0)
        self.custom_procs=[] # list of string with definition of each process
        super().__init__(*a,**k)

    def write(self):
        """
        Hook into DFlowModel initialization
        """
        super().write()
        if self.custom_procs:
            self.build_process_db()

    def write_inp(self):
        """
        Hook into WaqModel initialization
        """
        super().write_inp()
        if self.custom_procs:
            self.build_process_db() # will update self.waq_proc_def
        
    def build_process_db(self):
        """
        Compile any locally defined custom processes into proc_def.{def,dat}
        """
        # Clean slate:
        self.log.info("Copying source process tables")
        if os.path.exists(self.proc_table_dst_dir):
            shutil.rmtree(self.proc_table_dst_dir)
        shutil.copytree(self.proc_table_src_dir,self.proc_table_dst_dir)

        print("First call to waqpb_export - creating procesm.asc from decomposed tables")
        sys.stdout.flush()
        output = utils.call_with_path(self.waqpbexport,self.proc_table_dst_dir).decode('latin1') 
        print("Suppressed output")
        sys.stdout.flush()
        print("First call to waqpb_export DONE")
        sys.stdout.flush()
        # => proc_def.* and procesm.asc in proc_table_dst_dir

        # First line of proces_asc includes a process count. Rewrite that line, shove in
        # our new process, and copy the rest of the file.

        print(f"Transcribing procesm.asc -> proces.asc, adding {len(self.custom_procs)} custom processes")
        procesm_asc=os.path.join(self.proc_table_dst_dir,'procesm.asc')
        proces_asc=os.path.join(self.proc_table_dst_dir,'proces.asc')

        # funny characters in the incoming file.
        with open(procesm_asc,'rt',encoding='latin1') as fp_in:
            with open(proces_asc,'wt') as fp:
                hdr=fp_in.readline().split()
                count=int(hdr[0]) + len(self.custom_procs)
                fp.write(f"{count:10} {hdr[1]:>57} {hdr[2]:>11}\n")
                for proc in self.custom_procs:
                    fp.write(proc.strip()+"\n")
                for line in fp_in:
                    fp.write(line)
        sys.stdout.flush()
                    
        print("Calling waqpb_import to convert proces.asc back to tables")
        sys.stdout.flush()
        # This is the one likely to fail.
        output = utils.call_with_path(self.waqpbimport,self.proc_table_dst_dir).decode('latin1') 
        print("Suppressed output")
        #print(output) # uncomment this line to diagnose Fatal Error 
        sys.stdout.flush()
        print("Return from waqpb_import")
        sys.stdout.flush()

        print("Second call to waqpb_export to take the updated tables and write compiled form of tables")
        sys.stdout.flush()
        output = utils.call_with_path(self.waqpbexport,self.proc_table_dst_dir).decode('latin1') 
        print( "Suppressed output")
        #print("Output from waqpb_export")
        #print(output)
        sys.stdout.flush()
        print("Return from second call to waqpb_export")
        sys.stdout.flush()
        
        # Tell dwaq about the tables
        self.waq_proc_def=os.path.abspath(os.path.join(self.proc_table_dst_dir,'proc_def'))
        # if we're Scenario, it goes here
        self.proc_path=self.waq_proc_def

    def custom_add_process(self,name):        
        if isinstance(self,dflow_model.DFlowModel):
            self.dwaq.add_process(name)
        elif isinstance(self,waq_scenario.WaqModel):
            self.add_process(name)
        else: # waq_scenario.Scenario
            print(f"Assuming caller will activate process {name}")
        
    def custom_Decay(self,substance,rate):
        """
        Add a basic decay process for the given substance and rate.
        For now rate has to be given as a constant first order rate
        with units of d-1. In the future we could pass in partial.
        """
        proc_txt=self.CART(conc=substance,
                           conc_decay=rate)
        self.custom_procs.append(proc_txt)

    def custom_ExpFilter(self,sub_in,sub_out,rate):
        self.custom_Decay(sub_out,rate)
        proc_txt=self.CART(conc=sub_in, age_conc=sub_out, partial_default=rate)
        self.custom_procs.append(proc_txt)

    def custom_CART(self,**kw):
        proc_txt=self.CART(**kw)
        self.custom_procs.append(proc_txt)

    def CART(self, conc_decay=0.0, **kw):
        idx=self.copy_count['CART']
        self.copy_count['CART']+=1
        
        suffix=str(idx)
        p=dict(name="CART"+suffix,
               zero_order="ZAge"+suffix,
               partial="PartAge"+suffix,
               conc=f"Age{suffix}Conc",
               partial_default=1.0,
               age_conc=f"Age{suffix}AConc",
               flux="dAge"+suffix)
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        n_stoich=1
        if conc_decay!=0.0:
            n_stoich+=1

        # partial is a rate in d-1, defaults to 1.0.
        # flux = partial * conc

        # All of this is to have a pair of tracers that
        # integrate this:
        
        #     d conc / dt = -conc_decay*partial * conc
        # d age_conc / dt =             partial * conc

        # partial defaults to 1.0.

        process=f"""
{p['name']:10}                    Reuse nitrification as age                         
NITRIF    ; module name. 
123       ; TRswitch
        18; # input items for segments
{p['zero_order']:10}      0.00000     x zeroth-order flux          (g/m3/d)
{p['partial']   :10}      {p['partial_default']: 8g}     x set this to get partial age
RcNit20        0.100000     x ignored (b/c SWVnNit=0)
TcCART          1.00000     x not ignored, but separate name should keep it okay.
OXY             10.0000     x ignored
KsAmNit        0.500000     x ignored
KsOxNit         1.00000     x ignored
Temp            15.0000     x ignored
CTNit           3.00000     x ignored
Rc0NitOx        0.00000     x ignored
COXNIT          1.00000     x ignored
Poros           1.00000     x volumetric porosity                            (-)
SWVnNit         0.00000     x switch for old (0), new (1), TEWOR (2) version (-)
{p['conc']    :10}     0.100000     x concentration tracer
OOXNIT          5.00000     x ignored
CFLNIT          0.00000     x ignored
CurvNit         0.00000     x ignored
DELT           -999.000     x timestep for processes                         (d)
         0; # input items for exchanges
         1; # output items for segments
O2FuncNT1                   x oxygen function for nitrification              (-)
         0; # output items for exchanges
         1; # fluxes
{p['flux']:10}                  x nitrification flux                       (gN/m3/d)
         {n_stoich}; # stoichiometry lines. Could probably drop most of these.
{p['age_conc']:10}  {p['flux']:10}     1.00000
"""
        if conc_decay!=0.0:
            process+=f"{p['conc']:10}  {p['flux']:10}    {-conc_decay:.5f}\n"
        process+="""         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        return process

    def custom_fall_velocity(self,**kw):
        # name: name of process
        # sub: name of substance to apply settling to
        # fall_rate: name of parameter for settling speed.
        #   for online coupling only constant and time-varying will work.
        #   units are meters/day, positive down.
        proc_txt=self.fall_velocity(**kw)
        self.custom_procs.append(proc_txt)

    def fall_velocity(self, **kw):
        idx=self.copy_count['FALL']
        self.copy_count['FALL']+=1
        
        suffix=str(idx)
        p=dict(name="FALL"+suffix,
               # FIX BELOW
               sub=f"Fall{suffix}",
               fall_rate=f"FallRate{suffix}")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])

        # Two important things in the process table:
        #  - create an exchange-centered settling velocity,
        #  - assign that to the velocity array.
        process=f"""
{p['name']:10}                      Reuse algae settling 
SEDCAR    ; module name
123       ; TRswitch
         8; # input items for segments
{p['sub']:10}     -999.000     x concentration to settle                                (gC/m3)             
ZSedAlg         0.00000     x zeroth-order sedimentation flux algae                  (gC/m2/d)           
{p['fall_rate']:10}      0.00000     x sedimentation velocity algae type 04                   (m/d)               
Tau            0.100000     x total bottom shear stress                              (N/m2)              
TaucS          0.000000     x critical shear stress for sedimentation algae          (N/m2)              
Depth          -999.000     x depth of segment                                       (m)                 
DELT           -999.000     x timestep for processes                                 (d)                 
MinDepth       0.100000     x minimum waterdepth for sedimentation/resuspension      (m)                 
         1; # input items for exchanges
{p['fall_rate']:10}      0.00000       sedimentation velocity algae type 04                   (m/d)               
         2; # output items for segments
PSed{suffix:2}                      x sedimentation probability <0-1> algae type 04          (-)                 
fSed{suffix:2}                      x sedimentation flux algae type 04                       (gC/m2/d)           
         1; # output items for exchanges
VxFall{suffix}                     x sedimentation velocity algae type 04                   (m/s)               
         1; # fluxes
dFall{suffix}                      x sedimentation flux algae type 04                       (gC/m3/d)           
         0; # stoichiometry lines
         0; # stoichiometry lines dispersion arrays
         1; # stoichiometry lines velocity arrays
{p['sub']:10}  VxFall{suffix:4}     1.00000
END
"""
        return process

    def custom_DynDen(self,**kw):
        idx=self.copy_count['DynDen']
        self.copy_count['DynDen']+=1
        if self.copy_count['DynDyn']>1:
            raise Exception("custom_DynDen can only be used once")
        
        p=dict(name="DynDen")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        # With SW_Uitz=0.0:
        # SD = PAConstant / ExtVl
        process=f"""
{p['name']:10}                    Reuse Secchi depth for denit rate
SECCHI    ; module name
123       ; TRswitch
        22; # input items for segments
InvDenRate     -999.000     x POC                                                    (1/m)               
IM1             0.00000       inorganic matter (IM1)                                 (gDM/m3)            
IM2             0.00000       inorganic matter (IM2)                                 (gDM/m3)            
IM3             0.00000       inorganic matter (IM3)                                 (gDM/m3)            
POC1            0.00000       POC1 (fast decomposing fraction)                       (gC/m3)             
POC2            0.00000       POC2 (medium decomposing fraction)                     (gC/m3)             
POC3            0.00000       POC3 (slow decomposing fraction)                       (gC/m3)             
POC4            0.00000       POC4 (particulate refractory fraction)                 (gC/m3)             
ExtVlODS        0.00000       VL extinction by DOC                                   (1/m)               
Chlfa           0.00000       Chlorophyll-a concentration                            (mg/m3)             
SW_Uitz         0.00000       Extinction by Uitzicht On (1) or Off (0)               (-)                 
UitZDEPT1       1.20000       Z1 (depth)                                             (m)                 
UitZDEPT2       1.00000       Z2 (depth)                                             (m)                 
UitZCORCH       2.50000       CORa correction factor                                 (-)                 
UitZC_DET      0.260000E-01   C3 coeff. absorption ash weight & detritus             (-)                 
UitZC_GL1      0.730000       C1 coeff. absorption ash weight & detritus             (-)                 
UitZC_GL2       1.00000       C2 coeff. absorption ash weight & detritus             (-)                 
UitZHELHM      0.140000E-01   Hel_h constant                                         (1/nm)              
UitZTAU         7.80000       Tau constant calculation transparency                  (-)                 
UitZangle       30.0000       Angle of incidence solar radiation                     (degrees)           
DMCFDetC        2.50000       DM:C ratio DetC                                        (gDM/gC)            
DetCS1          1.70000     x Poole-Atkins constant                                  (-)                 
         0; # input items for exchanges
         1; # output items for segments
RcDenSed                    x 1st order denit m/d                                    (m)                 
         0; # output items for exchanges
         1; # fluxes
dDumDynDen                  x dummy flux to access Secchi                            (-)                 
         2; # stoichiometry lines
IM1         dDumDynDen     0.00000
POC1        dDumDynDen     0.00000
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)

    def custom_TFDiatAlt(self,**kw):
        idx=self.copy_count['TFDiatAlt']
        self.copy_count['TFDiatAlt']+=1
        if self.copy_count['TFDiatAlt']>1:
            raise Exception("custom_TFDiatAlt' can only be used once")
        
        p=dict(name="TFDiatAlt")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Use library developed from TFALGALT.f for asymmetric double exponential temperature correction for algal growth rate  
TFALGALT  ; module name
123       ; TRswitch
11        ; # input items for segments
Temp            20.000      x Water Temperature                                      (oC)               
TCGLD           4.0000      x lower temp limit for growth processes diatoms/greens   (oC)            
TCGUD           35.0000     x upper temp limit for growth processes diatoms/greens   (oC)          
TCGOD           20.0000     x optimal temp for growth processes diatoms/greens       (oC)            
TCGO_LD         18.0000     x lower range of optimal temp for growth processes       (oC)            
TCGO_UD         22.0000     x upper range of optimal temp for growth processes       (oC)             
K1D             0.01000     x fraction of growth rate at TCGL                        (-)   
K2D             0.95000     x fraction of growth rate at TCGO_L                      (-)    
K3D             0.95000     x fraction of growth rate at TCGO_U                      (-) 
K4D             0.01000     x fraction of growth rate at TCGU                        (-)
TcDecDiat       1.07000     x temp. coeff. for respiration and mortality Greens      (-)            
         0; # input items for exchanges
         2; # output items for segments
TFGroDiat                   x temperature function growth Greens <0-1>               (-)
TFMrtDiat                   x temperature function mortality Greens <0-1>            (-)                
         0; # output items for exchanges
         0; # fluxes
         0; # stoichiometry lines         
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)     

    def custom_TFGreenAlt(self,**kw):
        idx=self.copy_count['TFGreenAlt']
        self.copy_count['TFGreenAlt']+=1
        if self.copy_count['TFGreenAlt']>1:
            raise Exception("custom_TFGreenAlt can only be used once")
        
        p=dict(name="TFGreenAlt")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Use library developed from TFALGALT.f for asymmetric double exponential temperature correction for algal growth rate  
TFALGALT  ; module name
123       ; TRswitch
11        ; # input items for segments
Temp            20.000      x Water Temperature                                      (oC)               
TCGLG           4.0000      x lower temp limit for growth processes diatoms/greens   (oC)            
TCGUG           35.0000     x upper temp limit for growth processes diatoms/greens   (oC)          
TCGOG           20.0000     x optimal temp for growth processes diatoms/greens       (oC)            
TCGO_LG         18.0000     x lower range of optimal temp for growth processes       (oC)            
TCGO_UG         22.0000     x upper range of optimal temp for growth processes       (oC)             
K1G             0.05000     x fraction of growth rate at TCGL                        (-)   
K2G             0.95000     x fraction of growth rate at TCGO_L                      (-)    
K3G             0.95000     x fraction of growth rate at TCGO_L                      (-) 
K4G             0.95000     x fraction of growth rate at TCGO_L                      (-)
TcDecGreen      1.07000     x temp. coeff. for respiration and mortality Greens      (-)            
         0; # input items for exchanges
         2; # output items for segments
TFGroGreen                  x temperature function growth Greens <0-1>               (-)
TFMrtGreen                  x temperature function mortality Greens <0-1>            (-)                
         0; # output items for exchanges
         0; # fluxes
         0; # stoichiometry lines         
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)     
        
    def custom_sedmoddet1(self,**kw):
        idx=self.copy_count['sedmoddet1']
        self.copy_count['sedmoddet1']+=1
        if self.copy_count['sedmoddet1']>1:
            raise Exception("custom_sedmoddet1 can only be used once")
        
        p=dict(name="sedmoddet1")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Mineralization of Det1 substances
sedmod    ; module name
123       ; TRswitch
34        ; # input items for segments
DetCS1          -999.0      x Detrital carbon concentration                          (gC/m^2) 
DetNS1          -999.0      x Detrital nitrogen concentration                        (gC/m^2) 
DetPS1          -999.0      x Detrital phosphporus concentration                     (gC/m^2)  
OXY             -999.0      x oxygen concentration                                   (gO2/m^3)
NO3             -999.0      x nitrate concentration                                  (gN/m^3)
Temp            15.0000     x ambient water temperature                              (oC)
Depth           -999.0      x depth of segment                                       (m)
rsoptd1         0.00000     x option for same or sep rates for C,N,P, Si - 0 = same  (-)
rsdet1c         0.03000     x first order mineralization rate C (used for all if opt = 0) (1/d)
rsdet1n         0.03000     x first order mineralization rate                        (1/d)
rsdet1p         0.03000     x first order mineralization rate                        (1/d)
tcdet1          1.09000     x temp. correction factor for mineralization             (-)
frmindod1       0.80000     x fraction mineralization by DO                          (-) 
dbguntd1        0.00000     x option to print debug output (0 = no output, or specify 4 digit unit number > 0)           (-)
flgqmrt         0.00000     x flag for simulating quadratic mortality (0 = not used) (-)       
Diat            -999.0      x diatoms concentration                                  (gC/m^3)
Green           -999.0      x Greens concentration                                   (gC/m^3) 
kqmrtdiat       0.01000     x quadratic mortality rate for diatoms                   (m^3/gC.d)       
tmqdiat         1.07        x temperature correction factor for diatoms mortality    (-)       
frauqdiat       0.15        x autolysis fraction of diat mortality                   (-)
frrfqdiat       0.15        x fraction of diat mortality routed to refractory pool   (-) 
MinDiat         0.00000     x Minimum conc of diatoms to stop mortality              (g/m^3)  
NCRatDiat       0.16000     x N:C ratio Diatoms                                      (gN/gC)             
PCRatDiat       0.02000     x P:C ratio Diatoms                                      (gP/gC)             
SCRatDiat       0.49000     x Si:C ratio Diatoms                                     (gSi/gC)      
kqmrtgrn        0.01000     x quadratic mortality rate for greens                    (m^3/gC.d)
tmqgrn          1.07        x temperature correction factor for greens mortality     (-)             
frauqgrn        0.15        x autolysis fraction of mortality routed for diat        (-)
frrfqgrn        0.15        x fraction of diat mortality routed to refractory pool   (-)
MinGreen        0.00000     x Minimum conc of greens  to stop mortality              (g/m^3)
NCRatGreen      0.16000     x N:C ratio Greens                                       (gN/gC)             
PCRatGreen      0.02000     x P:C ratio Greens                                       (gP/gC) 
SCRatGreen      0.00000     x Si:C ratio Greens                                     (gSi/gC)        
optdbgq         0.00000     x option to print debug output for quadratic mortality (0 = no output)  (-)
         0; # input items for exchanges
         5; # output items for segments
fdetcs1min                  x sediment mineralization of C                           (gC/m^2/d) 
fdetns1min                  x sediment mineralization of N                           (gN/m^2/d)
fdetps1min                  x sediment mineralization of P                           (gP/m^2/d)
fdetcs1sod                  x sediment oxygen demand                                 (gO2/m^2/d)
fdetcs1nit                  x denitrification                                        (gN/m^2/d)
         0; # output items for exchanges
        29; # fluxes
ddetcs1min                  x mineralization                                         (gC/m^3/d)
ddetns1min                  x sediment mineralization of N                           (gN/m^3/d)
ddetps1min                  x sediment mineralization of P                           (gP/m^3/d)
ddetcs1sod                  x SOD                                                    (gO2/m^3/d)
ddetcs1nit                  x denitrification                                        (gN/m^3/d)
dqmtdiat                    x vol. quadratic mortality rate diatoms                  (gC/m^3/d)
dqmtdLC                     x vol. quadratic mortality rate diatoms C labile         (gC/m^3/d)
dqmtdRC                     x vol. quadratic mortality rate diatoms C refractory     (gC/m^3/d)      
dqmtdLN                     x vol. quadratic mortality rate diatoms N labile         (gN/m^3/d)
dqmtdRN                     x vol. quadratic mortality rate diatoms N refractory     (gN/m^3/d)  
dqmtdNH4                    x vol. quadratic mortality rate diatoms NH4 autolysis    (gN/m^3/d)      
dqmtdLP                     x vol. quadratic mortality rate diatoms P labile         (gP/m^3/d)
dqmtdRP                     x vol. quadratic mortality rate diatoms P refractory     (gP/m^3/d)    
dqmtdPO4                    x vol. quadratic mortality rate diatoms PO4 autolysis    (gP/m^3/d)      
dqmtdLSi                    x vol. quadratic mortality rate diatoms Si labile        (gSi/m^3/d) 
dqmtdRSi                    x vol. quadratic mortality rate diatoms Si refractory    (gSi/m^3/d)
dqmtdSi                     x vol. quadratic mortality rate diatoms Si autolysis     (gSi/m^3/d) 
dqmtgree                    x vol. quadratic mortality rate greens                   (gC/m^3/d)      
dqmtgLC                     x vol. quadratic mortality rate greens C labile          (gC/m^3/d)   
dqmtgRC                     x vol. quadratic mortality rate greens C refractory      (gC/m^3/d)      
dqmtgLN                     x vol. quadratic mortality rate greens N labile          (gN/m^3/d)
dqmtgRN                     x vol. quadratic mortality rate greens N refractory      (gN/m^3/d)   
dqmtgNH4                    x vol. quadratic mortality rate greens NH4 autolysis     (gN/m^3/d)       
dqmtgLP                     x vol. quadratic mortality rate greens P labile          (gP/m^3/d)
dqmtgRP                     x vol. quadratic mortality rate greens P refractory      (gP/m^3/d)      
dqmtgPO4                    x vol. quadratic mortality rate greens PO4 autolysis     (gP/m^3/d)      
dqmtgLSi                    x vol. quadratic mortality rate greens Si labile         (gSi/m^3/d)
dqmtgRSi                    x vol. quadratic mortality rate greens Si refractory     (gSi/m^3/d)
dqmtgSi                     x vol. quadratic mortality rate greens Si autolysis      (gSi/m^3/d)
        45; # stoichiometry lines      
DetCS1      ddetcs1min    -1.00000
TIC         ddetcs1min     1.00000
H2O         ddetcs1min     1.50000
DetNS1      ddetns1min    -1.00000
NH4         ddetns1min     1.00000
H+          ddetns1min    -0.07100
ALKA        ddetns1min     4.35700
DetPS1      ddetps1min    -1.00000
PO4         ddetps1min     1.00000
H+          ddetps1min     0.03200
ALKA        ddetps1min    -1.96800
OXY         ddetcs1sod    -1.00000
NO3         ddetcs1nit    -1.00000
Diat        dqmtdiat      -1.00000
POC1        dqmtdLC        1.00000
POC2        dqmtdRC        1.00000
PON1        dqmtdLN        1.00000
PON2        dqmtdRN        1.00000
NH4         dqmtdNH4       1.00000
H+          dqmtdNH4      -0.07100
ALKA        dqmtdNH4       4.35700
POP1        dqmtdLP        1.00000
POP2        dqmtdRP        1.00000 
PO4         dqmtdPO4       1.00000
H+          dqmtdPO4       0.03200
ALKA        dqmtdPO4      -1.96800
Opal        dqmtdLSi       1.00000
Opal        dqmtdRSi       1.00000
Si          dqmtdSi        1.00000      
Green       dqmtgree      -1.00000   
POC1        dqmtgLC        1.00000
POC2        dqmtgRC        1.00000
PON1        dqmtgLN        1.00000
PON2        dqmtgRN        1.00000
NH4         dqmtgNH4       1.00000
H+          dqmtgNH4      -0.07100
ALKA        dqmtgNH4       4.35700        
POP1        dqmtgLP        1.00000
POP2        dqmtgRP        1.00000
PO4         dqmtgPO4       1.00000
H+          dqmtgPO4       0.03200
ALKA        dqmtgPO4      -1.96800
Opal        dqmtgLSi       1.00000
Opal        dqmtgRSi       1.00000
Si          dqmtgSi        1.00000
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)  

    def custom_sedmoddet2(self,**kw):
        idx=self.copy_count['sedmoddet2']
        self.copy_count['sedmoddet2']+=1
        if self.copy_count['sedmoddet2']>1:
            raise Exception("custom_sedmoddet2 can only be used once")
        
        p=dict(name="sedmoddet2")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Mineralization of Det2 substances
sedmod    ; module name
123       ; TRswitch
34        ; # input items for segments
DetCS2          -999.0      x Detrital carbon concentration                          (gC/m^2) 
DetNS2          -999.0      x Detrital nitrogen concentration                        (gC/m^2) 
DetPS2          -999.0      x Detrital phosphporus concentration                     (gC/m^2) 
OXY             -999.0      x oxygen concentration                                   (gO2/m^3)
NO3             -999.0      x nitrate concentration                                  (gN/m^3)
Temp            15.0000     x ambient water temperature                              (oC)
Depth           -999.0      x depth of segment                                       (m)
rsoptd2         0.00000     x option for same or sep rates for C,N,P, Si - 0 = same  (-)
rsdet2c         0.03000     x first order mineralization rate C (used for all if opt = 0) (1/d)
rsdet2n         0.03000     x first order mineralization rate                        (1/d)
rsdet2p         0.03000     x first order mineralization rate                        (1/d)
tcdet2          1.09000     x temp. correction factor for mineralization             (-)
frmindod2       0.80000     x fraction mineralization by DO                          (-) 
dbguntd2        0.00000     x option to print debug output (0 = no output, or specify 4 digit unit number > 0)           (-)
flgqmrt2        0.00000     x flag for simulating quadratic mortality (0 = not used) (-)       
Diat            -999.0      x diatoms concentration                                  (gC/m^3)
Green           -999.0      x Greens concentration                                   (gC/m^3) 
kqmrtdiat       0.01000     x quadratic mortality rate for diatoms                   (m^3/gC.d)       
tmqdiat         1.07        x temperature correction factor for diatoms mortality    (-)       
frauqdiat       0.15        x autolysis fraction of diat mortality                   (-)
frrfqdiat       0.15        x fraction of diat mortality routed to refractory pool   (-) 
MinDiat         0.00000     x Minimum conc of diatoms to stop mortality              (g/m^3)  
NCRatDiat       0.16000     x N:C ratio Diatoms                                      (gN/gC)             
PCRatDiat       0.02000     x P:C ratio Diatoms                                      (gP/gC)             
SCRatDiat       0.49000     x Si:C ratio Diatoms                                     (gSi/gC)      
kqmrtgrn        0.01000     x quadratic mortality rate for greens                    (m^3/gC.d)
tmqgrn          1.07        x temperature correction factor for greens mortality     (-)             
frauqgrn        0.15        x autolysis fraction of mortality routed for diat        (-)
frrfqgrn        0.15        x fraction of diat mortality routed to refractory pool   (-)
MinGreen        0.00000     x Minimum conc of greens  to stop mortality              (g/m^3)
NCRatGreen      0.16000     x N:C ratio Greens                                       (gN/gC)             
PCRatGreen      0.02000     x P:C ratio Greens                                       (gP/gC) 
SCRatGreen      0.00000     x Si:C ratio Greens                                     (gSi/gC)        
optdbgq2        0.00000     x option to print debug output for quadratic mortality (0 = no output)  (-)
         0; # input items for exchanges
         5; # output items for segments
fdetcs2min                  x sediment mineralization of C                           (gC/m^2/d) 
fdetns2min                  x sediment mineralization of N                           (gN/m^2/d)
fdetps2min                  x sediment mineralization of P                           (gP/m^2/d)
fdetcs2sod                  x sediment oxygen demand                                 (gO2/m^2/d)
fdetcs2nit                  x denitrification                                        (gN/m^2/d)
         0; # output items for exchanges
        29; # fluxes
ddetcs2min                  x mineralization                                         (gC/m^3/d)
ddetns2min                  x sediment mineralization of N                           (gN/m^3/d)
ddetps2min                  x sediment mineralization of P                           (gP/m^3/d)
ddetcs2sod                  x SOD                                                    (gO2/m^3/d)
ddetcs2nit                  x denitrification                                        (gN/m^3/d)
ddumoo101                  x vol. quadratic mortality rate diatoms                  (gC/m^3/d)
ddumoo102                  x vol. quadratic mortality rate diatoms C labile         (gC/m^3/d)
ddumoo103                  x vol. quadratic mortality rate diatoms C refractory     (gC/m^3/d)      
ddumoo104                  x vol. quadratic mortality rate diatoms N labile         (gN/m^3/d)
ddumoo105                  x vol. quadratic mortality rate diatoms N refractory     (gN/m^3/d)  
ddumoo106                  x vol. quadratic mortality rate diatoms NH4 autolysis    (gN/m^3/d)      
ddumoo107                  x vol. quadratic mortality rate diatoms P labile         (gP/m^3/d)
ddumoo108                  x vol. quadratic mortality rate diatoms P refractory     (gP/m^3/d)    
ddumoo109                  x vol. quadratic mortality rate diatoms PO4 autolysis    (gP/m^3/d)      
ddumoo110                  x vol. quadratic mortality rate diatoms Si labile        (gSi/m^3/d) 
ddumoo111                  x vol. quadratic mortality rate diatoms Si refractory    (gSi/m^3/d)
ddumoo112                  x vol. quadratic mortality rate diatoms Si autolysis     (gSi/m^3/d) 
ddumoo113                  x vol. quadratic mortality rate greens                   (gC/m^3/d)      
ddumoo114                  x vol. quadratic mortality rate greens C labile          (gC/m^3/d)   
ddumoo115                  x vol. quadratic mortality rate greens C refractory      (gC/m^3/d)      
ddumoo116                  x vol. quadratic mortality rate greens N labile          (gN/m^3/d)
ddumoo117                  x vol. quadratic mortality rate greens N refractory      (gN/m^3/d)   
ddumoo118                  x vol. quadratic mortality rate greens NH4 autolysis     (gN/m^3/d)       
ddumoo119                  x vol. quadratic mortality rate greens P labile          (gP/m^3/d)
ddumoo120                  x vol. quadratic mortality rate greens P refractory      (gP/m^3/d)      
ddumoo121                  x vol. quadratic mortality rate greens PO4 autolysis     (gP/m^3/d)      
ddumoo122                  x vol. quadratic mortality rate greens Si labile         (gSi/m^3/d)
ddumoo123                  x vol. quadratic mortality rate greens Si refractory     (gSi/m^3/d)
ddumoo124                  x vol. quadratic mortality rate greens Si autolysis      (gSi/m^3/d)
        13; # stoichiometry lines      
DetCS2      ddetcs2min    -1.00000
TIC         ddetcs2min     1.00000
H2O         ddetcs2min     1.50000
DetNS2      ddetns2min    -1.00000
NH4         ddetns2min     1.00000
H+          ddetns2min    -0.07100
ALKA        ddetns2min     4.35700
DetPS2      ddetps2min    -1.00000
PO4         ddetps2min     1.00000
H+          ddetps2min     0.03200
ALKA        ddetps2min    -1.96800
OXY         ddetcs2sod    -1.00000
NO3         ddetcs2nit    -1.00000
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)

    def custom_sedmodoo1(self,**kw):
        idx=self.copy_count['sedmodoo1']
        self.copy_count['sedmodoo1']+=1
        if self.copy_count['sedmodoo1']>1:
            raise Exception("custom_sedmodoo1 can only be used once")
        
        p=dict(name="sedmodoo1")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Mineralization of OOX1 substances
sedmod    ; module name
123       ; TRswitch
34        ; # input items for segments
OOCS1           -999.0      x Detrital carbon concentration                          (gC/m^2) 
OONS1           -999.0      x Detrital nitrogen concentration                        (gN/m^2) 
OOPS1           -999.0      x Detrital phosphporus concentration                     (gP/m^2) 
OXY             -999.0      x oxygen concentration                                   (gO2/m^3)
NO3             -999.0      x nitrate concentration                                  (gN/m^3)
Temp            15.0000     x ambient water temperature                              (oC)
Depth           -999.0      x depth of segment                                       (m)
rsopto1         0.00000     x option for same or sep rates for C,N,P, Si - 0 = same  (-)
rsoo1c          0.00010     x first order mineralization rate C (used for all if opt = 0) (1/d)
rsoo1n          0.00010     x first order mineralization rate                        (1/d)
rsoo1p          0.00010     x first order mineralization rate                        (1/d)
tcoox1          1.09000     x temp. correction factor for mineralization             (-)
frmindoo1       0.80000     x fraction mineralization by DO                          (-) 
dbguntoo1       0.00000     x option to print debug output (0 = no output, or specify 4 digit unit number > 0)           (-)
flgqmrt3        0.00000     x flag for simulating quadratic mortality (0 = not used) (-)       
Diat            -999.0      x diatoms concentration                                  (gC/m^3)
Green           -999.0      x Greens concentration                                   (gC/m^3) 
kqmrtdiat       0.01000     x quadratic mortality rate for diatoms                   (m^3/gC.d)       
tmqdiat         1.07        x temperature correction factor for diatoms mortality    (-)       
frauqdiat       0.15        x autolysis fraction of diat mortality                   (-)
frrfqdiat       0.15        x fraction of diat mortality routed to refractory pool   (-) 
MinDiat         0.00000     x Minimum conc of diatoms to stop mortality              (g/m^3)  
NCRatDiat       0.16000     x N:C ratio Diatoms                                      (gN/gC)             
PCRatDiat       0.02000     x P:C ratio Diatoms                                      (gP/gC)             
SCRatDiat       0.49000     x Si:C ratio Diatoms                                     (gSi/gC)      
kqmrtgrn        0.01000     x quadratic mortality rate for greens                    (m^3/gC.d)
tmqgrn          1.07        x temperature correction factor for greens mortality     (-)             
frauqgrn        0.15        x autolysis fraction of mortality routed for diat        (-)
frrfqgrn        0.15        x fraction of diat mortality routed to refractory pool   (-)
MinGreen        0.00000     x Minimum conc of greens  to stop mortality              (g/m^3)
NCRatGreen      0.16000     x N:C ratio Greens                                       (gN/gC)             
PCRatGreen      0.02000     x P:C ratio Greens                                       (gP/gC) 
SCRatGreen      0.00000     x Si:C ratio Greens                                     (gSi/gC)        
optdbgq3        0.00000     x option to print debug output for quadratic mortality (0 = no output)  (-)
         0; # input items for exchanges
         5; # output items for segments
foocs1min                   x sediment mineralization of C                           (gC/m^2/d) 
foons1min                   x sediment mineralization of N                           (gN/m^2/d)
foops1min                   x sediment mineralization of P                           (gP/m^2/d)
foocs1sod                   x sediment oxygen demand                                 (gO2/m^2/d)
foocs1nit                   x denitrification                                        (gN/m^2/d)
         0; # output items for exchanges
        29; # fluxes
doocs1min                   x mineralization                                         (gC/m^3/d)
doons1min                   x sediment mineralization of N                           (gN/m^3/d)
doops1min                   x sediment mineralization of P                           (gP/m^3/d)
doocs1sod                   x SOD                                                    (gO2/m^3/d)
doocs1nit                   x denitrification                                        (gN/m^3/d)
ddumoo101                   x vol. quadratic mortality rate diatoms                  (gC/m^3/d)
ddumoo102                   x vol. quadratic mortality rate diatoms C labile         (gC/m^3/d)
ddumoo103                   x vol. quadratic mortality rate diatoms C refractory     (gC/m^3/d)      
ddumoo104                   x vol. quadratic mortality rate diatoms N labile         (gN/m^3/d)
ddumoo105                   x vol. quadratic mortality rate diatoms N refractory     (gN/m^3/d)  
ddumoo106                   x vol. quadratic mortality rate diatoms NH4 autolysis    (gN/m^3/d)      
ddumoo107                   x vol. quadratic mortality rate diatoms P labile         (gP/m^3/d)
ddumoo108                   x vol. quadratic mortality rate diatoms P refractory     (gP/m^3/d)    
ddumoo109                   x vol. quadratic mortality rate diatoms PO4 autolysis    (gP/m^3/d)      
ddumoo110                   x vol. quadratic mortality rate diatoms Si labile        (gSi/m^3/d) 
ddumoo111                   x vol. quadratic mortality rate diatoms Si refractory    (gSi/m^3/d)
ddumoo112                   x vol. quadratic mortality rate diatoms Si autolysis     (gSi/m^3/d) 
ddumoo113                   x vol. quadratic mortality rate greens                   (gC/m^3/d)      
ddumoo114                   x vol. quadratic mortality rate greens C labile          (gC/m^3/d)   
ddumoo115                   x vol. quadratic mortality rate greens C refractory      (gC/m^3/d)      
ddumoo116                   x vol. quadratic mortality rate greens N labile          (gN/m^3/d)
ddumoo117                   x vol. quadratic mortality rate greens N refractory      (gN/m^3/d)   
ddumoo118                   x vol. quadratic mortality rate greens NH4 autolysis     (gN/m^3/d)       
ddumoo119                   x vol. quadratic mortality rate greens P labile          (gP/m^3/d)
ddumoo120                   x vol. quadratic mortality rate greens P refractory      (gP/m^3/d)      
ddumoo121                   x vol. quadratic mortality rate greens PO4 autolysis     (gP/m^3/d)      
ddumoo122                   x vol. quadratic mortality rate greens Si labile         (gSi/m^3/d)
ddumoo123                   x vol. quadratic mortality rate greens Si refractory     (gSi/m^3/d)
ddumoo124                   x vol. quadratic mortality rate greens Si autolysis      (gSi/m^3/d)
        13; # stoichiometry lines      
OOCS1       doocs1min     -1.00000
TIC         doocs1min      1.00000
H2O         doocs1min      1.50000
OONS1       doons1min     -1.00000
NH4         doons1min      1.00000
H+          doons1min     -0.07100
ALKA        doons1min      4.35700
OOPS1       doops1min     -1.00000
PO4         doops1min      1.00000
H+          doops1min      0.03200
ALKA        doops1min     -1.96800
OXY         doocs1sod     -1.00000
NO3         doocs1nit     -1.00000
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)

    def custom_sedmodoo2(self,**kw):
        idx=self.copy_count['sedmodoo2']
        self.copy_count['sedmodoo2']+=1
        if self.copy_count['sedmodoo2']>1:
            raise Exception("custom_sedmodoo2 can only be used once")
        
        p=dict(name="sedmodoo2")
        p.update(kw)

        # Presumably want to enable it, though could be optional
        self.custom_add_process(p['name'])
        
        process=f"""
{p['name']:10}                    Mineralization of OOX2 substances
sedmod    ; module name
123       ; TRswitch
34        ; # input items for segments
OOCS2           -999.0      x Detrital carbon concentration                          (gC/m^2) 
OONS2           -999.0      x Detrital nitrogen concentration                        (gN/m^2) 
OOPS2           -999.0      x Detrital phosphporus concentration                     (gP/m^2) 
OXY             -999.0      x oxygen concentration                                   (gO2/m^3)
NO3             -999.0      x nitrate concentration                                  (gN/m^3)
Temp            15.0000     x ambient water temperature                              (oC)
Depth           -999.0      x depth of segment                                       (m)
rsopto2         0.00000     x option for same or sep rates for C,N,P, Si - 0 = same  (-)
rsoo2c          0.00010     x first order mineralization rate C (used for all if opt = 0) (1/d)
rsoo2n          0.00010     x first order mineralization rate                        (1/d)
rsoo2p          0.00010     x first order mineralization rate                        (1/d)
tcoox2          1.09000     x temp. correction factor for mineralization             (-)
frmindoo2       0.80000     x fraction mineralization by DO                          (-) 
dbguntoo2       0.00000     x option to print debug output (0 = no output, or specify 4 digit unit number > 0)           (-)
flgqmrt4        0.00000     x flag for simulating quadratic mortality (0 = not used) (-)       
Diat            -999.0      x diatoms concentration                                  (gC/m^3)
Green           -999.0      x Greens concentration                                   (gC/m^3) 
kqmrtdiat       0.01000     x quadratic mortality rate for diatoms                   (m^3/gC.d)       
tmqdiat         1.07        x temperature correction factor for diatoms mortality    (-)       
frauqdiat       0.15        x autolysis fraction of diat mortality                   (-)
frrfqdiat       0.15        x fraction of diat mortality routed to refractory pool   (-) 
MinDiat         0.00000     x Minimum conc of diatoms to stop mortality              (g/m^3)  
NCRatDiat       0.16000     x N:C ratio Diatoms                                      (gN/gC)             
PCRatDiat       0.02000     x P:C ratio Diatoms                                      (gP/gC)             
SCRatDiat       0.49000     x Si:C ratio Diatoms                                     (gSi/gC)      
kqmrtgrn        0.01000     x quadratic mortality rate for greens                    (m^3/gC.d)
tmqgrn          1.07        x temperature correction factor for greens mortality     (-)             
frauqgrn        0.15        x autolysis fraction of mortality routed for diat        (-)
frrfqgrn        0.15        x fraction of diat mortality routed to refractory pool   (-)
MinGreen        0.00000     x Minimum conc of greens  to stop mortality              (g/m^3)
NCRatGreen      0.16000     x N:C ratio Greens                                       (gN/gC)             
PCRatGreen      0.02000     x P:C ratio Greens                                       (gP/gC) 
SCRatGreen      0.00000     x Si:C ratio Greens                                     (gSi/gC)        
optdbgq4        0.00000     x option to print debug output for quadratic mortality (0 = no output)  (-)
         0; # input items for exchanges
         5; # output items for segments
foocs2min                   x sediment mineralization of C                           (gC/m^2/d) 
foons2min                   x sediment mineralization of N                           (gN/m^2/d)
foops2min                   x sediment mineralization of P                           (gP/m^2/d)
foocs2sod                   x sediment oxygen demand                                 (gO2/m^2/d)
foocs2nit                   x denitrification                                        (gN/m^2/d)
         0; # output items for exchanges
        29; # fluxes
doocs2min                   x mineralization                                         (gC/m^3/d)
doons2min                   x sediment mineralization of N                           (gN/m^3/d)
doops2min                   x sediment mineralization of P                           (gP/m^3/d)
doocs2sod                   x SOD                                                    (gO2/m^3/d)
doocs2nit                   x denitrification                                        (gN/m^3/d)
ddumoo201                   x vol. quadratic mortality rate diatoms                  (gC/m^3/d)
ddumoo202                   x vol. quadratic mortality rate diatoms C labile         (gC/m^3/d)
ddumoo203                   x vol. quadratic mortality rate diatoms C refractory     (gC/m^3/d)      
ddumoo204                   x vol. quadratic mortality rate diatoms N labile         (gN/m^3/d)
ddumoo205                   x vol. quadratic mortality rate diatoms N refractory     (gN/m^3/d)  
ddumoo206                   x vol. quadratic mortality rate diatoms NH4 autolysis    (gN/m^3/d)      
ddumoo207                   x vol. quadratic mortality rate diatoms P labile         (gP/m^3/d)
ddumoo208                   x vol. quadratic mortality rate diatoms P refractory     (gP/m^3/d)    
ddumoo209                   x vol. quadratic mortality rate diatoms PO4 autolysis    (gP/m^3/d)      
ddumoo210                   x vol. quadratic mortality rate diatoms Si labile        (gSi/m^3/d) 
ddumoo211                   x vol. quadratic mortality rate diatoms Si refractory    (gSi/m^3/d)
ddumoo212                   x vol. quadratic mortality rate diatoms Si autolysis     (gSi/m^3/d) 
ddumoo213                   x vol. quadratic mortality rate greens                   (gC/m^3/d)      
ddumoo214                   x vol. quadratic mortality rate greens C labile          (gC/m^3/d)   
ddumoo215                   x vol. quadratic mortality rate greens C refractory      (gC/m^3/d)      
ddumoo216                   x vol. quadratic mortality rate greens N labile          (gN/m^3/d)
ddumoo217                   x vol. quadratic mortality rate greens N refractory      (gN/m^3/d)   
ddumoo218                   x vol. quadratic mortality rate greens NH4 autolysis     (gN/m^3/d)       
ddumoo219                   x vol. quadratic mortality rate greens P labile          (gP/m^3/d)
ddumoo220                   x vol. quadratic mortality rate greens P refractory      (gP/m^3/d)      
ddumoo221                   x vol. quadratic mortality rate greens PO4 autolysis     (gP/m^3/d)      
ddumoo222                   x vol. quadratic mortality rate greens Si labile         (gSi/m^3/d)
ddumoo223                   x vol. quadratic mortality rate greens Si refractory     (gSi/m^3/d)
ddumoo224                   x vol. quadratic mortality rate greens Si autolysis      (gSi/m^3/d)
        13; # stoichiometry lines      
OOCS2       doocs2min     -1.00000
TIC         doocs2min      1.00000
H2O         doocs2min      1.50000
OONS2       doons2min     -1.00000
NH4         doons2min      1.00000
H+          doons2min     -0.07100
ALKA        doons2min      4.35700
OOPS2       doops2min     -1.00000
PO4         doops2min      1.00000
H+          doops2min      0.03200
ALKA        doops2min     -1.96800
OXY         doocs2sod     -1.00000
NO3         doocs2nit     -1.00000
         0; # stoichiometry lines dispersion arrays
         0; # stoichiometry lines velocity arrays
END
"""
        self.custom_procs.append(process)