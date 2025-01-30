import shutil
import sys
import os
from collections import defaultdict
from stompy import utils
from . import dflow_model


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
        if isinstance(self,dflow_model.DFlowModel):
            self.dwaq.add_process(p['name'])
        else: # WaqModel or related
            self.add_process(p['name'])
        
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
        self.dwaq.add_process(p['name'])

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
        if isinstance(self,dflow_model.DFlowModel):
            self.dwaq.add_process(p['name'])
        else: # WaqModel or related
            self.add_process(p['name'])
        
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

