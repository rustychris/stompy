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

        print("First call to waqpb_export")
        sys.stdout.flush()
        output = utils.call_with_path(self.waqpbexport,self.proc_table_dst_dir).decode('latin1') 
        print("Suppressed output")
        # That says Normal end, but then goes on to
        # make proces.asc
        sys.stdout.flush()
        print("First call to waqpb_export DONE")
        sys.stdout.flush()
        # => proc_def.* and procesm.asc in proc_table_dst_dir

        # First line of proces_asc includes a process count. Rewrite that line, shove in
        # our new process, and copy the rest of the file.

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
                    
        print("Calling waqpb_import")
        sys.stdout.flush()
        # Pretty sure this is the one that fails.
        output = utils.call_with_path(self.waqpbimport,self.proc_table_dst_dir).decode('latin1') 
        print("Suppressed output")
        sys.stdout.flush()
        print("Return from waqpb_import")
        sys.stdout.flush()
        print("Second call to waqpb_export")
        sys.stdout.flush()
        output = utils.call_with_path(self.waqpbexport,self.proc_table_dst_dir).decode('latin1') 
        print( "Suppressed output")
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
TcNit           1.00000     x ignored
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

# NOT YET IMPLEMENTED
#     def custom_settling(self,**kw):
#         proc_txt=self.settling(**kw)
#         self.custom_procs.append(proc_txt)

#      def settling(self, **kw):
#          idx=self.copy_count['SETTLING']
#          self.copy_count['SETTLING']+=1
#          
#          suffix=str(idx)
#          p=dict(name="SETTLE"+suffix,
#                 # FIX BELOW
#                 zero_order="ZAge"+suffix,
#                 partial="PartAge"+suffix,
#                 conc=f"Age{suffix}Conc",
#                 partial_default=1.0,
#                 age_conc=f"Age{suffix}AConc",
#                 flux="dAge"+suffix)
#          p.update(kw)
#  
#          # Presumably want to enable it, though could be optional
#          self.dwaq.add_process(p['name'])
#          
#          n_stoich=1
#          if conc_decay!=0.0:
#              n_stoich+=1
#  
#          # partial is a rate in d-1, defaults to 1.0.
#          # flux = partial * conc
#  
#          # All of this is to have a pair of tracers that
#          # integrate this:
#          
#          #     d conc / dt = -conc_decay*partial * conc
#          # d age_conc / dt =             partial * conc
#  
#          # partial defaults to 1.0.
#  
#          process=f"""
#  {p['name']:10}                    Reuse nitrification as age                         
#  NITRIF    ; module name. 
#  123       ; TRswitch
#          18; # input items for segments
#  {p['zero_order']:10}      0.00000     x zeroth-order flux          (g/m3/d)
#  {p['partial']   :10}      {p['partial_default']: 8g}     x set this to get partial age
#  RcNit20        0.100000     x ignored (b/c SWVnNit=0)
#  TcNit           1.00000     x ignored
#  OXY             10.0000     x ignored
#  KsAmNit        0.500000     x ignored
#  KsOxNit         1.00000     x ignored
#  Temp            15.0000     x ignored
#  CTNit           3.00000     x ignored
#  Rc0NitOx        0.00000     x ignored
#  COXNIT          1.00000     x ignored
#  Poros           1.00000     x volumetric porosity                            (-)
#  SWVnNit         0.00000     x switch for old (0), new (1), TEWOR (2) version (-)
#  {p['conc']    :10}     0.100000     x concentration tracer
#  OOXNIT          5.00000     x ignored
#  CFLNIT          0.00000     x ignored
#  CurvNit         0.00000     x ignored
#  DELT           -999.000     x timestep for processes                         (d)
#           0; # input items for exchanges
#           1; # output items for segments
#  O2FuncNT1                   x oxygen function for nitrification              (-)
#           0; # output items for exchanges
#           1; # fluxes
#  {p['flux']:10}                  x nitrification flux                       (gN/m3/d)
#           {n_stoich}; # stoichiometry lines. Could probably drop most of these.
#  {p['age_conc']:10}  {p['flux']:10}     1.00000
#  """
#          if conc_decay!=0.0:
#              process+=f"{p['conc']:10}  {p['flux']:10}    {-conc_decay:.5f}\n"
#          process+="""         0; # stoichiometry lines dispersion arrays
#           0; # stoichiometry lines velocity arrays
#  END
#  """
#          return process
    
