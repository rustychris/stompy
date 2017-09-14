from __future__ import print_function

import os
import shutil

import nose
import numpy as np
from delft import waq_scenario


# A basic water column setup with dynamo algae
Ndays=5
class ColumnHydro(waq_scenario.HydroStructured):
    # Hydro properties
    t_secs=np.arange(0,Ndays*86400,86400,dtype='i4') # daily output

    # HydroStructured properties
    num_x=num_y=1 # number of cells in horizontal directions
    num_z=5 # number of layers in vertical

    # Local properties
    dom_L=1000.0 # m size of analytical domain
    dom_W=1000.0 # m
    dom_D=5.0    # m

    # vertical coordinate system
    @property
    def z_levels(self):
        return np.linspace(0,self.dom_D,self.num_z+1)
    @property
    def z_centers(self):
        return 0.5*(self.z_levels[1:] + self.z_levels[:-1])

    def areas(self,t):
        """ areas of exchanges """
        A=np.zeros(self.num_exch,'f4')
        A[:]=self.dom_L*self.dom_W
        return A

    def flows(self,t):
        return np.zeros(self.num_exch,'f4') # closed box, no flows

    def vert_diffs(self,t):
        vdf=np.ones(self.num_seg,'f4')
        vdf[:]=1e-7  # basically nothing.
        return vdf

    @property
    def exchange_lengths(self):
        # write z exchanges
        z_exchanges=np.ones((self.num_exch_z,2),'f4')
        # unsure here - in the tut_fti_waq example, all vertical exchanges were 1.0 m.
        # maybe that's a lower bound?
        z_exchanges[:,:]=1.0
        return z_exchanges

    def volumes(self,t):
        V=np.zeros(self.seg_ids.shape,'f4')

        V[:,:,:]=self.dom_L*self.dom_W*np.diff(self.z_levels)
        return V.ravel()

    # these are written out as process parameters
    def planform_areas(self):
        a=np.ones(self.num_seg,'f4')
        a[:] = self.dom_L*self.dom_W
        return a
    def bottom_depth(self):
        d=np.ones(self.num_seg,'f4')
        d[:]=self.dom_D
        return d


PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial


class Bloombox(waq_scenario.Scenario):
    base_path=os.path.join(os.path.dirname(__file__),'test_waq')
    name="bloombox02-test"

    # this one work okay
    integration_option="15 NODISP-AT-BOUND LOWER-ORDER-AT-BOUND DISP-AT-NOFLOW"

    def init_substances(self):
        subs=super(Bloombox,self).init_substances()
        subs['Green']   =Sub(initial=IC(default=10.0))
        subs['ModTemp'] =Sub(initial=IC(default=15.0))
        subs['NO3']     =Sub(initial=IC(default=1.05))
        subs['NH4']     =Sub(initial=IC(default=0.17))
        subs['PO4']     =Sub(initial=IC(default=0.37))
        subs['IM1']     =Sub(initial=IC(default=0))
        return subs

    def init_parameters(self):
        params=super(Bloombox,self).init_parameters()
        params['SalM1Green']= PC(0.0) # g/kg
        params['SalM2Green']= PC(50.0) # g/kg
        params['OXY']= PC(10.0)  # mg/L
        params['ExtVlBak']= PC(0.07) # 1/m ?
        params['V0SedGreen']= PC(0.5) # m/d
        params['ACTIVE_VERTDISP']= PC(0.0)
        params['CLOSE_ERR']= PC(1.0)
        params['ScaleVDisp']= PC(1.0)
        params['MaxIter']= PC(100)
        params['Tolerance']= PC(1e-7)
        params['Iteration Report']= PC(0)
        params['ONLY_ACTIVE']= PC(1)
        params['ACTIVE_GroMrt_Gre']= PC(1)
        params['ACTIVE_PPrLim']= PC(1)
        params['ACTIVE_OXYMin']= PC(1)
        params['ACTIVE_Sed_Gre']= PC(1)
        params['ACTIVE_Daylength']= PC(1)
        params['ACTIVE_CalVS_Gree']= PC(1) # calculate sedimentation velocity of Green
        params['ACTIVE_CalTau']= PC(1) # calculate bed stress
        params['ACTIVE_DynDepth']= PC(1)
        params['ACTIVE_Compos']= PC(1) # composition
        params['ACTIVE_Veloc']= PC(1)
        params['ACTIVE_Chezy']= PC(1)
        params['ACTIVE_Extinc_VLG']= PC(1)
        params['ACTIVE_ExtPhDVL']= PC(1)
        params['ACTIVE_TotDepth']= PC(1)
        # # trying to get light extinction to work:
        params['ACTIVE_CalcRad']= PC(1)
        params['ACTIVE_Rad_Green']= PC(1)
        params['RadSurf']= PC( 150.0)
        return params

    def cmd_default(self):
        self.cmd_write_runid()
        self.cmd_write_hydro()
        self.cmd_write_inp()
        self.cmd_delwaq1()
        self.cmd_delwaq2()

    def __init__(self,*a,**k):
        super(Bloombox,self).__init__(*a,**k)
        # would like to get more info on the specific fluxes
        # can't just put dSedGreen in there - says it wasn't located.

        self.map_output += ('OXY','LimRadGree','fPPGreen',
                            'DayL','VSedGreen','Depth','V0SedGreen',
                            'Rad','RadBot',
                            'ExtVl','ExtVlPhyt',
                            # this gives sedimentation fluxes into the bed
                            'fSedGreen'
                        )


def bloom_setup():
    scen=Bloombox(hydro=ColumnHydro())
    # need at least 1 monitor entry in order to generate
    # monitor output
    scen.monitor_areas=[('test monitor',1)] 

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_default()
    return scen

def bloom_teardown(scen):
    shutil.rmtree(scen.base_path)

def test_bloom():
    scen=bloom_setup()

    hist_nef=scen.nef_history()
    map_nef =scen.nef_map()
    assert(hist_nef!=None)
    assert(map_nef!=None)

    hist_nef.close()
    map_nef.close()

    nc=scen.nc_map()
    assert(nc is not None)
    # make sure the data is compatible with on-disk format:
    nc.copy(fn='testout.nc')
    os.unlink('testout.nc')

    bloom_teardown(scen)

def test_bloom10():
    # Similar, but change number of output layers 
    hydro=ColumnHydro(num_z=10)
    scen=Bloombox(hydro)

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_default()

    shutil.rmtree(scen.base_path)

def test_bloom50():
    # seems to be failing with significantly larger number of layers-
    # I think it's a nefis problem.
    hydro=ColumnHydro(num_z=10)

    scen=Bloombox(hydro)

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_default()

    nc=scen.nc_map()
    
    shutil.rmtree(scen.base_path)



def test_delwaq1_fail():
    # cause delwaq1 to fail - make sure we can detect that.
    hydro=ColumnHydro(num_z=10)
    scen=Bloombox(hydro)

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_write_runid()
    scen.cmd_write_hydro()
    # skip writing inp file

    try:
        scen.cmd_delwaq1()
        assert(False)
    except waq_scenario.WaqException:
        pass

    shutil.rmtree(scen.base_path)

def test_delwaq2_fail():
    # cause delwaq2 to fail - make sure we can detect that.
    hydro=ColumnHydro(num_z=10)
    scen=Bloombox(hydro)

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_write_runid()
    scen.cmd_write_hydro()
    scen.cmd_write_inp()
    scen.cmd_delwaq1()
    os.unlink( scen.hydro.flo_filename )

    # should fail without a flow file
    try:
        scen.cmd_delwaq2()
        assert(False)
    except waq_scenario.WaqException:
        pass

    shutil.rmtree(scen.base_path)

    
class Foo(object):
    name='none'
    def __repr__(self):
        return "<{}>".format(self.name)

def test_named_objects():
    a=waq_scenario.NamedObjects('scenario')
    a['a']=Foo()
    a['b']=Foo()
    
    b=waq_scenario.NamedObjects('scenario')
    b['c']=Foo()
    b['d']=Foo()
    
    ab=a+b
    # simple check that ordering is preserved
    assert( ab.keys() == ['a','b','c','d'] )
    assert( (b+a).keys() == ['c','d','a','b'])

def test_named_objects_case_insensitive():
    # case insensitivity
    c=waq_scenario.NamedObjects('scenario')
    c['Hello']=hello=Foo()
    assert(c['Hello']==c['hello']==c['HELLO']==hello)
    del c['hELlo']

def test_named_objects_retain_case():
    # maintain case through addition
    a=waq_scenario.NamedObjects('scenario')
    a['A']=Foo()
    a['b']=Foo()
    
    b=waq_scenario.NamedObjects('scenario')
    b['C']=Foo()
    b['d']=Foo()

    ab=a+b
    assert(ab['a'].name=='A')
    assert(ab['d'].name=='d')
           



if __name__=='__main__':
    nose.main()
