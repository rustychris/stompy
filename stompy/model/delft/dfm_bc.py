"""
DFM specializations of generic boundary conditions

"""

from __future__ import print_function
import six


import os

import numpy as np
from . import io as dio
from ...io.local import noaa_coops
from ... import utils, filters
from . import dfm_grid

class BC(object):
    cache_dir=None
    @classmethod
    def set_cache_dir(klass,path,create=False):
        klass.cache_dir=path
        if create and not os.path.exists(klass.cache_dir):
            os.makedirs(klass.cache_dir)

    # @staticmethod
    # def factory(feat):
    #     code='[' + feat['src'] + ']'
    #     return eval(code)

    def set_feature(self,feature):
        """
        For boundary conditions derived from a shapefile feature, save
        that feature.  This is a separate step from __init__ since 
        the class def is evaluated, and then we can pass things to it.
        """
        self.feature=feature

class Scalar(BC):
    def __init__(self,name,value):
        """
        name: 'salinity','temperature'
        value: floating point
        """
        self.name=name
        self.value=value

    def write(self,mdu,feature,grid):
        print("Writing feature: %s"%(feature['name']))

        name=feature['name']
        old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )

        assert feature['geom'].type=='LineString'
        pli_data=[ (name, np.array(feature['geom'].coords)) ]
        base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,self.var_name))
        pli_fn=base_fn+'.pli'
        dio.write_pli(pli_fn,pli_data)

        if self.var_name=='salinity':
            quant='salinitybnd'
        elif self.var_name=='temperature':
            quant='temperaturebnd'
        else:
            assert False

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=%s"%quant,
                   "FILENAME=%s_%s.pli"%(name,self.var_name),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   ""]
            fp.write("\n".join(lines))

        self.write_data(mdu,feature,self.var_name,base_fn)

    def write_data(self,mdu,feature,var_name,base_fn):
        ref_date,start_date,end_date = mdu.time_range()
        period=np.array([start_date,end_date])
        elapsed_minutes=(period - ref_date)/np.timedelta64(60,'s')

        # just write a single node
        tim_fn=base_fn + "_0001.tim"
        with open(tim_fn,'wt') as fp:
            for t in elapsed_minutes:
                fp.write("%g %g\n"%(t,self.value))

class WaterLevel(BC):
    def __init__(self,z_offset=0.0):
        self.z_offset=z_offset

    def write(self,mdu,feature,grid):
        name=feature['name']
        print("Writing feature: %s"%name)

        bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )

        assert feature['geom'].type=='LineString'

        pli_data=[ (name, np.array(feature['geom'].coords)) ]
        base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,var_name))
        pli_fn=base_fn+'.pli'
        dio.write_pli(pli_fn,pli_data)

    def write(self,mdu,feature,grid):
        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=waterlevelbnd",
                   "FILENAME=%s_ssh.pli"%self.name,
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   ""]
            fp.write("\n".join(lines))

        self.write_data(mdu,feature,var_name,base_fn)

    def write_data(self,mdu,feature,var_name,base_fn):
        tides=noaa_coops.coops_dataset_product(self.station,'water_level',
                                               mdu.time_range()[1],mdu.time_range()[2],
                                               days_per_request='M',cache_dir=cache_dir)
        tide=tides.isel(station=0)
        water_level=utils.fill_tidal_data(tide.water_level) + self.z_offset
        # IIR butterworth.  Nicer than FIR, with minor artifacts at ends
        # 3 hours, defaults to 4th order.
        water_level[:] = filters.lowpass(water_level[:].values,
                                         utils.to_dnum(water_level.time),
                                         cutoff=3./24)

        ref_date=mdu.time_range()[0]
        elapsed_minutes=(tide.time.values - ref_date)/np.timedelta64(60,'s')

        # just write a single node
        tim_fn=base_fn + "_0001.tim"
        data=np.c_[elapsed_minutes,water_level]
        np.savetxt(tim_fn,data)
        
        
# class NoaaTides(WaterLevel):
#     def __init__(self,station,datum='NAVD88',*kw):
#         self.station=station
#         self.datum=datum
#         super(NoaaTides,self).__init__(self,*a,**kw)
        

class Flow(BC):
    dredge_depth=-1.0
    
    def __init__(self,flow=None):
        self.flow=flow
        
    def write(self,mdu,feature,grid):
        # obvious copy and paste from above.
        # not quite ready to abstract, though
        print("Writing feature: %s"%(feature['name']))

        name=feature['name']
        old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )

        for var_name in self.var_names:
            if feature['geom'].type=='LineString':
                pli_data=[ (name, np.array(feature['geom'].coords)) ]
                base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,var_name))
                pli_fn=base_fn+'.pli'
                dio.write_pli(pli_fn,pli_data)

                if var_name=='q':
                    quant='dischargebnd'
                else:
                    assert False

                with open(old_bc_fn,'at') as fp:
                    lines=["QUANTITY=%s"%quant,
                           "FILENAME=%s_%s.pli"%(name,var_name),
                           "FILETYPE=9",
                           "METHOD=3",
                           "OPERAND=O",
                           ""]
                    fp.write("\n".join(lines))

                self.write_data(mdu,feature,var_name,base_fn)

                dfm_grid.dredge_boundary(grid,pli_data[0][1],self.dredge_depth)
            else:
                assert False

    def write_data(self,mdu,feature,var_name,base_fn):
        ref_date,run_start,run_stop=mdu.time_range()

        def h_to_td64(h):
            # allows for decimal hours
            return int(h*3600) * np.timedelta64(1,'s')
        
        # trapezoid hydrograph
        times=np.array( [run_start,
                         run_start+h_to_td64(self.storm_start_h-1),
                         run_start+h_to_td64(self.storm_start_h),
                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h),
                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h+1),
                         run_stop+np.timedelta64(1,'D')] )
        flows=np.array( [0.0,0.0,
                         self.storm_flow,self.storm_flow,0.0,0.0] )
        elapsed_minutes=(times - ref_date)/np.timedelta64(60,'s')

        # just write a single node
        tim_fn=base_fn + "_0001.tim"
        data=np.c_[elapsed_minutes,flows]
        np.savetxt(tim_fn,data)

class Discharge(BC):
    """
    Similar to Storm, but implement as mass source, not a flow BC
    """
    def __init__(self,*a,**kw):
        self.salinity=kw.pop('salinity',None)
        self.temperature=kw.pop('temperature',None)
        
        super(Discharge,self).__init__(*a,**kw)
        
    def write(self,mdu,feature,grid):
        # obvious copy and paste from above.
        # not quite ready to abstract, though
        print("Writing feature: %s"%(feature['name']))

        name=feature['name']
        old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )

        assert feature['geom'].type=='LineString'
        
        pli_data=[ (name, np.array(feature['geom'].coords)) ]
        base_fn=os.path.join(mdu.base_path,"%s"%(name))
        pli_fn=base_fn+'.pli'
        dio.write_pli(pli_fn,pli_data)

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=discharge_salinity_temperature_sorsin",
                   "FILENAME=%s"%os.path.basename(pli_fn),
                   "FILETYPE=9",
                   "METHOD=1",
                   "OPERAND=O",
                   "AREA=0 # no momentum",
                   ""]
            fp.write("\n".join(lines))

        self.write_data(mdu,feature,base_fn)

        # Really just need to dredge the first and last nodes
        dfm_grid.dredge_discharge(grid,pli_data[0][1],self.dredge_depth)

    def write_data(self,mdu,feature,base_fn):
        ref_date,run_start,run_stop=mdu.time_range()

        def h_to_td64(h):
            # allows for decimal hours
            return int(h*3600) * np.timedelta64(1,'s')
        
        # trapezoid hydrograph
        times=np.array( [run_start,
                         run_start+h_to_td64(self.storm_start_h-1),
                         run_start+h_to_td64(self.storm_start_h),
                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h),
                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h+1),
                         run_stop+np.timedelta64(1,'D')] )
        flows=np.array( [0.0,0.0, 
                         self.storm_flow,self.storm_flow,0.0,0.0] )

        elapsed_minutes=(times - ref_date)/np.timedelta64(60,'s')
        items=[elapsed_minutes,flows]
        
        if self.salinity is not None:
            items.append(self.salinity * np.ones(len(times)))
            
        if self.temperature is not None:
            items.append(self.temperature * np.ones(len(times)))

        # just write a single node
        tim_fn=base_fn + ".tim"
        data=np.c_[tuple(items)]
        np.savetxt(tim_fn,data)


        
def set_from_shp(mdu,grid,shp_fn):
    features=wkb2shp.shp2geom(shp_fn)
    
    for feat in forcing_features:
        for data_src in factory(bc):
            data_src.write(mdu=mdu,feature=bc,grid=grid)
        
        
