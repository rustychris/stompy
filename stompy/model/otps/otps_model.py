import os
import subprocess
import numpy as np

import xarray as xr

from ... import tide_consts, utils, harm_decomp


class OTPS(object):
    """
    Rough interface for calling OTPS, the Oregon Tidal Prediction System(?)

    Requires that otps2 is installed along with a suitable atlas.
    Paths to these are specified on instantiation.

    The phases returned are in degrees, and follow the same convention
    as NOAA.
    """
    model_file="DATA/Model_atlas_v1"

    def __init__(self,bin_path,data_path,model_file=None):
        self.bin_path=bin_path
        self.data_path=data_path
        if model_file is not None:
            self.model_file=model_file

    @property
    def extract_HC_path(self):
        return os.path.join(self.bin_path,'extract_HC')

    @property
    def predict_tide(self):
        return os.path.join(self.bin_path,'predict_tide')

    def extract_HC(self,
                   lonlats,
                   constituents=['m2','s2','n2','k2','k1','o1','p1','q1'],
                   quant='z'):
        """
        quant: quantity to extract.
          z: waterlevel
          u,v: cm/s velocities
          U,V: m2/s transport
        """
        constituents=[c.lower() for c in constituents]
        cwd=os.getcwd()
        # just use a bunch of non-thread-safe temporary files 
        output_fn=os.path.join(cwd,"otps-output.inp")
        inp_fn=os.path.join(cwd,'otps-setup.inp')
        lltime_fn=os.path.join(cwd,'otps-latlontime.txt')
        with open(lltime_fn,'wt') as fp:
            for ll in lonlats:
                # we are called with lonlat, but otps wants lat,lon
                fp.write("%.6f %.6f\n"%(ll[1],ll[0]))

        inp_data="\n".join( ["%s      ! 1. tidal model control file"%self.model_file,
                             "%s      ! 2. latitude/longitude/<time> file"%lltime_fn,
                             "%s      ! 3. z/U/V/u/v"%quant,
                             "%s      ! 4. tidal constituents to include"%( ",".join(constituents) ),
                             "AP      ! 5. AP/RI",
                             "oce     ! 6. oce/geo",
                             "1       ! 7. 1/0 correct for minor constituents",
                             "%s      ! 8. output file (ASCII)"%output_fn,
                             "","",""] )

        with open(inp_fn,'wt') as fp:
            fp.write(inp_data)

        inp_fp=open(inp_fn,'rt')    

        # This seems to work.  Leaves stdout connected, so it gets some trash on console
        subprocess.call([self.extract_HC_path],cwd=self.data_path,stdin=inp_fp)

        with open(output_fn,'rt') as fp:
            atlas=fp.readline().strip() # atlas
            quantity=fp.readline().strip() # Elevation (m)
            headers=fp.readline().strip().split()
            recs=[]
            for line in fp:
                line=line.strip()
                if line.startswith('HC'):
                    continue
                rec=[float(s) for s in line.split()]
                recs.append(rec)
        recs=np.array(recs)

        result=xr.Dataset()

        for i,fld in enumerate(headers):
            result[fld]=('site',),recs[:,i]

        # Reformat that a bit:
        result['const']= ('const',),constituents
        speeds=[tide_consts.speeds[ tide_consts.const_names.index(c.upper()) ]
                for c in constituents ]
        result['speed']=('const',),speeds
        result['speed'].attrs['units']='deg s-1'

        amplitudes=np.zeros( (len(recs),len(constituents)), 'f8')
        phases=np.zeros( (len(recs),len(constituents)), 'f8')
        for ci,c in enumerate(constituents):
            amplitudes[:,ci] = result[c+'_amp'].values
            phases[:,ci] = result[c+'_ph'].values
        result['amp']=('site','const'),amplitudes
        result['ph'] =('site','const'),phases

        # Fix up missing outputs -- extract_HC drops repeated points,
        # so here we match all outputs to input locations by lat/lon
        ll_output=np.c_[result.Lon,result.Lat]

        remapping=[ np.argmin( utils.haversine(ll,ll_output) )
                    for ll in lonlats ]
        return result.isel(site=remapping)

def reconstruct(harms,times):
    t0=times[0].astype('M8[Y]')
    
    dnums=utils.to_dnum(times)
    dnum0=utils.to_dnum(t0)
    dt0=utils.to_datetime(dnum0)
    year0=dt0.year

    assert year0 in tide_consts.years

    year_i = tide_consts.years.searchsorted(year0)

    v0u=tide_consts.v0u[:,year_i]
    lun_nod=tide_consts.lun_nodes[:,year_i]

    const_idxs=[ tide_consts.const_names.index(c.upper())
                 for c in harms.const.values ]
    speeds=tide_consts.speeds[const_idxs]

    # extract just the relevant ones:
    v0u = v0u[const_idxs]
    lun_nod = lun_nod[const_idxs]

    # convert to radians:
    v0u_rads=(np.pi/180)*v0u # convert equilibrium argument to radians

    output=[]
    for site_i in range(len(harms.site)):
        harm=harms.isel(site=site_i)
        amps_out = lun_nod*harm.amp.values
        phase_in_rads = (np.pi/180)*harm.ph.values
        phase_out = v0u_rads - phase_in_rads
        phase_out *=-1 # conventions..

        # speeds are in degrees per hour, we want radians per day...
        omegas = speeds*(np.pi/180)*24

        comps=np.c_[amps_out,phase_out]

        time_series = harm_decomp.recompose(dnums-dnum0,comps,omegas)
        output.append(time_series)

    site_series = np.array(output)

    recon=xr.Dataset()

    recon['time']=('time',),times
    recon['dnums']=('time',),dnums
    recon['Lat']=('site',),harms.Lat
    recon['Lon']=('site',),harms.Lon
    recon['result']=('site','time'),site_series
    return recon
    
