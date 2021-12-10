"""
Streamline reading of RiverSurveyor data
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from stompy.spatial import proj_utils
from stompy import utils
import scipy.io
import re

import logging
log=logging.getLogger(__name__)

##

def surveyor_to_xr(rivr_fn,proj=None,source_preferences=['mat','csv'],
                   positive='down',
                   z_bed_preferences=['depth_bt','depth_vb']):
    """
    Read River Surveyor outputs (post-processed rivr file as either MAT
    or CSV).

    Return results in xarray Dataset.

    source_preferences: list of formats to check for, currently only
    mat or csv.

    positive: if 'up', flip the coordinates so that positive values are
    up.  otherwise default to positive=down
    """
    suffix=rivr_fn.split('.')[-1]
    assert suffix in ['rivr','riv']
    base=rivr_fn[:-len(suffix)] # includes trailing '.'

    ds=None
    for preference in source_preferences:
        if preference=='mat' and os.path.exists(base+'mat'):
            ds=_surveyor_mat_to_xr(base)
            if ds is not None:
                break
        elif preference=='csv' and os.path.exists(base+'vel'):
            ds=_surveyor_csv_to_xr(base)
            if ds is not None:
                break

    if ds is None:
        raise Exception("Couldn't find any post-processed files for %s"%rivr_fn)

    if proj is not None:
        mapper=proj_utils.mapper('WGS84',proj)
        ll=np.c_[ds.lon.values,ds.lat.values]
        # check for string values with degree, ', and "
        if not np.issubdtype(ll.dtype,np.floating):
            print("Attempting to deal with DMS formatted lat/lon")
            def dms_to_dec(s):
                s=s.replace('°','').replace("'","").replace('"','')
                deg,mn,sec=[float(x) for x in s.split()]
                return deg + np.sign(deg)*(mn/60 + sec/3600)
            ll=np.vectorize(dms_to_dec)(ll)
        xy=mapper(ll)
        ds['x_sample']=ds.lon.dims,xy[:,0]
        ds['y_sample']=ds.lon.dims,xy[:,1]

    # Other metadata:
    ds.attrs['rivr_filename']=rivr_fn
    ds.attrs['source']=rivr_fn

    if positive=='up':
        ds.z_ctr.values *= -1
        ds.z_ctr.attrs['positive']='up'

        ds['z_bed']=-ds[z_bed_preferences[0]]
        ds.z_bed.attrs['positive']='up'
    else:
        ds.z_ctr.attrs['positive']='down'

        ds['z_bed']=ds[z_bed_preferences[0]]
        ds.z_bed.attrs['positive']='down'

    log.info("Assuming Sontek data is relative to transducer, including depth")
    ds.z_bed.attrs['datum']='transducer'
    ds.z_ctr.attrs['datum']='transducer'

    return ds

def _surveyor_csv_to_xr(base):
    ds=xr.Dataset()

    snr_fn=base+'snr'
    sum_fn=base+'sum'
    vel_fn=base+'vel'

    # sometimes (?) river surveyor outputs lat/lon with degree symbol.
    # sometimes it's decimal.  thanks, RS.
    df_snr=pd.read_csv(base+'snr',parse_dates=['Date/Time'],encoding='latin-1')
    df_sum=pd.read_csv(base+'sum',encoding='latin-1')
    df_vel=pd.read_csv(base+'vel',encoding='latin-1')

    ds['sample']=('sample',),df_snr['Sample #']
    ds['beam']=('beam',),np.arange(4)
    n_cells=int(df_snr.columns[-1].split()[0][4:]) # 'Cell66 SNR4 (db)' => 66
    ds['layer']=('layer',),np.arange(n_cells) # 0-based!

    ds['time']=('sample',),df_snr['Date/Time']
    ds['frequency']=('sample',), [float(s.replace('MHz','')) for s in df_snr['Frequency (MHz)']]
    ds['profile_type']=('sample',),df_snr['Profile Type']
    ds['depth_m']=('sample',),df_snr['Depth (m)']

    snr_data=np.zeros( (len(ds.sample),
                        len(ds.layer),
                        len(ds.beam)), 'f4')
    for col in df_snr.columns:
        if not col.startswith('Cell') or ('SNR' not in col):
            continue
        cell,beam,_=col.split()
        cell_i=int(cell[4:])-1 # 0-based
        beam_i=int(beam[3:])-1 # 0-based
        snr_data[:,cell_i,beam_i]=df_snr[col].values

    ds['snr']=('sample','layer','beam'),snr_data
    ds['snr'].attrs['units']='dB'

    # velocity and cell geometry data
    ds['cell_start']=('sample',), df_vel['Cell Start (m)'].values
    ds['cell_size']=('sample',), df_vel['Cell Size (m)'].values

    x=np.zeros( ( len(ds.sample),
                  len(ds.layer) ), 'f4' )
    per_cell={'Ve':x.copy(),
              'Vn':x.copy(),
              'Vu':x.copy(),
              'Vd':x.copy(),
              'Location':x.copy(),
              'Spd':x.copy(),
              'Dir':x.copy() }

    for col in df_vel.columns:
        m=re.match(r'Cell(\d+) (\w+) (.*)',col)
        if not m:
            continue

        cell_i=int(m.group(1))-1 # 0-based
        key=m.group(2)
        units=m.group(3)

        per_cell[key][:,cell_i]=df_vel[col].values

    ds['Ve']=('sample','layer'),per_cell['Ve']
    ds['Vn']=('sample','layer'),per_cell['Vn']
    ds['Vu']=('sample','layer'),per_cell['Vu']
    ds['Vd']=('sample','layer'),per_cell['Vd']
    ds['z_ctr']=('sample','layer'),per_cell['Location']
    ds['water_speed']=('sample','layer'),per_cell['Spd']
    ds['water_dir']=('sample','layer'),per_cell['Dir']

    # summary data:
    for my_name,df_name in [ ('lat','Latitude (deg)'),
                             ('lon','Longitude (deg)'),
                             ('heading','Heading (deg)'),
                             ('pitch','Pitch (deg)'),
                             ('roll','Roll (deg)'),
                             ('depth_bt','BT Depth (m)'),
                             ('depth_vb','VB Depth (m)'),
                             ('track_dist','Track (m)'),
                             ('dmg','DMG (m)'),
                             ('mean_water_speed','Mean Speed (m/s)'),
                             ('boat_speed','Boat Speed (m/s)'),
                             ('mean_water_dir','Direction (deg)'),
                             ('boat_dir','Boat Direction (deg)') ]:
        ds[my_name]=('sample',),df_sum[df_name].values

    bt_depths=np.zeros( ( len(ds.sample),len(ds.beam) ), 'f4')

    bt_depths[:,0]=df_sum['BT Beam1 Depth (m)'].values
    bt_depths[:,1]=df_sum['BT Beam2 Depth (m)'].values
    bt_depths[:,2]=df_sum['BT Beam3 Depth (m)'].values
    bt_depths[:,3]=df_sum['BT Beam4 Depth (m)'].values
    ds['depth_bt_beam']=('sample','beam'), bt_depths

    return ds

def _surveyor_mat_to_xr(base,target_ref='bt'):
    mat_fn=base+'mat'

    try:
        mat=scipy.io.loadmat(mat_fn)
    except TypeError:
        print("MAT file may be corrupted. Back off")
        mat=scipy.io.loadmat(mat_fn,
                             variable_names=['Setup', 'SiteInfo', 'Processing', 'SystemHW',
                                             'Transformation_Matrices', 'System', 'Summary',
                                             'BottomTrack', 'GPS', 'WaterTrack'])

    if 'WaterTrack' not in mat:
        log.warning("MAT file %s does not contain velocity data"%mat_fn)
        return None

    # ['System',
    #  'WaterTrack',
    #  'SystemHW',
    #  '__globals__',
    #  'BottomTrack',
    #  'Setup',
    #  'Processing',
    #  '__header__',
    #  'SiteInfo',
    #  'RawGPSData',
    #  'Summary',
    #  'Compass',
    #  'Transformation_Matrices',
    #  '__version__',
    #  'GPS'

    ds=xr.Dataset()

    water_velocity=mat['WaterTrack'][0,0]['Velocity'] # 85,4,167   bins,beams,samples

    ds['sample']=('sample',),mat['System']['Sample'][0,0][:,0].astype('i4')
    ds['beam']=('beam',),np.arange(4)

    n_cells=water_velocity.shape[0] 
    ds['layer']=('layer',),np.arange(n_cells) # 0-based!

    ds['time_raw']=('sample',), mat['System']['Time'][0,0][:,0] # seconds since 2000-01-01
    ds['time']=ds['time_raw']*np.timedelta64(1,'s') + np.datetime64("2000-01-01 00:00")

    # reported in kHz.
    ds['frequency']=('sample',), mat['WaterTrack']['WT_Frequency'][0,0][:,0]/1000.

    # HD or IC... but not in the mat file
    ds['profile_type']=('sample',), ["n/a"] * len(ds.sample)


    ds['depth_m']=('sample',), mat['Summary']['Depth'][0,0][:,0]

    snr_data=np.zeros( (len(ds.sample),
                        len(ds.layer),
                        len(ds.beam)), 'f4')

    # mat['WaterTrack']['Correlation']
    ds['snr']=('sample','layer','beam'), mat['System']['SNR'][0,0].transpose(2,0,1)
    ds['snr'].attrs['units']='dB'

    # velocity and cell geometry data
    ds['cell_start']=('sample',), mat['System']['Cell_Start'][0,0][:,0]
    ds['cell_size']=('sample',),  mat['System']['Cell_Size'][0,0][:,0]

    ds['bottom_vel']= ('sample','beam'), mat['BottomTrack']['BT_Vel'][0,0] 
    ds['boat_vel'] = ('sample','beam'), mat['Summary']['Boat_Vel'][0,0]

    # 0: referenced to instrument
    # 1: bottom track
    # 2: GPS GGA
    # 3: GPS VTG
    velo_reference = mat['Setup']['velocityReference'][0,0][0,0]
    velo_label=['sys','bt','gps','gps'][velo_reference] # last two differ by GGA vs. VTG

    if velo_label!='gps':
        log.warning("Testing has only been with mat files using GPS reference")

    ds['Ve_'+velo_label]=('sample','layer'), water_velocity[:,0,:].T
    ds['Vn_'+velo_label]=('sample','layer'), water_velocity[:,1,:].T
    # Assuming that these do not need to be adjusted for BT vs GPS
    ds['Vu']=('sample','layer'), water_velocity[:,2,:].T
    ds['Vd']=('sample','layer'), water_velocity[:,3,:].T

    gps_to_bt=(ds['boat_vel'][:,:2] - ds['bottom_vel'][:,:2])
    if velo_label=='gps':
        ds['Ve_bt'] = ds['Ve_gps'] + gps_to_bt.isel(beam=0,drop=True)
        ds['Vn_bt'] = ds['Vn_gps'] + gps_to_bt.isel(beam=1,drop=True)
    elif velo_label=='bt':
        ds['Ve_gps'] = ds['Ve_bt'] - gps_to_bt.isel(beam=0,drop=True)
        ds['Vn_gps'] = ds['Vn_bt'] - gps_to_bt.isel(beam=1,drop=True)

    if target_ref=='bt':
        ds['Ve']=ds['Ve_bt']
        ds['Vn']=ds['Vn_bt']
    else:
        ds['Ve']=ds['Ve_gps']
        ds['Vn']=ds['Vn_gps']

    # matches CSV-based location, with roundoff up to 5mm.
    ds['z_ctr']=('sample','layer'), ( ds.cell_start.values[:,None]
                                      + ds.cell_size.values[:,None]
                                      * (0.5+np.arange(len(ds.layer))[None,:]) )

    # confirmed to match once the GPS vs. BT difference is applied.
    ds['water_speed']=np.sqrt(ds.Ve**2 + ds.Vn**2)
    ds['water_dir']=(90 - 180/np.pi*np.arctan2(ds.Vn,ds.Ve)) % 360.0

    ds['lat'] = ('sample',),mat['GPS']['Latitude'][0,0][:,0]
    ds['lon'] = ('sample',),mat['GPS']['Longitude'][0,0][:,0]
    ds['hdop']= ('sample',),mat['GPS']['HDOP'][0,0][:,0]
    # ds[mat['GPS']['UTM'] # they have UTM, but for consistency stick with our own projection
    # UTC is reported in seconds from an unknown time 0.
    # ds['utc'] = ('sample',),mat['GPS']['Utc'] # not sure what the reference is here

    # Could be used to get track distance
    # track_xy=mat['Summary']['Track'][0,0]
    ds['heading'] = ('sample',), mat['System']['Heading'][0,0][:,0]
    if 'Compass' in mat:
        ds['pitch']  = ('sample',), mat['Compass']['Pitch'][0,0][:,0]
        ds['roll']  = ('sample',), mat['Compass']['Roll'][0,0][:,0]
    else:
        print("No pitch or roll as Compass could not be read")

    ds['depth_bt'] = ('sample',), mat['BottomTrack']['BT_Depth'][0,0][:,0]
    ds['depth_vb'] = ('sample',), mat['BottomTrack']['VB_Depth'][0,0][:,0]


    ds['Ve_mean']= ('sample'),mat['Summary']['Mean_Vel'][0,0][:,0]
    ds['Vn_mean']= ('sample'),mat['Summary']['Mean_Vel'][0,0][:,1]

    # # summary data:
    # for my_name,df_name in [ ('track_dist','Track (m)'),
    #                          ('dmg','DMG (m)'),
    #                          ('mean_water_speed','Mean Speed (m/s)'),
    #                          ('boat_speed','Boat Speed (m/s)'),
    #                          ('mean_water_dir','Direction (deg)'),
    #                          ('boat_dir','Boat Direction (deg)') ]:
    #     ds[my_name]=('sample',),df_sum[df_name].values
    # 
    # bt_depths=np.zeros( ( len(ds.sample),len(ds.beam) ), 'f4')
    # 
    # bt_depths[:,0]=df_sum['BT Beam1 Depth (m)'].values
    # bt_depths[:,1]=df_sum['BT Beam2 Depth (m)'].values
    # bt_depths[:,2]=df_sum['BT Beam3 Depth (m)'].values
    # bt_depths[:,3]=df_sum['BT Beam4 Depth (m)'].values
    # ds['depth_bt_beam']=('sample','beam'), bt_depths

    return ds


