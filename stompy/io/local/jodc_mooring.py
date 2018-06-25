import datetime
import xarray as xr
import numpy as np

import datetime
import zipfile

def jodc_mooring_to_ds(filename):
    ds=xr.Dataset()
    ds.attrs['filename']=filename

    if filename.endswith(".zip"):
        zip=zipfile.ZipFile(filename)
        members=zip.namelist()
        if len(members)>1:
            logging.warning("Will only look at first member of zipfile")
        fpb=zip.open(members[0],'r')
        import codecs
        fp=codecs.getreader('Latin-1')(fpb)
    else:
        fp=open(filename,'rt')

    headers=[fp.readline(),
             fp.readline(),
             fp.readline() ]

    data=fp.readlines()
    fp.close()

    # Fields which might be concatenated later on are kept as 0-dimensional
    # variables.  Fields which are less useful metadata are attributes.

    # see http://jdoss.jodc.go.jp/jodcweb/JDOSS/formatMoor.html
    hdr1,hdr2,hdr3=headers
    assert hdr1[:2]=="HC"
    lat_s=hdr1[2:9] # DDMMSSn
    lon_s=hdr1[9:17]# DDDMMSSn
    signs=dict(N=1,S=-1,E=1,W=-1)
    lat=signs[lat_s[6]] * (float(lat_s[:2]) + float(lat_s[2:4])/60. + float(lat_s[4:6])/3600.)
    lon=signs[lon_s[7]] * (float(lon_s[:3]) + float(lon_s[3:5])/60. + float(lon_s[5:7])/3600.)
    ds['lat']=(),lat
    ds['lon']=(),lon

    ds.attrs['system_code']=(),hdr1[31] # 0: tidal current  9: mooring system?
    layer=hdr1[29:31] # 01 - layer 1
    ds['layer']=(),int(layer)
    sea_area=hdr1[42:58].strip()
    ds['sea_area']=(),sea_area

    station_depth=hdr2[22:27].strip()
    if station_depth!="":
        ds['station_depth']=(),station_depth

    adopt_dir=hdr2[42] # T[rue] or M[agnetic]
    declination=hdr2[43:47].strip()
    assert hdr2[47:50]=='deg'

    direc_digits=int(hdr2[50])

    speed_digits=int(hdr2[57])
    speed_decimal=int(hdr2[58])

    system_code2=hdr2[16]

    equipment_code_large = hdr2[28:30] # MS
    equipment_code_small = hdr2[30:33] # MTC

    times=[]
    direcs=[]
    speeds=[]

    for data_line in data:
        meas_time=data_line[18:32].strip() # YYYYMMDDHHmmss
        if len(meas_time)==12:
            fmt="%Y%m%d%H%M"
        elif len(meas_time)==14:
            fmt="%Y%m%d%H%M%S"
        else:
            assert False
        times.append( datetime.datetime.strptime(meas_time,fmt) )

        meas_direc=data_line[42:45] #
        if meas_direc=="999":
            d=np.nan
        else:
            d=int(meas_direc)
        direcs.append(d)

        meas_speed=data_line[48:48+speed_digits]
        if meas_speed=="9"*speed_digits:
            s=np.nan
        else:
            s=float(meas_speed) * (0.1**speed_decimal)
        speeds.append(s)

    times=np.array(times)
    direcs=np.array(direcs).astype('f8')
    speeds=np.array(speeds)

    # above is for cm/s
    speed_ms=speeds * 0.01

    rads=(90-direcs) * np.pi/180.

    Ve=speed_ms*np.cos(rads)
    Vn=speed_ms*np.sin(rads)

    ds['time']=('time',),times
    ds['speed']=('time',),speed_ms
    ds.speed.attrs['units']='m s-1'

    ds['Ve']=('time',),Ve
    ds['Vn']=('time',),Vn
    ds['direc']=('time',),direcs
    ds.direc.attrs['units']='deg'
    ds.direc.attrs['reference']=adopt_dir

    #  c. package as xarray
    return ds
