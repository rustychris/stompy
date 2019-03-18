import datetime
import numpy as np
import pdb
import functools
from .. import utils

class ParseError(Exception):
    pass

def my_float(s):
    try:
        return float(s)
    except ValueError:
        raise ParseError()

def my_int(s):
    try:
        return int(s)
    except ValueError:
        raise ParseError()

def parse_degmin(degmin,sign):
    if sign in "EW":
        if len(degmin) < 5:
            return np.nan
        dec=my_float(degmin[:3]) + my_float(degmin[3:])/60
    else:
        if len(degmin) < 4:
            return np.nan
        dec=my_float(degmin[:2]) + my_float(degmin[2:])/60
    if sign in "WS":
        dec*=-1
    elif sign in 'NE':
        pass
    else:
        return np.nan
    return dec

# line=fp.readline()
# line='$GPGGA,005744.000,0721.1657,N,08159.6998,W,2,11,0.8,20.7,M,3.1,M,0.8,0000*6C'
def parse_sentence(line):
    # This used to put a decimal day fraction into 'time', but for consistency
    # with other code, this is now in time_dn, and time, when possible, is a numpy
    # datetime64
    data_check=line.split('*')
    if len(data_check)==2:
        data,check=data_check
    else:
        data=data_check[0]
        check=''

    parts=data.split(',')

    try:
        to_check=line[ line.index('$')+1: line.index('*') ]
        if not to_check:
            checksummed=False
        else:
            xor=functools.reduce( lambda a,b: a^b, [ord(c) 
                                                    for c in to_check ] )
            calc=hex(xor)[2:].upper()
            checksummed=(calc==check.strip())
    except ValueError:
        checksummed=False

    result=dict(sentence=parts[0],checksum=checksummed)

    if parts[0]=='$GPGGA':
        # $GPGGA,005743.000,0721.1655,N,08159.6990,W,2,11,0.8,20.6,M,3.1,M,0.8,0000*60
        try:
            (time,lat,lat_sign,lon,lon_sign,fix,n_sats,hdop,alt,
             alt_units,geoid_sep,geoid_units,age,diff)=parts[1:]
        except ValueError: # not enough parts
            # print "Error parsing NMEA sentence, skipping"
            result['sentence']='BAD'
            return result
        # fractional day:
        time_dn=my_float(time[:2])/24 + my_float(time[2:4])/(24*60) + my_float(time[4:])/86400
        lat=parse_degmin(lat,lat_sign)    
        lon=parse_degmin(lon,lon_sign)
        result.update( dict(time_dn=time_dn,lat=lat,lon=lon) )
    elif parts[0]=='$RDENS':
        result['ensemble']=my_float(parts[1])
        result['pc_time']=my_float(parts[2])
        # result['src']=parts[3]
    elif parts[0]=='$GPGSA':
        pass # don't care about satellite status updates.
    elif parts[0]=='$GPGSV':
        pass # another satellite position report.
    elif parts[0]=='$GPRMC':
        # $GPRMC,005743.000,A,0721.1655,N,08159.6990,W,3.03,287.03,050415,,*1F
        try:
            (time,status,lat,lat_sign,lon,lon_sign,sog_kts,cog_deg,date,variation)=parts[1:11]
        except ValueError: # not enough parts
            # print "Error parsing NMEA sentence, skipping"
            result['sentence']='BAD'
            return result
        # fractional day:
        time_dn=my_float(time[:2])/24 + my_float(time[2:4])/(24*60) + my_float(time[4:])/86400
        day = my_int(date[:2])
        month=my_int(date[2:4])
        year=2000+my_int(date[4:6])
        try:
            date=datetime.date(year,month,day)
        except ValueError:
            raise ParseError()
        dn=date.toordinal() + time_dn
        lat=parse_degmin(lat,lat_sign)    
        lon=parse_degmin(lon,lon_sign)
        result.update( dict(time_dn=time_dn,lat=lat,lon=lon,dn=dn) )

        # Add a numpy timestamp, too
        time_us=int(time_dn*86400*1e6)
        time_dt64=utils.to_dt64(date) + np.timedelta64(time_us,'us')
        result['time']=time_dt64
    else:
        # print "Skip sentence type ",parts[0]
        pass
    return result


def parse_nmea(fn=None,fp=None,buff=None,trim_prefix=False):
    if buff is not None:
        lines=buff.split('\n')
    elif fp is not None:
        lines=fp # .readlines()
    elif fn is not None:
        fp=open(fn,'rt')
        lines=fp # .readlines()

    # Read GPS stream:
    nmea=[]
    for line in lines:
        try:
            if trim_prefix and '$' in line:
                line=line[line.index('$'):]
            sent=parse_sentence(line)
            if (not sent['checksum']) and (sent['sentence'] != '$RDENS'):
                raise ParseError()
            nmea.append(sent)
        except ParseError:
            # print "Ignoring mangled NMEA: %s"%line
            continue
    return nmea
