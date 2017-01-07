""" Some convenience routines for parsing interval inputs
"""
from __future__ import division
from __future__ import print_function

import datetime
from matplotlib.dates import date2num,num2date
import pytz

units_to_days = {'d':1.0,
                 'h':1.0/24.0,
                 'm':1.0/(60.0*24.0),
                 's':1.0/(3600.0*24.0)}

                 
def parse_date_or_interval(s):
    if s[0] in "+-":
        # it's an interval
        val = float(s[:-1])
        units = s[-1]

        print("Got interval: ",val,units)

        val = val * units_to_days[units]
        return val
    else:
        # Must be a date:
        year=0
        month=1
        day=1
        hour=0
        minute=0
        second=0

        if len(s) < 4:
            raise Exception("%s: bad date.  wanted YYYYMMDDHHmmss or prefix thereof"%s)
        
        if len(s) >= 4:
            year = int(s[:4])
            if len(s) >= 6:
                month = int(s[4:6])
                if len(s) >= 8:
                    day   = int(s[6:8])
                    if len(s) >= 10:
                        hour  = int(s[8:10])
                        if len(s) >= 12:
                            minute = int(s[10:12])
                            if len(s) >= 14:
                                second = int(s[12:14])
                                if len(s) > 14:
                                    raise Exception("Extra characters in date: %s"%s)
        d = datetime.datetime(year,month,day,hour,minute,second,tzinfo=pytz.timezone('utc'))

        return d
    
def parse_interval(s,default_start=None,default_end=None,default_days=None):
    """
    s: the string to parse.

    Format of s is  start_spec:end_spec
                    start_spec
                    :end_spec

    Each can be either a date in YYYYMMDDHHMMSS format or
    a signed offset  +/-XXXXunits
    the leading + or - *MUST* be included
    where units is d for days, h for hours, m for minutes, s for seconds

    returns a pair of python datetime instances
    """
    inputs = 0
    if default_start is not None:
        inputs += 1
    if default_end is not None:
        inputs += 1
    if default_days is not None:
        inputs += 1

    if inputs!=2:
        raise Exception("parse_interval expects exactly two of the defaults to be set")
    
    if default_start is None:
        default_start = default_end - datetime.timedelta(default_days)
    if default_end is None:
        default_end = default_start + datetime.timedelta(default_days)

    # do it.
    if ':' in s:
        a,b = s.split(':')

        if a != "":
            default_start = parse_date_or_interval(a)
        if b != "":
            default_end   = parse_date_or_interval(b)

        if not isinstance(default_start,datetime.datetime):
            default_start = default_end + default_start
        elif not isinstance(default_end,datetime.datetime):
            default_end = default_start + datetime.timedelta(default_end)
    elif s not in [None,'']:
        # This part is more slippery...
        # parse the one value we've got.
        # if it's a date, it's the start date and interval is unchanged
        # if it's an interval, then decide based on the sign.
        
        a = parse_date_or_interval(s)

        if isinstance(a,datetime.datetime):
            default_start = a
            default_end = a+datetime.timedelta(default_days)
        else:
            if a<0:
                default_start = default_end + datetime.timedelta(a)
            else:
                default_end = default_start + datetime.timedelta(a)
    else:
        print("No interval string specified")


    if date2num(default_start) > date2num(default_end):
        raise Exception("Bad date range: %s -> %s"%(default_start,default_end))

    return default_start,default_end

