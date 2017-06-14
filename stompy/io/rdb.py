"""
Tools for reading RDB files, the text-based format often used in USGS 
data.  See stompy/test/data for examples of this type of data.
"""

from __future__ import print_function

from functools import reduce

# the 2to3 stuff added some other cruft which I think actually makes it all worse...

# python import routines for rdb (USGS tab delimited) format

import re,string,time
import six
import io

import datetime
import numpy as np
import xarray as xr
from matplotlib.dates import date2num,num2date

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from rdb_datadescriptors import dd_to_synonyms

from . import rdb_codes


# best way to deal with dates:
# 64-bit float is plenty to cover milliseconds to a century.
# what is the best unit, though?
# days are nice, could pretty much ignore leap seconds
# leap years are still a problem, and one would want to be able
# to convert days to years reliably
# presumably the python datetime library can handle dates beyond
# the epoch.
# I think mxdatetime is probably the way to go.  If it's good enough
# for postgresql, it's good enough for me.  Looks like they use a
# pair of values - an int for days, and a float for seconds in the
# day.  Nicely compacted in an "absdays" attr


class Rdb(object):
    record_count = 0
    def __init__(self,
                 text=None,
                 source_file=None,
                 fp=None):
        self.source_filename = source_file
        if text is not None:
            try:
                text=text.decode()
            except AttributeError:
                pass
        self.text = text
        self.fp = fp

        if self.fp is None:
            if self.text is not None:
                self.fp = io.StringIO(self.text)
            elif self.source_filename:
                self.fp = open(self.source_filename,'rt')
        else:
            # coerce fp to be text, not binary
            try:
                self.fp.encoding
            except AttributeError:
                self.fp=io.StringIO(fp.read().decode())
        self.parse_source_file()

    def float_or_nan(self,s):
        if s in (None,'','Eqp'):
            return np.nan
        else:
            return float(s)

    def data(self):
        """ assuming that only one data type was requested, try to figure out which
        column it is, and return that data
        for single-valued columns, this will expand the data out to be the right
        length
        """
        for k in list(self.keys()):
            if re.match('^[0-9_]+$',k):
                d = self[k]
                try:
                    len(d)
                    return d
                except TypeError:
                    return d*ones(self.record_count)
            
        print(list(self.keys()))
        raise Exception("None of the keys looked good")
            
    def parse_date(self,s):
        """ parse a date like '2008-01-13 00:31' into a float representing
            absolute days since 0ad """

        d=None
        for fmt in ["%Y-%m-%d %H:%M","%Y-%m-%d"]:
            try:
                d = datetime.datetime.strptime(s,fmt)
            except ValueError:
                pass
        if d:
            return date2num(d)
        else:
            return None
        
    def parse_source_file(self):
        # eat the comments
        line = None

        for line in self.fp:
            if line[0]!='#':
                break
        if line is None:
            print("No data in source!")
            print("text:",self.text)
            print("source_filename:",self.source_filename)
            raise Exception("Empty RDB data?!")
            
        headers = line.strip().split("\t")
        specs = next(self.fp).strip().split("\t")

        # import the data into an array
        columns = [[] for each in headers]

        # split rows into columns
        self.record_count = 0
        for line in self.fp:
            data = line.strip("\r\n").split("\t")
            for column,datum in zip(columns,data):
                column.append(datum)
            self.record_count += 1

        self.fp.close()

        # fix up columns
        for i in range(len(headers)):
            if specs[i][-1] == 'n':
                columns[i] = np.array(list(map(self.float_or_nan,columns[i])))
            elif specs[i][-1] == 'd':
                # dates get mapped to days since AD 0
                columns[i] = np.array(list(map(self.parse_date,columns[i])))
            elif specs[i][-1] == 's':
                columns[i] = np.array(columns[i],dtype=object)

            # check for single-valued lists -
            # The logic here is a bit odd - since we often get columns
            # that have the same value (e.g. station_id) for every record,
            # it's convenient to detect that and replace them with a constant.
            # however, if there is just one record, it screws up code that
            # isn't expecting a single value...  
            if self.record_count > 1:
                try:
                    for elt in columns[i]:
                        if elt != columns[i][0]:
                            raise StopIteration
                    columns[i] = columns[i][0]
                except StopIteration:
                    pass

        # merge the columns into a hash
        self.compiled={}
        for header,column in zip(headers,columns):
            # main entry for given header title
            self.compiled[header] = column
            # other entries
            for syn in dd_to_synonyms(header):
                self.compiled[syn] = column

    # Dict interface:
    def __getitem__(self,key):
        if type(key) == tuple:
            # split into label and index
            if type(key[0]) == str:
                label,idx = key
            else:
                idx,label = key
            val = self.compiled[label]
            if type(val) in (list,ndarray):
                val = val[idx]
            return val
        else:
            return self.compiled[key]
    def keys(self):
        return list(self.compiled.keys())

    # Automatic removal of missing data:
    def series(self,*keys):
        """ return a tuple of vectors for the given keys,
        but only when all values have valid data
        """
        columns = [self.compiled[key] for key in keys]
        # create a mask for each:
        masks = [1-isnan(column) for column in columns]
        valid = reduce(lambda x,y: x&y, masks)

        return [compress(valid,column) for column in columns]




def rdb_to_dataset(usgs_fn):
    usgs_data=Rdb(source_file=usgs_fn)

    # Convert that xarray for consistency
    ds=xr.Dataset()
    ds['time']=( ('time',), usgs_data['datetime'])


    for key in usgs_data.keys():
        if key=='datetime':
            continue
        data=usgs_data[key]

        # attempt to find better name for the columns
        varname=key # default to original name

        m=re.match(r'(\d+)_(\d+)_(\d+)(_cd)?$',key)
        meta={}
        if m:
            meta['ts_code']=m.group(1)
            meta['parm_code']=m.group(2)
            meta['stat_code']=m.group(3)
            # print("ts: %s  parm: %s  stat: %s"%(meta['ts_code'],
            #                                     meta['parm_code'],
            #                                     meta['stat_code']))
            parameter=rdb_codes.parm_code_lookup(meta['parm_code'])
            if parameter is not None:
                # parm_nm is often really long!
                # srsname, when present, is shorter
                srsname=parameter['srsname']
                if srsname:
                    varname=srsname
                else:
                    varname=parameter['parameter_nm']
                varname=varname.lower().replace(' ','_').replace(',','').replace('.','')

                meta['units']=parameter['parameter_units']
            statistic=rdb_codes.stat_code_lookup(meta['stat_code'])
            if statistic is not None:
                meta['statistic']=statistic['name']

            if m.group(4) is not None:
                # TODO: save QA codes
                continue 

        if isinstance( usgs_data[key], np.ndarray ):
            if len(data)==len(ds.time):
                ds[varname] = ( ('time',), data)
            else:
                print("What to do with %s"%key)

            for k in meta:
                ds[varname].attrs[k]=meta[k]

        else: # probably should be an attribute
            ds.attrs[key]=data

    return ds
    
