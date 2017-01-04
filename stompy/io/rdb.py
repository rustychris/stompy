from __future__ import print_function
from future import standard_library
from functools import reduce
standard_library.install_aliases()
from builtins import zip
from builtins import map
from builtins import range
from builtins import object
# python import routines for rdb (USGS tab delimited) format

import re,string,time
import datetime
from matplotlib.dates import date2num,num2date

import rdb_codes
import six

import io

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from numpy import *
from rdb_datadescriptors import dd_to_synonyms


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
            return NaN
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
                columns[i] = array(list(map(self.float_or_nan,columns[i])))
            elif specs[i][-1] == 'd':
                # dates get mapped to days since AD 0
                columns[i] = array(list(map(self.parse_date,columns[i])))

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
