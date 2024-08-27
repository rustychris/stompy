import pandas as pd
import numpy as np

def parse_txt(filename):
    with open(filename,'rt') as fp:
        fields= fp.readline().strip('#').split()
        units = fp.readline().strip('#').split()
    data= pd.read_csv(filename, skiprows=2, names=fields, sep='\\s+',
                      parse_dates={ 'time':['YY','MM','DD','hh','mm'] },
                      date_format='%Y %m %d %H %M')

    no_data = [ ('WTMP',999.0) ]
    for fld,nan_value in no_data:
        if fld in data.columns:
            values = data[fld].values
            values[values==nan_value] = np.nan
            data[fld] = values
    return data
