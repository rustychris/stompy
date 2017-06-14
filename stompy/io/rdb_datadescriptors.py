# 2017-06-13: This file seems extraneous, in light of rdb_physical_codes.rdb

# a catalog of data descriptors for USGS rdb records


dd_dict = {
    '00060':['Discharge [cfs]'],
    '00095':['Conductance',"Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius"],
    '00065':['Gage height [ft]'],
    '63680':['Turbidity','Turbidity, water, unfiltered, monochrome near infra-red LED light, 780-900 nm, detection angle 90 +/ -2.5 degrees, formazin nephelometric units (FNU)']
    }

import re

def dd_to_synonyms(code):
    # print "Looking for synonyms of ",code
    m = re.match('\d\d_(.+)',code) 
    if m:
        short_code = m.group(1)
        # print "Got a short code ",short_code
    else:
        short_code = None
    # try straight lookup:
    if code in dd_dict:
        return dd_dict[code]
    elif short_code and short_code in dd_dict:
        return dd_dict[short_code]
    elif (code+"_cd") in dd_dict:
        return map(lambda s: s+" (extra code)",dd_dict[code])
    else:
        return []
    
