from __future__ import print_function
# reads in tide_consts.txt and sets basically the same info (a subset) of
# what tide_data.py sets

# lun_node
# v0u
# omegas
# const_names

import os, os.path
import numpy as np

fp = open(os.path.join(os.path.dirname(__file__),'tide_consts.txt'))

def tokenize(fp):
    """ generator for tokenizing files
    """
    for line in fp:
        line = line.strip()
        if line[0] == '#':
            continue
        for tok in line.split():
            yield tok

# awkward PY2/3 compat
gen=tokenize(fp)
gettok = lambda: next(gen)

nconsts = int(gettok())

const_names = [None]*nconsts
speeds = np.zeros( (nconsts,) ) # degrees per hour

for i in range(nconsts):
    const_names[i] = gettok()
    speeds[i] = float(gettok())


# equilibrium arguments:
start_year = int( gettok() )

def read_block():
    num_years =  int( gettok() )
    data = np.zeros( (nconsts,num_years), np.float64 )
    
    for i in range(nconsts):
        name = gettok()
        if name != const_names[i]:
            raise Exception("Expected %s, got %s"%(const_names[i],name))
        for year_offset in range(num_years):
            data[i,year_offset] = float(gettok())

    end = gettok()
    if end != '*END*':
        raise Exception("Expected *END*, got %s"%end)
    return data

v0u = read_block()
lun_nodes = read_block()

if lun_nodes.shape[1] != v0u.shape[1]:
    print("count of lunar nodes (%i) differs from equ. arguments (%i)"%(lun_nodes.shape[1],
                                                                        v0u.shape[1]))

num_years = min(lun_nodes.shape[1],v0u.shape[1])
years = np.arange(start_year,start_year+num_years)

        
        
