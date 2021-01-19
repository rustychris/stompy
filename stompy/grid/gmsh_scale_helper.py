"""
Handle scale information for a gmsh process.
Expects a single command line argument giving the path to a
pickled Field instance.

This probably doesn't work with python2 (gmsh docs provide some
additional options needed for python2)
"""
from stompy.spatial import field
import pickle
import struct
import sys
import math

pkl_file=sys.argv[1]
with open(pkl_file,'rb') as fp:
    scale=pickle.load(fp)
    
while True:
    xyz = struct.unpack("ddd", sys.stdin.buffer.read(24))
    if math.isnan(xyz[0]):
        break
    f = scale(xyz[:2])
    sys.stdout.buffer.write(struct.pack("d",f))
    sys.stdout.flush()
