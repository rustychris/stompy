# see how involved a NEFIS reader in native python/numpy would be
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# import memoize

## 
# os.chdir('/Users/rusty/research/sfei/models/delft/delft3d_repository/examples/01_standard')
d3d_libdir=os.path.join(os.environ['HOME'],
                        "code/delft/d3d/master/src/lib/")

nefis_lib=os.path.join(d3d_libdir,'libNefisSO')

if sys.platform.startswith('linux'):
    nefis_lib+='.so'
else:
    nefis_lib+=".dylib"


from ctypes import * 

nef=cdll.LoadLibrary(nefis_lib)

# # 

if 0:
    data_dir='/home/rusty/models/delft/tutorial/waq/friesian_tidal_inlet/all_flow_output'
    dat_fn=os.path.join(data_dir,'trih-tut_fti_waq.dat')
    def_fn=os.path.join(data_dir,'trih-tut_fti_waq.def')
elif 1:
    data_dir='/home/rusty/models/delft/tutorial/waq/friesian_tidal_inlet/all_flow_output'
    dat_fn=os.path.join(data_dir,'trim-tut_fti_waq.dat')
    def_fn=os.path.join(data_dir,'trim-tut_fti_waq.def')
else:
    dat_fn="trih-f34.dat"
    def_fn="trih-f34.def"


# # 
fd_nefis=c_int()
fd_nefis.value=0
endian=c_char()
endian.value="N"
access=c_char()
access.value="r"

# first param is "BInt4 *fd"  - how to pass that in ctypes??
err=nef.Crenef(byref(fd_nefis), dat_fn, def_fn,byref(endian), access)
if err:
    print(err)
    nef.nefis_error(1,None)

# # 

# get basic header info:
# init as long, null-terminated, so nefis knows it's long enough
header_buff=create_string_buffer(" "*129)
err=nef.Gethdf(byref(fd_nefis),byref(header_buff))
# => 'Deltares, NEFIS Definition File; 5.00.00'
print(header_buff.value)

if err:
    print(err)
    nef.nefis_error(1,None)

## 

grpnam=create_string_buffer(" "*16)
celnam=create_string_buffer(" "*16)
grpndm=c_int(5) # 5 is max - will be set to actual
grpdms=(c_int*5)()
grpord=(c_int*5)()

# try to read the first/next group from definition file

allocs=[]

# # nef.Inqfgr.argtypes=[c_void_p,c_char_p,c_char_p,c_void_p,c_void_p]
# for ii in range(2):
#     print "-"*20
#     err = nef.Inqfgr(byref(fd_nefis),
#                      byref(grpnam),
#                      byref(celnam),
#                      byref(grpndm),
#                      byref(grpdms),
#                      byref(grpord))
#     if err:
#         print err 
#         nef.nefis_error(1,None)
# 
#     print "  Group: %s   cell: %s"%(grpnam.value,celnam.value)
#     dims=",".join( ["%d"%grpdms[dim_i] for dim_i in range(grpndm.value)] )
#     print "  dims: ",dims
# 
#     # only crashes when there are some allocations going on.
#     allocs.append(np.zeros(10000))
# 
# print "DONE - okay"
## 

# try again, but using numpy arrays

grpnam=create_string_buffer(" "*16)
celnam=create_string_buffer(" "*16)
grpndm=c_int(5) # 5 is max - will be set to actual
grpdms=np.ones(10,'i4')
grpord=np.ones(10,'i4')

# try to read the first/next group from definition file

allocs=[]

err = nef.Inqfgr(byref(fd_nefis),
                 byref(grpnam),
                 byref(celnam),
                 byref(grpndm),
                 grpdms.ctypes,
                 grpord.ctypes)
if err:
    print(err)
    nef.nefis_error(1,None)

print("  Group: %s   cell: %s"%(grpnam.value,celnam.value))
dims=",".join( ["%d"%grpdms[dim_i] for dim_i in range(grpndm.value)] )
print("  dims: ",dims)
ords=",".join( ["%d"%grpord[dim_i] for dim_i in range(grpndm.value)] )
print("  order: ",ords)

## 

# as a function:
def inqfgr(fd):
    grpnam=create_string_buffer(" "*16)
    celnam=create_string_buffer(" "*16)
    grpndm=c_int(5) # 5 is max - will be set to actual
    grpdms=np.ones(10,'i4')
    grpord=np.ones(10,'i4')

    err = nef.Inqfgr(byref(fd),
                     byref(grpnam),
                     byref(celnam),
                     byref(grpndm),
                     grpdms.ctypes,
                     grpord.ctypes)
    if err:
        print(err)
        nef.nefis_error(1,None)

    print("  Group: %s   cell: %s"%(grpnam.value,celnam.value))
    dims=",".join( ["%d"%grpdms[dim_i] for dim_i in range(grpndm.value)] )
    print("  dims: ",dims)
    ords=",".join( ["%d"%grpord[dim_i] for dim_i in range(grpndm.value)] )
    print("  order: ",ords)

def inqngr(fd):
    grpnam=create_string_buffer(" "*16)
    celnam=create_string_buffer(" "*16)
    grpndm=c_int(5) # 5 is max - will be set to actual
    grpdms=np.ones(10,'i4')
    grpord=np.ones(10,'i4')

    err = nef.Inqngr(byref(fd),
                     byref(grpnam),
                     byref(celnam),
                     byref(grpndm),
                     grpdms.ctypes,
                     grpord.ctypes)
    if err:
        if err==-6028:
            print("End of groups")
            return
        print(err)
        nef.nefis_error(1,None)
        return None

    print("  Group: %s   cell: %s"%(grpnam.value,celnam.value))
    def fmt(l):
        if l:
            return "%d"%l
        else:
            return "0 (unlimited)"
    dims=",".join( [fmt(grpdms[dim_i]) for dim_i in range(grpndm.value)] )
    print("  dim lengths: ",dims)
    ords=",".join( ["%d"%grpord[dim_i] for dim_i in range(grpndm.value)] )
    print("  order: ",ords)
    
    return 1

# Check next group:
def group_summary(fd):
    inqfgr(fd)
    while inqngr(fd):
        pass

## 

def inquire_elt(fd,elt_name):
    # finally, need to go through the element definitions
    elmnam=(c_char*17)() ; elmnam.value=elt_name #  create_string_buffer("U1")
    elmtyp=create_string_buffer(" "*9)
    nbytsg=c_int(0)

    elmqty=(c_char*17)() ; elmqty.value=" "*16 # create_string_buffer(" "*17)
    elmunt=(c_char*17)() ; elmunt.value=" "*16 # create_string_buffer(" "*17)
    desc=(c_char*65)() ; desc.value=" "*64 # create_string_buffer(" "*65)
    elmndm=c_int(5)
    elmdms=(c_int*5)()

    err=nef.Inqelm(byref(fd),byref(elmnam), # inputs
                   byref(elmtyp), # outputs
                   byref(nbytsg),
                   byref(elmqty),byref(elmunt),
                   byref(desc),
                   byref(elmndm),
                   byref(elmdms))
    if err:
        print(err)
        nef.nefis_error(1,None)
        return

    print("     type: {}".format( elmtyp.value ))
    print("     nbytes/single element: {}".format( nbytsg.value ))
    print("     quantity {}  unit {}".format( elmqty.value,elmunt.value ))
    print("     desc: {}".format( desc.value ))
    print("     dimensions: {}".format( ",".join( [str(elmdms[i]) for i in range(elmndm.value)] ) ))

# inquire_elt(fd_nefis,"U1")

# what about cells?

def inqfcl(fd,describe_elements=False):
    celnam=create_string_buffer(" "*16)

    # tell it how many elements are allocated:
    nelems_alloc=100
    nelems=c_int(nelems_alloc) 

    bytes=c_int(0)
    elmnms=((c_char*17)*nelems_alloc)()
    
    err = nef.Inqfcl(byref(fd),
                     byref(celnam),
                     byref(nelems),
                     byref(bytes),
                     byref(elmnms))
                     
    if err:
        print(err)
        nef.nefis_error(1,None)

    print("Cell name: {}, bytes: {}".format(celnam.value,bytes.value))
    print("  elements: ")
    for elem_idx in range(nelems.value):
        # nefis doesn't null terminate - so drop last byte.
        print("   Element '{}'".format(elmnms[elem_idx].value[:16]))
        if describe_elements:
            inquire_elt(fd,elmnms[elem_idx].value[:16])

def inqncl(fd,describe_elements=False):
    celnam=create_string_buffer(" "*16)

    # tell it how many elements are allocated:
    nelems_alloc=100
    nelems=c_int(nelems_alloc) 

    bytes=c_int(0)
    elmnms=((c_char*17)*nelems_alloc)()
    
    err = nef.Inqncl(byref(fd),
                     byref(celnam),
                     byref(nelems),
                     byref(bytes),
                     byref(elmnms))
                     
    if err:
        if err == -6026:
            print("End of cells")
        else:
            print(err)
            nef.nefis_error(1,None)
        return None

    print("Cell name: {}, bytes: {}".format(celnam.value,bytes.value))
    print("  elements: ")
    for elem_idx in range(nelems.value):
        # nefis doesn't null terminate - so drop last byte.
        print("   Element {}".format(elmnms[elem_idx].value[:16]))
        if describe_elements:
            inquire_elt(fd,elmnms[elem_idx].value[:16])
    return 1

# Check next group:
def cell_summary(fd,describe_elements=False):
    inqfcl(fd,describe_elements=describe_elements)
    while inqncl(fd,describe_elements=describe_elements):
        pass

## 
if 1:
    group_summary(fd_nefis)
    cell_summary(fd_nefis,True)
## 

# find out length of map-series group
grpnam=create_string_buffer("map-series       ",17)
maxi=c_int(-1)
err=nef.Inqmxi(byref(fd_nefis),byref(grpnam),byref(maxi))
if err:
    print(err)
    nef.nefis_error(1,None)
print("Maximum index: ",maxi.value)

## 
# And finally read some data.
# try for U1

# group name and cell name are map-series

grp_name=create_string_buffer("map-series")
elm_name=create_string_buffer("U1")
ncells=3

# already know that map-series has 1 dimension, and it's unlimited.
grpndm=1
if 1:
    uindex=np.ones((grpndm,3),'i4')
    uindex[0,0]=7 # starting index - 1-based??
    uindex[0,1]=uindex[0,0]+ncells-1 # ending index, inclusive
    uindex[0,2]=1 # stride
    uindex=np.ascontiguousarray(uindex)
    uindex_ref=uindex.ctypes

if 1:
    uorder=np.arange(grpndm,dtype='i4')+1
    uorder=np.ascontiguousarray(uorder)
    uorder_ref=uorder.ctypes

# from cell definition:
cell_dims=[22,15,10]
itemsize=4
buflen=c_int(itemsize*np.prod(cell_dims)*ncells) # the two is in case ending index is inclusive
if 1:
    data=np.ones(np.prod(cell_dims)*ncells,'f4') # (2,) + tuple(cell_dims)
    data=np.ascontiguousarray(data)
    data_ref=data.ctypes
else:
    data=(c_float*(np.prod(cell_dims)*ncells))()
    data_ref=byref(data)


print("Buffer length: %d"%(buflen.value))
# #  
print("HOLD YER BREATH")
err = nef.Getelt(byref(fd_nefis),
                 byref(grp_name), byref(elm_name),
                 uindex_ref, uorder_ref,
                 byref(buflen),
                 data_ref)
print("WELL??")
if err:
    print(err)
    nef.nefis_error(1,None)
else:
    print("Appears successful")

## 

# look at how data was actually written:
# with uindex=[ [0,2,1] ] - we get two time steps of data.
#      uindex=[ [1,2,1] ] - also get two time steps of data.
#      uindex=[ [1,1,1] ] - get *one* time step of data

# so the order given by the cell info is fortran order - have to 
# reverse for python.

u1=data.reshape( ncells,10,15,22 ) # cell_dims

u1_frame=u1[0]

# presumably this is [x-ish,y-ish,z] - nope.

plt.clf()
plt.imshow(u1_frame[1,:,:],aspect='auto',interpolation='nearest',cmap='RdBu',
           vmin=-0.2,vmax=0.2)
plt.colorbar()

## 

# And what about some geometry?
# that's in group map-const, XCOR/YCOR for nodes, XZ/YZ for centers
# also has some mask arrays


## 

import nefis
import test_nefis
reload(nefis)
reload(test_nefis)

nef=test_nefis.get_nef()
try:
    u1=nef['map-series'].getelt('U1')
finally:
    nef.close()

## 
plt.clf()
for t in range(11):
    plt.imshow(u1[t,9],interpolation='none',aspect='auto',
               vmin=-0.5,vmax=0.5,cmap='RdBu')
    plt.pause(0.2)

## 

# And what about some geometry?
# that's in group map-const, XCOR/YCOR for nodes, XZ/YZ for centers
# also has some mask arrays

import nefis
import test_nefis
reload(nefis)
reload(test_nefis)

nef=test_nefis.get_nef()

## 
u1=nef['map-series'].getelt('U1',[10])
rho=nef['map-series'].getelt('RHO',[10])
xcor=nef['map-const'].getelt('XCOR',[0])
ycor=nef['map-const'].getelt('YCOR',[0])
dp0=nef['map-const'].getelt('DP0',[0])
dps0=nef['map-const'].getelt('DPS0',[0])

xz=nef['map-const'].getelt('XZ',[0])
yz=nef['map-const'].getelt('YZ',[0])

active=nef['map-const'].getelt('KCS',[0])
# nef.close()

## 

plt.figure(2).clf()
fig,axs=plt.subplots(2,2,sharex=True,sharey=True,num=2)

# dp0 has the same masking as xcor - presumably dp0 is defined
# at the nodes.
axs[0,0].imshow(dp0,aspect='auto',interpolation='none')
axs[0,1].imshow(dps0,aspect='auto',interpolation='none')
axs[1,0].imshow(active,aspect='auto',interpolation='none')
axs[1,1].imshow(xcor,aspect='auto',interpolation='none')
## 


xcor_m=xcor.copy()
ycor_m=ycor.copy()
xcor_m[ xcor==0]=np.nan
ycor_m[ ycor==0]=np.nan
xcor_m=np.ma.masked_invalid(xcor_m)
ycor_m=np.ma.masked_invalid(ycor_m)


# Masking -
# all arrays are [15,22]
# that's 2 more than the actual number of cell/volumes
# from the delwaw tutorial grid, appears that the actual grid has 20x13 cells.
# FLOW documentation suggests that 

# xcor doesn't have any data for the last row or column
# so assume that the corners of the lower-left cell are xcor[{0,1},{0,1}] 

dp0_m=dp0.copy()
dp0_m[ dp0==-999 ] =np.nan
dp0_m=np.ma.masked_invalid(dp0_m)

plt.figure(1).clf()
mesh=plt.pcolormesh( xcor_m[:-1,:-1],
                     ycor_m[:-1,:-1],
                     dp0_m[:-1,:-1],
                     edgecolor='k',lw=0.2)
plt.axis('equal')

plt.axis( (165934.04627395154, 233318.9728923004, 586053.18933475052, 639279.13222990348) )


## 

# plot each version of depth as a scatter - make sure things are
# consistent
plt.figure(3).clf()
mesh=plt.pcolormesh( xcor_m[:-1,:-1],
                     ycor_m[:-1,:-1],
                     dp0_m[:-1,:-1],
                     facecolors='none',
                     edgecolors='k',lw=0.2)
mesh.set_array(None)

plt.scatter(xcor[:-1,:-1],ycor[:-1,:-1],40,dp0[:-1,:-1],
            vmin=0,vmax=5,lw=0)

plt.scatter(xz,yz,100,active,lw=0)

plt.scatter(xz,yz,40,dps0,
            vmin=0,vmax=5,lw=0)

plt.axis('equal')
plt.axis( (165934.04627395154, 233318.9728923004, 586053.18933475052, 639279.13222990348) )


# I think z means zeta means center.

# active corresponds to cells/volumes, and xz/yz are set for all cells
# with nonzero active.



# active==2 for 'ghost' cells - i.e. cells outside the domain adjacent to
# a water level boundary.


# Best ways to plot filled scalar field?

# plt.contourf(xcor_m,ycor_m,dp0_m)

# Make it into a bunch of triangles
# don't worry about which way the diagonal inside each quad
# goes.

# at this point, don't dwell on the ultimate solution (which might
# go towards https://publicwiki.deltares.nl/display/NETCDF/Deltares+proposal+for+Staggered+Grid+data+model+(SGRID)
# or Filipe/Rich's sgrid convention.

## 
import matplotlib.tri as tri

class DflowGrid2D(object):
    def __init__(self,xcor,ycor,xz,yz,active):
        self.xcor=xcor
        self.ycor=ycor
        self.xz=xz
        self.yz=yz
        self.active=active
        comp=(active==1) # computational cells
        self.node_active=comp.copy()
        self.node_active[:-1,:-1] = comp[1:,1:] | comp[:-1,1:] | comp[1:,:-1] | comp[:-1,:-1]

    @staticmethod
    def from_nef_map(self):
        xcor=nef['map-const'].getelt('XCOR',[0])
        ycor=nef['map-const'].getelt('YCOR',[0])
        xz=nef['map-const'].getelt('XZ',[0])
        yz=nef['map-const'].getelt('YZ',[0])
        active=nef['map-const'].getelt('KCS',[0])
        return DflowGrid2D(xcor,ycor,xz,yz,active)

    def wireframe(self):
        # this method double-draws interior lines.
        coll=self.face_centered(0*self.xcor)
        coll.set_array(None)
        coll.set_facecolors('none')
        return coll

    def pcolor_face(self,v):
        vmsk=np.ma.masked_where(self.active!=1,v)
        return plt.pcolor( self.xcor[:-1,:-1],
                           self.ycor[:-1,:-1],
                           vmsk[1:-1,1:-1],
                           edgecolors='k',lw=0.2)

    def pcolormesh_node(self,v):
        vmsk=np.ma.masked_where(~self.node_active,v)
        return plt.pcolormesh( self.xcor[:-1,:-1],
                               self.ycor[:-1,:-1],
                               vmsk[:-1,:-1],
                               edgecolors='k',lw=0.2,
                               shading='gouraud')

    # @memoize.memoize()
    def to_triangulation(self):
        # map 2d node index to 1d index - not yet compressed.
        idx1d=arange(np.prod(self.xcor.shape)).reshape(self.xcor.shape)

        tris=[]
        comp=(self.active==1)[1:-1,1:-1]
        for row,col in zip(*np.nonzero(comp)):
            # may not be CCW
            tris.append( [ idx1d[row,col],
                           idx1d[row,col+1],
                           idx1d[row+1,col+1] ] )
            tris.append( [ idx1d[row+1,col+1],
                           idx1d[row+1,col],
                           idx1d[row,col] ] )

        used=np.unique(tris)
        x=self.xcor.ravel()[used]
        y=self.ycor.ravel()[used]
        remap=np.zeros(np.prod(self.xcor.shape))
        remap[used] = np.arange(len(used))
        orig_tris=tris
        tris=remap[tris] # these now index 

        T=tri.Triangulation(x,y,tris)
        idx_map=used

        return T,idx_map
    def pcolor_node(self,v,**kws):
        my_kws=dict(shading='gouraud')
        my_kws.update(kws)
        T,idx_map = self.to_triangulation()
        return plt.tripcolor(T,v.ravel()[idx_map],**my_kws)
    def contourf_node(self,v,*args,**kws):
        T,idx_map = self.to_triangulation()
        return plt.tricontourf(T,v.ravel()[idx_map],*args,**kws)
        

g=DflowGrid2D.from_nef_map(nef)
plt.clf()
# g.face_centered(rho[1])
# g.wireframe()
# coll=g.pcolor_node(dp0)
cnt=g.contourf_node(dp0,50)
plt.axis('equal')
plt.axis( (165934.04627395154, 233318.9728923004, 586053.18933475052, 639279.13222990348) )

## 

