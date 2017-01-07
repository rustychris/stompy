# see how involved a NEFIS reader in native python/numpy would be
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import memoize

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
    def from_nef_map(nef):
        xcor=nef['map-const'].getelt('XCOR',[0])
        ycor=nef['map-const'].getelt('YCOR',[0])
        xz=nef['map-const'].getelt('XZ',[0])
        yz=nef['map-const'].getelt('YZ',[0])
        active=nef['map-const'].getelt('KCS',[0])
        return DflowGrid2D(xcor,ycor,xz,yz,active)

    def wireframe(self):
        # this method double-draws interior lines.
        coll=self.pcolor_face(0*self.xcor)
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

    @memoize.memoize()
    def to_triangulation(self):
        # map 2d node index to 1d index - not yet compressed.
        idx1d=np.arange(np.prod(self.xcor.shape)).reshape(self.xcor.shape)

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
        

