import os
from .. import utils
from scipy import sparse
from scipy import ndimage

from pylab import *

#############

class DemToGraph(object):
    def __init__(self,dem,max_elevation=inf,degree=16,forced=None,regional_factors=[],**kwargs):
        """
        dem: a field.SimpleGrid
        max_elevation: positive up value, above which
          cells are considered dry and discarded
        degree: how connected nodes are -
           4 would be cardinal directions (e.g. N)
           8 adds ordinal directions (e.g. NW)
           16 adds knights-jump directions
        regional_factors: list of [ ((xmin,xmax,ymin,myax),factor), ...]
           where all edges whose centers fall within the bounding box will have
           their 'admittance' scaled by the given factor.  use factor in [0,1]
           to 'slow down' diffusion or make distances appear longer.
        """
        self.dem=dem
        self.max_elevation=max_elevation
        self.degree=degree
        self.forced=forced
        self.regional_factors=regional_factors
        self.K_j = None # will be constructed as a constant if not supplied.
        
        # this is reasonable for thalwegs, but for extrapolation 
        # should remove the depth dependence
        self.weight_function=lambda d_j,K_j,edge_depths: d_j/(K_j*edge_depths**2)

        self.__dict__.update(**kwargs)
        
        self.process()
        
    def init_F(self):
        self.F=F=self.dem.F.copy()
        
        F[F>self.max_elevation] = nan # down to 387k
        eps=0.1 # shot in the dark fixing a bug
        if self.max_elevation>-eps:
            # code below assumes that "valid" cells are at or below 0.0
            self.F -= self.max_elevation+eps
        mask=isfinite(F) # label it 
        labels,Nlabels=ndimage.label(mask)

        comps=labels[ self.forced_dem_idx[:,0], self.forced_dem_idx[:,1] ]
        if comps[0] != comps[1]:
            # for the moment, ignore that SJ is not connected
            raise Exception("GG and Sac are not in the same connected component")

        F[labels!=comps[0]] = nan # down to 368k

    def init_stencils(self):
        # convolve different bitmaps over valid to get the
        # edges:
        self.kerns=kerns=[]
        self.offsets=offsets=[]

        if self.degree>=4:
            # # vertical edges
            kerns.append( array([[1],[1]]) )
            offsets.append( array( [ [0,0],[1,0] ] ) )

            # horizontal edges:
            kerns.append( array([[1,1]]) )
            offsets.append( array( [ [0,0],[0,1] ] ) )
        if self.degree>=8:
            # NE dense diagonals
            kerns.append( array([[1,1],[1,1]]) )
            offsets.append( array( [ [0,0],[1,1] ] ) )

            # NW dense diagonals
            kerns.append( array([[1,1],[1,1]]) )
            offsets.append( array( [ [0,1],[1,0] ] ) )
        if self.degree>=16:
            # Knight jumps - apparently the filter origin is chosen at the
            # center, rounding toward zero.  So for 2,1 or 2,2 filters, it looks
            # like row=0,col=0 is the center of the filter, but here it's offset
            kerns.append( array([[1,0],
                                 [1,1],
                                 [0,1]]) )
            offsets.append( array( [ [-1,0],[1,1] ] ) ) # should be [0,0],[2,1]

            kerns.append( array([[0,1],
                                 [1,1],
                                 [1,0]]) )
            offsets.append( array( [ [-1,1],[1,0] ] ) )

            kerns.append( array([[1,1,0],
                                 [0,1,1]] ))
            offsets.append( array( [ [0,-1],[1,1] ] ) )

            kerns.append( array([[0,1,1],
                                 [1,1,0]] ))
            offsets.append( array( [ [1,-1],[0,1] ] ) )
        
    def find_edges(self):
        all_A=[]
        all_B=[]
        all_depth=[]
        for kern,offset in zip(self.kerns,self.offsets):
            conv=ndimage.correlate(self.depth,kern,mode='constant',cval=1e20) 
            # so if edges[r,c] is true, then there should be an edge from
            #  dem.F[r+offsets0[0,0],c+offsets0[0,1]] to
            #  dem.F[r+offsets0[1,0],c+offsets0[1,1]]

            # the matches for the kernel:
            match=conv<0
            mrow,mcol=nonzero(match)
            Arow=mrow-offset[0,0]
            Acol=mcol-offset[0,1]
            Brow=mrow-offset[1,0]
            Bcol=mcol-offset[1,1]

            if 1:
                if (Arow.min()<0) or (Arow.max()>=self.depth.shape[0]):
                    raise Exception("Bad Arows")
                if (Acol.min()<0) or (Acol.max()>=self.depth.shape[1]):
                    raise Exception("Bad Acols")
                if (Brow.min()<0) or (Brow.max()>=self.depth.shape[0]):
                    raise Exception("Bad Brows")
                if (Bcol.min()<0) or (Bcol.max()>=self.depth.shape[1]):
                    raise Exception("Bad Bcols")

                if not all( self.valid[Arow,Acol] ):
                    raise Exception("Bad A truth")
                if not all( self.valid[Brow,Bcol] ):
                    raise Exception("Bad B truth")

            Aidx=self.idxs[Arow,Acol]
            Bidx=self.idxs[Brow,Bcol]

            if any(Aidx<0):
                raise Exception("Some A indices missed")
            if any(Bidx<0):
                raise Exception("Some B indices missed")

            all_A.append(Aidx)
            all_B.append(Bidx)
            all_depth.append( conv[match]/sum(kern) ) #  assigns mean depth
            # would be nice to use min depth, but that gets complicated and slow

        self.c1=concatenate(all_A)
        self.c2=concatenate(all_B)
        self.edge_depths=concatenate(all_depth)

        if any(self.c1<0):
            raise Exception("Bad c1 values")
        if any(self.c2<0):
            raise Exception("Bad c2 values")
        return self.c1,self.c2,self.edge_depths

    def process(self):
        self.forced_dem_idx=self.dem.point_to_index(self.forced).round().astype(int32)
        
        self.init_F()

        self.valid=valid=isfinite(self.F)
        self.depth=depth=self.F.copy()
        depth[ isnan(depth) ] = 1e20 # valid values are small and negative

        # map row,col to a linear cell index:
        self.idxs=idxs=zeros(depth.shape,int32)-1
        idxs[ depth<=0 ] = arange(np.sum(depth<=0 ))

        self.forced_lin_idx=forced_lin_idx=idxs[self.forced_dem_idx[:,0],self.forced_dem_idx[:,1]]

        self.init_stencils() # define connectivity for edges

        c1,c2,edge_depths = self.find_edges()

        # centers of dem cells:
        X,Y=self.dem.XY()
        x=X[valid]
        y=Y[valid]
        self.vc=vc=array([x,y]).T
 
        self.d_j=utils.dist( vc[c1] - vc[c2] )
        self.ec=ec=0.5*(vc[c1]+vc[c2])

        self.build_matrix()
        return self.sg

    def build_matrix(self):
        """
        these steps separated out so the K_j vector can be modified, and the
        graph rebuilt
        """
        Ncells=len(self.vc)
        if self.K_j is None:
            self.K_j=K_j=100*ones(len(self.d_j),np.float64)
            for xxyy,factor in self.regional_factors:
                sel=(self.ec[:,0]>xxyy[0])&(self.ec[:,0]<xxyy[1])&(self.ec[:,1]>xxyy[2])&(self.ec[:,1]<xxyy[3])
                K_j[sel] *= factor
        else:
            K_j = self.K_j

        weights=self.weight_function(self.d_j,K_j,self.edge_depths)

        ij=array([self.c1,self.c2]).T

        self.weights=weights
        self.ij=ij

        ij_undirected=concatenate( (ij,ij[:,::-1]) )
        weights_undirected=concatenate( (weights,weights) )
        self.sg=sparse.coo_matrix( (weights_undirected,(ij_undirected[:,0],ij_undirected[:,1])), (Ncells,Ncells) )
        return self.sg

